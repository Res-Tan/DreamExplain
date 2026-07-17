# World Model Decision Explanation — Design Spec

## 0. Environment & Execution Paths

- **Single machine (`matsutake`), everything lives here now**: repo root is
  `/home/tanyixin/worldmodel/` on host `matsutake` (user `tanyixin`). This is a
  direct-access multi-GPU server (10 GPUs visible via `nvidia-smi`: RTX A6000 /
  RTX 6000 Ada) — there is no separate edit-locally/run-remotely split anymore
  and no job queue in front of the GPUs (`qsub`/`qstat` binaries exist under
  `/opt/pbs/bin` but the scheduler is not reachable — `qstat` returns
  "connection refused" — so just run jobs directly, no `qrsh`/`qsub` needed).
  This file lives at `method/DreamExplain/readme.md`.
- **DreamerV3 codebase**: `dreamerv3/`, sibling of `method/`
  (`/home/tanyixin/worldmodel/dreamerv3`). This is what `rollout.py` (§10)
  wraps to get `π_θ`, the RSSM, and the reward/continue/critic heads.
- **Trained checkpoints (local, migrated from the previous server together
  with the rest of this repo)**:
  `/home/tanyixin/worldmodel/models/<task>_<timestamp>/`. Directory names
  encode task + training start time; each run dir contains a `latest` file
  (pointer/marker, not a checkpoint itself) plus a timestamped subdir (e.g.
  `20260704T103718F585051/`) holding the actual checkpoint. For each task, the
  run with the latest timestamp is the checkpoint to load for this method. As
  of 2026-07-14, the available runs (one per task so far):

  | Task | Run dir |
  |---|---|
  | `atari_pong` | `atari_pong_20260703_105122` |
  | `crafter` | `crafter_20260704_130112` |
  | `dmc_walker_walk` | `dmc_walker_walk_20260706_113519` |

  Re-run `ls /home/tanyixin/worldmodel/models/` before use — once more runs
  accumulate per task, pick the one with the latest `<timestamp>` in the
  dirname.
- **Disk note**: `/home` is at 98% capacity (5.7T/5.9T used, ~179G free) as of
  2026-07-14 — be mindful of checkpoint/log growth when running new training or
  large diagnostic sweeps (§8).
- **Tip**: running multiple tasks concurrently across GPUs can cause tiny
  numerical drift in results (results.md 2026-07-15) — negligible for
  aggregate conclusions, use concurrent GPUs freely; only worth re-running a
  task solo if a specific cited number needs to match exactly.
- **No sync step needed anymore**: since code, `dreamerv3/`, and `models/` are
  all local on `matsutake`, the old rsync/scp pull-back-to-Mac step is gone.
  The explanation method (rollout, masking, search, diagnostic experiments in
  §10) runs directly against checkpoints under `models/` on this machine.

## 1. Problem Statement

We want to explain the decisions of a **DreamerV3-based planner** at inference time.
Given a current internal state `h0`, the planner considers `N` candidate actions
`c1, ..., cN` (e.g., the top-N modes/samples from the actor's action distribution
`π_θ(· | h0)`, including the actually-selected action).

For each candidate `c_i`, the world model (RSSM) rolls out an imagined trajectory of
length `T`:

```
Y_i = { h_{i,t}, r̂_{i,t}, û_{i,t}, d̂_{i,t} }_{t=1..T}
```

- `h_{i,t}`: imagined latent state (RSSM deterministic + stochastic state)
- `r̂_{i,t}`: predicted reward (reward head)
- `û_{i,t}`: predicted value / utility (critic; note — in many λ-return
  formulations this is only used as a bootstrap at the final step, not per-step —
  **must confirm which form `G(·)` uses before implementing masking on it**)
- `d̂_{i,t}`: predicted continuation probability (continue head, replaces
  done/discount flag)

A return operator `G(·)` (λ-return style aggregation) turns each trajectory into a
scalar decision score `J_i = G(Y_i)`, giving the preference vector
`j = [J1, ..., JN]`. The model's actual decision is `argmax_i J_i`.

**Goal:** find a compact set of *shared* temporal segments `B` (positions in
`1..T`, applied identically across all N candidate trajectories) such that keeping
only the information inside `B` — and removing (masking) everything else — is
*sufficient* to reconstruct the original decision-relevant structure of `j`.

This is a **segment-level, decision-faithful explanation**, not a per-timestep
saliency score and not a per-candidate importance score. The unit of explanation
is a *time segment*, shared across candidates.

---

## 2. Candidate Segment Pool

Multi-resolution segments over the horizon `T`:

```
P = { [p, p+d-1] | d ∈ D, 1 ≤ p ≤ T-d+1 },   D = {1, 2, 4, 8}
```

An explanation is a subset `B ⊆ P` (a set of possibly-overlapping or disjoint
intervals). Define the binary temporal indicator:

```
b_t = 1 if t is covered by some segment in B, else 0
```

`b_t` is **shared across all N candidates** (same mask positions applied to every
trajectory).

---

## 3. Masking (open design question — implement multiple variants)

For each candidate `i` and each scalar signal `y ∈ {r̂, û, d̂}` at time `t`:

```
ỹ_{i,t} = b_t * y_{i,t} + (1 - b_t) * fill(y, i, t)
```

`fill(...)` is the baseline/replacement value for masked-out timesteps. **Do not
hardcode a single baseline** — implement as a pluggable strategy, because the
choice materially affects results. Implement all of the following, selectable via
config:

| Name | Definition | Notes |
|---|---|---|
| `self_mean` | `μ_i(y) = mean_t(y_{i,t})` per-trajectory | Original design. Known issue: self-referential, leaks the trajectory's overall magnitude into the "removed" region, which can artificially preserve ranking/decision without any real segment being informative. Keep only as a baseline-of-baselines for comparison. |
| `global_prior` | Mean of `y` computed over a large offline set of rollouts (population-level, not per-trajectory) | Recommended default. Removes self-leakage. |
| `zero` | `fill = 0` | Only apply to `r̂`. Do **not** apply to `d̂` (see below). |
| `shuffle` | Randomly permute the masked-out values of `y_{i,t}` among themselves (values preserved, order destroyed), or swap in same-timestep values from other candidates | Diagnostic: separates "does the *value* matter" from "does the *timing/order* matter". |
| `counterfactual_reimagine` | Re-run RSSM rollout for the masked region under a fixed/default action (e.g., no-op), replacing that segment's `h_{i,t}` (and downstream heads) with genuinely model-generated content | Most expensive but avoids OOD inputs to `G`; use only for small-scale case studies, not full search. |

### Special handling of `d̂` (continuation)

`d̂` typically participates in `G` recursively/multiplicatively (discount-like
role in λ-return), not additively. **Do not silently apply the arithmetic mean or
zero to `d̂` the same way as `r̂`/`û`.** Implementation requirements:

- Never use `zero` fill for `d̂` (implies forced episode termination at masked
  steps — corrupts all downstream discounting).
  - `global_prior` fill (mean continuation probability from offline data) is the
    safe default for `d̂`.
- Add a unit test that checks the effect of masking `d̂` alone (all else
  unmasked) on `J_i`, to quantify how sensitive `G` actually is to this signal
  before trusting search results involving it.

### `h_{i,t}` (latent state) — must resolve before coding `G`

Determine and document explicitly whether `G(·)`:

(a) is a **pure scalar-sequence function** over stored `{r̂, û, d̂}` (a closed-form
    λ-return formula, no network calls) — in this case `h_{i,t}` never enters `G`
    directly and does not need masking, **or**

(b) requires **re-invoking the critic network** on `h_{i,t}` (e.g., bootstrap value
    at the final step is computed live, not read from the stored rollout) — in
    this case `h_T` (or whichever steps are re-queried) must also be masked/
    replaced, or the "removed" information leaks back in through the network call.

Implement `G` so this is an explicit, inspectable choice (a config flag /
docstring), not implicit behavior buried in code.

### `û` per-step vs. bootstrap-only

Confirm whether `û_t` is used at every timestep in `G`, or only as a terminal
bootstrap value `û_T`. If only terminal, masking intermediate `û_t` is a no-op —
do not let the search algorithm interpret "doesn't matter" for those steps as a
meaningful finding; log/flag this distinction in output.

---

## 4. Decision Re-Evaluation

Recompute masked trajectories `Ỹ_i^B`, re-score:

```
J̃_i(B) = G(Ỹ_i^B)
j̃_B = [J̃_1(B), ..., J̃_N(B)]
```

## 5. Decision-Faithfulness Objective — implement as THREE separate variants, not one blended score

Do not sum D_sel + D_rank + D_margin into a single weighted objective by
default. Implement each as an **independent, separately-optimized objective**
(this was an explicit design decision — see rationale below), plus optionally a
combined variant for comparison.

- **`H_sel(B) = D_sel + λc·R(B)`**
  `D_sel`: binary/soft indicator that `argmax_i J̃_i(B) == argmax_i J_i` (original
  top-1 choice preserved). Answers: *"why did the model pick this action over the
  others?"* — cheapest to satisfy, expect smallest `B`.

- **`H_rank(B) = D_rank + λc·R(B)`**
  `D_rank`: full ranking of candidates preserved (e.g., Kendall-tau / Spearman
  between original and reconstructed ordering). Answers: *"why is the full
  preference ordering what it is?"*

- **`H_margin(B) = D_margin + λc·R(B)`**
  `D_margin`: pairwise score differences preserved (e.g., L2/relative error
  between `(J_i - J_j)` and `(J̃_i - J̃_j)` for all pairs). Answers: *"why is the
  model's confidence gap this large?"* — expect largest `B`.

- **`H_full(B)` (optional, for comparison)**: `α_sel·D_sel + α_rank·D_rank +
  α_margin·D_margin + α_reg·D_reg + λc·R(B)`. Run only after the three
  individual variants are working, as a sanity check for whether it converges
  near the union of the three individual `B`s.

`D_reg`: utility loss penalty when top-1 does change (only nonzero if `D_sel`
constraint fails) — mainly relevant to `H_full`, can be omitted/logged-only in the
three individual variants.

### Rationale (for documentation, not code)
These three questions are genuinely different explanations and may highlight
different evidence (e.g., a single decisive event vs. cumulative sustained
factors). Running them separately, then comparing the resulting `B_sel`,
`B_rank`, `B_margin` (overlap / Jaccard similarity, visualized positions), is
itself a planned experiment — not just a hyperparameter sweep.

## 6. Compactness / Coherence Regularizer

```
R(B) = βdur·Rdur + βnum·Rnum + βsep·Rsep + βdyn·Rdyn
```

- `Rdur`: total covered timesteps (discourage large B)
- `Rnum`: number of disjoint segments (discourage fragmentation)
- `Rsep`: penalize segments scattered far apart
- `Rdyn`: penalize segment boundaries that cut through steep transitions in the
  imagined trajectory (needs a definition of "transition strength" — e.g., local
  derivative magnitude of `r̂`/`û` — implement as a configurable function)

Use the **same `R(B)` and same `β` weights** across all H_sel/H_rank/H_margin
runs so the comparison of resulting `B` sizes is fair (this is the point of the
experiment — comparing "evidence efficiency" across the three criteria).

---

## 7. Search Algorithm (needs concrete implementation, currently undefined)

`P` has on the order of ~100+ candidate segments for `T=30, D={1,2,4,8}`; `B ⊆
P` is a combinatorial subset-selection problem. Implement:

1. **Baseline: greedy forward selection.** Start `B = ∅`, at each step add the
   segment from `P \ B` that most improves `H(B)`, stop when no candidate
   improves it (or a max-size budget is hit). Simple, fast, good default.
2. Log whether `R(B)`'s structure gives any submodularity-like guarantees for
   greedy (nice-to-have theoretical note, not required for first implementation).
3. Leave a clean interface to swap in beam search / CEM-style search over subsets
   later if greedy proves too weak.

---

## 8. Required Diagnostic / Sanity-Check Experiments

These are not optional — implement alongside the main method, they are needed to
validate the approach itself:

1. **`B = ∅` sanity check**: for each `fill` strategy, compute `D_dec` (or each of
   `D_sel/D_rank/D_margin`) when nothing is retained. If this is already low-cost
   (near the satisfaction threshold) for a given baseline, that baseline is
   suspect — flag it in output/report.
2. **Cross-baseline agreement**: run full search with each `fill` strategy
   (`self_mean`, `global_prior`, `shuffle`, `zero` [reward only],
   `counterfactual_reimagine` [small-scale only]) on the same `(h0, candidates)`
   instance, compute pairwise Jaccard similarity / IoU between resulting `B`s,
   output as a heatmap.
3. **Cross-objective agreement**: compare `B_sel` vs `B_rank` vs `B_margin` (same
   fill strategy) — Jaccard similarity + visualization of segment positions along
   the T-step horizon.
4. **Compute-cost accounting**: log wall-clock / number of `G` evaluations per
   `fill` strategy and per objective variant, for a cost-vs-quality table.

### Suggested experiment scope (to control compute)

Do NOT run all 5 fill strategies × 3 objectives × full dataset (15 combos) as the
main sweep. Instead:

- Main-line results: `global_prior` fill × all three objectives (`H_sel`,
  `H_rank`, `H_margin`) on the full evaluation set.
- Robustness ablation: all fill strategies × `H_sel` only (cheapest objective),
  smaller subset of instances, to show conclusions are robust to baseline choice.
- `counterfactual_reimagine`: a handful of qualitative case studies only (too
  expensive for the full sweep).

---

## 9. Open Items to Confirm Before/During Implementation

- [x] Exact functional form of `G(·)` — resolved: pure closed-form λ-return over
      stored `{r̂, û, d̂}`, no network calls (see `scoring.py`).
- [x] Whether `û` is per-step or bootstrap-only — resolved: per-step.
- [x] How `c_i` are generated — resolved: top-N modes (discrete) / N samples
      (continuous); continuation for t=2..T only implements policy-sampling so far.
- [x] Source/scale of data for `global_prior` baseline statistics — resolved
      2026-07-15: per-timestep population means over ~1000 sampled imagined
      candidates per task, built by `experiments/build_global_prior.py` and
      loaded automatically by `pipeline.Decision` (see `results.md`).
- [x] Definition of "transition strength" for `Rdyn` — resolved: local derivative
      magnitude of `r̂`/`û` (see `objectives.py`).
- [x] `D_rank`'s tie-breaking was structurally blind to candidate-independent
      fills (`global_prior`) — resolved 2026-07-15: `objectives.d_rank`'s
      Kendall-tau used to fall back to τ=1 ("ranking fully preserved")
      whenever there were zero concordant/discordant pairs to compare, which
      a candidate-independent fill like `global_prior` trivially hits at
      `B=∅` (all candidates' masked scores collapse to one value). Fixed by
      defaulting to τ=-1 (worst case) instead; `global_prior`'s `H_rank`
      non-empty-`B` rate went from 0% to 100% on all three tasks once fixed
      (see `results.md`).

See `results.md` (same folder) for the experiment-progress log, including open items found
only once evaluation started (degenerate decision points, decision-point
reproducibility).

---

## 10. Suggested Code Structure

Lives alongside this doc, i.e. `method/worldmodel_explain/` (so
`.../worldmodel/dreamerv3` and `.../worldmodel/method/worldmodel_explain` are
siblings under `worldmodel/`); `rollout.py` imports the `dreamerv3` package
from there. Checkpoint paths passed into `rollout.py` point at
`/home/tanyixin/worldmodel/models/<task>_<timestamp>/` on `matsutake` (§0) —
do not hardcode a specific timestamp, take it as a config/CLI arg.

```
worldmodel_explain/
  rollout.py          # wraps DreamerV3 to produce Y_i for each candidate c_i
  segments.py          # builds P, defines B, indicator b_t
  masking.py            # fill strategies: self_mean, global_prior, zero, shuffle, counterfactual
  scoring.py             # G(·) implementation(s); explicit flag for h-dependence
  objectives.py          # D_sel, D_rank, D_margin, D_reg, R(B); H_sel/H_rank/H_margin/H_full
  search.py               # greedy forward selection (extensible)
  experiments/
    sanity_check_empty_B.py
    cross_baseline_agreement.py
    cross_objective_agreement.py
    cost_accounting.py
  config.yaml              # T, D, β/α weights, fill strategy, objective variant, N candidates
```