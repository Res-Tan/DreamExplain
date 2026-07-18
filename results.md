# Results Log

Running log of eval / experiment results for the checkpoints under `models/`.
Append a new dated section each time a new result is worth recording — don't
edit past entries except to fix errors.

---

## 2026-07-14

### Checkpoint sanity-check eval (`dreamerv3/eval.sh`, 50000 steps, GPU 0)

| Task | Metric | Result | Checkpoint |
|---|---|---|---|
| Crafter | Crafter Score (`crafter_score.py`, achievement success rates) | **50.16%** over 22 achievements | `crafter_20260704_130112/20260704T223212F348053` |
| Atari Pong | `episode/score` mean, 16 episodes | **20.375** (min 19.0 / max 21.0) | `atari_pong_20260703_105122/20260704T103718F585051` |
| DMC Walker Walk | `episode/score` mean, 48 episodes | **962.4** (min 898.3 / max 990.3) | `dmc_walker_walk_20260706_113519/20260706T192124F187841` |

Crafter achievement breakdown: 100% on basic-tier (collect_wood/stone/drink/sapling,
place_table/stone/plant/furnace, make_wood_pickaxe/sword, wake_up), 89% collect_coal,
36% defeat_zombie, 37% eat_cow, 5% defeat_skeleton, 0.45% make_stone_pickaxe, and 0%
on the iron/diamond tier (collect_iron/diamond, make_stone_sword, make_iron_pickaxe/sword,
eat_plant) — model mastered the early tech tree but never progressed past it.

Compared against the published DreamerV3 curves bundled in `dreamerv3/scores/*.json.gz`:
- **Pong**: training stopped at step 18,784,560 (target was `5.1e7` in `configs.yaml`,
  likely cut short by the old TSUBAME 24h job limit) — but the eval score already sits in
  the paper's converged range (~20.3–20.8 across seeds at 200M frames), because Pong
  saturates early. **Converged, usable as-is.**
- **Walker Walk**: training completed essentially exactly at target
  (1,095,488 / 1.1e6 steps). Eval score matches the paper's 10-seed converged range
  (~887–991). **Fully converged, best-trained of the three.**
- **Crafter**: no step-count read done (config target `1.1e6`); score reflects a model
  that has NOT explored the deeper tech tree — usable, but imagined rollouts into
  iron/diamond-adjacent states should be treated as low-confidence (RSSM never trained
  on those transitions).

### `worldmodel_explain` pipeline smoke test

First real run of `method/worldmodel_explain/` against these 3 checkpoints.

Config used: `N=8` candidates, `T=30` horizon, `D={1,2,4,8}`, `fill=self_mean` (default
`global_prior` fill is not usable yet — needs offline rollout stats precomputed, see Open
Items below), `objective=H_sel` (`D_sel + λc·R(B)`, `λc=1.0`,
`β_dur=0.01, β_num=0.05, β_sep=0.01, β_dyn=0.05`), `search=greedy forward selection`.

| Task | J (8 candidates, unmasked) | B found | n_evals |
|---|---|---|---|
| Crafter | `[7.004, 6.780, 8.817, 7.633, 7.150, 6.424, 7.494, 6.999]` | `[(4,11), (0,3)]` | 325 |
| Atari Pong | `[2.954, 2.949, 2.979, 2.920, 2.966, 2.908, 2.902, 2.989]` | `[]` (empty) | 110 |
| DMC Walker Walk | `[313.9, 313.9, 320.2, 313.2, 320.8, 315.2, 313.1, 322.5]` | `[]` (empty) | 110 |

This is effectively a check of whether `self_mean` fill forces real work out of the
search. **Crafter passed** (candidates well-separated, J range 6.4–8.8; masking flipped
the argmax, `D_sel(∅)=1`, so search added `[(4,11), (0,3)]` back in). **Pong/Walker
Walk did not** (candidate J's packed within ~0.03–0.09 of each other; even fully-masked
`self_mean` reproduced the original argmax, `D_sel(∅)=0`, so `H_sel(∅)=0` was already the
minimum and search stopped at `B=∅`) — this is the intended §8 sanity check working
correctly, not a pipeline bug: these two decision points just weren't decisive enough.

**Conclusion**: all 3 checkpoints are usable as trained world models for this method —
rollout, scoring, and search all behave sensibly, including correctly flagging the
degenerate `B=∅` cases.

Note: this Crafter run and the §8 diagnostic-experiment Crafter run below sampled
different decision points (different J vectors on the same checkpoint) — decision-point
sampling isn't currently reproducible run-to-run, see Open items.

### §8 diagnostic experiments — first full run, all 3 tasks in parallel (GPU 0/1/3)

Ran all four `method/worldmodel_explain/experiments/*.py` scripts (readme.md §8.1-8.4)
for all three checkpoints in parallel (one GPU each), `fill=self_mean` (per-task config
copies under `worldmodel_explain/config_<task>.yaml`, since `global_prior` still isn't
implemented). Full logs + PNGs under `experiments/out/<task>/`.

**8.1 B=∅ sanity check** (`self_mean`/`shuffle`/`zero` fills, single decision point each):

| Task | self_mean D_sel/D_rank/D_margin | shuffle | zero | Suspect fills? |
|---|---|---|---|---|
| Crafter | 0.000 / 0.250 / 0.924 | 1.000 / 0.357 / 1.455 | 1.000 / 0.571 / 1.945 | none |
| Atari Pong | 0.000 / 0.000 / 0.083 | 0.000 / 0.179 / 0.554 | 0.000 / 0.000 / 0.083 | **all 3** |
| DMC Walker Walk | 1.000 / 0.036 / 0.160 | 1.000 / 0.750 / 4.060 | 0.000 / 0.036 / 0.163 | **zero** |

**Conclusion**: Pong's decision point is degenerate across all three fills (can't
demonstrate the method here). Walker Walk's `zero` fill leaks (consistent with
readme.md §3's known risk); `self_mean`/`shuffle` behave correctly. Crafter: none
flagged.

**8.2 Cross-baseline agreement** (IoU of B across fill strategies, `H_sel`):

| Crafter | Atari Pong | DMC Walker Walk |
|---|---|---|
| ![Crafter cross-baseline agreement](images/crafter/2026-07-14_cross_baseline_agreement.png) | ![Pong cross-baseline agreement](images/atari_pong/2026-07-14_cross_baseline_agreement.png) | ![Walker Walk cross-baseline agreement](images/dmc_walker_walk/2026-07-14_cross_baseline_agreement.png) |

**Conclusion**: Crafter's fills disagree (`self_mean` |B|=0, `shuffle` |B|=1, `zero`
|B|=4 covering the full horizon) — `zero` is the outlier/leaking, not evidence the
method itself is unstable. Pong's "agreement" (all `B=∅`) and Walker Walk's partial
agreement (`self_mean`/`zero` empty, `shuffle` 2/30 covered) are both downstream of the
degenerate-decision-point issue, not genuine cross-baseline robustness.

**8.3 Cross-objective agreement** (`B_sel` vs `B_rank` vs `B_margin`, `self_mean` fill):

| Crafter | Atari Pong | DMC Walker Walk |
|---|---|---|
| ![Crafter cross-objective agreement](images/crafter/2026-07-14_cross_objective_agreement.png) | ![Pong cross-objective agreement](images/atari_pong/2026-07-14_cross_objective_agreement.png) | ![Walker Walk cross-objective agreement](images/dmc_walker_walk/2026-07-14_cross_objective_agreement.png) |

**Conclusion**: Crafter confirms readme.md §5's design assumption — `sel`/`rank`/`margin`
point at genuinely different evidence (`sel=[(9,12)]`, `rank=[]`,
`margin=[(2,9),(13,14),(0,1)]`, pairwise IoU 0.00–0.07). Pong/Walker Walk mostly
collapse to empty/near-identical B across objectives, again downstream of the
degenerate-decision-point issue rather than the objectives being redundant.

**8.4 Cost accounting**: cheap across the board — 0.08-0.79s and 110-743 `G` evaluations
per (fill, objective) combo, all three tasks. `H_margin`/`shuffle` combos are consistently
the most expensive (more segments end up retained → more evals before greedy stops); `sel`
+ `self_mean`/`zero` are the cheapest. No scalability concern at this N=8/T=30 scale.

### Open items (carried over from readme.md §9, still unresolved)

- `global_prior` fill needs an offline-rollout precompute step (`config.yaml`
  `masking.global_prior.rollouts_dir` is empty) — not implemented yet.
- `H_full` and `counterfactual_reimagine` still haven't been exercised by any experiment
  script.
- **Pong and Walker Walk's sampled decision point is degenerate** (candidate J's within
  ~0.03-0.16 of each other) and makes the method's output trivial (`B=∅` or near-empty)
  for most fill/objective combos — this is a decision-point-sampling issue, not a model
  or pipeline bug (Crafter's decision point, by contrast, produced clearly non-trivial,
  differentiated B's across objectives). Next step: sample multiple decision points per
  task (raise `get_decision_point`'s `warmup_steps`, or loop it) and pick/report on ones
  where candidates actually disagree, rather than judging the method on one arbitrary
  frame per task.
- **Decision-point sampling isn't reproducible run-to-run**: the Crafter smoke test and
  the Crafter §8 diagnostic run above used the same checkpoint/config/seed but landed on
  different decision points (different J vectors) — needs investigation before results
  from separate runs can be compared directly.

---

## 2026-07-17

*Tip: decision-point sampling (env reset + policy action sampling) is now
seeded and verified reproducible — the same `(seed, step)` always yields the
same imagined rollout and `J`, for all three tasks.*

### Multi-point sampling sweep (`fill=self_mean`, `objective=H_sel`, 35 points/task)

5 env seeds × 7 steps (1/5/10/15/20/25/30) = 35 decision points per task.
Per point: `J range` = max−min candidate score ("how decisive is this
point"), and whether greedy search finds any evidence at all (`B` non-empty).

| Task | Max J range found | Non-empty `B` rate | Reference point |
|---|---|---|---|
| Crafter | 5.72 | 20/35 (57%) | seed=3, step=25 |
| Atari Pong | 0.10 | 11/35 (31%) | seed=2, step=30 |
| DMC Walker Walk | 18.32 | 8/35 (23%) | seed=0, step=25 |

**Conclusions:**

- **Walker Walk is not a degenerate task** (J range up to 18.3 — an order of
  magnitude above the single arbitrary point checked on 2026-07-14), but
  `self_mean` fails *worse* the more decisive the point: all 5 top points by
  J range give `B=∅`; every non-empty `B` in the whole sweep comes from a
  point with J range < 3.5. This is `self_mean`'s self-leakage failure mode
  (readme.md §3) — masking with a trajectory's own mean reconstructs a
  sustained, global signal like Walker Walk's average-velocity return, and
  does so *better* the larger that signal is.
- **Pong's low decisiveness is real** (max J range 0.10 across all 35
  points, uniformly small) — consistent with its near-ceiling eval score,
  not a fill or sampling artifact.
- **Crafter**: `self_mean` works reasonably well (57%), with no such inverse
  relationship between decisiveness and `B` size.

### Fill × objective grid, all 35 decision points per task

Comprehensive version of the diagnostic above: is `self_mean`'s weakness
specific to that one (fill, objective) pair, or does it hold for the other
fills (`shuffle`, `zero`, and — once built, see below — `global_prior`) and
the other two objectives (`H_rank`, `H_margin`)? All 4×3 fill/objective
combinations evaluated on *all 35* decision points per task (not just the
most decisive few — `experiments/fill_objective_sweep.py --top-k 1000`).
`global_prior` uses population-level per-timestep means for `{r, u, d}`,
estimated from ~1000 sampled imagined candidates per task
(`experiments/build_global_prior.py`, saved to
`worldmodel_explain/priors/<task>/prior.npz`, closing the readme.md §9 open
item). Cell = % of the 35 points where greedy search returns a non-empty `B`:

**Crafter**

| fill | H_sel | H_rank | H_margin |
|---|---|---|---|
| self_mean | 54% | 80% | 94% |
| shuffle | 66% | 94% | 97% |
| zero | 71% | 100% | 100% |
| global_prior | 80% | 100% | 97% |

**Atari Pong**

| fill | H_sel | H_rank | H_margin |
|---|---|---|---|
| self_mean | 31% | 46% | 51% |
| shuffle | 66% | 94% | 100% |
| zero | 31% | 46% | 51% |
| global_prior | 94% | 100% | 100% |

**DMC Walker Walk**

| fill | H_sel | H_rank | H_margin |
|---|---|---|---|
| self_mean | 23% | 34% | 80% |
| shuffle | 54% | 91% | 100% |
| zero | 54% | 57% | 94% |
| global_prior | 83% | 100% | 100% |

*(`shuffle`'s numbers move a few points between re-runs — it draws an
unseeded random permutation each call; doesn't change any conclusion below.)*

**Conclusions:**

1. **`global_prior` is the best fill overall, on every task and every
   objective** (`H_sel` 80–94%, `H_rank` 100%, `H_margin` 97–100%) — it fixes
   `self_mean`'s self-leakage failure (readme.md §3): the replacement value
   for a masked step no longer depends on that candidate's own magnitude, so
   it can't "leak" the answer for free.
2. **`global_prior`'s `H_rank` initially measured 0% on every task — a
   metric bug, now fixed.** Fully masking with `global_prior` collapses all 8
   candidates' scores to the *same* value (e.g. `J̃ = [5.838, 5.838, ...]`),
   leaving no pair of candidates to compare. `objectives.d_rank`'s
   Kendall-tau used to default to τ=1 ("ranking preserved") whenever there
   were zero comparable pairs, so a fully-collapsed, uninformative ranking
   scored identically to a perfectly-preserved one and greedy search never
   had a reason to add anything back. Fixed by defaulting to τ=-1
   (worst case) instead — the table above already reflects the fix. Only
   candidate-independent fills ever hit this (`self_mean`/`shuffle`/`zero`
   keep per-candidate structure at `B=∅` and don't).
3. **`self_mean` is the weakest fill on every task/objective**, and
   specifically fails worse the more decisive the point is on Walker Walk:
   restricting to just its 5 most decisive points (J range 7.5–18.3) drops
   its `H_sel` rate to 0% (vs. 23% pooled over all 35) — direct evidence the
   self-leakage effect scales with how much of the signal is a sustained,
   global quantity (Walker Walk's return ≈ average velocity) rather than a
   local event.
4. **`self_mean` and `zero` behave identically on Pong** (identical rates in
   every cell) — Pong's reward is sparse/near-zero at most steps, so masking
   it to zero and masking it to its own mean converge.

**Practical takeaway**: `global_prior` should be the default fill across all
three objectives (already set in the task configs) — it's no longer just the
`H_sel`/`H_margin` recommendation, the `H_rank` weakness was a measurement
artifact, not a real limitation.

### Cross-baseline agreement (readme.md §8.2), `H_sel`, single-point case study

Heatmap of pairwise IoU between the `B`s found by each fill strategy, on one
decision point per task (`experiments/cross_baseline_agreement.py` — this is
a single-point qualitative case study by design, readme.md §8.2; the
aggregate non-empty-`B` rates per fill are the "Fill × objective grid" above).

*Caveat: IoU between two **empty** `B`s is defined as 1.0 (∅ literally equals
∅), which is correct but easy to misread as "these fills found the same
evidence" when really neither found any — check `|B|` before reading a
yellow cell as agreement.*

| Task | Point | self_mean | shuffle | zero | global_prior |
|---|---|---|---|---|---|
| Crafter | seed=3, step=25 | \|B\|=1 | \|B\|=0 | \|B\|=0 | \|B\|=1 |
| Atari Pong | seed=2, step=30 | \|B\|=1 | \|B\|=1 | \|B\|=1 | \|B\|=1 |
| DMC Walker Walk | seed=0, step=5 | \|B\|=1 | \|B\|=1 | \|B\|=1 | \|B\|=1 |

| Crafter | Atari Pong | DMC Walker Walk |
|---|---|---|
| ![Crafter cross-baseline agreement](images/crafter/cross_baseline_agreement.png) | ![Pong cross-baseline agreement](images/atari_pong/cross_baseline_agreement.png) | ![Walker Walk cross-baseline agreement](images/dmc_walker_walk/cross_baseline_agreement.png) |

**Conclusions:**

1. **Crafter**: `self_mean` and `global_prior` agree on the same segment
   (IoU=1.0); `shuffle` and `zero` both return `B=∅` at this point (the
   "agreement" between them is the vacuous empty-set case above, not real
   shared evidence).
2. **Atari Pong**: `self_mean`/`shuffle`/`zero` all agree on the same segment;
   `global_prior` finds a *different* one (IoU=0 vs. the other three) — a
   real, non-trivial disagreement.
3. **DMC Walker Walk**: the task's usual reference point (seed=0, step=25)
   turned out to be degenerate for *every* fill under `H_sel` (`B=∅` across
   the board — an all-yellow, uninformative heatmap, same caveat as above),
   so this case study uses a different, non-degenerate point from the
   35-point sweep (seed=0, step=5) instead. There, `self_mean`/`zero`/
   `global_prior` agree on the same segment and `shuffle` disagrees — same
   qualitative pattern as Crafter (the non-self-referential-but-still-
   candidate-blind fills agreeing, `shuffle`'s destroyed ordering picking out
   something different).
4. `cross_baseline_agreement.py` now takes `--seed`/`--warmup-steps` to target
   a specific decision point instead of only the config's default reference
   point — needed to work around case 3 above, kept as a general option.

### Cross-objective agreement (readme.md §5's planned comparison), `global_prior`

A single point's `B_sel`/`B_rank`/`B_margin` overlap can be an outlier (the
reference-point case study below turned up an IoU of exactly 1.0 for Pong),
so this was run two ways: comprehensively across all 35 sampled decision
points per task, and as a single-point qualitative case study with a
timeline visualization.

**Comprehensive (35 points/task, `experiments/cross_objective_agreement_sweep.py`):**

| Task | mean \|B_sel\| | mean \|B_rank\| | mean \|B_margin\| | sel↔rank IoU (mean/median) | sel↔margin IoU | rank↔margin IoU |
|---|---|---|---|---|---|---|
| Crafter | 4.3 | 12.2 | 14.4 | 0.14 / 0.00 | 0.12 / 0.03 | 0.27 / 0.20 |
| Atari Pong | 1.0 | 1.7 | 29.7 | 0.08 / 0.00 | 0.04 / 0.03 | 0.06 / 0.03 |
| DMC Walker Walk | 3.3 | 10.5 | 27.7 | 0.13 / 0.00 | 0.11 / 0.07 | 0.38 / 0.27 |

**Conclusions:**

1. **`|B_sel| < |B_rank| < |B_margin|` holds on average, on every task** —
   confirms readme.md §5's expected size ordering, but **`H_margin`'s large
   `B` is mostly degenerate, not a genuinely larger-but-still-compact
   explanation.** Mean `B_margin` coverage is 99% of the horizon for Pong,
   92% for Walker Walk, 48% for Crafter — i.e. for Pong/Walker Walk, `H_margin`
   almost always ends up keeping nearly *everything*. With nearly nothing
   masked, `J̃ ≈ J` trivially, so `D_sel`/`D_rank`/`D_margin` are all trivially
   near their best values too — this isn't "evidence found," it's the
   objective being satisfied by not really masking anything. Root cause:
   readme.md §6 mandates the *same* regularizer weights (`β_dur=0.01` etc.,
   `λc=1.0`) across all three objectives for a fair size comparison, but
   `D_margin` is a relative-error quantity where keeping one more real
   timestep almost always buys a real, non-negligible improvement — the
   `β_dur=0.01`-scale cost of covering one more step is far too cheap to stop
   greedy search before it's covered nearly all of `T`. See "Regularizer
   strength vs. `H_margin` compactness" below for how sensitive this is to
   the regularizer weights. Crafter's lower average (48%) shows this isn't
   universal — the failure mode is specific to how much of Pong's/Walker
   Walk's return is a *sustained* signal (nothing to gain from masking a
   sustained trend, so keep it all) vs. Crafter's more localized rewards.
2. **The three objectives mostly surface different evidence** — median IoU
   is 0.00–0.27 for every pair, on every task. `rank↔margin` is consistently
   the most overlapping pair (0.20–0.38 median) — plausible since `H_margin`
   is the strictest constraint and its evidence set most often ends up
   covering whatever `H_rank` already needed, plus more.
3. **Pong's sel==rank coincidence (IoU=1.0) at its reference point is real
   but uncommon** — happens at 2/35 points for Pong, 1/35 for Walker Walk,
   0/35 for Crafter. Not a general pattern, just occasionally true.
4. **`B_sel` is empty** at 7/35 (20%) Crafter points, 6/35 (17%) Walker Walk,
   2/35 (6%) Pong — matches the `global_prior` `H_sel` non-empty rates from
   the fill × objective grid above almost exactly (80%/83%/94% non-empty ⟺
   20%/17%/6% empty), a good cross-check between the two experiments.

**Single-point case study** (reference decision point, `experiments/cross_objective_agreement.py`).
**Caveat confirmed directly: `B_margin` covers 100% of the 30-step horizon
(0–29) at all three points below** — a clean illustration of conclusion 1,
not three genuinely-informative margin explanations.

| Task | `B_sel` | `B_rank` | `B_margin` (covers 30/30 in all 3 cases) |
|---|---|---|---|
| Crafter | `[(29,29)]` | `[(15,16),(17,20)]` | `[(0,7),(8,15),(15,22),(22,29)]` |
| Atari Pong | `[(7,7)]` | `[(7,7)]` | `[(0,0),(1,8),(9,16),(14,21),(22,29)]` |
| DMC Walker Walk | `[]` | `[(28,29)]` | `[(0,7),(6,13),(14,21),(22,29)]` |

| Crafter | Atari Pong | DMC Walker Walk |
|---|---|---|
| ![Crafter cross-objective agreement](images/crafter/cross_objective_agreement.png) | ![Pong cross-objective agreement](images/atari_pong/cross_objective_agreement.png) | ![Walker Walk cross-objective agreement](images/dmc_walker_walk/cross_objective_agreement.png) |

Walker Walk's `B_sel` happens to be empty at this specific reference point
(one of its 6/35 empty cases) — not a useful point for an `H_sel` case study
on this task; a different point from the 35-point sweep should be picked if
one is needed for a write-up figure.

### Regularizer strength vs. `H_margin` compactness

Follow-up to the near-full-horizon `B_margin` finding above: does simply
strengthening the regularizer (readme.md §6's `β_dur/β_num/β_sep/β_dyn`,
scaled together by a multiplier — equivalent to scaling `λc`, since `R(B)` is
linear in the betas) recover a compact `B_margin`? First tried a coarse
multiplier grid (1/3/10/30/100) and initially concluded there was no middle
ground, only a cliff between ×3 and ×10 — that conclusion turned out to be
an artifact of the coarse spacing skipping over the transition. Re-ran with a
finer grid (1–20) and **evaluated each task independently rather than
assuming one multiplier has to work for all three** — the transition point is
genuinely different per task. Mean coverage (of `T=30`) / mean `D_margin` at
the search's final `B`, `global_prior` fill, all 35 points per task:

| Multiplier | Crafter cov / D_margin | Pong cov / D_margin | Walker Walk cov / D_margin |
|---|---|---|---|
| ×1 (baseline) | 14.8/30 / 0.40 | 29.7/30 / 0.00 | 27.7/30 / 0.03 |
| ×3 | 13.1/30 / 0.49 | 29.8/30 / 0.00 | 27.8/30 / 0.03 |
| ×5 | 12.7/30 / 0.52 | 29.9/30 / 0.00 | 21.5/30 / 0.25 |
| ×7 | 6.3/30 / 0.78 | 27.3/30 / 0.10 | 13.8/30 / 0.55 |
| ×9 | 2.4/30 / 0.98 | 15.3/30 / 0.56 | 4.3/30 / 0.92 |
| ×12 | 0.0/30 / 1.14 | 2.6/30 / 1.05 | 2.5/30 / 0.99 |
| ×15–20 | 0.0/30 / 1.14 | 0.0/30 / 1.14 | 0.0–1.1/30 / 1.06–1.14 |

**Corrected conclusion: there is a real, usable middle ground — the earlier
"cliff" was a sampling artifact, not a property of `D_margin`.** All three
tasks show a genuine gradual decline, not a step function. But **the
transition happens at a different multiplier for each task, so a shared
multiplier across tasks isn't the right framing** (readme.md §6's
"same weights for a fair comparison" applies to comparing `H_sel`/`H_rank`/
`H_margin` *within* a task, not to reusing one `H_margin` compactness setting
*across* tasks):

- **Pong needs the strongest push**: completely flat from ×1–×6 (the default
  weights are simply too weak to register at all), only starts declining at
  ×7, reaching a moderate compactness/faithfulness point around ×8–9
  (24→15 out of 30 covered, `D_margin` 0.23→0.56).
- **Crafter and Walker Walk respond earlier and more smoothly**, with a
  usable middle ground around ×5–×7 (Crafter: 6–13 of 30 covered, `D_margin`
  0.49–0.78; Walker Walk: 14–22 of 30, `D_margin` 0.25–0.55).

**Practical takeaway**: a per-task regularizer multiplier (roughly ×5–7 for
Crafter/Walker Walk, ×8–9 for Pong) gives a genuinely more compact `B_margin`
at a moderate, non-catastrophic `D_margin` cost — worth adopting per-task if
`H_margin` explanations are needed for a write-up, rather than the shared ×1
weights that currently make it degenerate to near-full-horizon coverage.

### Case study with randomly-selected points, not the max-`J`-range one

The reference points used everywhere above (picked by "largest `J` range"
back in the first multi-point sweep) are a biased choice for a case study —
"largest `J` range" was the right criterion for its original purpose
(showing a task isn't degenerate), but not for "what does a representative
`H_margin` explanation look like." Drew **3 points per task genuinely at
random** (Python's `random`, OS entropy, no cherry-picking) and ran the full
regularizer-multiplier sweep (×1–20, `experiments/margin_regularizer_sweep.py`
on a single point at a time) on each individually, instead of just an
×1-vs-one-tuned-value snapshot:

| Task | Point (seed,step) | Coverage @×1 | Behavior across ×1–20 |
|---|---|---|---|
| Crafter | (4,25) | 30/30 | flat at 30/30 through ×6, clean jump to 0/30 at ×7 |
| Crafter | (2,25) | 8/30 | **already compact**, drops to 0/30 by ×2 |
| Crafter | (4,30) | 12/30 | **non-monotonic**: rises to 30/30 at ×2–5, then drops to 0/30 at ×6 |
| Atari Pong | (3,20) | 29/30 | flat through ×9, clean jump to 0/30 at ×10 |
| Atari Pong | (4,1) | 30/30 | flat through ×8, clean jump to 0/30 at ×9 |
| Atari Pong | (0,5) | 29/30 | flat through ×8, clean jump to 0/30 at ×9 |
| Walker Walk | (1,25) | 30/30 | flat through ×10, clean jump to 0/30 by ×12 |
| Walker Walk | (3,5) | 14/30 | **stable and compact** at 14/30 across ×1–8, drops to 0/30 at ×9 |
| Walker Walk | (4,25) | 30/30 | flat through ×5, clean jump to 0/30 at ×6 |

Timeline images for one representative point per task:
`images/crafter/cross_objective_agreement_random2.png` (seed=2,step=25),
`images/atari_pong/cross_objective_agreement_random2.png` (seed=4,step=1),
`images/dmc_walker_walk/cross_objective_agreement_random2.png` (seed=3,step=5).

**Conclusions:**

1. **7 of 9 randomly-drawn points are pure step functions** — flat at full
   (30/30, or 29/30) coverage, then a single clean jump straight to `B=∅`
   at a point-specific multiplier threshold, which itself varies a lot even
   within one task (Walker Walk's three points jump at ×6, ×9, and ×12).
   This reconfirms, with 9 fresh points instead of 1, that individual points
   generally don't degrade gradually.
2. **2 of 9 points are genuinely compact already, no tuning needed**:
   Crafter (2,25) at 8/30, and — the best example found so far — **Walker
   Walk (3,5), stably compact at 14/30 across the entire ×1–8 range**
   before eventually collapsing at ×9. This is a real, non-degenerate,
   tuning-robust `H_margin` explanation, not a knife-edge artifact.
3. **One point, Crafter (4,30), was non-monotonic**: coverage went
   30→12→30→30→30→30→0 as the multiplier increased from ×1 to ×7 (rose
   *before* falling). Greedy forward selection is a heuristic, not an exact
   optimizer, so it isn't guaranteed to shrink `B` monotonically as
   regularization strengthens — noted as an observed quirk, not
   investigated further today.

**Calibrated takeaway**: representative (random) points confirm the
population-level story — most individual points are either already fine or
flip to empty at their own threshold, and only a minority (here ~2/9) give a
genuinely compact, non-degenerate `B_margin` without needing to hit exactly
the right multiplier. `H_margin` explanations should be presented with this
caveat; Walker Walk (3,5) is the cleanest illustrative example available so
far if one compact case is needed for a write-up.

### Open items (updated)

- ~~Decision-point sampling isn't reproducible run-to-run~~ — **resolved**.
- ~~Pong and Walker Walk's sampled decision point is degenerate~~ — **resolved
  for Walker Walk** (it isn't degenerate; `self_mean` leakage was masking
  that). **Still true for Pong**, confirmed to be a property of the
  task/checkpoint rather than a sampling issue.
- ~~`global_prior` fill needs an offline-rollout precompute step~~ —
  **resolved**, see "Fill × objective grid, all 35 decision points per task" above.
- ~~`D_rank`'s tie-breaking makes it structurally blind to candidate-independent
  fills like `global_prior`~~ — **resolved**, see conclusion 2 above.
- ~~Cross-objective agreement hasn't been re-run against the new reference
  points / `global_prior`~~ — **resolved**, see "Cross-objective agreement" above.
- ~~`cross_baseline_agreement.py`'s heatmap hasn't been re-run against the new
  reference points / `global_prior`~~ — **resolved**, see "Cross-baseline
  agreement" above.
- ~~`H_margin`'s large `B` is mostly degenerate (near-full-horizon coverage,
  especially Pong/Walker Walk) with the shared (×1) regularizer weights~~ —
  **partially resolved**: a per-task regularizer multiplier (×5–7 for
  Crafter/Walker Walk, ×8–9 for Pong) recovers a genuinely compact
  `B_margin` *on average* — see "Regularizer strength vs. `H_margin`
  compactness" and "Case study with randomly-selected points" above.
  **Caveat**: not a guaranteed per-point fix — 7/9 randomly-drawn points are
  pure step functions with their own threshold, so the tuned multiplier can
  still land a specific point on the wrong side (full or empty) rather than
  a nice middle; only 2/9 random points were genuinely compact without
  hitting an exact threshold. Not yet adopted as the actual per-task config
  default (still ×1 everywhere).
- `H_full` and `counterfactual_reimagine` still haven't been exercised by any experiment
  script.
- The §8 cost-accounting experiment hasn't been re-run against the new
  reference points / `global_prior` yet.
- `shuffle`'s fill is unseeded (fresh RNG per call, see note above) — low
  priority (doesn't change conclusions) but worth seeding if exact numbers
  need to match run-to-run.
