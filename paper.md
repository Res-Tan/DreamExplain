# Seeing What the Model Imagined: Decision-Faithful Explanations for World-Model-Based Planning

*Working draft. Method, experiments, and framing are still evolving; update this
file as the paper develops rather than treating it as final. Title is a
placeholder, not final.*

**TL;DR:** We propose a method for identifying the compact, shared moments in a
world model's imagined rollouts that actually explain why it chose one action
over another.

---

## Abstract

A world model is a learned simulator of an environment's dynamics: given a current state and a hypothetical sequence of actions, it predicts how future observations, rewards, and outcomes would unfold. 
Modern model-based reinforcement learning agents use this capability to plan, imagining several candidate futures and selecting the one with the highest predicted return, yet it remains opaque which parts of an imagined future actually drove a given choice. 
We introduce a framework for explaining such decisions by identifying a compact set of temporal segments, shared across all candidate rollouts, that are sufficient to reconstruct the decision-relevant structure of the model's preferences. 
We formalize this as a search over the imagined trajectories for a minimal, regularized subset of evidence that preserves one of three complementary notions of decision faithfulness: the selected action, the full preference ranking, or the pairwise confidence margins between candidates. 
We further show that how the "removed" information is represented is itself a critical design choice, and that naive strategies can silently leak the answer and produce spuriously faithful explanations. 
We validate the framework on trained world models across several domains with different reward structures.
Our analysis shows that the three faithfulness criteria surface largely non-overlapping evidence, that explanation quality depends strongly on how removed information is represented, and that the recovered explanations reflect meaningful structure in each task's underlying dynamics. 
These results establish a general, task-agnostic methodology for auditing and interpreting the imagined reasoning of world-model-based agents.
