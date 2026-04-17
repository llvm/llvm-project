====================================================
Loop Fusion in LLVM
====================================================

1. Introduction
===============

Loop fusion (also called loop jamming) is a compiler optimization that
merges two adjacent loops into a single loop, provided the
transformation preserves the program's original semantics. The
motivation is straightforward: by executing the bodies of two loops in
a single pass over the iteration space, we reduce loop overhead (fewer
branch instructions, fewer induction variable updates), improve
temporal data locality (data written by the first loop body and read
by the second is still in cache), and create new opportunities for
subsequent optimizations such as instruction scheduling and register
allocation.

LLVM's implementation resides in
``llvm/lib/Transforms/Scalar/LoopFuse.cpp`` and is based on
Christopher Barton's MSc thesis, *"Code Transformations to Augment the
Scope of Loop Fusion in a Production Compiler"*. The pass operates one
LLVM IR, leveraging several core analysis frameworks -- 
Scalar Evolution (SCEV), Dependence Analysis
(DA), and Dominator/Post-Dominator Trees -- to determine legality and
perform the CFG rewiring that fuses two loops into one.

2. Prerequisite Concepts
========================

Before describing the algorithm, we define the terms and LLVM IR
concepts it relies on.

2.1 Loop Canonical Form (Simplified Form)
------------------------------------------

A loop in LLVM is said to be in *simplified form*
(``Loop::isLoopSimplifyForm()``) when it satisfies three structural
properties:

1. **Preheader**: A single basic block that is the sole predecessor
   of the loop header from outside the loop. It serves as the "entry
   gate" and is a convenient place to hoist loop-invariant
   computations.
2. **Dedicated exit blocks**: Every exit block (a block outside the
   loop reached from inside) has all its predecessors inside the loop.
3. **Single latch**: Exactly one back-edge targets the header. The
   block that contains this back-edge is called the *latch*.

2.2 Rotated Form
-----------------

A loop is in *rotated form* (``Loop::isRotatedForm()``) when the
latch block is a conditional branch that decides whether to re-enter
the header or exit the loop. In source-level terms, this corresponds
to a do-while style loop. Loop rotation transforms a while-loop
(guard-test-at-top) into a do-while (test-at-bottom), which is the
form the loop fusion pass requires.

2.3 Loop Guard
--------------

Some loops are preceded by a *guard branch* -- a conditional branch
that checks whether the loop should execute at all (e.g., when the
trip count might be zero). The guard branch sits before the preheader
and either jumps to the preheader (entering the loop) or to a
*non-loop block* that bypasses the loop entirely. The pass must handle
both guarded and unguarded loops.

2.4 Dominator and Post-Dominator Trees
---------------------------------------

The *dominator tree* (DT) encodes the dominance relation: basic block
A *dominates* B if every path from the function entry to B must pass
through A. The *post-dominator tree* (PDT) is the dual: A
*post-dominates* B if every path from B to the function exit must pass
through A.

Two blocks are *control-flow equivalent* if each dominates and
post-dominates the other. If loop L0 dominates loop L1 and L1
post-dominates L0, then whenever one executes, the other is guaranteed
to execute as well. This is a necessary condition for fusion.

2.5 Scalar Evolution (SCEV)
----------------------------

SCEV is LLVM's framework for symbolically analyzing how scalar
expressions evolve across loop iterations. It can express induction
variables, trip counts, and pointer access functions as closed-form
recurrences. The fusion pass uses SCEV to:

- Compute and compare loop trip counts (backedge-taken counts).
- Rewrite access functions from one loop into the iteration space of
  another to compare memory addresses.

2.6 Dependence Analysis (DA)
-----------------------------

DA determines the dependence relation between pairs of memory
accesses. A dependence from instruction S1 to S2 is characterized by:

- **Flow (true) dependence**: S1 writes, S2 reads the same location
  (read-after-write).
- **Anti dependence**: S1 reads, S2 writes (write-after-read).
- **Output dependence**: Both S1 and S2 write (write-after-write).

Each dependence carries a *direction vector* at each loop nesting
level, indicating whether the source iteration is less than (``<``),
equal to (``=``), or greater than (``>``) the sink iteration at that
level. A dependence with a ``>`` component at the current loop level
represents a *backward loop-carried dependence* (also called a
negative-distance dependence). Such dependences are the critical
hazard that loop fusion must respect.

2.7 Trip Count and Peeling
---------------------------

The *trip count* is the number of times the loop body executes. Two
loops can only be fused if they iterate the same number of times. When
trip counts differ by a small constant, the pass can *peel* iterations
from the first loop -- extracting leading iterations into
straight-line code before the loop -- to equalize the counts.


3. High-Level Algorithm
=======================

The pass operates in a top-down, level-by-level fashion over the loop
nest tree. At each nesting depth, it:

1. **Collects fusion candidates**: Wraps each eligible loop in a
   ``FusionCandidate`` structure, then groups candidates into lists of
   *control-flow equivalent, strictly adjacent* loops sorted in
   dominance order.
2. **Attempts pairwise fusion**: Walks each candidate list linearly,
   testing every consecutive pair ``(FC0, FC1)`` against the four
   legality conditions. If all conditions hold, the pair is fused and
   replaced by a single new candidate in the list, which is then
   considered for further fusion with its neighbor.
3. **Descends to inner loops**: After processing all sibling groups at
   the current depth, the pass descends one level and repeats.

This strategy means outermost loops are fused first. Fusing inner
loops is handled in subsequent iterations of the outer while-loop.


4. Data Structures
==================

4.1 FusionCandidate
--------------------

This is the central abstraction. It caches loop components that are
queried repeatedly:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``Preheader``
     - The single entry block into the loop
   * - ``Header``
     - The loop header (target of the back-edge)
   * - ``ExitingBlock``
     - The block inside the loop that branches out
   * - ``ExitBlock``
     - The first block outside the loop after exit
   * - ``Latch``
     - The block containing the back-edge
   * - ``GuardBranch``
     - The conditional branch guarding the loop, if any
   * - ``MemReads`` / ``MemWrites``
     - All memory-accessing instructions in the loop
   * - ``PP`` / ``AbleToPeel`` / ``Peeled``
     - Peeling metadata from ``TargetTransformInfo``

During construction, the candidate scans every block and every
instruction. If it encounters a block whose address is taken, an
instruction that may throw, or a volatile memory access, it marks
itself invalid immediately. These are hard blockers: the pass cannot
reason about exception-based control flow or volatile ordering.

4.2 FusionCandidateList and FusionCandidateCollection
------------------------------------------------------

A ``FusionCandidateList`` is a ``std::list<FusionCandidate>``
representing an ordered sequence of strictly adjacent, control-flow
equivalent candidates. The list is used (rather than a vector) because
fusion replaces two adjacent elements with one, and ``std::list``
supports O(1) insertion and erasure without invalidating other
iterators.

A ``FusionCandidateCollection`` is a ``SmallVector`` of such lists --
one list per group of mutually adjacent candidates at the current
nesting level.

4.3 LoopDepthTree
------------------

This structure organizes the function's loops by nesting depth. It
starts with the outermost loops (depth 1) and supports a ``descend()``
operation that replaces the current level with the children of all
non-removed loops. It also tracks which loops have been removed due to
fusion (via ``RemovedLoops``), preventing the pass from operating on
stale loop objects.


5. Phase 1: Candidate Collection
=================================

The entry point is ``LoopFuser::fuseLoops()``. For each group of
sibling loops (loops sharing a parent) at the current depth, it calls
``collectFusionCandidates()``.

5.1 Eligibility Check
----------------------

Each loop is first wrapped in a ``FusionCandidate``. The
``isEligibleForFusion()`` method checks:

1. **Structural validity**: Preheader, header, exiting block, exit
   block, and latch all exist. This implicitly requires simplified
   form.
2. **Computable trip count**: SCEV must be able to compute a
   loop-invariant backedge-taken count. If not, the pass cannot
   compare trip counts.
3. **Simplified form**: ``Loop::isLoopSimplifyForm()`` must hold.
4. **Rotated form**: ``Loop::isRotatedForm()`` must hold. The exiting
   block must be the latch.

If any check fails, the candidate is discarded and an optimization
remark is emitted explaining why.

5.2 Grouping by Adjacency
---------------------------

Eligible candidates are grouped into ``FusionCandidateList`` instances
based on *strict adjacency*. Two candidates FC0 and FC1 are strictly
adjacent if:

- **Unguarded loops**: FC0's exit block is exactly FC1's preheader
  (i.e., there is no intervening code between the two loops).
- **Guarded loops**: FC0's entry block dominates FC1's entry block,
  and FC0's exit block has a unique successor that is FC1's entry
  block.

The algorithm appends each new candidate to the first existing list
whose last element is strictly adjacent to it. If no such list exists,
a new list is created. Because the input loops are already sorted in
dominance order (they come from ``LoopInfo`` which provides them in
program order), this single-pass grouping correctly partitions
candidates into maximal chains of adjacent loops.


6. Phase 2: Pairwise Fusion Attempts
=====================================

``fuseCandidates()`` iterates over each candidate list and attempts to
fuse consecutive pairs ``(FC0, FC1)``. The following legality
conditions are checked in order, with early exit on failure.

6.1 Condition 1: Identical Trip Counts (Conformance)
------------------------------------------------------

``haveIdenticalTripCounts()`` uses SCEV to retrieve the
backedge-taken count of both loops. If the SCEV expressions are
identical (pointer equality after canonicalization), the loops are
conforming.

If they differ but both are small constants, the pass computes the
arithmetic difference. If the first loop has more iterations than the
second, and the difference does not exceed ``FusionPeelMaxCount`` (a
command-line parameter, default 0), the pass marks the pair as
eligible for peeling. Peeling the first loop by the difference will
equalize the trip counts.

The current implementation does not support the case where the second
loop has more iterations than the first.

6.2 Condition 2: Compatible Guard Structure
--------------------------------------------

Both loops must be either both guarded or both unguarded. If one is
guarded and the other is not, fusion is rejected. When both are
guarded, ``haveIdenticalGuards()`` checks that:

1. The guard condition instructions are identical
   (``Instruction::isIdenticalTo``).
2. The true/false successors have the same polarity relative to the
   loop (i.e., both guards branch to the preheader on the same
   condition outcome).

Additionally, for guarded loops, the pass verifies that instructions
in FC0's exit block can be safely moved before FC1's exit block, and
that FC1's guard block instructions can be safely moved before FC0's
guard block terminator. This uses ``isSafeToMoveBefore()`` from
``CodeMoverUtils``.

6.3 Condition 3: No Negative-Distance Dependencies
----------------------------------------------------

``dependencesAllowFusion()`` performs the most involved legality
analysis. It checks all pairs of memory accesses where at least one is
a write:

- (FC0 writes) x (FC1 writes) -- output dependences
- (FC0 writes) x (FC1 reads) -- flow dependences
- (FC1 writes) x (FC0 reads) -- anti dependences

Additionally, it checks for *cross-loop def-use chains*: if any
instruction in FC1 uses a value defined inside FC0, fusion is rejected
outright. This is because after fusion, the definition from a given
iteration of the original FC0 would need to be available to the same
iteration of FC1, but the interleaved execution order may not preserve
this.

For each memory pair, the pass invokes one of three dependence
analysis strategies (selectable via
``--loop-fusion-dependence-analysis``):

**SCEV-based analysis** (``accessDiffIsPositive``):
Rewrites the access function of FC0's instruction into FC1's loop
iteration space using ``AddRecLoopReplacer`` (a
``SCEVRewriteVisitor`` that substitutes the loop in
``SCEVAddRecExpr`` nodes). Then uses ``SE.isKnownPredicate()`` to
prove that FC0's address is always >= FC1's address. A non-negative
difference means no backward dependence.

**DA-based analysis**:
Invokes ``DependenceInfo::depends()`` on the pair. If no dependence
exists, fusion is safe. If a dependence exists, the pass examines its
direction vector:

1. At outer loop levels (levels above the current fusion level): if
   any level has a direction that excludes equality (``EQ``), the
   outer indices differ and the dependence does not constrain fusion
   at the current level.
2. At the current level: if the direction excludes ``GT``
   (greater-than), there is no backward loop-carried dependence, and
   fusion is safe. For example, a pure ``LT`` direction indicates a
   forward dependence like ``A[i] = ...; ... = A[i-1]``, which
   remains valid after fusion.
3. Loop-invariant (scalar) non-anti dependences at the current level
   are also safe.

**Combined analysis** (``FUSION_DEPENDENCE_ANALYSIS_ALL``):
Accepts the pair if either SCEV-based or DA-based analysis approves
it.

6.4 Condition 4: Empty or Movable Preheader
---------------------------------------------

FC1's preheader must be empty (containing only the terminator branch)
or all its instructions must be safely movable.
``collectMovablePreheaderInsts()`` classifies each preheader
instruction as either hoistable (to FC0's preheader) or sinkable
(into FC1's body after the fused loop):

- **Hoisting**: An instruction can be hoisted if all its operands
  either dominate FC0's preheader target or have already been
  classified as hoistable. PHI nodes cannot be hoisted. Memory
  instructions can be hoisted only if they have no flow, anti, or
  output dependences with non-hoisted preheader instructions or with
  FC0's memory accesses.
- **Sinking**: An instruction can be sunk if none of its users are
  inside FC1's loop body, and it has no flow or output dependences
  with FC1's memory accesses.
- Atomic and volatile instructions cannot be moved.

If any instruction is neither hoistable nor sinkable, fusion is
abandoned for this pair.

6.5 Profitability
------------------

``isBeneficialFusion()`` currently always returns ``true``. The
comment notes this is intentional for testing coverage and will evolve
to include cost-model heuristics (e.g., register pressure estimation,
cache footprint analysis) in the future.


7. Phase 3: The Fusion Transformation
======================================

Once all legality checks pass, the transformation proceeds in two
stages: an optional peeling step and the actual CFG rewiring.

7.1 Loop Peeling (Optional)
-----------------------------

If the trip counts differ by a constant ``d``,
``peelFusionCandidate()`` peels ``d`` iterations from FC0. This
extracts the first ``d`` iterations as straight-line code before the
loop, so the remaining loop has the same trip count as FC1. After
peeling:

1. The post-dominator tree is recalculated (peeling does not update
   it).
2. FC0's cached block pointers are refreshed via
   ``updateAfterPeeling()``.
3. The peeled iteration blocks' branches are rewritten to remove edges
   to FC1's preheader, ensuring FC0's entry block still dominates
   FC1's entry block.

7.2 CFG Rewiring (Non-Guarded Loops)
--------------------------------------

The ``performFusion()`` method for non-guarded loops performs these
steps:

1. **Merge preheaders**: Move all instructions from FC1's preheader
   into FC0's preheader (before the terminator).

2. **Rewire FC0's exiting block**: Instead of branching to FC1's
   preheader (which is about to be deleted), the exiting block now
   targets FC1's header directly.

3. **Delete FC1's preheader**: It has no predecessors left; replace
   its terminator with ``unreachable``.

4. **Move PHI nodes**: All PHI nodes from FC1's header are moved to
   FC0's header. If a PHI has no uses, it is deleted.

5. **Introduce intermediate PHI nodes**: If FC0's exiting block is
   not the latch (a rare case given the rotated form requirement),
   new PHI nodes are inserted in FC1's header. These select the
   loop-carried value from FC0 when arriving via FC0's latch, or
   ``poison`` when arriving via FC0's exiting block (which means the
   loop is exiting and the value is dead).

6. **Reconnect latches**:

   - FC0's latch, which previously branched back to FC0's header, now
     branches to FC1's header.
   - FC1's latch, which previously branched back to FC1's header, now
     branches to FC0's header.
   - FC0's latch branch is simplified to unconditional (both targets
     are now FC1's header).

7. **Transfer loop blocks**: All basic blocks belonging to FC1 are
   moved into FC0's ``Loop`` object. All child loops of FC1 are
   re-parented under FC0.

8. **Merge latches**: Instructions from FC0's old latch are moved to
   FC1's latch (which is now the single latch of the fused loop). If
   FC0's latch has a unique successor, the blocks are merged.

9. **Cleanup**: FC1's ``Loop`` object is erased from ``LoopInfo``.
   SCEV caches for both loops are invalidated. The dominator and
   post-dominator trees are updated via batched ``DomTreeUpdater``
   operations.

The resulting fused loop has:

- FC0's preheader as its preheader.
- FC0's header as its header.
- A body containing blocks from both FC0 and FC1.
- FC1's latch as its latch.
- FC1's exit block as its exit block.

7.3 CFG Rewiring (Guarded Loops)
----------------------------------

``fuseGuardedLoops()`` handles the additional complexity of guard
branches and exit blocks:

1. FC0's guard is updated to use FC1's non-loop block as its bypass
   target, effectively making FC0's guard protect both loops.
2. FC0's exit block instructions are moved to FC1's exit block.
3. FC1's guard block is disconnected and deleted.
4. FC0's exit block is disconnected and deleted.
5. The latch rewiring and block transfer proceed identically to the
   non-guarded case.

7.4 Post-Fusion Bookkeeping
-----------------------------

After fusion, the fused loop is wrapped in a new
``FusionCandidate``. The two original candidates are removed from the
candidate list, and the fused candidate is inserted in their place.
The iteration variable ``NextIt`` is set to point at this new
candidate, so the next iteration will attempt to fuse it with its
successor in the list. This enables *cascading fusion*: if three
adjacent loops A, B, C are all fusible, A+B is fused first, then
(A+B)+C is attempted.

FC1's loop is recorded in ``LoopDepthTree::RemovedLoops`` to prevent
it from being processed when the pass descends to inner nesting
levels.
