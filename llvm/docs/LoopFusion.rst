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
Scope of Loop Fusion in a Production Compiler"*. The pass operates on
LLVM IR, leveraging several core analysis frameworks -- 
Scalar Evolution (SCEV), Dependence Analysis
(DA), and Dominator/Post-Dominator Trees -- to determine legality and
perform the CFG rewiring that fuses two loops into one.

2. Prerequisite Concepts
========================

The fusion pass relies on several standard LLVM loop concepts that
are documented elsewhere: simplified loop form, rotated loop form,
loop guard branches, dominator and post-dominator trees, Scalar
Evolution (SCEV), and trip count with peeling. The analysis that
drives the most involved legality check -- Dependence Analysis (DA)
-- is summarized below, because later sections refer directly to its
dependence kinds and direction vectors.

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


3. High-Level Algorithm
=======================

The pass operates in a top-down, level-by-level fashion over the loop
nest tree. At each nesting depth, it:

1. **Collects fusion candidates**: Identifies each eligible loop and
   partitions the eligible loops into ordered chains of
   *control-flow equivalent, strictly adjacent* loops sorted in
   dominance order.
2. **Attempts pairwise fusion**: Walks each chain linearly, testing
   every consecutive pair ``(FC0, FC1)`` against the four legality
   conditions. If all conditions hold, the pair is fused and replaced
   by the fused loop in the chain, which is then considered for
   further fusion with its successor.
3. **Descends to inner loops**: After processing all sibling groups at
   the current depth, the pass descends one level and repeats.

This strategy means outermost loops are fused first. Fusing inner
loops is handled in subsequent iterations of the outer while-loop.


4. Phase 1: Candidate Collection
=================================

For each group of sibling loops (loops sharing a parent) at the
current depth, the pass gathers the loops that are eligible for
fusion.

4.1 Eligibility Check
----------------------

Each loop is first scanned for disqualifying properties. A loop is
rejected immediately if any of its blocks has its address taken, if
any instruction in the loop may throw, or if the loop contains a
volatile memory access. The pass cannot reason about exception-based
control flow or volatile ordering, so these are hard blockers.

Surviving loops must additionally satisfy four structural
requirements:

1. **Structural validity**: preheader, header, exiting block, exit
   block, and latch all exist. This implicitly requires simplified
   form.
2. **Computable trip count**: SCEV must be able to compute a
   loop-invariant backedge-taken count. If not, the pass cannot
   compare trip counts.
3. **Simplified form**: the loop must be in simplified form (single
   preheader, dedicated exit blocks, single latch).
4. **Rotated form**: the loop must be in rotated form (do-while
   shape, with the exit test at the latch). The exiting block must
   be the latch.

If any check fails, the loop is discarded and an optimization remark
is emitted explaining why.

4.2 Grouping by Adjacency
---------------------------

Eligible loops are partitioned into ordered chains based on *strict
adjacency*. Two loops FC0 and FC1 are strictly adjacent if:

- **Unguarded loops**: FC0's exit block is exactly FC1's preheader
  (i.e., there is no intervening code between the two loops).
- **Guarded loops**: FC0's entry block dominates FC1's entry block,
  and FC0's exit block has a unique successor that is FC1's entry
  block.

The algorithm appends each new loop to the first existing chain whose
last element is strictly adjacent to it. If no such chain exists, a
new chain is started. Because the input loops are already supplied in
dominance (program) order, this single-pass grouping correctly
partitions eligible loops into maximal chains of adjacent loops.


5. Phase 2: Pairwise Fusion Attempts
=====================================

The pass iterates over each chain and attempts to fuse consecutive
pairs ``(FC0, FC1)``. The following legality conditions are checked
in order, with early exit on failure.

5.1 Condition 1: Identical Trip Counts (Conformance)
------------------------------------------------------

The conformance check uses SCEV to retrieve the backedge-taken count
of both loops. If the SCEV expressions are identical (pointer equality
after canonicalization), the loops are conforming.

If they differ but both are small constants, the pass computes the
arithmetic difference. If the first loop has more iterations than the
second, and the difference does not exceed the command-line limit
``-loop-fusion-peel-max-count`` (default 0), the pass marks the pair
as eligible for peeling. Peeling the first loop by the difference
will equalize the trip counts.

The current implementation does not support the case where the second
loop has more iterations than the first.

5.2 Condition 2: Compatible Guard Structure
--------------------------------------------

Both loops must be either both guarded or both unguarded. If one is
guarded and the other is not, fusion is rejected. When both are
guarded, the guard-comparison check ensures that:

1. The guard condition instructions are structurally identical
   (same opcode, same operands).
2. The true/false successors have the same polarity relative to the
   loop (i.e., both guards branch to the preheader on the same
   condition outcome).

Additionally, for guarded loops, the pass verifies that instructions
in FC0's exit block can be safely moved before FC1's exit block, and
that FC1's guard block instructions can be safely moved before FC0's
guard block terminator. These moves reuse LLVM's generic code-motion
safety helpers.

5.3 Condition 3: No Negative-Distance Dependencies
----------------------------------------------------

The dependence check is the most involved legality analysis. It
examines all pairs of memory accesses where at least one is a write:

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
analysis strategies, selectable via
``--loop-fusion-dependence-analysis``. The default is DA-based
analysis (``da``); the SCEV-based and combined modes are retained as
opt-ins. The SCEV-based path is expected to be removed in a future
change.

**SCEV-based analysis**:
Rewrites the access function of FC0's instruction into FC1's loop
iteration space by substituting FC0's loop with FC1's loop inside
every add-recurrence subexpression. The pass then asks SCEV whether
FC0's address is known to be greater than or equal to FC1's address.
A non-negative difference means no backward dependence.

**DA-based analysis**:
Queries LLVM's dependence analysis on the pair. If no dependence
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

**Combined analysis**:
Accepts the pair if either SCEV-based or DA-based analysis approves
it.

5.4 Condition 4: Empty or Movable Preheader
---------------------------------------------

FC1's preheader must be empty (containing only the terminator branch)
or all its instructions must be safely movable. The preheader-motion
check classifies each preheader instruction as either hoistable (to
FC0's preheader) or sinkable (into FC1's body after the fused loop):

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

5.5 Profitability
------------------

The profitability check currently always reports that fusion is
beneficial. This is intentional for testing coverage and is expected
to evolve to include cost-model heuristics (e.g., register pressure
estimation, cache footprint analysis) in the future.


6. Phase 3: The Fusion Transformation
======================================

Once all legality checks pass, the transformation proceeds in two
stages: an optional peeling step and the actual CFG rewiring.

6.1 Loop Peeling (Optional)
-----------------------------

If the trip counts differ by a constant ``d``, the pass peels ``d``
iterations from FC0. This extracts the first ``d`` iterations as
straight-line code before the loop, so the remaining loop has the
same trip count as FC1. After peeling:

1. The post-dominator tree is recalculated. 
2. FC0's cached block pointers are refreshed.
3. The peeled iteration blocks' branches are rewritten to remove edges
   to FC1's preheader, ensuring FC0's entry block still dominates
   FC1's entry block.

6.2 CFG Rewiring (Non-Guarded Loops)
--------------------------------------

For non-guarded loops, the CFG transformation performs these steps:

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
   moved into FC0. All child loops of FC1 are re-parented under FC0.

8. **Merge latches**: Instructions from FC0's old latch are moved to
   FC1's latch (which is now the single latch of the fused loop). If
   FC0's latch has a unique successor, the blocks are merged.

9. **Cleanup**: FC1 is erased from the function's loop information.
   SCEV caches for both loops are invalidated, and the dominator and
   post-dominator trees are updated in a single batched transaction.

The resulting fused loop has:

- FC0's preheader as its preheader.
- FC0's header as its header.
- A body containing blocks from both FC0 and FC1.
- FC1's latch as its latch.
- FC1's exit block as its exit block.

6.3 CFG Rewiring (Guarded Loops)
----------------------------------

The guarded-loop fusion path handles the additional complexity of
guard branches and exit blocks:

1. FC0's guard is updated to use FC1's non-loop block as its bypass
   target, effectively making FC0's guard protect both loops.
2. FC0's exit block instructions are moved to FC1's exit block.
3. FC1's guard block is disconnected and deleted.
4. FC0's exit block is disconnected and deleted.
5. The latch rewiring and block transfer proceed identically to the
   non-guarded case.

6.4 Post-Fusion Bookkeeping
-----------------------------

After fusion, the fused loop replaces the original pair in the chain
and becomes the new left operand for the next pairwise attempt. This
enables *cascading fusion*: if three adjacent loops A, B, C are all
fusible, A+B is fused first, then (A+B)+C is attempted.

FC1's loop is also recorded as removed so that it is skipped when the
pass descends to inner nesting levels.


7. Limitations
===============

The current implementation is purely opportunistic: it fuses only
loop pairs that already satisfy the four legality conditions. It does
not reshape the surrounding code to create new fusion opportunities,
and several legality checks are intentionally conservative.

7.1 Algorithmic Scope
----------------------

- **No loop reshaping to enable fusion.** The pass does not insert
  guards, rotate loops, run loop-simplify, or otherwise modify loops
  that miss a legality condition. Loops that are not already in
  rotated, simplified form with a computable trip count are discarded
  as ineligible.
- **No profitability cost model.** The profitability check
  unconditionally reports every legal pair as beneficial. Fusion
  proceeds regardless of cache footprint, register pressure, or any
  interaction with downstream vectorization.
- **Cross-loop def-use chains are always rejected.** If any
  instruction in FC1 uses a value produced inside FC0, the pair is
  rejected outright. Patterns such as a reduction whose result is
  consumed by the next loop are blocked even when the producing value
  is loop-invariant and could legally be hoisted.

7.2 Trip-Count Equalization
----------------------------

- **Peeling is disabled by default.** The maximum number of
  iterations the pass may peel is controlled by the command-line
  option ``-loop-fusion-peel-max-count``, which defaults to ``0``.
  Peeling therefore never fires unless the user opts in explicitly.
- **Peeling shrinks only the first loop.** If the second loop has
  more iterations than the first, fusion is rejected. There is no
  attempt to peel the second loop or extend the first.
- **Peeling requires constant trip counts with a single exit.** Both
  loops must have a small constant trip count for the peel distance
  to be computed. Symbolic trip-count differences are not handled.


7.4 Preheader and Block Handling
---------------------------------

- **Volatile or atomic preheader instructions block fusion.** Such
  instructions cannot be moved, so a non-empty preheader that
  contains them prevents fusion.
- **Memory-reading preheader instructions are handled
  conservatively.** The pass does try to move memory-reading
  preheader instructions, but only when dependence checks prove
  hoisting or sinking is safe. This conservative filtering can still
  reject fusible-looking cases.
- **FC1's preheader must reduce to a single terminator.** After
  hoisting and sinking, the simple (non-guarded) fusion path requires
  FC1's preheader to contain exactly one instruction -- its branch.
  Any instruction that is neither hoistable nor sinkable prevents
  fusion rather than being preserved by block merging.
- **Blocks are rewired, not merged.** Preheaders, latches, and exit
  blocks are reconnected to new successors during fusion, but never
  merged with their neighbors. The fused Control-Flow Graph (CFG)
  therefore retains more basic blocks than strictly necessary.

7.5 Eligibility Blockers
-------------------------

A loop is rejected during candidate construction if any of the
following properties holds:

- A block in the loop has its address taken (for example, used as a
  ``blockaddress`` operand).
- Any instruction in the loop may throw.
- The loop contains a volatile ``load`` or ``store``.

These conditions are hard blockers, not soft preferences: the pass
never attempts to work around them.

**Atomic operations are not reasoned about.** Atomic loads, stores,
``atomicrmw``, ``cmpxchg``, and ``fence`` instructions inside the
loop body are tracked only as ordinary memory reads or writes. The
pass does not inspect their memory ordering (``unordered``,
``monotonic``, ``acquire``, ``release``, ``seq_cst``), and the
underlying dependence analysis reasons about address-based data
dependence rather than inter-thread synchronization. Fusing two
loops that contain non-``unordered`` atomics can therefore change
observable multi-threaded behavior even when the dependence check
reports no conflict.

