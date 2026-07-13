//===- LoopSplitUtils.h - Split a loop's iteration space --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Splits a counted loop's iteration space into a chain of per-partition
// sub-loops. See LoopSplitUtils.cpp for the structure produced.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPSPLITUTILS_H
#define LLVM_TRANSFORMS_UTILS_LOOPSPLITUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <memory>

namespace llvm {

class BasicBlock;
class DominatorTree;
class ICmpInst;
class Instruction;
class Loop;
class LoopInfo;
class PHINode;
class SCEV;
class SCEVAddRecExpr;
class ScalarEvolution;
class Value;

/// Splits a counted loop into a chain of per-partition sub-loops.
///
/// Usage:
/// \code
///   LoopSplitUtils LSU(L, LI, SE, DT);
///   if (!LSU.isLegal())
///     return false;
///   LSU.addPartition(S0, E0);   // one call per partition, in order
///   LSU.addPartition(S1, E1);
///   LSU.split();
/// \endcode
class LoopSplitUtils {
public:
  LoopSplitUtils(Loop *L, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT)
      : L(L), LI(LI), SE(SE), DT(DT) {}

  /// Analyze \p L and return true if it is a counted loop this utility can split:
  /// a bottom-tested single-exit loop in LCSSA form with a unique unit-step
  /// integer induction and a computable trip count. Must succeed before split().
  LLVM_ABI bool isLegal();

  /// Return the loop's induction variable. Valid only after isLegal() succeeds.
  PHINode *getInductionVariable() const { return Induction; }

  /// Append an inclusive partition range [Start, End] in iteration order.
  /// Partitions must tile the whole space: first Start = induction start, each
  /// later Start = previous End +/- step, last End = induction end (desc: S >= E).
  ///
  /// Bounds must be loop-invariant and representable in the induction type
  /// without wrapping: a Start +/- offset that wraps past TYPE_MAX/MIN/0 looks
  /// in-range and silently miscompiles. See LoopSplitUtils.cpp for the rationale.
  ///
  /// Every partition is guarded by default; use avoidPartitionGuard() to opt out.
  LLVM_ABI void addPartition(const SCEV *Start, const SCEV *End);

  /// Suppress the entry guard for partition \p PartitionIndex (already added). Use
  /// only for a partition the caller can prove runs at least once; for a runtime-
  /// empty partition this is incorrect and yields one spurious iteration.
  LLVM_ABI void avoidPartitionGuard(unsigned PartitionIndex);

  unsigned getNumPartitions() const { return Partitions.size(); }

  /// Perform the split. Requires a successful isLegal() and at least two
  /// partitions. Returns true if the loop was rewritten.
  LLVM_ABI bool split();

  /// Return the counterpart of original-loop value \p V in partition
  /// \p PartitionIndex (0-based). Partition 0 maps values to themselves; a later
  /// partition returns the clone, or null if not cloned. Valid only after split().
  LLVM_ABI Value *getPartitionValue(const Value *V,
                                    unsigned PartitionIndex) const;

  /// Return the original-to-clone value map for the partition at
  /// \p PartitionIndex, for callers that want to remap many values. Null for
  /// partition 0 (identity) and for any partition that was not cloned.
  LLVM_ABI const ValueToValueMapTy *
  getPartitionValueMap(unsigned PartitionIndex) const;

private:
  /// Everything known about one partition: the caller-supplied range plus the
  /// state split() derives. Indexed by partition number in \c Partitions.
  struct PartitionInfo {
    // Set by addPartition() / avoidPartitionGuard() before split():
    const SCEV *StartExpr = nullptr; // inclusive iteration range [Start, End].
    const SCEV *EndExpr = nullptr;
    bool Guarded = true; // emit an entry guard?

    // Filled in by split():
    std::unique_ptr<ValueToValueMapTy> VMap; // null for partition 0 (identity).
    Value *StartVal = nullptr;               // expanded start.
    Value *SelEnd = nullptr;                 // clamped end min(End, indEnd).
    bool Empty = false;                      // provably zero-iteration.
    BasicBlock *GuardBlock = nullptr;
    BasicBlock *Preheader = nullptr;
    BasicBlock *Exit = nullptr;
    Loop *SubLoop = nullptr;
    Value *LatchIndOp = nullptr; // induction operand of the latch compare.
  };

  /// Per-split() scratch threaded through the phase helpers (the escaping
  /// values, new blocks, etc.). A pure transform internal, so it is defined in
  /// the implementation file.
  struct SplitState;

  Loop *L;
  LoopInfo *LI;
  ScalarEvolution *SE;
  DominatorTree *DT;

  // Induction analysis, populated by isLegal().
  PHINode *Induction = nullptr;
  ICmpInst *LatchCmp = nullptr;        // the loop's latch exit compare.
  Value *LatchIndOperand = nullptr;    // induction operand of the latch compare.
  bool LatchUsesInductionPHI = false;  // latch compares the PHI, not the step.
  bool InductionIsSigned = false;      // iteration ordering signedness.
  bool InductionIsDescending = false;  // step is -1 (loop counts down).
  const SCEV *InductionEnd = nullptr;

  /// One record per partition, in add order.
  SmallVector<PartitionInfo, 4> Partitions;

  /// Find and validate the induction recurrence; returns its add-recurrence, or
  /// null if the loop has no suitable induction.
  const SCEVAddRecExpr *analyzeInduction();
  /// Determine the signedness of the iteration ordering from the latch compare
  /// and the recurrence's no-wrap flags; returns false if it cannot be proven.
  bool computeSignedness(const SCEVAddRecExpr *IndAR);

  // split() phase helpers, run in order; each is documented at its definition.
  /// Collect loop-carried and live-out values and split off the final exit.
  void collectEscapingValues(SplitState &S);
  /// Insert the entry guard ahead of partition 0 and update the dominator tree.
  void buildEntryGuard(SplitState &S);
  /// Expand each partition's start and clamped end into the entry guard.
  void expandPartitionBounds(SplitState &S);
  /// Pass 1: clone each later partition's sub-loop and create its guard/exit.
  void clonePartitions(SplitState &S);
  /// Pass 2: emit each guard, clamp each latch, and chain the partitions.
  void chainPartitions(SplitState &S);
  /// Rebuild SSA for every escaping value with a per-value SSAUpdater.
  void reconstructSSA(SplitState &S);
  /// Clamp \p PL's latch so it iterates only within [start, \p SelEnd].
  void rewriteLatch(Loop *PL, Value *IndOp, Value *SelEnd, BasicBlock *Exit);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOPSPLITUTILS_H
