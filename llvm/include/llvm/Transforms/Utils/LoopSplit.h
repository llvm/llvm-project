//===- LoopSplit.h - Split a loop's iteration space -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Splits a counted loop's iteration space into a chain of per-partition
// sub-loops. See LoopSplit.cpp for the structure produced.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPSPLIT_H
#define LLVM_TRANSFORMS_UTILS_LOOPSPLIT_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {

class DominatorTree;
class SCEV;
class SCEVExpander;
class ScalarEvolution;

/// Splits a counted loop into a chain of per-partition sub-loops.
///
/// Usage:
/// \code
///   if (auto LS = LoopSplit::get(L, LI, SE, DT)) {
///     LS->addPartition(S0, E0);   // one call per partition, in order
///     LS->addPartition(S1, E1);
///     LS->split();
///   }
/// \endcode
class LoopSplit {
public:
  /// Analyze \p L and, if it is a counted loop this utility can split, return a
  /// LoopSplit ready for addPartition() and split(). Otherwise return
  /// std::nullopt. Eligible loops are bottom-tested single-exit loops in LCSSA
  /// form with dedicated exits, no loop-carried and no escaping values, a
  /// unique unit-step integer induction, and a computable trip count that
  /// cannot wrap.
  LLVM_ABI static std::optional<LoopSplit>
  get(Loop *L, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT);

  /// Return the loop's induction variable. Valid only on a legal LoopSplit.
  LLVM_ABI PHINode *getInductionVariable() const {
    return L->getInductionVariable(*SE);
  }

  /// The induction value on the last iteration, which the final partition must
  /// end at. Valid only on a legal LoopSplit.
  LLVM_ABI const SCEV *getInductionEnd() const { return InductionEnd; }

  /// Append an inclusive partition range [Start, End] in iteration order.
  /// Partitions must tile the whole space: first Start = induction start, each
  /// later Start = previous End +/- step, last End = induction end (desc: S >=
  /// E).
  ///
  /// Both bounds must have the induction type and be loop-invariant. They must
  /// also stay within the iteration space, extended by the one step past its
  /// start that an empty partition needs; legality analysis has proven that
  /// much representable. Reaching further wraps past TYPE_MAX/MIN/0 into a
  /// bound that still looks in range, which silently miscompiles. See
  /// LoopSplit.cpp for the rationale.
  LLVM_ABI void addPartition(const SCEV *Start, const SCEV *End);

  LLVM_ABI size_t getNumPartitions() const { return Partitions.size(); }

  /// Perform the split. Requires at least two partitions. Returns true if the
  /// loop was rewritten.
  LLVM_ABI bool split();

private:
  LoopSplit(Loop *L, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT,
            const SCEV *InductionEnd, bool InductionIsSigned, bool Descending)
      : L(L), LI(LI), SE(SE), DT(DT), InductionEnd(InductionEnd),
        InductionIsSigned(InductionIsSigned), Descending(Descending) {}

  /// Everything known about one partition: the caller-supplied range plus the
  /// state split() derives. Indexed by partition number in \c Partitions.
  struct PartitionInfo {
    PartitionInfo(const SCEV *StartExpr, const SCEV *EndExpr)
        : StartExpr(StartExpr), EndExpr(EndExpr) {}

    // Set by addPartition() before split():
    const SCEV *StartExpr; // inclusive iteration range [Start, End].
    const SCEV *EndExpr;

    // Filled in by split():
    Value *StartVal = nullptr; // expanded start.
    Value *SelEnd = nullptr;   // clamped end min(End, indEnd).
    BasicBlock *GuardBlock = nullptr;
    BasicBlock *Preheader = nullptr;
    BasicBlock *Exit = nullptr;
    Loop *SubLoop = nullptr;
    PHINode *IndPHI = nullptr; // this partition's induction variable.
  };

  /// Per-split() scratch threaded through the phase helpers: the blocks the
  /// transform creates. A pure transform internal, so it is defined in the
  /// implementation file.
  struct SplitState;

  Loop *L;
  LoopInfo *LI;
  ScalarEvolution *SE;
  DominatorTree *DT;

  // Induction analysis, populated during legality analysis.
  const SCEV *InductionEnd = nullptr; // value on the last iteration.
  bool InductionIsSigned = false;     // iteration ordering signedness.
  bool Descending = false;            // step is -1 (the loop counts down).

  /// One record per partition, in add order.
  SmallVector<PartitionInfo, 4> Partitions;

  // split() phase helpers, run in order; each is documented at its definition.
  /// Split the final exit off the loop exit block.
  void splitFinalExit(SplitState &S);
  /// Expand each partition's start and clamped end into the entry guard.
  void expandPartitionBounds(SplitState &S, SCEVExpander &Expander);
  /// Clone each later partition's sub-loop and create its guard/exit.
  void clonePartitions(SplitState &S);
  /// Emit each guard, clamp each latch, and chain the partitions.
  void chainPartitions(SplitState &S);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOPSPLIT_H
