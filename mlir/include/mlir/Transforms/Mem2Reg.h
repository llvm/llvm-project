//===-- Mem2Reg.h - Mem2Reg definitions -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_MEM2REG_H
#define MLIR_TRANSFORMS_MEM2REG_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

namespace mlir {

/// Information computed during promotion analysis used to perform actual
/// promotion.
struct MemorySlotPromotionInfo {
  /// Blocks for which at least two definitions of the slot values clash.
  SmallPtrSet<Block *, 8> mergePoints;
  /// Contains, for each operation, which uses must be eliminated by promotion.
  /// This is a DAG structure because if an operation must eliminate some of
  /// its uses, it is because the defining ops of the blocking uses requested
  /// it. The defining ops therefore must also have blocking uses or be the
  /// starting point of the bloccking uses.
  DenseMap<Operation *, SmallPtrSet<OpOperand *, 4>> userToBlockingUses;
};

/// Computes information for basic slot promotion. This will check that direct
/// slot promotion can be performed, and provide the information to execute the
/// promotion. This does not mutate IR.
class MemorySlotPromotionAnalyzer {
public:
  MemorySlotPromotionAnalyzer(MemorySlot slot, DominanceInfo &dominance)
      : slot(slot), dominance(dominance) {}

  /// Computes the information for slot promotion if promotion is possible,
  /// returns nothing otherwise.
  std::optional<MemorySlotPromotionInfo> computeInfo();

private:
  /// Computes the transitive uses of the slot that block promotion. This finds
  /// uses that would block the promotion, checks that the operation has a
  /// solution to remove the blocking use, and potentially forwards the analysis
  /// if the operation needs further blocking uses resolved to resolve its own
  /// uses (typically, removing its users because it will delete itself to
  /// resolve its own blocking uses). This will fail if one of the transitive
  /// users cannot remove a requested use, and should prevent promotion.
  LogicalResult computeBlockingUses(
      DenseMap<Operation *, SmallPtrSet<OpOperand *, 4>> &userToBlockingUses);

  /// Computes in which blocks the value stored in the slot is actually used,
  /// meaning blocks leading to a load. This method uses `definingBlocks`, the
  /// set of blocks containing a store to the slot (defining the value of the
  /// slot).
  SmallPtrSet<Block *, 16>
  computeSlotLiveIn(SmallPtrSetImpl<Block *> &definingBlocks);

  /// Computes the points in which multiple re-definitions of the slot's value
  /// (stores) may conflict.
  void computeMergePoints(SmallPtrSetImpl<Block *> &mergePoints);

  /// Ensures predecessors of merge points can properly provide their current
  /// definition of the value stored in the slot to the merge point. This can
  /// notably be an issue if the terminator used does not have the ability to
  /// forward values through block operands.
  bool areMergePointsUsable(SmallPtrSetImpl<Block *> &mergePoints);

  MemorySlot slot;
  DominanceInfo &dominance;
};

/// The MemorySlotPromoter handles the state of promoting a memory slot. It
/// wraps a slot and its associated allocator. This will perform the mutation of
/// IR.
class MemorySlotPromoter {
public:
  MemorySlotPromoter(MemorySlot slot, PromotableAllocationOpInterface allocator,
                     OpBuilder &builder, DominanceInfo &dominance,
                     MemorySlotPromotionInfo info);

  /// Actually promotes the slot by mutating IR. Promoting a slot does not
  /// invalidate the MemorySlotPromotionInfo of other slots.
  void promoteSlot();

private:
  /// Computes the reaching definition for all the operations that require
  /// promotion. `reachingDef` is the value the slot should contain at the
  /// beginning of the block. This method returns the reached definition at the
  /// end of the block.
  Value computeReachingDefInBlock(Block *block, Value reachingDef);

  /// Computes the reaching definition for all the operations that require
  /// promotion. `reachingDef` corresponds to the initial value the
  /// slot will contain before any write, typically a poison value.
  void computeReachingDefInRegion(Region *region, Value reachingDef);

  /// Removes the blocking uses of the slot, in topological order.
  void removeBlockingUses();

  /// Lazily-constructed default value representing the content of the slot when
  /// no store has been executed. This function may mutate IR.
  Value getLazyDefaultValue();

  MemorySlot slot;
  PromotableAllocationOpInterface allocator;
  OpBuilder &builder;
  /// Potentially non-initialized default value. Use `getLazyDefaultValue` to
  /// initialize it on demand.
  Value defaultValue;
  /// Contains the reaching definition at this operation. Reaching definitions
  /// are only computed for promotable memory operations with blocking uses.
  DenseMap<PromotableMemOpInterface, Value> reachingDefs;
  DominanceInfo &dominance;
  MemorySlotPromotionInfo info;
};

/// Pattern applying mem2reg to the regions of the operations on which it
/// matches.
class Mem2RegPattern : public RewritePattern {
public:
  using RewritePattern::RewritePattern;

  Mem2RegPattern(MLIRContext *ctx, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

/// Attempts to promote the memory slots of the provided allocators. Succeeds if
/// at least one memory slot was promoted.
LogicalResult
tryToPromoteMemorySlots(ArrayRef<PromotableAllocationOpInterface> allocators,
                        OpBuilder &builder, DominanceInfo &dominance);

} // namespace mlir

#endif // MLIR_TRANSFORMS_MEM2REG_H
