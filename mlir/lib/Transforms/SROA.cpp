//===-- SROA.cpp - Scalar Replacement Of Aggregates -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/SROA.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_SROA
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "sroa"

using namespace mlir;

namespace {

/// Information computed by destructurable memory slot analysis used to perform
/// actual destructuring of the slot. This struct is only constructed if
/// destructuring is possible, and contains the necessary data to perform it.
struct MemorySlotDestructuringInfo {
  /// Set of the indices that are actually used when accessing the subelements.
  SmallPtrSet<Attribute, 8> usedIndices;
  /// Blocking uses of a given user of the memory slot that must be eliminated.
  DenseMap<Operation *, SmallPtrSet<OpOperand *, 4>> userToBlockingUses;
  /// List of potentially indirect accessors of the memory slot that need
  /// rewiring.
  SmallVector<DestructurableAccessorOpInterface> accessors;
};

} // namespace

/// Computes information for slot destructuring. This will compute whether this
/// slot can be destructured and data to perform the destructuring. Returns
/// nothing if the slot cannot be destructured or if there is no useful work to
/// be done.
static std::optional<MemorySlotDestructuringInfo>
computeDestructuringInfo(DestructurableMemorySlot &slot) {
  assert(isa<DestructurableTypeInterface>(slot.elemType));

  if (slot.ptr.use_empty())
    return {};

  MemorySlotDestructuringInfo info;

  SmallVector<MemorySlot> usedSafelyWorklist;

  auto scheduleAsBlockingUse = [&](OpOperand &use) {
    SmallPtrSetImpl<OpOperand *> &blockingUses =
        info.userToBlockingUses.getOrInsertDefault(use.getOwner());
    blockingUses.insert(&use);
  };

  // Initialize the analysis with the immediate users of the slot.
  for (OpOperand &use : slot.ptr.getUses()) {
    if (auto accessor =
            dyn_cast<DestructurableAccessorOpInterface>(use.getOwner())) {
      if (accessor.canRewire(slot, info.usedIndices, usedSafelyWorklist)) {
        info.accessors.push_back(accessor);
        continue;
      }
    }

    // If it cannot be shown that the operation uses the slot safely, maybe it
    // can be promoted out of using the slot?
    scheduleAsBlockingUse(use);
  }

  SmallPtrSet<OpOperand *, 16> visited;
  while (!usedSafelyWorklist.empty()) {
    MemorySlot mustBeUsedSafely = usedSafelyWorklist.pop_back_val();
    for (OpOperand &subslotUse : mustBeUsedSafely.ptr.getUses()) {
      if (!visited.insert(&subslotUse).second)
        continue;
      Operation *subslotUser = subslotUse.getOwner();

      if (auto memOp = dyn_cast<SafeMemorySlotAccessOpInterface>(subslotUser))
        if (succeeded(memOp.ensureOnlySafeAccesses(mustBeUsedSafely,
                                                   usedSafelyWorklist)))
          continue;

      // If it cannot be shown that the operation uses the slot safely, maybe it
      // can be promoted out of using the slot?
      scheduleAsBlockingUse(subslotUse);
    }
  }

  SetVector<Operation *> forwardSlice;
  mlir::getForwardSlice(slot.ptr, &forwardSlice);
  for (Operation *user : forwardSlice) {
    // If the next operation has no blocking uses, everything is fine.
    if (!info.userToBlockingUses.contains(user))
      continue;

    SmallPtrSet<OpOperand *, 4> &blockingUses = info.userToBlockingUses[user];
    auto promotable = dyn_cast<PromotableOpInterface>(user);

    // An operation that has blocking uses must be promoted. If it is not
    // promotable, destructuring must fail.
    if (!promotable)
      return {};

    SmallVector<OpOperand *> newBlockingUses;
    // If the operation decides it cannot deal with removing the blocking uses,
    // destructuring must fail.
    if (!promotable.canUsesBeRemoved(blockingUses, newBlockingUses))
      return {};

    // Then, register any new blocking uses for coming operations.
    for (OpOperand *blockingUse : newBlockingUses) {
      assert(llvm::is_contained(user->getResults(), blockingUse->get()));

      SmallPtrSetImpl<OpOperand *> &newUserBlockingUseSet =
          info.userToBlockingUses.getOrInsertDefault(blockingUse->getOwner());
      newUserBlockingUseSet.insert(blockingUse);
    }
  }

  return info;
}

/// Performs the destructuring of a destructible slot given associated
/// destructuring information. The provided slot will be destructured in
/// subslots as specified by its allocator.
static void destructureSlot(DestructurableMemorySlot &slot,
                            DestructurableAllocationOpInterface allocator,
                            RewriterBase &rewriter,
                            MemorySlotDestructuringInfo &info,
                            const SROAStatistics &statistics) {
  RewriterBase::InsertionGuard guard(rewriter);

  rewriter.setInsertionPointToStart(slot.ptr.getParentBlock());
  DenseMap<Attribute, MemorySlot> subslots =
      allocator.destructure(slot, info.usedIndices, rewriter);

  if (statistics.slotsWithMemoryBenefit &&
      slot.elementPtrs.size() != info.usedIndices.size())
    (*statistics.slotsWithMemoryBenefit)++;

  if (statistics.maxSubelementAmount)
    statistics.maxSubelementAmount->updateMax(slot.elementPtrs.size());

  SetVector<Operation *> usersToRewire;
  for (Operation *user : llvm::make_first_range(info.userToBlockingUses))
    usersToRewire.insert(user);
  for (DestructurableAccessorOpInterface accessor : info.accessors)
    usersToRewire.insert(accessor);
  usersToRewire = mlir::topologicalSort(usersToRewire);

  llvm::SmallVector<Operation *> toErase;
  for (Operation *toRewire : llvm::reverse(usersToRewire)) {
    rewriter.setInsertionPointAfter(toRewire);
    if (auto accessor = dyn_cast<DestructurableAccessorOpInterface>(toRewire)) {
      if (accessor.rewire(slot, subslots, rewriter) == DeletionKind::Delete)
        toErase.push_back(accessor);
      continue;
    }

    auto promotable = cast<PromotableOpInterface>(toRewire);
    if (promotable.removeBlockingUses(info.userToBlockingUses[promotable],
                                      rewriter) == DeletionKind::Delete)
      toErase.push_back(promotable);
  }

  for (Operation *toEraseOp : toErase)
    rewriter.eraseOp(toEraseOp);

  assert(slot.ptr.use_empty() && "after destructuring, the original slot "
                                 "pointer should no longer be used");

  LLVM_DEBUG(llvm::dbgs() << "[sroa] Destructured memory slot: " << slot.ptr
                          << "\n");

  if (statistics.destructuredAmount)
    (*statistics.destructuredAmount)++;

  allocator.handleDestructuringComplete(slot, rewriter);
}

LogicalResult mlir::tryToDestructureMemorySlots(
    ArrayRef<DestructurableAllocationOpInterface> allocators,
    RewriterBase &rewriter, SROAStatistics statistics) {
  bool destructuredAny = false;

  for (DestructurableAllocationOpInterface allocator : allocators) {
    for (DestructurableMemorySlot slot : allocator.getDestructurableSlots()) {
      std::optional<MemorySlotDestructuringInfo> info =
          computeDestructuringInfo(slot);
      if (!info)
        continue;

      destructureSlot(slot, allocator, rewriter, *info, statistics);
      destructuredAny = true;
    }
  }

  return success(destructuredAny);
}

namespace {

struct SROA : public impl::SROABase<SROA> {
  using impl::SROABase<SROA>::SROABase;

  void runOnOperation() override {
    Operation *scopeOp = getOperation();

    SROAStatistics statistics{&destructuredAmount, &slotsWithMemoryBenefit,
                              &maxSubelementAmount};

    bool changed = false;

    for (Region &region : scopeOp->getRegions()) {
      if (region.getBlocks().empty())
        continue;

      OpBuilder builder(&region.front(), region.front().begin());
      IRRewriter rewriter(builder);

      // Destructuring a slot can allow for further destructuring of other
      // slots, destructuring is tried until no destructuring succeeds.
      while (true) {
        SmallVector<DestructurableAllocationOpInterface> allocators;
        // Build a list of allocators to attempt to destructure the slots of.
        // TODO: Update list on the fly to avoid repeated visiting of the same
        // allocators.
        region.walk([&](DestructurableAllocationOpInterface allocator) {
          allocators.emplace_back(allocator);
        });

        if (failed(
                tryToDestructureMemorySlots(allocators, rewriter, statistics)))
          break;

        changed = true;
      }
    }
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
