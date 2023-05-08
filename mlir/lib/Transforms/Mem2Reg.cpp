//===- Mem2Reg.cpp - Promotes memory slots into values ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Mem2Reg.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

namespace mlir {
#define GEN_PASS_DEF_MEM2REG
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// mem2reg
///
/// This pass turns unnecessary uses of automatically allocated memory slots
/// into direct Value-based operations. For example, it will simplify storing a
/// constant in a memory slot to immediately load it to a direct use of that
/// constant. In other words, given a memory slot addressed by a non-aliased
/// "pointer" Value, mem2reg removes all the uses of that pointer.
///
/// Within a block, this is done by following the chain of stores and loads of
/// the slot and replacing the results of loads with the values previously
/// stored. If a load happens before any other store, a poison value is used
/// instead.
///
/// Control flow can create situations where a load could be replaced by
/// multiple possible stores depending on the control flow path taken. As a
/// result, this pass must introduce new block arguments in some blocks to
/// accomodate for the multiple possible definitions. Each predecessor will
/// populate the block argument with the definition reached at its end. With
/// this, the value stored can be well defined at block boundaries, allowing
/// the propagation of replacement through blocks.
///
/// This pass computes this transformation in four main steps:
/// - A first step computes the list of operations that transitively use the
/// memory slot we would like to promote. The purpose of this phase is to
/// identify which uses must be removed to promote the slot, either by rewiring
/// the user or deleting it. Naturally, direct uses of the slot must be removed.
/// Sometimes additional uses must also be removed: this is notably the case
/// when a direct user of the slot cannot rewire its use and must delete itself,
/// and thus must make its users no longer use it. If any of those uses cannot
/// be removed by their users in any way, promotion cannot continue: this is
/// decided at this step.
/// - A second step computes the list of blocks where a block argument will be
/// needed ("merge points") without mutating the IR. These blocks are the blocks
/// leading to a definition clash between two predecessors. Such blocks happen
/// to be the Iterated Dominance Frontier (IDF) of the set of blocks containing
/// a store, as they represent the point where a clear defining dominator stops
/// existing. Computing this information in advance allows making sure the
/// terminators that will forward values are capable of doing so (inability to
/// do so aborts promotion at this step).
/// - A third step computes the reaching definition of the memory slot at each
/// blocking user. This is the core of the mem2reg algorithm, also known as
/// load-store forwarding. This analyses loads and stores and propagates which
/// value must be stored in the slot at each blocking user.  This is achieved by
/// doing a depth-first walk of the dominator tree of the function. This is
/// sufficient because the reaching definition at the beginning of a block is
/// either its new block argument if it is a merge block, or the definition
/// reaching the end of its immediate dominator (parent in the dominator tree).
/// We can therefore propagate this information down the dominator tree to
/// proceed with renaming within blocks.
/// - The final fourth step uses the reaching definition to remove blocking uses
/// in topological order.
///
/// The two first steps do not mutate IR because promotion can still be aborted
/// at this point. Once the two last steps are reached, promotion is guaranteed
/// to succeed, allowing to start mutating IR.
///
/// For further reading, chapter three of SSA-based Compiler Design [1]
/// showcases SSA construction, where mem2reg is an adaptation of the same
/// process.
///
/// [1]: Rastello F. & Bouchez Tichadou F., SSA-based Compiler Design (2022),
///      Springer.

namespace {

/// The SlotPromoter handles the state of promoting a memory slot. It wraps a
/// slot and its associated allocator, along with analysis results related to
/// the slot.
class SlotPromoter {
public:
  SlotPromoter(MemorySlot slot, PromotableAllocationOpInterface allocator,
               OpBuilder &builder, DominanceInfo &dominance);

  /// Prepare data for the promotion of the slot while checking if it can be
  /// promoted. Succeeds if the slot can be promoted. This method does not
  /// mutate IR.
  LogicalResult prepareSlotPromotion();

  /// Actually promotes the slot by mutating IR. This method must only be
  /// called after a successful call to `SlotPromoter::prepareSlotPromotion`.
  /// Promoting a slot does not invalidate the preparation of other slots.
  void promoteSlot();

private:
  /// This is the first step of the promotion algorithm.
  /// Computes the transitive uses of the slot that block promotion. This finds
  /// uses that would block the promotion, checks that the operation has a
  /// solution to remove the blocking use, and potentially forwards the analysis
  /// if the operation needs further blocking uses resolved to resolve its own
  /// uses (typically, removing its users because it will delete itself to
  /// resolve its own blocking uses). This will fail if one of the transitive
  /// users cannot remove a requested use, and should prevent promotion.
  LogicalResult computeBlockingUses();

  /// Computes in which blocks the value stored in the slot is actually used,
  /// meaning blocks leading to a load. This method uses `definingBlocks`, the
  /// set of blocks containing a store to the slot (defining the value of the
  /// slot).
  SmallPtrSet<Block *, 16>
  computeSlotLiveIn(SmallPtrSetImpl<Block *> &definingBlocks);

  /// This is the second step of the promotion algorithm.
  /// Computes the points in which multiple re-definitions of the slot's value
  /// (stores) may conflict.
  void computeMergePoints();

  /// Ensures predecessors of merge points can properly provide their current
  /// definition of the value stored in the slot to the merge point. This can
  /// notably be an issue if the terminator used does not have the ability to
  /// forward values through block operands.
  bool areMergePointsUsable();

  /// Computes the reaching definition for all the operations that require
  /// promotion. `reachingDef` is the value the slot should contain at the
  /// beginning of the block. This method returns the reached definition at the
  /// end of the block.
  Value computeReachingDefInBlock(Block *block, Value reachingDef);

  /// This is the third step of the promotion algorithm.
  /// Computes the reaching definition for all the operations that require
  /// promotion. `reachingDef` corresponds to the initial value the
  /// slot will contain before any write, typically a poison value.
  void computeReachingDefInRegion(Region *region, Value reachingDef);

  /// This is the fourth step of the promotion algorithm.
  /// Removes the blocking uses of the slot, in topological order.
  void removeBlockingUses();

  /// Lazily-constructed default value representing the content of the slot when
  /// no store has been executed. This function may mutate IR.
  Value getLazyDefaultValue();

  MemorySlot slot;
  PromotableAllocationOpInterface allocator;
  OpBuilder &builder;
  /// Potentially non-initialized default value. Use `lazyDefaultValue` to
  /// initialize it on demand.
  Value defaultValue;
  /// Blocks where multiple definitions of the slot value clash.
  SmallPtrSet<Block *, 8> mergePoints;
  /// Contains, for each operation, which uses must be eliminated by promotion.
  /// This is a DAG structure because an operation that must eliminate some of
  /// its uses always comes from a request from an operation that must
  /// eliminate some of its own uses.
  DenseMap<Operation *, SmallPtrSet<OpOperand *, 4>> userToBlockingUses;
  /// Contains the reaching definition at this operation. Reaching definitions
  /// are only computed for promotable memory operations with blocking uses.
  DenseMap<PromotableMemOpInterface, Value> reachingDefs;
  DominanceInfo &dominance;
};

} // namespace

SlotPromoter::SlotPromoter(MemorySlot slot,
                           PromotableAllocationOpInterface allocator,
                           OpBuilder &builder, DominanceInfo &dominance)
    : slot(slot), allocator(allocator), builder(builder), dominance(dominance) {
#ifndef NDEBUG
  auto isResultOrNewBlockArgument = [&]() {
    if (BlockArgument arg = slot.ptr.dyn_cast<BlockArgument>())
      return arg.getOwner()->getParentOp() == allocator;
    return slot.ptr.getDefiningOp() == allocator;
  };

  assert(isResultOrNewBlockArgument() &&
         "a slot must be a result of the allocator or an argument of the child "
         "regions of the allocator");
#endif // NDEBUG
}

Value SlotPromoter::getLazyDefaultValue() {
  if (defaultValue)
    return defaultValue;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(slot.ptr.getParentBlock());
  return defaultValue = allocator.getDefaultValue(slot, builder);
}

LogicalResult SlotPromoter::computeBlockingUses() {
  // The promotion of an operation may require the promotion of further
  // operations (typically, removing operations that use an operation that must
  // delete itself). We thus need to start from the use of the slot pointer and
  // propagate further requests through the forward slice.

  // First insert that all immediate users of the slot pointer must no longer
  // use it.
  for (OpOperand &use : slot.ptr.getUses()) {
    SmallPtrSet<OpOperand *, 4> &blockingUses =
        userToBlockingUses.getOrInsertDefault(use.getOwner());
    blockingUses.insert(&use);
  }

  // Then, propagate the requirements for the removal of uses. The
  // topologically-sorted forward slice allows for all blocking uses of an
  // operation to have been computed before we reach it. Operations are
  // traversed in topological order of their uses, starting from the slot
  // pointer.
  SetVector<Operation *> forwardSlice;
  mlir::getForwardSlice(slot.ptr, &forwardSlice);
  for (Operation *user : forwardSlice) {
    // If the next operation has no blocking uses, everything is fine.
    if (!userToBlockingUses.contains(user))
      continue;

    SmallPtrSet<OpOperand *, 4> &blockingUses = userToBlockingUses[user];

    SmallVector<OpOperand *> newBlockingUses;
    // If the operation decides it cannot deal with removing the blocking uses,
    // promotion must fail.
    if (auto promotable = dyn_cast<PromotableOpInterface>(user)) {
      if (!promotable.canUsesBeRemoved(slot, blockingUses, newBlockingUses))
        return failure();
    } else if (auto promotable = dyn_cast<PromotableMemOpInterface>(user)) {
      if (!promotable.canUsesBeRemoved(slot, blockingUses, newBlockingUses))
        return failure();
    } else {
      // An operation that has blocking uses must be promoted. If it is not
      // promotable, promotion must fail.
      return failure();
    }

    // Then, register any new blocking uses for coming operations.
    for (OpOperand *blockingUse : newBlockingUses) {
      assert(llvm::is_contained(user->getResults(), blockingUse->get()));

      SmallPtrSetImpl<OpOperand *> &newUserBlockingUseSet =
          userToBlockingUses.getOrInsertDefault(blockingUse->getOwner());
      newUserBlockingUseSet.insert(blockingUse);
    }
  }

  // Because this pass currently only supports analysing the parent region of
  // the slot pointer, if a promotable memory op that needs promotion is
  // outside of this region, promotion must fail because it will be impossible
  // to provide a valid `reachingDef` for it.
  for (auto &[toPromote, _] : userToBlockingUses)
    if (isa<PromotableMemOpInterface>(toPromote) &&
        toPromote->getParentRegion() != slot.ptr.getParentRegion())
      return failure();

  return success();
}

SmallPtrSet<Block *, 16>
SlotPromoter::computeSlotLiveIn(SmallPtrSetImpl<Block *> &definingBlocks) {
  SmallPtrSet<Block *, 16> liveIn;

  // The worklist contains blocks in which it is known that the slot value is
  // live-in. The further blocks where this value is live-in will be inferred
  // from these.
  SmallVector<Block *> liveInWorkList;

  // Blocks with a load before any other store to the slot are the starting
  // points of the analysis. The slot value is definitely live-in in those
  // blocks.
  SmallPtrSet<Block *, 16> visited;
  for (Operation *user : slot.ptr.getUsers()) {
    if (visited.contains(user->getBlock()))
      continue;
    visited.insert(user->getBlock());

    for (Operation &op : user->getBlock()->getOperations()) {
      if (auto memOp = dyn_cast<PromotableMemOpInterface>(op)) {
        // If this operation loads the slot, it is loading from it before
        // ever writing to it, so the value is live-in in this block.
        if (memOp.loadsFrom(slot)) {
          liveInWorkList.push_back(user->getBlock());
          break;
        }

        // If we store to the slot, further loads will see that value.
        // Because we did not meet any load before, the value is not live-in.
        if (memOp.getStored(slot))
          break;
      }
    }
  }

  // The information is then propagated to the predecessors until a def site
  // (store) is found.
  while (!liveInWorkList.empty()) {
    Block *liveInBlock = liveInWorkList.pop_back_val();

    if (!liveIn.insert(liveInBlock).second)
      continue;

    // If a predecessor is a defining block, either:
    // - It has a load before its first store, in which case it is live-in but
    // has already been processed in the initialisation step.
    // - It has a store before any load, in which case it is not live-in.
    // We can thus at this stage insert to the worklist only predecessors that
    // are not defining blocks.
    for (Block *pred : liveInBlock->getPredecessors())
      if (!definingBlocks.contains(pred))
        liveInWorkList.push_back(pred);
  }

  return liveIn;
}

using IDFCalculator = llvm::IDFCalculatorBase<Block, false>;
void SlotPromoter::computeMergePoints() {
  if (slot.ptr.getParentRegion()->hasOneBlock())
    return;

  IDFCalculator idfCalculator(dominance.getDomTree(slot.ptr.getParentRegion()));

  SmallPtrSet<Block *, 16> definingBlocks;
  for (Operation *user : slot.ptr.getUsers())
    if (auto storeOp = dyn_cast<PromotableMemOpInterface>(user))
      if (storeOp.getStored(slot))
        definingBlocks.insert(user->getBlock());

  idfCalculator.setDefiningBlocks(definingBlocks);

  SmallPtrSet<Block *, 16> liveIn = computeSlotLiveIn(definingBlocks);
  idfCalculator.setLiveInBlocks(liveIn);

  SmallVector<Block *> mergePointsVec;
  idfCalculator.calculate(mergePointsVec);

  mergePoints.insert(mergePointsVec.begin(), mergePointsVec.end());
}

bool SlotPromoter::areMergePointsUsable() {
  for (Block *mergePoint : mergePoints)
    for (Block *pred : mergePoint->getPredecessors())
      if (!isa<BranchOpInterface>(pred->getTerminator()))
        return false;

  return true;
}

Value SlotPromoter::computeReachingDefInBlock(Block *block, Value reachingDef) {
  for (Operation &op : block->getOperations()) {
    if (auto memOp = dyn_cast<PromotableMemOpInterface>(op)) {
      if (userToBlockingUses.contains(memOp))
        reachingDefs.insert({memOp, reachingDef});

      if (Value stored = memOp.getStored(slot))
        reachingDef = stored;
    }
  }

  return reachingDef;
}

void SlotPromoter::computeReachingDefInRegion(Region *region,
                                              Value reachingDef) {
  if (region->hasOneBlock()) {
    computeReachingDefInBlock(&region->front(), reachingDef);
    return;
  }

  struct DfsJob {
    llvm::DomTreeNodeBase<Block> *block;
    Value reachingDef;
  };

  SmallVector<DfsJob> dfsStack;

  auto &domTree = dominance.getDomTree(slot.ptr.getParentRegion());

  dfsStack.emplace_back<DfsJob>(
      {domTree.getNode(&region->front()), reachingDef});

  while (!dfsStack.empty()) {
    DfsJob job = dfsStack.pop_back_val();
    Block *block = job.block->getBlock();

    if (mergePoints.contains(block)) {
      BlockArgument blockArgument =
          block->addArgument(slot.elemType, slot.ptr.getLoc());
      builder.setInsertionPointToStart(block);
      allocator.handleBlockArgument(slot, blockArgument, builder);
      job.reachingDef = blockArgument;
    }

    job.reachingDef = computeReachingDefInBlock(block, job.reachingDef);

    if (auto terminator = dyn_cast<BranchOpInterface>(block->getTerminator())) {
      for (BlockOperand &blockOperand : terminator->getBlockOperands()) {
        if (mergePoints.contains(blockOperand.get())) {
          if (!job.reachingDef)
            job.reachingDef = getLazyDefaultValue();
          terminator.getSuccessorOperands(blockOperand.getOperandNumber())
              .append(job.reachingDef);
        }
      }
    }

    for (auto *child : job.block->children())
      dfsStack.emplace_back<DfsJob>({child, job.reachingDef});
  }
}

void SlotPromoter::removeBlockingUses() {
  llvm::SetVector<Operation *> usersToRemoveUses;
  for (auto &user : llvm::make_first_range(userToBlockingUses))
    usersToRemoveUses.insert(user);
  SetVector<Operation *> sortedUsersToRemoveUses =
      mlir::topologicalSort(usersToRemoveUses);

  llvm::SmallVector<Operation *> toErase;
  for (Operation *toPromote : llvm::reverse(sortedUsersToRemoveUses)) {
    if (auto toPromoteMemOp = dyn_cast<PromotableMemOpInterface>(toPromote)) {
      Value reachingDef = reachingDefs.lookup(toPromoteMemOp);
      // If no reaching definition is known, this use is outside the reach of
      // the slot. The default value should thus be used.
      if (!reachingDef)
        reachingDef = getLazyDefaultValue();

      builder.setInsertionPointAfter(toPromote);
      if (toPromoteMemOp.removeBlockingUses(slot, userToBlockingUses[toPromote],
                                            builder, reachingDef) ==
          DeletionKind::Delete)
        toErase.push_back(toPromote);

      continue;
    }

    auto toPromoteBasic = cast<PromotableOpInterface>(toPromote);
    builder.setInsertionPointAfter(toPromote);
    if (toPromoteBasic.removeBlockingUses(slot, userToBlockingUses[toPromote],
                                          builder) == DeletionKind::Delete)
      toErase.push_back(toPromote);
  }

  for (Operation *toEraseOp : toErase)
    toEraseOp->erase();

  assert(slot.ptr.use_empty() &&
         "after promotion, the slot pointer should not be used anymore");
}

void SlotPromoter::promoteSlot() {
  computeReachingDefInRegion(slot.ptr.getParentRegion(), {});

  // Now that reaching definitions are known, remove all users.
  removeBlockingUses();

  // Update terminators in dead branches to forward default if they are
  // succeeded by a merge points.
  for (Block *mergePoint : mergePoints) {
    for (BlockOperand &use : mergePoint->getUses()) {
      auto user = cast<BranchOpInterface>(use.getOwner());
      SuccessorOperands succOperands =
          user.getSuccessorOperands(use.getOperandNumber());
      assert(succOperands.size() == mergePoint->getNumArguments() ||
             succOperands.size() + 1 == mergePoint->getNumArguments());
      if (succOperands.size() + 1 == mergePoint->getNumArguments())
        succOperands.append(getLazyDefaultValue());
    }
  }

  allocator.handlePromotionComplete(slot, defaultValue);
}

LogicalResult SlotPromoter::prepareSlotPromotion() {
  // First, find the set of operations that will need to be changed for the
  // promotion to happen. These operations need to resolve some of their uses,
  // either by rewiring them or simply deleting themselves. If any of them
  // cannot find a way to resolve their blocking uses, we abort the promotion.
  if (failed(computeBlockingUses()))
    return failure();

  // Then, compute blocks in which two or more definitions of the allocated
  // variable may conflict. These blocks will need a new block argument to
  // accomodate this.
  computeMergePoints();

  // The slot can be promoted if the block arguments to be created can
  // actually be populated with values, which may not be possible depending
  // on their predecessors.
  return success(areMergePointsUsable());
}

LogicalResult mlir::tryToPromoteMemorySlots(
    ArrayRef<PromotableAllocationOpInterface> allocators, OpBuilder &builder,
    DominanceInfo &dominance) {
  // Actual promotion may invalidate the dominance analysis, so slot promotion
  // is prepated in batches.
  SmallVector<SlotPromoter> toPromote;
  for (PromotableAllocationOpInterface allocator : allocators) {
    for (MemorySlot slot : allocator.getPromotableSlots()) {
      if (slot.ptr.use_empty())
        continue;

      SlotPromoter promoter(slot, allocator, builder, dominance);
      if (succeeded(promoter.prepareSlotPromotion()))
        toPromote.emplace_back(std::move(promoter));
    }
  }

  for (SlotPromoter &promoter : toPromote)
    promoter.promoteSlot();

  return success(!toPromote.empty());
}

namespace {

struct Mem2Reg : impl::Mem2RegBase<Mem2Reg> {
  void runOnOperation() override {
    Operation *scopeOp = getOperation();
    bool changed = false;

    for (Region &region : scopeOp->getRegions()) {
      if (region.getBlocks().empty())
        continue;

      OpBuilder builder(&region.front(), region.front().begin());

      // Promoting a slot can allow for further promotion of other slots,
      // promotion is tried until no promotion succeeds.
      while (true) {
        DominanceInfo &dominance = getAnalysis<DominanceInfo>();

        SmallVector<PromotableAllocationOpInterface> allocators;
        // Build a list of allocators to attempt to promote the slots of.
        for (Block &block : region)
          for (Operation &op : block.getOperations())
            if (auto allocator = dyn_cast<PromotableAllocationOpInterface>(op))
              allocators.emplace_back(allocator);

        // Attempt promoting until no promotion succeeds.
        if (failed(tryToPromoteMemorySlots(allocators, builder, dominance)))
          break;

        changed = true;
        getAnalysisManager().invalidate({});
      }
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
