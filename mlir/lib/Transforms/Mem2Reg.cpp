//===- Mem2Reg.cpp - Promotes memory slots into values ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Mem2Reg.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

namespace mlir {
#define GEN_PASS_DEF_MEM2REG
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "mem2reg"

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
/// accommodate for the multiple possible definitions. Each predecessor will
/// populate the block argument with the definition reached at its end. With
/// this, the value stored can be well defined at block boundaries, allowing
/// the propagation of replacement through blocks.
///
/// The way regions are handled in the transformation is by offering an
/// interface to express the behavior of the allocation value at the edges of
/// the regions: from a particular definition reaching the region operation, the
/// operation will specify what the reaching definition at the entry of its
/// regions are (potentially mutating itself, for example to add region
/// arguments). Likewise, provided a reaching definition at the end of the
/// blocks in the regions, the region operation will provide the reaching
/// definition right after itself.
///
/// This pass computes this transformation in two main phases: an analysis
/// phase that does not mutate IR, and a transformation phase where mutation
/// happens. Each phase is handled by the `MemorySlotPromotionAnalyzer` and
/// `MemorySlotPromoter` classes respectively.
///
/// The two steps of the analysis phase are the following:
/// - A first step computes the list of operations that transitively use the
/// memory slot we would like to promote. The purpose of this phase is to
/// identify which uses must be removed to promote the slot, either by rewiring
/// the user or deleting it. Naturally, direct uses of the slot must be removed.
/// Sometimes additional uses must also be removed: this is notably the case
/// when a direct user of the slot cannot rewire its use and must delete itself,
/// and thus must make its users no longer use it. If the allocation is used in
/// nested regions, it is also ensured the region operations provide the right
/// interface to analyze the values of the allocation at the edges of its
/// regions. If any of those constraints cannot be satisfied, promotion cannot
/// continue: this is decided at this step.
/// - A second step computes the list of blocks where a block argument will be
/// needed ("merge points") without mutating the IR. These blocks are the blocks
/// leading to a definition clash between two predecessors. Such blocks happen
/// to be the Iterated Dominance Frontier (IDF) of the set of blocks containing
/// a store, as they represent the points where a clear defining dominator stops
/// existing. Computing this information in advance allows making sure the
/// terminators that will forward values are capable of doing so (inability to
/// do so aborts promotion at this step).
///
/// At this point, promotion is guaranteed to happen, and the transformation
/// phase can begin. For each region of the program, a two step process is
/// carried out.
/// - The first step of the per-region process computes the reaching definition
/// of the memory slot at each blocking user. This is the core of the mem2reg
/// algorithm, also known as load-store forwarding. This analyses loads and
/// stores and propagates which value must be stored in the slot at each
/// blocking user. This is achieved by doing a depth-first walk of the dominator
/// tree of the function. This is sufficient because the reaching definition at
/// the beginning of a block is either its new block argument if it is a merge
/// block, or the definition reaching the end of its immediate dominator (parent
/// in the dominator tree). We can therefore propagate this information down the
/// dominator tree to proceed with renaming within blocks. If at any point a
/// region operation that contains a use of the allocation is encountered, the
/// transformation process is triggered on the child regions of the encountered
/// operation, to obtain the reaching definition at its end and carry on with
/// the value forwarding.
/// - The second step of the per-region process uses the reaching definition to
/// remove blocking uses in topological order. Some reaching definitions may
/// be values that will be removed or modified during the blocking use removal
/// step (typically, in the case of a store that stores the result of a load).
/// To properly handle such values, this step traverses the operations to modify
/// in reverse topological order. This way, if a value that will disappear is
/// used in place of reaching definition, the logic to make it disappear will be
/// executed after the value has been used to replace an operation. For regions
/// within a PromotableRegionOpInterface, in order to correctly handle cases
/// where the finalization logic would use a reaching definition that will be
/// replaced, the finalization logic must be called before the blocking use
/// removal step, so that any use of a value that will be removed gets properly
/// replaced.
///
/// For further reading, chapter three of SSA-based Compiler Design [1]
/// showcases SSA construction for control-flow graphs, where mem2reg is an
/// adaptation of the same process.
///
/// [1]: Rastello F. & Bouchez Tichadou F., SSA-based Compiler Design (2022),
///      Springer.

namespace {

using BlockingUsesMap =
    llvm::MapVector<Operation *, SmallPtrSet<OpOperand *, 4>>;
using RegionBlockingUsesMap =
    llvm::SmallMapVector<Region *, BlockingUsesMap, 2>;

using RegionSet = SmallPtrSet<Region *, 32>;

/// Information about regions that will be traversed for promotion, computed
/// during promotion analysis.
struct RegionPromotionInfo {
  /// True if an operation storing to the slot is present in the region.
  bool hasValueStores;
};

/// Information computed during promotion analysis used to perform actual
/// promotion.
struct MemorySlotPromotionInfo {
  /// Blocks for which at least two definitions of the slot values clash.
  SmallPtrSet<Block *, 8> mergePoints;
  /// Contains, for each each region, the blocking uses for its operations. The
  /// blocking uses are the uses that must be eliminated by promotion. For each
  /// region, this is a DAG structure because if an operation must eliminate
  /// some of its uses, it is because the defining ops of the blocking uses
  /// requested it. The defining ops therefore must also have blocking uses or
  /// be the starting point of the blocking uses.
  RegionBlockingUsesMap userToBlockingUses;
  /// Regions of which the edges must be analyzed for promotion. All regions
  /// are guaranteed to be held by a PromotableRegionOpInterface, and to be
  /// nested within the parent region of the slot pointer.
  DenseMap<Region *, RegionPromotionInfo> regionsToPromote;
};

/// Computes information for basic slot promotion. This will check that direct
/// slot promotion can be performed, and provide the information to execute the
/// promotion. This does not mutate IR.
class MemorySlotPromotionAnalyzer {
public:
  MemorySlotPromotionAnalyzer(MemorySlot slot, DominanceInfo &dominance,
                              const DataLayout &dataLayout)
      : slot(slot), dominance(dominance), dataLayout(dataLayout) {}

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
  /// Resulting blocking uses are grouped by region.
  /// This also ensures all the uses are within promotable regions, adding
  /// information about regions to be promoted to the `regionsToPromote` map.
  LogicalResult computeBlockingUses(
      RegionBlockingUsesMap &userToBlockingUses,
      DenseMap<Region *, RegionPromotionInfo> &regionsToPromote);

  /// Computes the points in the provided region where multiple re-definitions
  /// of the slot's value (stores) may conflict.
  /// `definingBlocks` is the set of blocks containing a store to the slot,
  /// either directly or inherited from a nested region.
  void computeMergePoints(Region *region,
                          SmallPtrSetImpl<Block *> &definingBlocks,
                          SmallPtrSetImpl<Block *> &mergePoints);

  /// Ensures predecessors of merge points can properly provide their current
  /// definition of the value stored in the slot to the merge point. This can
  /// notably be an issue if the terminator used does not have the ability to
  /// forward values through block operands.
  bool areMergePointsUsable(SmallPtrSetImpl<Block *> &mergePoints);

  MemorySlot slot;

  DominanceInfo &dominance;
  const DataLayout &dataLayout;
};

/// Maps a region to a map of blocks to their index in the region.
/// The region is identified by its entry block pointer instead of its region
/// pointer to not need to invalidate the cache when region content is moved to
/// a new region. This only supports moves of all the blocks of a region to
/// an empty region.
using BlockIndexCache = DenseMap<Block *, DenseMap<Block *, size_t>>;

/// The MemorySlotPromoter handles the state of promoting a memory slot. It
/// wraps a slot and its associated allocator. This will perform the mutation of
/// IR.
class MemorySlotPromoter {
public:
  MemorySlotPromoter(MemorySlot slot, PromotableAllocationOpInterface allocator,
                     OpBuilder &builder, DominanceInfo &dominance,
                     const DataLayout &dataLayout, MemorySlotPromotionInfo info,
                     const Mem2RegStatistics &statistics,
                     BlockIndexCache &blockIndexCache);

  /// Actually promotes the slot by mutating IR. Promoting a slot DOES
  /// invalidate the MemorySlotPromotionInfo of other slots. Preparation of
  /// promotion info should NOT be performed in batches.
  /// Returns a promotable allocation op if a new allocator was created, nullopt
  /// otherwise.
  std::optional<PromotableAllocationOpInterface> promoteSlot();

private:
  /// Computes the reaching definition for all the operations that require
  /// promotion, including within nested regions needing promotion.
  /// `reachingDef` is the value the slot contains at the beginning of the
  /// block. This member function returns the reached definition at the end of
  /// the block. If the block contains a region that needs promotion, the
  /// blocking uses of that region will have been removed. This member function
  /// will not remove the blocking uses contained directly in the block.
  ///
  /// The `reachingDef` may be a null value. In that case, a lazily-created
  /// default value will be used.
  ///
  /// This member function must only be called at most once per block.
  Value promoteInBlock(Block *block, Value reachingDef);

  /// Computes the reaching definition for all the operations that require
  /// promotion, including within nested regions needing promotion, and removes
  /// the blocking uses of the slot within the region.
  /// `reachingDef` is the value the slot contains at the beginning of the
  /// region.
  ///
  /// The `reachingDef` may be a null value. In that case, a lazily-created
  /// default value will be used.
  ///
  /// This member function must only be called at most once per region.
  void promoteInRegion(Region *region, Value reachingDef);

  /// Removes the blocking uses of the slot within the given region, in
  /// reverse topological order. If the content of the region was moved out
  /// to a different region, the new region will be processed instead.
  void removeBlockingUses(Region *region);

  /// Removes operations and merge point block arguments that ended up not being
  /// necessary.
  void removeUnusedItems();

  /// Lazily-constructed default value representing the content of the slot when
  /// no store has been executed. This function may mutate IR.
  Value getOrCreateDefaultValue();

  MemorySlot slot;
  PromotableAllocationOpInterface allocator;
  OpBuilder &builder;
  /// Potentially non-initialized default value. Use `getOrCreateDefaultValue`
  /// to initialize it on demand.
  Value defaultValue;
  /// Contains the reaching definition at this operation. Reaching definitions
  /// are only computed for promotable memory operations with blocking uses.
  DenseMap<PromotableMemOpInterface, Value> reachingDefs;
  DenseMap<PromotableMemOpInterface, Value> replacedValuesMap;

  /// Contains the reaching definition at the end of the blocks visited so far.
  DenseMap<Block *, Value> reachingAtBlockEnd;

  /// Lists all the values that have been set by a memory operation as a
  /// reaching definition at one point during the promotion. The accompanying
  /// operation is the memory operation that originally stored the value.
  llvm::SmallVector<std::pair<Operation *, Value>> replacedValues;
  /// Operations to visit with the `visitReplacedValues` method at the end of
  /// the promotion.
  llvm::SmallVector<PromotableOpInterface> toVisitReplacedValues;
  /// Operations to be erased at the end of the promotion.
  llvm::SmallSetVector<Operation *, 8> toErase;

  DominanceInfo &dominance;
  const DataLayout &dataLayout;
  MemorySlotPromotionInfo info;
  const Mem2RegStatistics &statistics;

  /// Shared cache of block indices of specific regions.
  /// Cache entries must be invalidated before any addition, removal or
  /// reordering of blocks in the corresponding region.
  /// Cache entries are *NOT* invalidated if all the blocks of the corresponding
  /// region are moved to an empty region.
  BlockIndexCache &blockIndexCache;
};

} // namespace

MemorySlotPromoter::MemorySlotPromoter(
    MemorySlot slot, PromotableAllocationOpInterface allocator,
    OpBuilder &builder, DominanceInfo &dominance, const DataLayout &dataLayout,
    MemorySlotPromotionInfo info, const Mem2RegStatistics &statistics,
    BlockIndexCache &blockIndexCache)
    : slot(slot), allocator(allocator), builder(builder), dominance(dominance),
      dataLayout(dataLayout), info(std::move(info)), statistics(statistics),
      blockIndexCache(blockIndexCache) {
#ifndef NDEBUG
  auto isResultOrNewBlockArgument = [&]() {
    if (BlockArgument arg = dyn_cast<BlockArgument>(slot.ptr))
      return arg.getOwner()->getParentOp() == allocator;
    return slot.ptr.getDefiningOp() == allocator;
  };

  assert(isResultOrNewBlockArgument() &&
         "a slot must be a result of the allocator or an argument of the child "
         "regions of the allocator");
#endif // NDEBUG
}

Value MemorySlotPromoter::getOrCreateDefaultValue() {
  if (defaultValue)
    return defaultValue;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(slot.ptr.getParentBlock());
  return defaultValue = allocator.getDefaultValue(slot, builder);
}

LogicalResult MemorySlotPromotionAnalyzer::computeBlockingUses(
    RegionBlockingUsesMap &userToBlockingUses,
    DenseMap<Region *, RegionPromotionInfo> &regionsToPromote) {
  // The promotion of an operation may require the promotion of further
  // operations (typically, removing operations that use an operation that must
  // delete itself). We thus need to start from the use of the slot pointer and
  // propagate further requests through the forward slice.

  // Graph regions are not supported.
  Region *slotPtrRegion = slot.ptr.getParentRegion();
  auto slotPtrRegionOp =
      dyn_cast<RegionKindInterface>(slotPtrRegion->getParentOp());
  if (slotPtrRegionOp &&
      slotPtrRegionOp.getRegionKind(slotPtrRegion->getRegionNumber()) ==
          RegionKind::Graph)
    return failure();

  // First insert that all immediate users of the slot pointer must no longer
  // use it.
  for (OpOperand &use : slot.ptr.getUses()) {
    SmallPtrSet<OpOperand *, 4> &blockingUses =
        userToBlockingUses[use.getOwner()->getParentRegion()][use.getOwner()];
    blockingUses.insert(&use);
  }

  // Regions that immediately contain a slot memory use that is not a store.
  RegionSet regionsWithDirectUse;
  // Regions that immediately contain a slot memory use that is a store.
  RegionSet regionsWithDirectStore;

  // Then, propagate the requirements for the removal of uses. The
  // topologically-sorted forward slice allows for all blocking uses of an
  // operation to have been computed before it is reached. Operations are
  // traversed in topological order of their uses, starting from the slot
  // pointer.
  SetVector<Operation *> forwardSlice;
  mlir::getForwardSlice(slot.ptr, &forwardSlice);
  for (Operation *user : forwardSlice) {
    // If the next operation has no blocking uses, everything is fine.
    auto *blockingUsesMapIt = userToBlockingUses.find(user->getParentRegion());
    if (blockingUsesMapIt == userToBlockingUses.end())
      continue;
    BlockingUsesMap &blockingUsesMap = blockingUsesMapIt->second;
    auto *it = blockingUsesMap.find(user);
    if (it == blockingUsesMap.end())
      continue;

    SmallPtrSet<OpOperand *, 4> &blockingUses = it->second;

    SmallVector<OpOperand *> newBlockingUses;
    // If the operation decides it cannot deal with removing the blocking uses,
    // promotion must fail.
    if (auto promotable = dyn_cast<PromotableOpInterface>(user)) {
      if (!promotable.canUsesBeRemoved(blockingUses, newBlockingUses,
                                       dataLayout))
        return failure();
      regionsWithDirectUse.insert(user->getParentRegion());
    } else if (auto promotable = dyn_cast<PromotableMemOpInterface>(user)) {
      if (!promotable.canUsesBeRemoved(slot, blockingUses, newBlockingUses,
                                       dataLayout))
        return failure();

      // Operations that interact with the slot's memory will be promoted using
      // a reaching definition. Therefore, the operation must be within a region
      // where the reaching definition can be computed.
      if (promotable.storesTo(slot))
        regionsWithDirectStore.insert(user->getParentRegion());
      else
        regionsWithDirectUse.insert(user->getParentRegion());
    } else {
      // An operation that has blocking uses must be promoted. If it is not
      // promotable, promotion must fail.
      return failure();
    }

    // Then, register any new blocking uses for coming operations.
    for (OpOperand *blockingUse : newBlockingUses) {
      assert(llvm::is_contained(user->getResults(), blockingUse->get()));

      SmallPtrSetImpl<OpOperand *> &newUserBlockingUseSet =
          blockingUsesMap[blockingUse->getOwner()];
      newUserBlockingUseSet.insert(blockingUse);
    }
  }

  // Finally, check that all the regions needed are promotable, and propagate
  // the constraint to their parent regions.
  auto visitRegions = [&](SmallVector<Region *> &regionsToPropagateFrom,
                          bool hasValueStores) {
    while (!regionsToPropagateFrom.empty()) {
      Region *region = regionsToPropagateFrom.pop_back_val();

      if (region == slot.ptr.getParentRegion() ||
          regionsToPromote.contains(region))
        continue;

      RegionPromotionInfo &regionInfo = regionsToPromote[region];
      regionInfo.hasValueStores = hasValueStores;

      auto promotableParentOp =
          dyn_cast<PromotableRegionOpInterface>(region->getParentOp());
      if (!promotableParentOp)
        return failure();

      if (!promotableParentOp.isRegionPromotable(slot, region, hasValueStores))
        return failure();

      regionsToPropagateFrom.push_back(region->getParentRegion());
    }

    return success();
  };

  // Start with the regions that directly contain a store to give priority
  // to stores in the propagation of `hasValueStores` information.
  SmallVector<Region *> regionsToPropagateFrom(regionsWithDirectStore.begin(),
                                               regionsWithDirectStore.end());
  if (failed(visitRegions(regionsToPropagateFrom, true)))
    return failure();

  // Then, propagate from the regions that directly contain non-store uses.
  regionsToPropagateFrom.clear();
  regionsToPropagateFrom.append(regionsWithDirectUse.begin(),
                                regionsWithDirectUse.end());
  if (failed(visitRegions(regionsToPropagateFrom, false)))
    return failure();

  return success();
}

using IDFCalculator = llvm::IDFCalculatorBase<Block, false>;
void MemorySlotPromotionAnalyzer::computeMergePoints(
    Region *region, SmallPtrSetImpl<Block *> &definingBlocks,
    SmallPtrSetImpl<Block *> &mergePoints) {
  if (region->hasOneBlock())
    return;

  IDFCalculator idfCalculator(dominance.getDomTree(region));
  idfCalculator.setDefiningBlocks(definingBlocks);

  SmallVector<Block *> mergePointsVec;
  idfCalculator.calculate(mergePointsVec);

  mergePoints.insert_range(mergePointsVec);
}

bool MemorySlotPromotionAnalyzer::areMergePointsUsable(
    SmallPtrSetImpl<Block *> &mergePoints) {
  for (Block *mergePoint : mergePoints)
    for (Block *pred : mergePoint->getPredecessors())
      if (!isa<BranchOpInterface>(pred->getTerminator()))
        return false;

  return true;
}

std::optional<MemorySlotPromotionInfo>
MemorySlotPromotionAnalyzer::computeInfo() {
  MemorySlotPromotionInfo info;

  // First, find the set of operations that will need to be changed for the
  // promotion to happen. These operations need to resolve some of their uses,
  // either by rewiring them or simply deleting themselves. If any of them
  // cannot find a way to resolve their blocking uses, we abort the promotion.
  // We also compute at this stage the regions that will be analyzed for
  // reaching definition information.
  if (failed(
          computeBlockingUses(info.userToBlockingUses, info.regionsToPromote)))
    return {};

  // Compute the blocks containing a store for each region, either directly or
  // inherited from a nested region. As a side effect, `definingBlocks` contains
  // all regions with at least one store.
  DenseMap<Region *, SmallPtrSet<Block *, 16>> definingBlocks;
  for (Operation *user : slot.ptr.getUsers())
    if (auto storeOp = dyn_cast<PromotableMemOpInterface>(user))
      if (storeOp.storesTo(slot))
        definingBlocks[user->getParentRegion()].insert(user->getBlock());
  for (auto &[region, regionInfo] : info.regionsToPromote)
    if (regionInfo.hasValueStores)
      definingBlocks[region->getParentRegion()].insert(
          region->getParentOp()->getBlock());

  // Then, compute blocks in which two or more definitions of the allocated
  // variable may conflict. These blocks will need a new block argument to
  // accommodate this.
  for (auto &[region, defBlocks] : definingBlocks)
    computeMergePoints(region, defBlocks, info.mergePoints);

  // The slot can be promoted if the block arguments to be created can
  // actually be populated with values, which may not be possible depending
  // on their predecessors.
  if (!areMergePointsUsable(info.mergePoints))
    return {};

  return info;
}

Value MemorySlotPromoter::promoteInBlock(Block *block, Value reachingDef) {
  SmallVector<Operation *> blockOps;
  for (Operation &op : block->getOperations())
    blockOps.push_back(&op);
  for (Operation *op : blockOps) {
    // Promote operations that interact with the slot's memory.
    if (auto memOp = dyn_cast<PromotableMemOpInterface>(op)) {
      if (info.userToBlockingUses[memOp->getParentRegion()].contains(memOp))
        reachingDefs.insert({memOp, reachingDef});

      if (memOp.storesTo(slot)) {
        builder.setInsertionPointAfter(memOp);
        // To not expose default value creation to the interfaces, if we have
        // no reaching definition by now, we set it to the default value.
        // This is slightly too eager as `getStored` may not need it.
        if (!reachingDef)
          reachingDef = getOrCreateDefaultValue();
        Value stored = memOp.getStored(slot, builder, reachingDef, dataLayout);
        assert(stored && "a memory operation storing to a slot must provide a "
                         "new definition of the slot");
        reachingDef = stored;
        replacedValuesMap[memOp] = stored;
      }
    }

    // Promote regions that contain operations that interact with the slot's
    // memory.
    if (auto promotableRegionOp = dyn_cast<PromotableRegionOpInterface>(op)) {
      bool needsPromotion = false;
      bool hasValueStores = false;
      for (Region &region : op->getRegions()) {
        auto regionInfoIt = info.regionsToPromote.find(&region);
        if (regionInfoIt == info.regionsToPromote.end())
          continue;
        needsPromotion = true;
        if (!regionInfoIt->second.hasValueStores)
          continue;

        hasValueStores = true;
        break;
      }

      if (needsPromotion) {
        llvm::SmallMapVector<Region *, Value, 2> regionsToProcess;

        // To not expose default value creation to the interfaces, if we have
        // no reaching definition by now, we set it to the default value.
        // This is slightly too eager as `setupPromotion` may not need it.
        if (!reachingDef)
          reachingDef = getOrCreateDefaultValue();

        promotableRegionOp.setupPromotion(slot, reachingDef, hasValueStores,
                                          regionsToProcess);

#ifndef NDEBUG
        for (Region &region : op->getRegions())
          if (info.regionsToPromote.contains(&region))
            assert(
                regionsToProcess.contains(&region) &&
                "reaching definition must be provided for a required region");
#endif // NDEBUG

        for (auto &[region, reachingDef] : regionsToProcess) {
          assert(region->getParentOp() == op &&
                 "region must be part of the operation");
          if (!info.regionsToPromote.contains(region))
            continue;
          promoteInRegion(region, reachingDef);
        }

        // TODO: Currently we have to invalidate the dominance information of
        // the regions of the operation because finalizePromotion may move their
        // content. We might want to support moving dominance information
        // accross regions as this can be detected.
        for (Region &region : op->getRegions())
          dominance.invalidate(&region);

        builder.setInsertionPointAfter(op);
        reachingDef = promotableRegionOp.finalizePromotion(
            slot, reachingDef, hasValueStores, reachingAtBlockEnd, builder);

        // Blocking uses can then be removed for the regions that were promoted.
        // Even though `finalizePromotion` may have moved regions to a new
        // operation, `removeBlockingUses` handles this case and will redirect
        // processing to the correct region.
        for (auto &[region, reachingDef] : regionsToProcess)
          removeBlockingUses(region);
      }
    }
  }

  reachingAtBlockEnd[block] = reachingDef;
  return reachingDef;
}

void MemorySlotPromoter::promoteInRegion(Region *region, Value reachingDef) {
  if (region->hasOneBlock()) {
    promoteInBlock(&region->front(), reachingDef);
    return;
  }

  struct DfsJob {
    llvm::DomTreeNodeBase<Block> *block;
    Value reachingDef;
  };

  SmallVector<DfsJob> dfsStack;

  auto &domTree = dominance.getDomTree(region);

  dfsStack.emplace_back<DfsJob>(
      {domTree.getNode(&region->front()), reachingDef});

  while (!dfsStack.empty()) {
    DfsJob job = dfsStack.pop_back_val();
    Block *block = job.block->getBlock();

    if (info.mergePoints.contains(block)) {
      BlockArgument blockArgument =
          block->addArgument(slot.elemType, slot.ptr.getLoc());
      job.reachingDef = blockArgument;
    }

    job.reachingDef = promoteInBlock(block, job.reachingDef);

    if (auto terminator = dyn_cast<BranchOpInterface>(block->getTerminator())) {
      for (BlockOperand &blockOperand : terminator->getBlockOperands()) {
        if (info.mergePoints.contains(blockOperand.get())) {
          if (!job.reachingDef)
            job.reachingDef = getOrCreateDefaultValue();

          terminator.getSuccessorOperands(blockOperand.getOperandNumber())
              .append(job.reachingDef);
        }
      }
    }

    for (auto *child : job.block->children())
      dfsStack.emplace_back<DfsJob>({child, job.reachingDef});
  }
}

/// Gets or creates a block index mapping for the region of which the entry
/// block is `regionEntryBlock`.
static const DenseMap<Block *, size_t> &
getOrCreateBlockIndices(BlockIndexCache &blockIndexCache,
                        Block *regionEntryBlock) {
  auto [it, inserted] = blockIndexCache.try_emplace(regionEntryBlock);
  if (!inserted)
    return it->second;

  DenseMap<Block *, size_t> &blockIndices = it->second;
  SetVector<Block *> topologicalOrder =
      getBlocksSortedByDominance(*regionEntryBlock->getParent());
  for (auto [index, block] : llvm::enumerate(topologicalOrder))
    blockIndices[block] = index;
  return blockIndices;
}

/// Sorts `ops` according to dominance. Relies on the topological order of basic
/// blocks to get a deterministic ordering. Uses `blockIndexCache` to avoid the
/// potentially expensive recomputation of a block index map.
/// This function assumes no blocks are ever deleted or entry block changed
/// during the lifetime of the block index cache.
static void dominanceSort(SmallVector<Operation *> &ops, Region &region,
                          BlockIndexCache &blockIndexCache) {
  if (region.empty())
    return;

  // Produce a topological block order and construct a map to lookup the indices
  // of blocks.
  const DenseMap<Block *, size_t> &topoBlockIndices =
      getOrCreateBlockIndices(blockIndexCache, &region.front());

  // Combining the topological order of the basic blocks together with block
  // internal operation order guarantees a deterministic, dominance respecting
  // order.
  llvm::sort(ops, [&](Operation *lhs, Operation *rhs) {
    size_t lhsBlockIndex = topoBlockIndices.at(lhs->getBlock());
    size_t rhsBlockIndex = topoBlockIndices.at(rhs->getBlock());
    if (lhsBlockIndex == rhsBlockIndex)
      return lhs->isBeforeInBlock(rhs);
    return lhsBlockIndex < rhsBlockIndex;
  });
}

void MemorySlotPromoter::removeBlockingUses(Region *region) {
  auto *blockingUsesMapIt = info.userToBlockingUses.find(region);
  if (blockingUsesMapIt == info.userToBlockingUses.end())
    return;
  BlockingUsesMap &blockingUsesMap = blockingUsesMapIt->second;
  if (blockingUsesMap.empty())
    return;

  // Operations may have been moved to a different region at this point.
  // To cover this, we process the current region of an operation to remove
  // instead of the provided region.
  region = blockingUsesMap.front().first->getParentRegion();
#ifndef NDEBUG
  for (auto &[op, blockingUses] : blockingUsesMap)
    assert(op->getParentRegion() == region &&
           "all operations must still be in the same region");
#endif // NDEBUG

  llvm::SmallVector<Operation *> usersToRemoveUses(
      llvm::make_first_range(blockingUsesMap));

  // Sort according to dominance.
  dominanceSort(usersToRemoveUses, *region, blockIndexCache);

  // Iterate over the operations to rewrite in reverse dominance order.
  for (Operation *toPromote : llvm::reverse(usersToRemoveUses)) {
    if (auto toPromoteMemOp = dyn_cast<PromotableMemOpInterface>(toPromote)) {
      Value reachingDef = reachingDefs.lookup(toPromoteMemOp);
      // If no reaching definition is known, this use is outside the reach of
      // the slot. The default value should thus be used.
      // FIXME: This is too eager, and will generate default values even for
      // pure stores. This cannot be removed easily as partial stores may
      // still require a default value to complete.
      if (!reachingDef)
        reachingDef = getOrCreateDefaultValue();

      builder.setInsertionPointAfter(toPromote);
      if (toPromoteMemOp.removeBlockingUses(slot, blockingUsesMap[toPromote],
                                            builder, reachingDef,
                                            dataLayout) == DeletionKind::Delete)
        toErase.insert(toPromote);
      if (toPromoteMemOp.storesTo(slot))
        if (Value replacedValue = replacedValuesMap[toPromoteMemOp])
          replacedValues.push_back({toPromoteMemOp, replacedValue});
      continue;
    }

    auto toPromoteBasic = cast<PromotableOpInterface>(toPromote);
    builder.setInsertionPointAfter(toPromote);
    if (toPromoteBasic.removeBlockingUses(blockingUsesMap[toPromote],
                                          builder) == DeletionKind::Delete)
      toErase.insert(toPromote);
    if (toPromoteBasic.requiresReplacedValues())
      toVisitReplacedValues.push_back(toPromoteBasic);
  }
}

void MemorySlotPromoter::removeUnusedItems() {
  // We want to eliminate unused block arguments. Because block arguments can be
  // used to populate other block arguments, there might be cycles of arguments
  // that are only used to populate each-other. We therefore need a small
  // dataflow analysis to identify which block arguments are truly used.

  SmallPtrSet<BlockArgument, 8> mergePointArgsUnused;
  SmallVector<BlockArgument> usedMergePointArgsToProcess;

  // First, separate the block arguments that are not used or only used for the
  // purpose of populating a merge point block argument from the others. These
  // block arguments are potentially unused. Meanwhile, arguments that are
  // definitely used will be the starting point of the propagation of the
  // analysis.
  auto isDefinitelyUsed = [&](BlockArgument arg) {
    for (auto &use : arg.getUses()) {
      if (llvm::is_contained(toErase, use.getOwner()))
        continue;

      // We now want to detect whether the use is to populate a merge point
      // block argument. If it is not, the argument is definitely used.

      auto branchOp = dyn_cast<BranchOpInterface>(use.getOwner());
      if (!branchOp)
        return true;

      std::optional<BlockArgument> successorArgument =
          branchOp.getSuccessorBlockArgument(use.getOperandNumber());
      if (!successorArgument)
        return true;

      if (!info.mergePoints.contains(successorArgument->getOwner()))
        return true;

      // The last block argument of a merge point is its reaching definition
      // argument. If the argument being populated is not the last one, it is a
      // genuine use of the value.
      bool isLastBlockArgument =
          successorArgument->getArgNumber() ==
          successorArgument->getOwner()->getNumArguments() - 1;
      if (!isLastBlockArgument)
        return true;
    }
    return false;
  };

  for (Block *mergePoint : info.mergePoints) {
    BlockArgument arg = mergePoint->getArguments().back();
    if (isDefinitelyUsed(arg))
      usedMergePointArgsToProcess.push_back(arg);
    else
      mergePointArgsUnused.insert(arg);
  }

  // We now refine mergePointArgsUnused from the information of which block
  // arguments are definitely used.
  while (!usedMergePointArgsToProcess.empty()) {
    BlockArgument arg = usedMergePointArgsToProcess.pop_back_val();
    Block *mergePoint = arg.getOwner();

    assert(arg.getArgNumber() == mergePoint->getNumArguments() - 1 &&
           "merge point argument must be the last argument of the merge point");

    for (BlockOperand &use : mergePoint->getUses()) {
      // If a value used to populate this used merge point argument is another
      // merge point block argument that is currently considered unused, it must
      // now be considered used and processed as such later.

      auto branch = cast<BranchOpInterface>(use.getOwner());
      SuccessorOperands succOperands =
          branch.getSuccessorOperands(use.getOperandNumber());

      // The successor operand is either the last one or is not present if the
      // user block is dead.
      assert(succOperands.size() == mergePoint->getNumArguments() ||
             succOperands.size() + 1 == mergePoint->getNumArguments());

      // If the user block is dead, the default value acts as a placeholder
      // dummy value.
      if (succOperands.size() + 1 == mergePoint->getNumArguments())
        succOperands.append(getOrCreateDefaultValue());

      Value populatedValue = succOperands[arg.getArgNumber()];
      auto populatedValueAsArg = dyn_cast<BlockArgument>(populatedValue);
      if (populatedValueAsArg &&
          mergePointArgsUnused.erase(populatedValueAsArg))
        usedMergePointArgsToProcess.push_back(populatedValueAsArg);
    }

    builder.setInsertionPointToStart(mergePoint);
    allocator.handleBlockArgument(slot, arg, builder);
    if (statistics.newBlockArgumentAmount)
      (*statistics.newBlockArgumentAmount)++;
  }

  for (Operation *toEraseOp : toErase)
    toEraseOp->erase();

  for (BlockArgument arg : mergePointArgsUnused) {
    Block *mergePoint = arg.getOwner();
    for (BlockOperand &use : mergePoint->getUses()) {
      auto branch = cast<BranchOpInterface>(use.getOwner());
      SuccessorOperands succOperands =
          branch.getSuccessorOperands(use.getOperandNumber());
      succOperands.erase(arg.getArgNumber());
    }
    mergePoint->eraseArgument(mergePoint->getNumArguments() - 1);
  }
}

std::optional<PromotableAllocationOpInterface>
MemorySlotPromoter::promoteSlot() {
  // Perform the promotion recursively through nested regions. The reaching
  // definition starts with a null value that will be replaced by a
  // lazily-created default value if the value must be passed to a promotion
  // interface while no store has been encountered yet.
  // Innermost regions will see their blocking uses be removed, but not the
  // outermost region which we have to remove manually afterwards. This is
  // because PromotableRegionOpInterface::finalizePromotion must be called
  // before removeBlockingUses.
  promoteInRegion(slot.ptr.getParentRegion(), nullptr);

  // Blocking uses can then be removed for the outermost region.
  removeBlockingUses(slot.ptr.getParentRegion());

  // Notify operations that requested it of the reaching definitions set by
  // storing memory operations.
  for (PromotableOpInterface op : toVisitReplacedValues) {
    builder.setInsertionPointAfter(op);
    op.visitReplacedValues(replacedValues, builder);
  }

  // Finally, remove unused operations and merge point block arguments.
  removeUnusedItems();

  assert(slot.ptr.use_empty() &&
         "after promotion, the slot pointer should not be used anymore");

  LDBG() << "Promoted memory slot: " << slot.ptr;

  if (statistics.promotedAmount)
    (*statistics.promotedAmount)++;

  return allocator.handlePromotionComplete(slot, defaultValue, builder);
}

LogicalResult mlir::tryToPromoteMemorySlots(
    ArrayRef<PromotableAllocationOpInterface> allocators, OpBuilder &builder,
    const DataLayout &dataLayout, DominanceInfo &dominance,
    Mem2RegStatistics statistics) {
  bool promotedAny = false;

  // A cache that stores deterministic block indices which are used to determine
  // a valid operation modification order. The block index maps are computed
  // lazily and cached to avoid expensive recomputation.
  BlockIndexCache blockIndexCache;

  SmallVector<PromotableAllocationOpInterface> workList(allocators);

  SmallVector<PromotableAllocationOpInterface> newWorkList;
  newWorkList.reserve(workList.size());
  while (true) {
    bool changesInThisRound = false;
    for (PromotableAllocationOpInterface allocator : workList) {
      bool changedAllocator = false;
      for (MemorySlot slot : allocator.getPromotableSlots()) {
        if (slot.ptr.use_empty())
          continue;

        MemorySlotPromotionAnalyzer analyzer(slot, dominance, dataLayout);
        std::optional<MemorySlotPromotionInfo> info = analyzer.computeInfo();
        if (info) {
          std::optional<PromotableAllocationOpInterface> newAllocator =
              MemorySlotPromoter(slot, allocator, builder, dominance,
                                 dataLayout, std::move(*info), statistics,
                                 blockIndexCache)
                  .promoteSlot();
          changedAllocator = true;
          // Add newly created allocators to the worklist for further
          // processing.
          if (newAllocator)
            newWorkList.push_back(*newAllocator);

          // A break is required, since promoting a slot may invalidate the
          // remaining slots of an allocator.
          break;
        }
      }
      if (!changedAllocator)
        newWorkList.push_back(allocator);
      changesInThisRound |= changedAllocator;
    }
    if (!changesInThisRound)
      break;
    promotedAny = true;

    // Swap the vector's backing memory and clear the entries in newWorkList
    // afterwards. This ensures that additional heap allocations can be avoided.
    workList.swap(newWorkList);
    newWorkList.clear();
  }

  return success(promotedAny);
}

namespace {

struct Mem2Reg : impl::Mem2RegBase<Mem2Reg> {
  using impl::Mem2RegBase<Mem2Reg>::Mem2RegBase;

  void runOnOperation() override {
    Operation *scopeOp = getOperation();

    Mem2RegStatistics statistics{&promotedAmount, &newBlockArgumentAmount};

    bool changed = false;

    auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(scopeOp);
    auto &dominance = getAnalysis<DominanceInfo>();

    for (Region &region : scopeOp->getRegions()) {
      if (region.getBlocks().empty())
        continue;

      OpBuilder builder(&region.front(), region.front().begin());

      SmallVector<PromotableAllocationOpInterface> allocators;
      // Build a list of allocators to attempt to promote the slots of.
      region.walk([&](PromotableAllocationOpInterface allocator) {
        allocators.emplace_back(allocator);
      });

      // Attempt promoting as many of the slots as possible.
      if (succeeded(tryToPromoteMemorySlots(allocators, builder, dataLayout,
                                            dominance, statistics)))
        changed = true;
    }
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
