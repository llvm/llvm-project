//===- RegionUtils.cpp - Region-related transformation utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;

void mlir::replaceAllUsesInRegionWith(Value orig, Value replacement,
                                      Region &region) {
  for (auto &use : llvm::make_early_inc_range(orig.getUses())) {
    if (region.isAncestor(use.getOwner()->getParentRegion()))
      use.set(replacement);
  }
}

void mlir::visitUsedValuesDefinedAbove(
    Region &region, Region &limit, function_ref<void(OpOperand *)> callback) {
  assert(limit.isAncestor(&region) &&
         "expected isolation limit to be an ancestor of the given region");

  // Collect proper ancestors of `limit` upfront to avoid traversing the region
  // tree for every value.
  SmallPtrSet<Region *, 4> properAncestors;
  for (auto *reg = limit.getParentRegion(); reg != nullptr;
       reg = reg->getParentRegion()) {
    properAncestors.insert(reg);
  }

  region.walk([callback, &properAncestors](Operation *op) {
    for (OpOperand &operand : op->getOpOperands())
      // Callback on values defined in a proper ancestor of region.
      if (properAncestors.count(operand.get().getParentRegion()))
        callback(&operand);
  });
}

void mlir::visitUsedValuesDefinedAbove(
    MutableArrayRef<Region> regions, function_ref<void(OpOperand *)> callback) {
  for (Region &region : regions)
    visitUsedValuesDefinedAbove(region, region, callback);
}

void mlir::getUsedValuesDefinedAbove(Region &region, Region &limit,
                                     llvm::SetVector<Value> &values) {
  visitUsedValuesDefinedAbove(region, limit, [&](OpOperand *operand) {
    values.insert(operand->get());
  });
}

void mlir::getUsedValuesDefinedAbove(MutableArrayRef<Region> regions,
                                     llvm::SetVector<Value> &values) {
  for (Region &region : regions)
    getUsedValuesDefinedAbove(region, region, values);
}

//===----------------------------------------------------------------------===//
// Unreachable Block Elimination
//===----------------------------------------------------------------------===//

/// Erase the unreachable blocks within the provided regions. Returns success
/// if any blocks were erased, failure otherwise.
// TODO: We could likely merge this with the DCE algorithm below.
static LogicalResult eraseUnreachableBlocks(MutableArrayRef<Region> regions) {
  // Set of blocks found to be reachable within a given region.
  llvm::df_iterator_default_set<Block *, 16> reachable;
  // If any blocks were found to be dead.
  bool erasedDeadBlocks = false;

  SmallVector<Region *, 1> worklist;
  worklist.reserve(regions.size());
  for (Region &region : regions)
    worklist.push_back(&region);
  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    if (region->empty())
      continue;

    // If this is a single block region, just collect the nested regions.
    if (std::next(region->begin()) == region->end()) {
      for (Operation &op : region->front())
        for (Region &region : op.getRegions())
          worklist.push_back(&region);
      continue;
    }

    // Mark all reachable blocks.
    reachable.clear();
    for (Block *block : depth_first_ext(&region->front(), reachable))
      (void)block /* Mark all reachable blocks */;

    // Collect all of the dead blocks and push the live regions onto the
    // worklist.
    for (Block &block : llvm::make_early_inc_range(*region)) {
      if (!reachable.count(&block)) {
        block.dropAllDefinedValueUses();
        block.erase();
        erasedDeadBlocks = true;
        continue;
      }

      // Walk any regions within this block.
      for (Operation &op : block)
        for (Region &region : op.getRegions())
          worklist.push_back(&region);
    }
  }

  return success(erasedDeadBlocks);
}

//===----------------------------------------------------------------------===//
// Dead Code Elimination
//===----------------------------------------------------------------------===//

namespace {
/// Data structure used to track which values have already been proved live.
///
/// Because Operation's can have multiple results, this data structure tracks
/// liveness for both Value's and Operation's to avoid having to look through
/// all Operation results when analyzing a use.
///
/// This data structure essentially tracks the dataflow lattice.
/// The set of values/ops proved live increases monotonically to a fixed-point.
class LiveMap {
public:
  /// Value methods.
  bool wasProvenLive(Value value) { return liveValues.count(value); }
  void setProvedLive(Value value) {
    changed |= liveValues.insert(value).second;
  }

  /// Operation methods.
  bool wasProvenLive(Operation *op) { return liveOps.count(op); }
  void setProvedLive(Operation *op) { changed |= liveOps.insert(op).second; }

  /// Methods for tracking if we have reached a fixed-point.
  void resetChanged() { changed = false; }
  bool hasChanged() { return changed; }

private:
  bool changed = false;
  DenseSet<Value> liveValues;
  DenseSet<Operation *> liveOps;
};
} // namespace

static bool isUseSpeciallyKnownDead(OpOperand &use, LiveMap &liveMap) {
  Operation *owner = use.getOwner();
  unsigned operandIndex = use.getOperandNumber();
  // This pass generally treats all uses of an op as live if the op itself is
  // considered live. However, for successor operands to terminators we need a
  // finer-grained notion where we deduce liveness for operands individually.
  // The reason for this is easiest to think about in terms of a classical phi
  // node based SSA IR, where each successor operand is really an operand to a
  // *separate* phi node, rather than all operands to the branch itself as with
  // the block argument representation that MLIR uses.
  //
  // And similarly, because each successor operand is really an operand to a phi
  // node, rather than to the terminator op itself, a terminator op can't e.g.
  // "print" the value of a successor operand.
  if (owner->hasTrait<OpTrait::IsTerminator>()) {
    if (BranchOpInterface branchInterface = dyn_cast<BranchOpInterface>(owner))
      if (auto arg = branchInterface.getSuccessorBlockArgument(operandIndex))
        return !liveMap.wasProvenLive(*arg);
    return false;
  }
  return false;
}

static void processValue(Value value, LiveMap &liveMap) {
  bool provedLive = llvm::any_of(value.getUses(), [&](OpOperand &use) {
    if (isUseSpeciallyKnownDead(use, liveMap))
      return false;
    return liveMap.wasProvenLive(use.getOwner());
  });
  if (provedLive)
    liveMap.setProvedLive(value);
}

static bool isOpIntrinsicallyLive(Operation *op) {
  // This pass doesn't modify the CFG, so terminators are never deleted.
  if (op->mightHaveTrait<OpTrait::IsTerminator>())
    return true;
  // If the op has a side effect, we treat it as live.
  // TODO: Properly handle region side effects.
  return !MemoryEffectOpInterface::hasNoEffect(op) || op->getNumRegions() != 0;
}

static void propagateLiveness(Region &region, LiveMap &liveMap);

static void propagateTerminatorLiveness(Operation *op, LiveMap &liveMap) {
  // Terminators are always live.
  liveMap.setProvedLive(op);

  // Check to see if we can reason about the successor operands and mutate them.
  BranchOpInterface branchInterface = dyn_cast<BranchOpInterface>(op);
  if (!branchInterface) {
    for (Block *successor : op->getSuccessors())
      for (BlockArgument arg : successor->getArguments())
        liveMap.setProvedLive(arg);
    return;
  }

  // If we can't reason about the operands to a successor, conservatively mark
  // all arguments as live.
  for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
    if (!branchInterface.getMutableSuccessorOperands(i))
      for (BlockArgument arg : op->getSuccessor(i)->getArguments())
        liveMap.setProvedLive(arg);
  }
}

static void propagateLiveness(Operation *op, LiveMap &liveMap) {
  // All Value's are either a block argument or an op result.
  // We call processValue on those cases.

  // Recurse on any regions the op has.
  for (Region &region : op->getRegions())
    propagateLiveness(region, liveMap);

  // Process terminator operations.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return propagateTerminatorLiveness(op, liveMap);

  // Process the op itself.
  if (isOpIntrinsicallyLive(op)) {
    liveMap.setProvedLive(op);
    return;
  }
  for (Value value : op->getResults())
    processValue(value, liveMap);
  bool provedLive = llvm::any_of(op->getResults(), [&](Value value) {
    return liveMap.wasProvenLive(value);
  });
  if (provedLive)
    liveMap.setProvedLive(op);
}

static void propagateLiveness(Region &region, LiveMap &liveMap) {
  if (region.empty())
    return;

  for (Block *block : llvm::post_order(&region.front())) {
    // We process block arguments after the ops in the block, to promote
    // faster convergence to a fixed point (we try to visit uses before defs).
    for (Operation &op : llvm::reverse(block->getOperations()))
      propagateLiveness(&op, liveMap);
    for (Value value : block->getArguments())
      processValue(value, liveMap);
  }
}

static void eraseTerminatorSuccessorOperands(Operation *terminator,
                                             LiveMap &liveMap) {
  BranchOpInterface branchOp = dyn_cast<BranchOpInterface>(terminator);
  if (!branchOp)
    return;

  for (unsigned succI = 0, succE = terminator->getNumSuccessors();
       succI < succE; succI++) {
    // Iterating successors in reverse is not strictly needed, since we
    // aren't erasing any successors. But it is slightly more efficient
    // since it will promote later operands of the terminator being erased
    // first, reducing the quadratic-ness.
    unsigned succ = succE - succI - 1;
    Optional<MutableOperandRange> succOperands =
        branchOp.getMutableSuccessorOperands(succ);
    if (!succOperands)
      continue;
    Block *successor = terminator->getSuccessor(succ);

    for (unsigned argI = 0, argE = succOperands->size(); argI < argE; ++argI) {
      // Iterating args in reverse is needed for correctness, to avoid
      // shifting later args when earlier args are erased.
      unsigned arg = argE - argI - 1;
      if (!liveMap.wasProvenLive(successor->getArgument(arg)))
        succOperands->erase(arg);
    }
  }
}

static LogicalResult deleteDeadness(MutableArrayRef<Region> regions,
                                    LiveMap &liveMap) {
  bool erasedAnything = false;
  for (Region &region : regions) {
    if (region.empty())
      continue;

    // We do the deletion in an order that deletes all uses before deleting
    // defs.
    // MLIR's SSA structural invariants guarantee that except for block
    // arguments, the use-def graph is acyclic, so this is possible with a
    // single walk of ops and then a final pass to clean up block arguments.
    //
    // To do this, we visit ops in an order that visits domtree children
    // before domtree parents. A CFG post-order (with reverse iteration with a
    // block) satisfies that without needing an explicit domtree calculation.
    for (Block *block : llvm::post_order(&region.front())) {
      eraseTerminatorSuccessorOperands(block->getTerminator(), liveMap);
      for (Operation &childOp :
           llvm::make_early_inc_range(llvm::reverse(block->getOperations()))) {
        erasedAnything |=
            succeeded(deleteDeadness(childOp.getRegions(), liveMap));
        if (!liveMap.wasProvenLive(&childOp)) {
          erasedAnything = true;
          childOp.erase();
        }
      }
    }
    // Delete block arguments.
    // The entry block has an unknown contract with their enclosing block, so
    // skip it.
    for (Block &block : llvm::drop_begin(region.getBlocks(), 1)) {
      // Iterate in reverse to avoid shifting later arguments when deleting
      // earlier arguments.
      for (unsigned i = 0, e = block.getNumArguments(); i < e; i++)
        if (!liveMap.wasProvenLive(block.getArgument(e - i - 1))) {
          block.eraseArgument(e - i - 1);
          erasedAnything = true;
        }
    }
  }
  return success(erasedAnything);
}

// This function performs a simple dead code elimination algorithm over the
// given regions.
//
// The overall goal is to prove that Values are dead, which allows deleting ops
// and block arguments.
//
// This uses an optimistic algorithm that assumes everything is dead until
// proved otherwise, allowing it to delete recursively dead cycles.
//
// This is a simple fixed-point dataflow analysis algorithm on a lattice
// {Dead,Alive}. Because liveness flows backward, we generally try to
// iterate everything backward to speed up convergence to the fixed-point. This
// allows for being able to delete recursively dead cycles of the use-def graph,
// including block arguments.
//
// This function returns success if any operations or arguments were deleted,
// failure otherwise.
static LogicalResult runRegionDCE(MutableArrayRef<Region> regions) {
  LiveMap liveMap;
  do {
    liveMap.resetChanged();

    for (Region &region : regions)
      propagateLiveness(region, liveMap);
  } while (liveMap.hasChanged());

  return deleteDeadness(regions, liveMap);
}

//===----------------------------------------------------------------------===//
// Block Merging
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BlockEquivalenceData

namespace {
/// This class contains the information for comparing the equivalencies of two
/// blocks. Blocks are considered equivalent if they contain the same operations
/// in the same order. The only allowed divergence is for operands that come
/// from sources outside of the parent block, i.e. the uses of values produced
/// within the block must be equivalent.
///   e.g.,
/// Equivalent:
///  ^bb1(%arg0: i32)
///    return %arg0, %foo : i32, i32
///  ^bb2(%arg1: i32)
///    return %arg1, %bar : i32, i32
/// Not Equivalent:
///  ^bb1(%arg0: i32)
///    return %foo, %arg0 : i32, i32
///  ^bb2(%arg1: i32)
///    return %arg1, %bar : i32, i32
struct BlockEquivalenceData {
  BlockEquivalenceData(Block *block);

  /// Return the order index for the given value that is within the block of
  /// this data.
  unsigned getOrderOf(Value value) const;

  /// The block this data refers to.
  Block *block;
  /// A hash value for this block.
  llvm::hash_code hash;
  /// A map of result producing operations to their relative orders within this
  /// block. The order of an operation is the number of defined values that are
  /// produced within the block before this operation.
  DenseMap<Operation *, unsigned> opOrderIndex;
};
} // end anonymous namespace

BlockEquivalenceData::BlockEquivalenceData(Block *block)
    : block(block), hash(0) {
  unsigned orderIt = block->getNumArguments();
  for (Operation &op : *block) {
    if (unsigned numResults = op.getNumResults()) {
      opOrderIndex.try_emplace(&op, orderIt);
      orderIt += numResults;
    }
    auto opHash = OperationEquivalence::computeHash(
        &op, OperationEquivalence::Flags::IgnoreOperands);
    hash = llvm::hash_combine(hash, opHash);
  }
}

unsigned BlockEquivalenceData::getOrderOf(Value value) const {
  assert(value.getParentBlock() == block && "expected value of this block");

  // Arguments use the argument number as the order index.
  if (BlockArgument arg = value.dyn_cast<BlockArgument>())
    return arg.getArgNumber();

  // Otherwise, the result order is offset from the parent op's order.
  OpResult result = value.cast<OpResult>();
  auto opOrderIt = opOrderIndex.find(result.getDefiningOp());
  assert(opOrderIt != opOrderIndex.end() && "expected op to have an order");
  return opOrderIt->second + result.getResultNumber();
}

//===----------------------------------------------------------------------===//
// BlockMergeCluster

namespace {
/// This class represents a cluster of blocks to be merged together.
class BlockMergeCluster {
public:
  BlockMergeCluster(BlockEquivalenceData &&leaderData)
      : leaderData(std::move(leaderData)) {}

  /// Attempt to add the given block to this cluster. Returns success if the
  /// block was merged, failure otherwise.
  LogicalResult addToCluster(BlockEquivalenceData &blockData);

  /// Try to merge all of the blocks within this cluster into the leader block.
  LogicalResult merge();

private:
  /// The equivalence data for the leader of the cluster.
  BlockEquivalenceData leaderData;

  /// The set of blocks that can be merged into the leader.
  llvm::SmallSetVector<Block *, 1> blocksToMerge;

  /// A set of operand+index pairs that correspond to operands that need to be
  /// replaced by arguments when the cluster gets merged.
  std::set<std::pair<int, int>> operandsToMerge;
};
} // end anonymous namespace

LogicalResult BlockMergeCluster::addToCluster(BlockEquivalenceData &blockData) {
  if (leaderData.hash != blockData.hash)
    return failure();
  Block *leaderBlock = leaderData.block, *mergeBlock = blockData.block;
  if (leaderBlock->getArgumentTypes() != mergeBlock->getArgumentTypes())
    return failure();

  // A set of operands that mismatch between the leader and the new block.
  SmallVector<std::pair<int, int>, 8> mismatchedOperands;
  auto lhsIt = leaderBlock->begin(), lhsE = leaderBlock->end();
  auto rhsIt = blockData.block->begin(), rhsE = blockData.block->end();
  for (int opI = 0; lhsIt != lhsE && rhsIt != rhsE; ++lhsIt, ++rhsIt, ++opI) {
    // Check that the operations are equivalent.
    if (!OperationEquivalence::isEquivalentTo(
            &*lhsIt, &*rhsIt, OperationEquivalence::Flags::IgnoreOperands))
      return failure();

    // Compare the operands of the two operations. If the operand is within
    // the block, it must refer to the same operation.
    auto lhsOperands = lhsIt->getOperands(), rhsOperands = rhsIt->getOperands();
    for (int operand : llvm::seq<int>(0, lhsIt->getNumOperands())) {
      Value lhsOperand = lhsOperands[operand];
      Value rhsOperand = rhsOperands[operand];
      if (lhsOperand == rhsOperand)
        continue;
      // Check that the types of the operands match.
      if (lhsOperand.getType() != rhsOperand.getType())
        return failure();

      // Check that these uses are both external, or both internal.
      bool lhsIsInBlock = lhsOperand.getParentBlock() == leaderBlock;
      bool rhsIsInBlock = rhsOperand.getParentBlock() == mergeBlock;
      if (lhsIsInBlock != rhsIsInBlock)
        return failure();
      // Let the operands differ if they are defined in a different block. These
      // will become new arguments if the blocks get merged.
      if (!lhsIsInBlock) {
        mismatchedOperands.emplace_back(opI, operand);
        continue;
      }

      // Otherwise, these operands must have the same logical order within the
      // parent block.
      if (leaderData.getOrderOf(lhsOperand) != blockData.getOrderOf(rhsOperand))
        return failure();
    }

    // If the lhs or rhs has external uses, the blocks cannot be merged as the
    // merged version of this operation will not be either the lhs or rhs
    // alone (thus semantically incorrect), but some mix dependending on which
    // block preceeded this.
    // TODO allow merging of operations when one block does not dominate the
    // other
    if (rhsIt->isUsedOutsideOfBlock(mergeBlock) ||
        lhsIt->isUsedOutsideOfBlock(leaderBlock)) {
      return failure();
    }
  }
  // Make sure that the block sizes are equivalent.
  if (lhsIt != lhsE || rhsIt != rhsE)
    return failure();

  // If we get here, the blocks are equivalent and can be merged.
  operandsToMerge.insert(mismatchedOperands.begin(), mismatchedOperands.end());
  blocksToMerge.insert(blockData.block);
  return success();
}

/// Returns true if the predecessor terminators of the given block can not have
/// their operands updated.
static bool ableToUpdatePredOperands(Block *block) {
  for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
    auto branch = dyn_cast<BranchOpInterface>((*it)->getTerminator());
    if (!branch || !branch.getMutableSuccessorOperands(it.getSuccessorIndex()))
      return false;
  }
  return true;
}

LogicalResult BlockMergeCluster::merge() {
  // Don't consider clusters that don't have blocks to merge.
  if (blocksToMerge.empty())
    return failure();

  Block *leaderBlock = leaderData.block;
  if (!operandsToMerge.empty()) {
    // If the cluster has operands to merge, verify that the predecessor
    // terminators of each of the blocks can have their successor operands
    // updated.
    // TODO: We could try and sub-partition this cluster if only some blocks
    // cause the mismatch.
    if (!ableToUpdatePredOperands(leaderBlock) ||
        !llvm::all_of(blocksToMerge, ableToUpdatePredOperands))
      return failure();

    // Collect the iterators for each of the blocks to merge. We will walk all
    // of the iterators at once to avoid operand index invalidation.
    SmallVector<Block::iterator, 2> blockIterators;
    blockIterators.reserve(blocksToMerge.size() + 1);
    blockIterators.push_back(leaderBlock->begin());
    for (Block *mergeBlock : blocksToMerge)
      blockIterators.push_back(mergeBlock->begin());

    // Update each of the predecessor terminators with the new arguments.
    SmallVector<SmallVector<Value, 8>, 2> newArguments(
        1 + blocksToMerge.size(),
        SmallVector<Value, 8>(operandsToMerge.size()));
    unsigned curOpIndex = 0;
    for (auto it : llvm::enumerate(operandsToMerge)) {
      unsigned nextOpOffset = it.value().first - curOpIndex;
      curOpIndex = it.value().first;

      // Process the operand for each of the block iterators.
      for (unsigned i = 0, e = blockIterators.size(); i != e; ++i) {
        Block::iterator &blockIter = blockIterators[i];
        std::advance(blockIter, nextOpOffset);
        auto &operand = blockIter->getOpOperand(it.value().second);
        newArguments[i][it.index()] = operand.get();

        // Update the operand and insert an argument if this is the leader.
        if (i == 0)
          operand.set(leaderBlock->addArgument(operand.get().getType()));
      }
    }
    // Update the predecessors for each of the blocks.
    auto updatePredecessors = [&](Block *block, unsigned clusterIndex) {
      for (auto predIt = block->pred_begin(), predE = block->pred_end();
           predIt != predE; ++predIt) {
        auto branch = cast<BranchOpInterface>((*predIt)->getTerminator());
        unsigned succIndex = predIt.getSuccessorIndex();
        branch.getMutableSuccessorOperands(succIndex)->append(
            newArguments[clusterIndex]);
      }
    };
    updatePredecessors(leaderBlock, /*clusterIndex=*/0);
    for (unsigned i = 0, e = blocksToMerge.size(); i != e; ++i)
      updatePredecessors(blocksToMerge[i], /*clusterIndex=*/i + 1);
  }

  // Replace all uses of the merged blocks with the leader and erase them.
  for (Block *block : blocksToMerge) {
    block->replaceAllUsesWith(leaderBlock);
    block->erase();
  }
  return success();
}

/// Identify identical blocks within the given region and merge them, inserting
/// new block arguments as necessary. Returns success if any blocks were merged,
/// failure otherwise.
static LogicalResult mergeIdenticalBlocks(Region &region) {
  if (region.empty() || llvm::hasSingleElement(region))
    return failure();

  // Identify sets of blocks, other than the entry block, that branch to the
  // same successors. We will use these groups to create clusters of equivalent
  // blocks.
  DenseMap<SuccessorRange, SmallVector<Block *, 1>> matchingSuccessors;
  for (Block &block : llvm::drop_begin(region, 1))
    matchingSuccessors[block.getSuccessors()].push_back(&block);

  bool mergedAnyBlocks = false;
  for (ArrayRef<Block *> blocks : llvm::make_second_range(matchingSuccessors)) {
    if (blocks.size() == 1)
      continue;

    SmallVector<BlockMergeCluster, 1> clusters;
    for (Block *block : blocks) {
      BlockEquivalenceData data(block);

      // Don't allow merging if this block has any regions.
      // TODO: Add support for regions if necessary.
      bool hasNonEmptyRegion = llvm::any_of(*block, [](Operation &op) {
        return llvm::any_of(op.getRegions(),
                            [](Region &region) { return !region.empty(); });
      });
      if (hasNonEmptyRegion)
        continue;

      // Try to add this block to an existing cluster.
      bool addedToCluster = false;
      for (auto &cluster : clusters)
        if ((addedToCluster = succeeded(cluster.addToCluster(data))))
          break;
      if (!addedToCluster)
        clusters.emplace_back(std::move(data));
    }
    for (auto &cluster : clusters)
      mergedAnyBlocks |= succeeded(cluster.merge());
  }

  return success(mergedAnyBlocks);
}

/// Identify identical blocks within the given regions and merge them, inserting
/// new block arguments as necessary.
static LogicalResult mergeIdenticalBlocks(MutableArrayRef<Region> regions) {
  llvm::SmallSetVector<Region *, 1> worklist;
  for (auto &region : regions)
    worklist.insert(&region);
  bool anyChanged = false;
  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    if (succeeded(mergeIdenticalBlocks(*region))) {
      worklist.insert(region);
      anyChanged = true;
    }

    // Add any nested regions to the worklist.
    for (Block &block : *region)
      for (auto &op : block)
        for (auto &nestedRegion : op.getRegions())
          worklist.insert(&nestedRegion);
  }

  return success(anyChanged);
}

//===----------------------------------------------------------------------===//
// Region Simplification
//===----------------------------------------------------------------------===//

/// Run a set of structural simplifications over the given regions. This
/// includes transformations like unreachable block elimination, dead argument
/// elimination, as well as some other DCE. This function returns success if any
/// of the regions were simplified, failure otherwise.
LogicalResult mlir::simplifyRegions(MutableArrayRef<Region> regions) {
  bool eliminatedBlocks = succeeded(eraseUnreachableBlocks(regions));
  bool eliminatedOpsOrArgs = succeeded(runRegionDCE(regions));
  bool mergedIdenticalBlocks = succeeded(mergeIdenticalBlocks(regions));
  return success(eliminatedBlocks || eliminatedOpsOrArgs ||
                 mergedIdenticalBlocks);
}
