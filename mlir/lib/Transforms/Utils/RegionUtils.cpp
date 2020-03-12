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
#include "mlir/Interfaces/SideEffects.h"

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
  if (owner->isKnownTerminator()) {
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
  if (!op->isKnownNonTerminator())
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
  if (!branchInterface || !branchInterface.canEraseSuccessorOperand()) {
    for (Block *successor : op->getSuccessors())
      for (BlockArgument arg : successor->getArguments())
        liveMap.setProvedLive(arg);
    return;
  }

  // If we can't reason about the operands to a successor, conservatively mark
  // all arguments as live.
  for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
    if (!branchInterface.getSuccessorOperands(i))
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
  if (op->isKnownTerminator())
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
    Optional<OperandRange> succOperands = branchOp.getSuccessorOperands(succ);
    if (!succOperands)
      continue;
    Block *successor = terminator->getSuccessor(succ);

    for (unsigned argI = 0, argE = succOperands->size(); argI < argE; ++argI) {
      // Iterating args in reverse is needed for correctness, to avoid
      // shifting later args when earlier args are erased.
      unsigned arg = argE - argI - 1;
      if (!liveMap.wasProvenLive(successor->getArgument(arg)))
        branchOp.eraseSuccessorOperand(succ, arg);
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
// Region Simplification
//===----------------------------------------------------------------------===//

/// Run a set of structural simplifications over the given regions. This
/// includes transformations like unreachable block elimination, dead argument
/// elimination, as well as some other DCE. This function returns success if any
/// of the regions were simplified, failure otherwise.
LogicalResult mlir::simplifyRegions(MutableArrayRef<Region> regions) {
  LogicalResult eliminatedBlocks = eraseUnreachableBlocks(regions);
  LogicalResult eliminatedOpsOrArgs = runRegionDCE(regions);
  return success(succeeded(eliminatedBlocks) || succeeded(eliminatedOpsOrArgs));
}
