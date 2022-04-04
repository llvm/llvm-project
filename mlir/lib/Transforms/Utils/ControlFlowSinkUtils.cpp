//===- ControlFlowSinkUtils.cpp - Code to perform control-flow sinking ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for control-flow sinking. Control-flow
// sinking moves operations whose only uses are in conditionally-executed blocks
// into those blocks so that they aren't executed on paths where their results
// are not needed.
//
// Control-flow sinking is not implemented on BranchOpInterface because
// sinking ops into the successors of branch operations may move ops into loops.
// It is idiomatic MLIR to perform optimizations at IR levels that readily
// provide the necessary information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <vector>

#define DEBUG_TYPE "cf-sink"

using namespace mlir;

namespace {
/// A helper struct for control-flow sinking.
class Sinker {
public:
  /// Create an operation sinker with given dominance info.
  Sinker(function_ref<bool(Operation *, Region *)> shouldMoveIntoRegion,
         function_ref<void(Operation *, Region *)> moveIntoRegion,
         DominanceInfo &domInfo)
      : shouldMoveIntoRegion(shouldMoveIntoRegion),
        moveIntoRegion(moveIntoRegion), domInfo(domInfo) {}

  /// Given a list of regions, find operations to sink and sink them. Return the
  /// number of operations sunk.
  size_t sinkRegions(ArrayRef<Region *> regions);

private:
  /// Given a region and an op which dominates the region, returns true if all
  /// users of the given op are dominated by the entry block of the region, and
  /// thus the operation can be sunk into the region.
  bool allUsersDominatedBy(Operation *op, Region *region);

  /// Given a region and a top-level op (an op whose parent region is the given
  /// region), determine whether the defining ops of the op's operands can be
  /// sunk into the region.
  ///
  /// Add moved ops to the work queue.
  void tryToSinkPredecessors(Operation *user, Region *region,
                             std::vector<Operation *> &stack);

  /// Iterate over all the ops in a region and try to sink their predecessors.
  /// Recurse on subgraphs using a work queue.
  void sinkRegion(Region *region);

  /// The callback to determine whether an op should be moved in to a region.
  function_ref<bool(Operation *, Region *)> shouldMoveIntoRegion;
  /// The calback to move an operation into the region.
  function_ref<void(Operation *, Region *)> moveIntoRegion;
  /// Dominance info to determine op user dominance with respect to regions.
  DominanceInfo &domInfo;
  /// The number of operations sunk.
  size_t numSunk = 0;
};
} // end anonymous namespace

bool Sinker::allUsersDominatedBy(Operation *op, Region *region) {
  assert(region->findAncestorOpInRegion(*op) == nullptr &&
         "expected op to be defined outside the region");
  return llvm::all_of(op->getUsers(), [&](Operation *user) {
    // The user is dominated by the region if its containing block is dominated
    // by the region's entry block.
    return domInfo.dominates(&region->front(), user->getBlock());
  });
}

void Sinker::tryToSinkPredecessors(Operation *user, Region *region,
                                   std::vector<Operation *> &stack) {
  LLVM_DEBUG(user->print(llvm::dbgs() << "\nContained op:\n"));
  for (Value value : user->getOperands()) {
    Operation *op = value.getDefiningOp();
    // Ignore block arguments and ops that are already inside the region.
    if (!op || op->getParentRegion() == region)
      continue;
    LLVM_DEBUG(op->print(llvm::dbgs() << "\nTry to sink:\n"));

    // If the op's users are all in the region and it can be moved, then do so.
    if (allUsersDominatedBy(op, region) && shouldMoveIntoRegion(op, region)) {
      moveIntoRegion(op, region);
      ++numSunk;
      // Add the op to the work queue.
      stack.push_back(op);
    }
  }
}

void Sinker::sinkRegion(Region *region) {
  // Initialize the work queue with all the ops in the region.
  std::vector<Operation *> stack;
  for (Operation &op : region->getOps())
    stack.push_back(&op);

  // Process all the ops depth-first. This ensures that nodes of subgraphs are
  // sunk in the correct order.
  while (!stack.empty()) {
    Operation *op = stack.back();
    stack.pop_back();
    tryToSinkPredecessors(op, region, stack);
  }
}

size_t Sinker::sinkRegions(ArrayRef<Region *> regions) {
  for (Region *region : regions)
    if (!region->empty())
      sinkRegion(region);
  return numSunk;
}

size_t mlir::controlFlowSink(
    ArrayRef<Region *> regions, DominanceInfo &domInfo,
    function_ref<bool(Operation *, Region *)> shouldMoveIntoRegion,
    function_ref<void(Operation *, Region *)> moveIntoRegion) {
  return Sinker(shouldMoveIntoRegion, moveIntoRegion, domInfo)
      .sinkRegions(regions);
}

void mlir::getSinglyExecutedRegionsToSink(RegionBranchOpInterface branch,
                                          SmallVectorImpl<Region *> &regions) {
  // Collect constant operands.
  SmallVector<Attribute> operands(branch->getNumOperands(), Attribute());
  for (auto &it : llvm::enumerate(branch->getOperands()))
    (void)matchPattern(it.value(), m_Constant(&operands[it.index()]));

  // Get the invocation bounds.
  SmallVector<InvocationBounds> bounds;
  branch.getRegionInvocationBounds(operands, bounds);

  // For a simple control-flow sink, only consider regions that are executed at
  // most once.
  for (auto it : llvm::zip(branch->getRegions(), bounds)) {
    const InvocationBounds &bound = std::get<1>(it);
    if (bound.getUpperBound() && *bound.getUpperBound() <= 1)
      regions.push_back(&std::get<0>(it));
  }
}
