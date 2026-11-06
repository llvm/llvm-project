//===- RegionExitInterfaces.cpp - Region exit interfaces ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/RegionExitInterfaces.h"
#include "mlir/Support/WalkResult.h"

using namespace mlir;

#include "mlir/IR/RegionExitInterfaces.cpp.inc"

namespace {
// Worklist entry for walking block terminators in nested regions.
// Tracks the current position within a region and the nesting depth.
struct NestedOpIterator {
  NestedOpIterator(Region *region, int nestedLevel)
      : region(region), nestedLevel(nestedLevel) {
    regionIt = region->begin();
    if (regionIt != region->end())
      blockIt = regionIt->begin();
  }
  // Advance the iterator to the next reachable operation.
  void advance() {
    assert(regionIt != region->end());
    if (blockIt == regionIt->end()) {
      ++regionIt;
      if (regionIt != region->end())
        blockIt = regionIt->begin();
      return;
    }
    ++blockIt;
  }

  // The region we're iterating over.
  Region *region;
  // The Block currently being iterated over.
  Region::iterator regionIt;
  // The Operation currently being iterated over.
  Block::iterator blockIt;
  // The nested level of the current region relative to the starting region.
  int nestedLevel = 0;
};
} // namespace

/// Walk block terminators nested within `rootOp` through propagators,
/// receivers, and unregistered operations. The callback receives the terminator
/// and its 1-based nesting level relative to `rootOp`. Callers filter based on
/// breaking-control-flow properties as needed.
static void walkNestedBlockTerminators(
    Operation *rootOp, function_ref<WalkResult(Operation *, int)> callback) {
  // Worklist of regions to visit to drive the traversal.
  SmallVector<NestedOpIterator> worklist;

  // Perform a traversal of the regions, visiting each
  // reachable operation.
  for (Region &region : rootOp->getRegions()) {
    if (region.empty())
      continue;
    worklist.push_back({&region, 1});
  }
  while (!worklist.empty()) {
    NestedOpIterator &it = worklist.back();
    if (it.regionIt == it.region->end()) {
      // We're done with this region.
      worklist.pop_back();
      continue;
    }
    if (it.blockIt == it.regionIt->end()) {
      // We're done with this block.
      it.advance();
      continue;
    }
    Operation *op = &*it.blockIt;

    // Only call the callback if we're at the end of the block.
    if (std::next(it.blockIt) == it.regionIt->end() &&
        callback(op, it.nestedLevel).wasInterrupted())
      return;

    // Advance before pushing nested regions to avoid reference invalidation.
    int currentNestedLevel = it.nestedLevel;
    it.advance();

    // A registered op without either capability is a propagation boundary.
    if (op->isRegistered() &&
        !op->hasTrait<OpTrait::PropagateControlFlowBreak>() &&
        !isa<HasBreakingControlFlowOpInterface>(op))
      continue;

    // Recursively visit the nested regions.
    for (Region &nestedRegion : op->getRegions()) {
      if (nestedRegion.empty())
        continue;
      worklist.push_back({&nestedRegion, currentNestedLevel + 1});
    }
  }
}

NestedBreakingControlFlowInfo
mlir::getNestedBreakingControlFlowInfo(Operation *op) {
  NestedBreakingControlFlowInfo info;
  walkNestedBlockTerminators(op, [&](Operation *visitedOp, int nestedLevel) {
    bool directlyTargetsOp = false;
    bool escapesThroughOp = false;
    for (HasBreakingControlFlowOpInterface target :
         findPotentialBreakTargets(visitedOp)) {
      if (!target)
        continue;
      Operation *targetOp = target.getOperation();
      if (targetOp == op)
        directlyTargetsOp = true;
      else if (targetOp->isProperAncestor(op))
        escapesThroughOp = true;
    }

    if (directlyTargetsOp) {
      info.predecessors.push_back(visitedOp);
      if (nestedLevel > 1)
        info.hasNestedPredecessors = true;
    }
    if (escapesThroughOp)
      info.hasEscapingBreakingControlFlow = true;

    return WalkResult::advance();
  });
  return info;
}

bool mlir::hasNestedPredecessors(Operation *op) {
  return getNestedBreakingControlFlowInfo(op).hasNestedPredecessors;
}

bool mlir::hasEscapingBreakingControlFlow(Operation *op) {
  return getNestedBreakingControlFlowInfo(op).hasEscapingBreakingControlFlow;
}

void mlir::detail::visitNestedBreakingControlFlowOpsImpl(
    Operation *op,
    function_ref<WalkResult(RegionExitTerminatorOpInterface, int nestedLevel)>
        callback) {
  walkNestedBlockTerminators(op, [&](Operation *visitedOp, int nestedLevel) {
    for (HasBreakingControlFlowOpInterface target :
         findPotentialBreakTargets(visitedOp)) {
      if (target && (target.getOperation() == op ||
                     target.getOperation()->isProperAncestor(op)))
        return callback(cast<RegionExitTerminatorOpInterface>(visitedOp),
                        nestedLevel);
    }
    return WalkResult::advance();
  });
}

SmallVector<HasBreakingControlFlowOpInterface>
mlir::findPotentialBreakTargets(Operation *terminator) {
  auto breakingTerminator =
      dyn_cast<RegionExitTerminatorOpInterface>(terminator);
  if (!breakingTerminator)
    return {};
  SmallVector<HasBreakingControlFlowOpInterface> targets;
  for (Operation *target : breakingTerminator.getPotentialTargets()) {
    if (!target)
      continue;
    if (auto breakTarget = dyn_cast<HasBreakingControlFlowOpInterface>(target))
      targets.push_back(breakTarget);
  }
  return targets;
}
