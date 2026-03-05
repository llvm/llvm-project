//===- RegionKindInterface.cpp - Region Kind Interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the region kind interfaces defined in
// `RegionKindInterface.td`.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "region-kind-interface"

using namespace mlir;

#include "mlir/IR/RegionKindInterface.cpp.inc"

bool mlir::mayHaveSSADominance(Region &region) {
  auto regionKindOp = dyn_cast<RegionKindInterface>(region.getParentOp());
  if (!regionKindOp)
    return true;
  return regionKindOp.hasSSADominance(region.getRegionNumber());
}

bool mlir::mayBeGraphRegion(Region &region) {
  if (!region.getParentOp()->isRegistered())
    return true;
  auto regionKindOp = dyn_cast<RegionKindInterface>(region.getParentOp());
  if (!regionKindOp)
    return false;
  return !regionKindOp.hasSSADominance(region.getRegionNumber());
}

namespace {
// Iterator on all reachable operations in the region.
// Also keep track if we visited the nested regions of the current op
// already to drive the traversal.
struct NestedOpIterator {
  NestedOpIterator(Region *region, int nestedLevel)
      : region(region), nestedLevel(nestedLevel) {
    regionIt = region->begin();
    blockIt = regionIt->end();
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
    if (blockIt != regionIt->end()) {
      LDBG() << this << " - Incrementing block iterator, next op: "
             << OpWithFlags(&*blockIt, OpPrintingFlags().skipRegions());
    }
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

/// Recursive walk that calls the callback only for terminator operation which
/// are breaking control flow.
static void walk(Operation *rootOp,
                 function_ref<WalkResult(Operation *, int)> callback) {
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

    // Recursively visit the nested regions.
    for (Region &nestedRegion : op->getRegions()) {
      if (nestedRegion.empty())
        continue;
      worklist.push_back({&nestedRegion, currentNestedLevel + 1});
    }
  }
}

/// Return true if `op` has at least one RegionTerminator nested inside it
/// that directly targets `op` as its control-flow destination. A terminator
/// directly targets `op` when its num-breaking-regions equals the nesting
/// depth at which it appears inside `op`'s regions, AND that depth is > 1
/// (depth 1 would mean the terminator exits only the immediately enclosing
/// region, going to `op`'s parent rather than `op` itself — that case is
/// handled by the normal RegionBranchOpInterface path).
bool mlir::hasNestedPredecessors(Operation *op) {
  bool found = false;
  walk(op, [&](Operation *visitedOp, int nestedLevel) {
    if (nestedLevel > 1 &&
        nestedLevel ==
            static_cast<int>(visitedOp->getNumBreakingControlRegions()))
      found = true;
    return found ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found;
}

/// Return true if `op` contains any RegionTerminator whose num-breaking-
/// regions value would carry it *past* `op` toward an outer ancestor. Such a
/// terminator's nestedLevel (depth relative to `op`'s body) is strictly less
/// than its num-breaking-regions, meaning `op` is one of the intermediate
/// PropagateControlFlowBreak ops that is bypassed by this early exit.
bool mlir::hasBreakingControlFlowOps(Operation *op) {
  bool found = false;
  walk(op, [&](Operation *visitedOp, int nestedLevel) {
    if (nestedLevel <
        static_cast<int>(visitedOp->getNumBreakingControlRegions()))
      found = true;
    return found ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found;
}

/// Invoke `callback` for every RegionTerminator inside `op` whose
/// num-breaking-regions is >= its current nesting depth (i.e. the terminator
/// either terminates directly into `op` or propagates further upward). The
/// `nestedLevel` passed to the callback is the 1-based depth of the terminator
/// relative to `op`'s outermost region.
void mlir::detail::visitNestedBreakingControlFlowOpsImpl(
    Operation *op,
    function_ref<WalkResult(Operation *, int nestedLevel)> callback) {
  ::walk(op, [&](Operation *visitedOp, int nestedLevel) {
    if (nestedLevel <=
        static_cast<int>(visitedOp->getNumBreakingControlRegions()))
      return callback(visitedOp, nestedLevel);
    return WalkResult::advance();
  });
}

/// Collect all RegionTerminator ops nested inside `op` that directly target
/// `op` as their control-flow destination (num-breaking-regions ==
/// nestedLevel). These are the ops that transfer control to `op` on an early
/// exit path; they are the "nested predecessors" of `op`.
void mlir::collectAllNestedPredecessors(
    Operation *op, SmallVector<Operation *> &predecessors) {
  visitNestedBreakingControlFlowOps(
      op, [&](Operation *visitedOp, int nestedLevel) {
        if (nestedLevel ==
            static_cast<int>(visitedOp->getNumBreakingControlRegions()))
          predecessors.push_back(visitedOp);
        return WalkResult::advance();
      });
}