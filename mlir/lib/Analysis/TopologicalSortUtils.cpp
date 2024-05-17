//===- TopologicalSortUtils.cpp - Topological sort utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionGraphTraits.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

/// Return `true` if the given operation is ready to be scheduled.
static bool isOpReady(Operation *op, DenseSet<Operation *> &unscheduledOps,
                      function_ref<bool(Value, Operation *)> isOperandReady) {
  // An operation is ready to be scheduled if all its operands are ready. An
  // operation is ready if:
  const auto isReady = [&](Value value) {
    // - the user-provided callback marks it as ready,
    if (isOperandReady && isOperandReady(value, op))
      return true;
    Operation *parent = value.getDefiningOp();
    // - it is a block argument,
    if (!parent)
      return true;
    // - or it is not defined by an unscheduled op (and also not nested within
    //   an unscheduled op).
    do {
      // Stop traversal when op under examination is reached.
      if (parent == op)
        return true;
      if (unscheduledOps.contains(parent))
        return false;
    } while ((parent = parent->getParentOp()));
    // No unscheduled op found.
    return true;
  };

  // An operation is recursively ready to be scheduled of it and its nested
  // operations are ready.
  WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
    return llvm::all_of(nestedOp->getOperands(),
                        [&](Value operand) { return isReady(operand); })
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return !readyToSchedule.wasInterrupted();
}

bool mlir::sortTopologically(
    Block *block, llvm::iterator_range<Block::iterator> ops,
    function_ref<bool(Value, Operation *)> isOperandReady) {
  if (ops.empty())
    return true;

  // The set of operations that have not yet been scheduled.
  DenseSet<Operation *> unscheduledOps;
  // Mark all operations as unscheduled.
  for (Operation &op : ops)
    unscheduledOps.insert(&op);

  Block::iterator nextScheduledOp = ops.begin();
  Block::iterator end = ops.end();

  bool allOpsScheduled = true;
  while (!unscheduledOps.empty()) {
    bool scheduledAtLeastOnce = false;

    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (move it before the `nextScheduledOp`).
    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
      if (!isOpReady(&op, unscheduledOps, isOperandReady))
        continue;

      // Schedule the operation by moving it to the start.
      unscheduledOps.erase(&op);
      op.moveBefore(block, nextScheduledOp);
      scheduledAtLeastOnce = true;
      // Move the iterator forward if we schedule the operation at the front.
      if (&op == &*nextScheduledOp)
        ++nextScheduledOp;
    }
    // If no operations were scheduled, give up and advance the iterator.
    if (!scheduledAtLeastOnce) {
      allOpsScheduled = false;
      unscheduledOps.erase(&*nextScheduledOp);
      ++nextScheduledOp;
    }
  }

  return allOpsScheduled;
}

bool mlir::sortTopologically(
    Block *block, function_ref<bool(Value, Operation *)> isOperandReady) {
  if (block->empty())
    return true;
  if (block->back().hasTrait<OpTrait::IsTerminator>())
    return sortTopologically(block, block->without_terminator(),
                             isOperandReady);
  return sortTopologically(block, *block, isOperandReady);
}

bool mlir::computeTopologicalSorting(
    MutableArrayRef<Operation *> ops,
    function_ref<bool(Value, Operation *)> isOperandReady) {
  if (ops.empty())
    return true;

  // The set of operations that have not yet been scheduled.
  DenseSet<Operation *> unscheduledOps;

  // Mark all operations as unscheduled.
  for (Operation *op : ops)
    unscheduledOps.insert(op);

  unsigned nextScheduledOp = 0;

  bool allOpsScheduled = true;
  while (!unscheduledOps.empty()) {
    bool scheduledAtLeastOnce = false;

    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (swap it with the op at `nextScheduledOp`).
    for (unsigned i = nextScheduledOp; i < ops.size(); ++i) {
      if (!isOpReady(ops[i], unscheduledOps, isOperandReady))
        continue;

      // Schedule the operation by moving it to the start.
      unscheduledOps.erase(ops[i]);
      std::swap(ops[i], ops[nextScheduledOp]);
      scheduledAtLeastOnce = true;
      ++nextScheduledOp;
    }

    // If no operations were scheduled, just schedule the first op and continue.
    if (!scheduledAtLeastOnce) {
      allOpsScheduled = false;
      unscheduledOps.erase(ops[nextScheduledOp++]);
    }
  }

  return allOpsScheduled;
}

SetVector<Block *> mlir::getBlocksSortedByDominance(Region &region) {
  // For each block that has not been visited yet (i.e. that has no
  // predecessors), add it to the list as well as its successors.
  SetVector<Block *> blocks;
  for (Block &b : region) {
    if (blocks.count(&b) == 0) {
      llvm::ReversePostOrderTraversal<Block *> traversal(&b);
      blocks.insert(traversal.begin(), traversal.end());
    }
  }
  assert(blocks.size() == region.getBlocks().size() &&
         "some blocks are not sorted");

  return blocks;
}

/// Computes the common ancestor region of all operations in `ops`. Remembers
/// all the traversed regions in `traversedRegions`.
static Region *findCommonParentRegion(const SetVector<Operation *> &ops,
                                      DenseSet<Region *> &traversedRegions) {
  // Map to count the number of times a region was encountered.
  llvm::DenseMap<Region *, size_t> regionCounts;
  size_t expectedCount = ops.size();

  // Walk the region tree for each operation towards the root and add to the
  // region count.
  Region *res = nullptr;
  for (Operation *op : ops) {
    Region *current = op->getParentRegion();
    while (current) {
      // Insert or get the count.
      auto it = regionCounts.try_emplace(current, 0).first;
      size_t count = ++it->getSecond();
      if (count == expectedCount) {
        res = current;
        break;
      }
      current = current->getParentRegion();
    }
  }
  auto firstRange = llvm::make_first_range(regionCounts);
  traversedRegions.insert(firstRange.begin(), firstRange.end());
  return res;
}

/// Topologically traverses `region` and insers all encountered operations in
/// `toSort` into the result. Recursively traverses regions when they are
/// present in `relevantRegions`.
static void topoSortRegion(Region &region,
                           const DenseSet<Region *> &relevantRegions,
                           const SetVector<Operation *> &toSort,
                           SetVector<Operation *> &result) {
  SetVector<Block *> sortedBlocks = getBlocksSortedByDominance(region);
  for (Block *block : sortedBlocks) {
    for (Operation &op : *block) {
      if (toSort.contains(&op))
        result.insert(&op);
      for (Region &subRegion : op.getRegions()) {
        // Skip regions that do not contain operations from `toSort`.
        if (!relevantRegions.contains(&region))
          continue;
        topoSortRegion(subRegion, relevantRegions, toSort, result);
      }
    }
  }
}

SetVector<Operation *>
mlir::topologicalSort(const SetVector<Operation *> &toSort) {
  if (toSort.size() <= 1)
    return toSort;

  assert(llvm::all_of(toSort,
                      [&](Operation *op) { return toSort.count(op) == 1; }) &&
         "expected only unique set entries");

  // First, find the root region to start the recursive traversal through the
  // IR.
  DenseSet<Region *> relevantRegions;
  Region *rootRegion = findCommonParentRegion(toSort, relevantRegions);
  assert(rootRegion && "expected all ops to have a common ancestor");

  // Sort all element in `toSort` by recursively traversing the IR.
  SetVector<Operation *> result;
  topoSortRegion(*rootRegion, relevantRegions, toSort, result);
  assert(result.size() == toSort.size() &&
         "expected all operations to be present in the result");
  return result;
}
