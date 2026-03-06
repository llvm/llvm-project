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
  return sortTopologically(block, block->without_terminator(), isOperandReady);
}

bool mlir::computeTopologicalSorting(
    MutableArrayRef<Operation *> ops,
    function_ref<bool(Value, Operation *)> isOperandReady) {
  if (ops.empty())
    return true;

  // The set of operations that have not yet been scheduled.
  // Mark all operations as unscheduled.
  DenseSet<Operation *> unscheduledOps(llvm::from_range, ops);

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
      blocks.insert_range(traversal);
    }
  }
  assert(blocks.size() == region.getBlocks().size() &&
         "some blocks are not sorted");

  return blocks;
}

namespace {
class TopoSortHelper {
public:
  explicit TopoSortHelper(const SetVector<Operation *> &toSort)
      : toSort(toSort) {}

  /// Executes the topological sort of the operations this instance was
  /// constructed with. This function will destroy the internal state of the
  /// instance.
  SetVector<Operation *> sort() {
    if (toSort.size() <= 1) {
      // Note: Creates a copy on purpose.
      return toSort;
    }

    // First, find the root region to start the traversal through the IR. This
    // additionally enriches the internal caches with all relevant ancestor
    // regions and blocks.
    Region *rootRegion = findCommonAncestorRegion();
    assert(rootRegion && "expected all ops to have a common ancestor");

    // Sort all elements in `toSort` by traversing the IR in the appropriate
    // order.
    SetVector<Operation *> result = topoSortRegion(*rootRegion);
    assert(result.size() == toSort.size() &&
           "expected all operations to be present in the result");
    return result;
  }

private:
  /// Computes the closest common ancestor region of all operations in `toSort`.
  Region *findCommonAncestorRegion() {
    // Map to count the number of times a region was encountered.
    DenseMap<Region *, size_t> regionCounts;
    size_t expectedCount = toSort.size();

    // Walk the region tree for each operation towards the root and add to the
    // region count.
    Region *res = nullptr;
    for (Operation *op : toSort) {
      Region *current = op->getParentRegion();
      // Store the block as an ancestor block.
      ancestorBlocks.insert(op->getBlock());
      while (current) {
        // Insert or update the count and compare it.
        if (++regionCounts[current] == expectedCount) {
          res = current;
          break;
        }
        ancestorBlocks.insert(current->getParentOp()->getBlock());
        current = current->getParentRegion();
      }
    }
    auto firstRange = llvm::make_first_range(regionCounts);
    ancestorRegions.insert_range(firstRange);
    return res;
  }

  /// Performs the dominance respecting IR walk to collect the topological order
  /// of the operation to sort.
  SetVector<Operation *> topoSortRegion(Region &rootRegion) {
    using StackT = PointerUnion<Region *, Block *, Operation *>;

    SetVector<Operation *> result;
    // Stack that stores the different IR constructs to traverse.
    SmallVector<StackT> stack;
    stack.push_back(&rootRegion);

    // Traverse the IR in a dominance respecting pre-order walk.
    while (!stack.empty()) {
      StackT current = stack.pop_back_val();
      if (auto *region = dyn_cast<Region *>(current)) {
        // A region's blocks need to be traversed in dominance order.
        SetVector<Block *> sortedBlocks = getBlocksSortedByDominance(*region);
        for (Block *block : llvm::reverse(sortedBlocks)) {
          // Only add blocks to the stack that are ancestors of the operations
          // to sort.
          if (ancestorBlocks.contains(block))
            stack.push_back(block);
        }
        continue;
      }

      if (auto *block = dyn_cast<Block *>(current)) {
        // Add all of the blocks operations to the stack.
        for (Operation &op : llvm::reverse(*block))
          stack.push_back(&op);
        continue;
      }

      auto *op = cast<Operation *>(current);
      if (toSort.contains(op))
        result.insert(op);

      // Add all the subregions that are ancestors of the operations to sort.
      for (Region &subRegion : op->getRegions())
        if (ancestorRegions.contains(&subRegion))
          stack.push_back(&subRegion);
    }
    return result;
  }

  /// Operations to sort.
  const SetVector<Operation *> &toSort;
  /// Set containing all the ancestor regions of the operations to sort.
  DenseSet<Region *> ancestorRegions;
  /// Set containing all the ancestor blocks of the operations to sort.
  DenseSet<Block *> ancestorBlocks;
};
} // namespace

SetVector<Operation *>
mlir::topologicalSort(const SetVector<Operation *> &toSort) {
  return TopoSortHelper(toSort).sort();
}
