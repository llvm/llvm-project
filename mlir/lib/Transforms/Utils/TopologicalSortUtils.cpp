//===- TopologicalSortUtils.h - Topological sort utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/TopologicalSortUtils.h"
#include "mlir/IR/OpDefinition.h"

using namespace mlir;

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

  // An operation is ready to be scheduled if all its operands are ready. An
  // operation is ready if:
  const auto isReady = [&](Value value, Operation *top) {
    // - the user-provided callback marks it as ready,
    if (isOperandReady && isOperandReady(value, top))
      return true;
    Operation *parent = value.getDefiningOp();
    // - it is a block argument,
    if (!parent)
      return true;
    Operation *ancestor = block->findAncestorOpInBlock(*parent);
    // - it is an implicit capture,
    if (!ancestor)
      return true;
    // - it is defined in a nested region, or
    if (ancestor == top)
      return true;
    // - its ancestor in the block is scheduled.
    return !unscheduledOps.contains(ancestor);
  };

  bool allOpsScheduled = true;
  while (!unscheduledOps.empty()) {
    bool scheduledAtLeastOnce = false;

    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (move it before the `nextScheduledOp`).
    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
      // An operation is recursively ready to be scheduled of it and its nested
      // operations are ready.
      WalkResult readyToSchedule = op.walk([&](Operation *nestedOp) {
        return llvm::all_of(
                   nestedOp->getOperands(),
                   [&](Value operand) { return isReady(operand, &op); })
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      });
      if (readyToSchedule.wasInterrupted())
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
