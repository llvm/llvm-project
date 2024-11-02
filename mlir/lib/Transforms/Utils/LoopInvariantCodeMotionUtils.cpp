//===- LoopInvariantCodeMotionUtils.cpp - LICM Utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the core LICM algorithm.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/SideEffectUtils.h"
#include "llvm/Support/Debug.h"
#include <queue>

#define DEBUG_TYPE "licm"

using namespace mlir;

/// Checks whether the given op can be hoisted by checking that
/// - the op and none of its contained operations depend on values inside of the
///   loop (by means of calling definedOutside).
/// - the op has no side-effects.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  // Do not move terminators.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;

  // Walk the nested operations and check that all used values are either
  // defined outside of the loop or in a nested region, but not at the level of
  // the loop body.
  auto walkFn = [&](Operation *child) {
    for (Value operand : child->getOperands()) {
      // Ignore values defined in a nested region.
      if (op->isAncestor(operand.getParentRegion()->getParentOp()))
        continue;
      if (!definedOutside(operand))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  return !op->walk(walkFn).wasInterrupted();
}

size_t mlir::moveLoopInvariantCode(
    RegionRange regions,
    function_ref<bool(Value, Region *)> isDefinedOutsideRegion,
    function_ref<bool(Operation *, Region *)> shouldMoveOutOfRegion,
    function_ref<void(Operation *, Region *)> moveOutOfRegion) {
  size_t numMoved = 0;

  for (Region *region : regions) {
    LLVM_DEBUG(llvm::dbgs() << "Original loop:\n"
                            << *region->getParentOp() << "\n");

    std::queue<Operation *> worklist;
    // Add top-level operations in the loop body to the worklist.
    for (Operation &op : region->getOps())
      worklist.push(&op);

    auto definedOutside = [&](Value value) {
      return isDefinedOutsideRegion(value, region);
    };

    while (!worklist.empty()) {
      Operation *op = worklist.front();
      worklist.pop();
      // Skip ops that have already been moved. Check if the op can be hoisted.
      if (op->getParentRegion() != region)
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Checking op: " << *op << "\n");
      if (!shouldMoveOutOfRegion(op, region) ||
          !canBeHoisted(op, definedOutside))
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Moving loop-invariant op: " << *op << "\n");
      moveOutOfRegion(op, region);
      ++numMoved;

      // Since the op has been moved, we need to check its users within the
      // top-level of the loop body.
      for (Operation *user : op->getUsers())
        if (user->getParentRegion() == region)
          worklist.push(user);
    }
  }

  return numMoved;
}

size_t mlir::moveLoopInvariantCode(LoopLikeOpInterface loopLike) {
  return moveLoopInvariantCode(
      &loopLike.getLoopBody(),
      [&](Value value, Region *) {
        return loopLike.isDefinedOutsideOfLoop(value);
      },
      [&](Operation *op, Region *) {
        return isMemoryEffectFree(op) && isSpeculatable(op);
      },
      [&](Operation *op, Region *) { loopLike.moveOutOfLoop(op); });
}
