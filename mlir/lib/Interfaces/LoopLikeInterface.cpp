//===- LoopLikeInterface.cpp - Loop-like operations in MLIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;

#define DEBUG_TYPE "loop-like"

//===----------------------------------------------------------------------===//
// LoopLike Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the loop-like interfaces.
#include "mlir/Interfaces/LoopLikeInterface.cpp.inc"

//===----------------------------------------------------------------------===//
// LoopLike Utilities
//===----------------------------------------------------------------------===//

/// Returns true if the given operation is side-effect free as are all of its
/// nested operations.
///
/// TODO: There is a duplicate function in ControlFlowSink. Move
/// `moveLoopInvariantCode` to TransformUtils and then factor out this function.
static bool isSideEffectFree(Operation *op) {
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // If the op has side-effects, it cannot be moved.
    if (!memInterface.hasNoEffect())
      return false;
    // If the op does not have recursive side effects, then it can be moved.
    if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>())
      return true;
  } else if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
    // Otherwise, if the op does not implement the memory effect interface and
    // it does not have recursive side effects, then it cannot be known that the
    // op is moveable.
    return false;
  }

  // Recurse into the regions and ensure that all nested ops can also be moved.
  for (Region &region : op->getRegions())
    for (Operation &op : region.getOps())
      if (!isSideEffectFree(&op))
        return false;
  return true;
}

/// Checks whether the given op can be hoisted by checking that
/// - the op and none of its contained operations depend on values inside of the
///   loop (by means of calling definedOutside).
/// - the op has no side-effects.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  if (!isSideEffectFree(op))
    return false;

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

void mlir::moveLoopInvariantCode(LoopLikeOpInterface looplike) {
  Region *loopBody = &looplike.getLoopBody();

  std::queue<Operation *> worklist;
  // Add top-level operations in the loop body to the worklist.
  for (Operation &op : loopBody->getOps())
    worklist.push(&op);

  auto definedOutside = [&](Value value) {
    return looplike.isDefinedOutsideOfLoop(value);
  };

  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop();
    // Skip ops that have already been moved. Check if the op can be hoisted.
    if (op->getParentRegion() != loopBody || !canBeHoisted(op, definedOutside))
      continue;

    looplike.moveOutOfLoop(op);

    // Since the op has been moved, we need to check its users within the
    // top-level of the loop body.
    for (Operation *user : op->getUsers())
      if (user->getParentRegion() == loopBody)
        worklist.push(user);
  }
}
