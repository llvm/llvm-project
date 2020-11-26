//===- LoopInvariantCodeMotion.cpp - Code to perform loop fusion-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop invariant code motion.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "licm"

using namespace mlir;

namespace {
/// Loop invariant code motion (LICM) pass.
struct LoopInvariantCodeMotion
    : public LoopInvariantCodeMotionBase<LoopInvariantCodeMotion> {
  void runOnOperation() override;
};
} // end anonymous namespace

// Checks whether the given op can be hoisted by checking that
// - the op and any of its contained operations do not depend on SSA values
//   defined inside of the loop (by means of calling definedOutside).
// - the op has no side-effects. If sideEffecting is Never, sideeffects of this
//   op and its nested ops are ignored.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  // Check that dependencies are defined outside of loop.
  if (!llvm::all_of(op->getOperands(), definedOutside))
    return false;
  // Check whether this op is side-effect free. If we already know that there
  // can be no side-effects because the surrounding op has claimed so, we can
  // (and have to) skip this step.
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!memInterface.hasNoEffect())
      return false;
    // If the operation doesn't have side effects and it doesn't recursively
    // have side effects, it can always be hoisted.
    if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>())
      return true;

    // Otherwise, if the operation doesn't provide the memory effect interface
    // and it doesn't have recursive side effects we treat it conservatively as
    // side-effecting.
  } else if (!op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
    return false;
  }

  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted.
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto &innerOp : block.without_terminator())
        if (!canBeHoisted(&innerOp, definedOutside))
          return false;
    }
  }
  return true;
}


LogicalResult mlir::moveLoopInvariantCode(LoopLikeOpInterface looplike) {
  auto &loopBody = looplike.getLoopBody();

  // We use two collections here as we need to preserve the order for insertion
  // and this is easiest.
  SmallPtrSet<Operation *, 8> willBeMovedSet;
  SmallVector<Operation *, 8> opsToMove;

  // Helper to check whether an operation is loop invariant wrt. SSA properties.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto definingOp = value.getDefiningOp();
    return (definingOp && !!willBeMovedSet.count(definingOp)) ||
           looplike.isDefinedOutsideOfLoop(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  for (auto &block : loopBody) {
    for (auto &op : block.without_terminator()) {
      if (canBeHoisted(&op, isDefinedOutsideOfBody)) {
        opsToMove.push_back(&op);
        willBeMovedSet.insert(&op);
      }
    }
  }

  // For all instructions that we found to be invariant, move outside of the
  // loop.
  auto result = looplike.moveOutOfLoop(opsToMove);
  LLVM_DEBUG(looplike.print(llvm::dbgs() << "\n\nModified loop:\n"));
  return result;
}

void LoopInvariantCodeMotion::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation()->walk([&](LoopLikeOpInterface loopLike) {
    LLVM_DEBUG(loopLike.print(llvm::dbgs() << "\nOriginal loop:\n"));
    if (failed(moveLoopInvariantCode(loopLike)))
      signalPassFailure();
  });
}

std::unique_ptr<Pass> mlir::createLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotion>();
}
