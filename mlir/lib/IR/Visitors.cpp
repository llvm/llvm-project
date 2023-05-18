//===- Visitors.cpp - MLIR Visitor Utilities ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Visitors.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

WalkStage::WalkStage(Operation *op)
    : numRegions(op->getNumRegions()), nextRegion(0) {}

MutableArrayRef<Region> ForwardIterator::makeIterable(Operation &range) {
  return range.getRegions();
}

void detail::walk(Operation *op,
                  function_ref<void(Operation *, const WalkStage &)> callback) {
  WalkStage stage(op);

  for (Region &region : op->getRegions()) {
    // Invoke callback on the parent op before visiting each child region.
    callback(op, stage);
    stage.advance();

    for (Block &block : region) {
      for (Operation &nestedOp : block)
        walk(&nestedOp, callback);
    }
  }

  // Invoke callback after all regions have been visited.
  callback(op, stage);
}

WalkResult detail::walk(
    Operation *op,
    function_ref<WalkResult(Operation *, const WalkStage &)> callback) {
  WalkStage stage(op);

  for (Region &region : op->getRegions()) {
    // Invoke callback on the parent op before visiting each child region.
    WalkResult result = callback(op, stage);

    if (result.wasSkipped())
      return WalkResult::advance();
    if (result.wasInterrupted())
      return WalkResult::interrupt();

    stage.advance();

    for (Block &block : region) {
      // Early increment here in the case where the operation is erased.
      for (Operation &nestedOp : llvm::make_early_inc_range(block))
        if (walk(&nestedOp, callback).wasInterrupted())
          return WalkResult::interrupt();
    }
  }
  return callback(op, stage);
}
