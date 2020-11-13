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

/// Walk all of the regions/blocks/operations nested under and including the
/// given operation.
void detail::walk(Operation *op, function_ref<void(Region *)> callback) {
  for (auto &region : op->getRegions()) {
    callback(&region);
    for (auto &block : region) {
      for (auto &nestedOp : block)
        walk(&nestedOp, callback);
    }
  }
}

void detail::walk(Operation *op, function_ref<void(Block *)> callback) {
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      callback(&block);
      for (auto &nestedOp : block)
        walk(&nestedOp, callback);
    }
  }
}

void detail::walk(Operation *op, function_ref<void(Operation *op)> callback) {
  // TODO: This walk should be iterative over the operations.
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block))
        walk(&nestedOp, callback);
    }
  }
  callback(op);
}

/// Walk all of the regions/blocks/operations nested under and including the
/// given operation. These functions walk operations until an interrupt result
/// is returned by the callback.
WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Region *op)> callback) {
  for (auto &region : op->getRegions()) {
    if (callback(&region).wasInterrupted())
      return WalkResult::interrupt();
    for (auto &block : region) {
      for (auto &nestedOp : block)
        walk(&nestedOp, callback);
    }
  }
  return WalkResult::advance();
}

WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Block *op)> callback) {
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      if (callback(&block).wasInterrupted())
        return WalkResult::interrupt();
      for (auto &nestedOp : block)
        walk(&nestedOp, callback);
    }
  }
  return WalkResult::advance();
}

WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Operation *op)> callback) {
  // TODO: This walk should be iterative over the operations.
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block)) {
        if (walk(&nestedOp, callback).wasInterrupted())
          return WalkResult::interrupt();
      }
    }
  }
  return callback(op);
}
