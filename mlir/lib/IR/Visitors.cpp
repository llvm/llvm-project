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
/// given operation. The walk order is specified by 'Order'.

void detail::walk(Operation *op, function_ref<void(Region *)> callback,
                  WalkOrder order) {
  for (auto &region : op->getRegions()) {
    if (order == WalkOrder::PreOrder)
      callback(&region);
    for (auto &block : region) {
      for (auto &nestedOp : block)
        walk(&nestedOp, callback, order);
    }
    if (order == WalkOrder::PostOrder)
      callback(&region);
  }
}

void detail::walk(Operation *op, function_ref<void(Block *)> callback,
                  WalkOrder order) {
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      if (order == WalkOrder::PreOrder)
        callback(&block);
      for (auto &nestedOp : block)
        walk(&nestedOp, callback, order);
      if (order == WalkOrder::PostOrder)
        callback(&block);
    }
  }
}

void detail::walk(Operation *op, function_ref<void(Operation *)> callback,
                  WalkOrder order) {
  if (order == WalkOrder::PreOrder)
    callback(op);

  // TODO: This walk should be iterative over the operations.
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block))
        walk(&nestedOp, callback, order);
    }
  }

  if (order == WalkOrder::PostOrder)
    callback(op);
}

/// Walk all of the regions/blocks/operations nested under and including the
/// given operation. The walk order is specified by 'order'. These functions
/// walk operations until an interrupt result is returned by the callback.
WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Region *)> callback,
                        WalkOrder order) {
  for (auto &region : op->getRegions()) {
    if (order == WalkOrder::PreOrder)
      if (callback(&region).wasInterrupted())
        return WalkResult::interrupt();
    for (auto &block : region) {
      for (auto &nestedOp : block)
        walk(&nestedOp, callback, order);
    }
    if (order == WalkOrder::PostOrder)
      if (callback(&region).wasInterrupted())
        return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Block *)> callback,
                        WalkOrder order) {
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      if (order == WalkOrder::PreOrder)
        if (callback(&block).wasInterrupted())
          return WalkResult::interrupt();
      for (auto &nestedOp : block)
        walk(&nestedOp, callback, order);
      if (order == WalkOrder::PostOrder)
        if (callback(&block).wasInterrupted())
          return WalkResult::interrupt();
    }
  }
  return WalkResult::advance();
}

WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Operation *)> callback,
                        WalkOrder order) {
  if (order == WalkOrder::PreOrder)
    if (callback(op).wasInterrupted())
      return WalkResult::interrupt();

  // TODO: This walk should be iterative over the operations.
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block)) {
        if (walk(&nestedOp, callback, order).wasInterrupted())
          return WalkResult::interrupt();
      }
    }
  }

  if (order == WalkOrder::PostOrder)
    return callback(op);
  return WalkResult::advance();
}
