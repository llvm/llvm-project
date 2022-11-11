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

/// Walk all of the regions/blocks/operations nested under and including the
/// given operation. Regions, blocks and operations at the same nesting level
/// are visited in lexicographical order. The walk order for enclosing regions,
/// blocks and operations with respect to their nested ones is specified by
/// 'order'. These methods are invoked for void-returning callbacks. A callback
/// on a block or operation is allowed to erase that block or operation only if
/// the walk is in post-order. See non-void method for pre-order erasure.
void detail::walk(Operation *op, function_ref<void(Region *)> callback,
                  WalkOrder order) {
  // We don't use early increment for regions because they can't be erased from
  // a callback.
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
    // Early increment here in the case where the block is erased.
    for (auto &block : llvm::make_early_inc_range(region)) {
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

/// Walk all of the regions/blocks/operations nested under and including the
/// given operation. These functions walk operations until an interrupt result
/// is returned by the callback. Walks on regions, blocks and operations may
/// also be skipped if the callback returns a skip result. Regions, blocks and
/// operations at the same nesting level are visited in lexicographical order.
/// The walk order for enclosing regions, blocks and operations with respect to
/// their nested ones is specified by 'order'. A callback on a block or
/// operation is allowed to erase that block or operation if either:
///   * the walk is in post-order, or
///   * the walk is in pre-order and the walk is skipped after the erasure.
WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Region *)> callback,
                        WalkOrder order) {
  // We don't use early increment for regions because they can't be erased from
  // a callback.
  for (auto &region : op->getRegions()) {
    if (order == WalkOrder::PreOrder) {
      WalkResult result = callback(&region);
      if (result.wasSkipped())
        continue;
      if (result.wasInterrupted())
        return WalkResult::interrupt();
    }
    for (auto &block : region) {
      for (auto &nestedOp : block)
        if (walk(&nestedOp, callback, order).wasInterrupted())
          return WalkResult::interrupt();
    }
    if (order == WalkOrder::PostOrder) {
      if (callback(&region).wasInterrupted())
        return WalkResult::interrupt();
      // We don't check if this region was skipped because its walk already
      // finished and the walk will continue with the next region.
    }
  }
  return WalkResult::advance();
}

WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Block *)> callback,
                        WalkOrder order) {
  for (auto &region : op->getRegions()) {
    // Early increment here in the case where the block is erased.
    for (auto &block : llvm::make_early_inc_range(region)) {
      if (order == WalkOrder::PreOrder) {
        WalkResult result = callback(&block);
        if (result.wasSkipped())
          continue;
        if (result.wasInterrupted())
          return WalkResult::interrupt();
      }
      for (auto &nestedOp : block)
        if (walk(&nestedOp, callback, order).wasInterrupted())
          return WalkResult::interrupt();
      if (order == WalkOrder::PostOrder) {
        if (callback(&block).wasInterrupted())
          return WalkResult::interrupt();
        // We don't check if this block was skipped because its walk already
        // finished and the walk will continue with the next block.
      }
    }
  }
  return WalkResult::advance();
}

WalkResult detail::walk(Operation *op,
                        function_ref<WalkResult(Operation *)> callback,
                        WalkOrder order) {
  if (order == WalkOrder::PreOrder) {
    WalkResult result = callback(op);
    // If skipped, caller will continue the walk on the next operation.
    if (result.wasSkipped())
      return WalkResult::advance();
    if (result.wasInterrupted())
      return WalkResult::interrupt();
  }

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
