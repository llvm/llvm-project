//===- Matchers.h - Various common matchers ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements specific matchers
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::detail {

bool BackwardSliceMatcher::matches(Operation *rootOp,
                                   llvm::SetVector<Operation *> &backwardSlice,
                                   query::QueryOptions &options,
                                   int64_t maxDepth) {
  backwardSlice.clear();
  llvm::DenseMap<Operation *, int64_t> opDepths;
  // Initialize the map with the root operation
  // and set its depth to 0
  opDepths[rootOp] = 0;
  options.filter = [&](Operation *op) {
    if (opDepths[op] > maxDepth)
      return false;
    // Begins by checking the previous operation's arguments
    // and computing their depth
    for (auto operand : op->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        // If the operation is in the map, it means
        // we have already computed its depth
        if (!opDepths.contains(definingOp)) {
          // The operation's depth is 1 level above its root op
          opDepths[definingOp] = opDepths[op] + 1;
          if (opDepths[op] > maxDepth)
            return false;
        }
      } else {
        auto blockArgument = cast<BlockArgument>(operand);
        Operation *parentOp = blockArgument.getOwner()->getParentOp();
        if (!opDepths.contains(parentOp)) {
          opDepths[parentOp] = opDepths[op] + 1;
          if (opDepths[op] > maxDepth)
            return false;
        }
      }
    }
    return true;
  };
  getBackwardSlice(rootOp, &backwardSlice, options);
  return true;
}
} // namespace mlir::detail
