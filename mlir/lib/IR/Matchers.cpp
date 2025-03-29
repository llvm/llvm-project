//===- Matchers.cpp - Various common matchers -------------------*- C++ -*-===//
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
  // The starting point is the root op, therfore we set its depth to 0
  opDepths[rootOp] = 0;
  options.filter = [&](Operation *subOp) {
    // If the subOp’s depth exceeds maxDepth, we can stop further computing the
    // slice for the current branch
    if (opDepths[subOp] > maxDepth)
      return false;
    // Examining subOp's operands to compute the depths of their defining
    // operations
    for (auto operand : subOp->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        // If the defining operation is already in the map, its depth has been
        // computed; recomputation can be skipped
        if (!opDepths.contains(definingOp)) {
          // Set the defining operation's depth to one level greater than
          // subOp's depth
          opDepths[definingOp] = opDepths[subOp] + 1;
          if (opDepths[subOp] > maxDepth)
            return false;
        }
      } else {
        auto blockArgument = cast<BlockArgument>(operand);
        Operation *parentOp = blockArgument.getOwner()->getParentOp();
        if (!opDepths.contains(parentOp)) {
          opDepths[parentOp] = opDepths[subOp] + 1;
          if (opDepths[subOp] > maxDepth)
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
