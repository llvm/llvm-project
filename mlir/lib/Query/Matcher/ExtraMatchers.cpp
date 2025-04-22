//===- ExtraMatchers.cpp - Various common matchers ---------------*- C++-*-===//
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

#include "mlir/Query/Matcher/ExtraMatchers.h"

namespace mlir::query::matcher {

bool BackwardSliceMatcher::matches(Operation *rootOp,
                                   llvm::SetVector<Operation *> &backwardSlice,
                                   BackwardSliceOptions &options,
                                   int64_t maxDepth) {
  options.inclusive = inclusive;
  options.omitUsesFromAbove = omitUsesFromAbove;
  options.omitBlockArguments = omitBlockArguments;
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
        // Set the defining operation's depth to one level greater than
        // subOp's depth
        int64_t newDepth = opDepths[subOp] + 1;
        if (!opDepths.contains(definingOp)) {
          opDepths[definingOp] = newDepth;
        } else {
          opDepths[definingOp] = std::min(opDepths[definingOp], newDepth);
        }
        return !(opDepths[subOp] > maxDepth);
      } else {
        auto blockArgument = cast<BlockArgument>(operand);
        Operation *parentOp = blockArgument.getOwner()->getParentOp();
        if (!parentOp)
          continue;
        int64_t newDepth = opDepths[subOp] + 1;
        if (!opDepths.contains(parentOp)) {
          opDepths[parentOp] = newDepth;
        } else {
          opDepths[parentOp] = std::min(opDepths[parentOp], newDepth);
        }
        return !(opDepths[parentOp] > maxDepth);
      }
    }
    return true;
  };
  getBackwardSlice(rootOp, &backwardSlice, options);
  return true;
}

} // namespace mlir::query::matcher
