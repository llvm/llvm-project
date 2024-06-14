//===- LoopUtils.cpp - Helpers related to loop operations -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;

/// Calculate the normalized loop upper bounds with lower bound equal to zero
/// and step equal to one.
LoopParams mlir::emitNormalizedLoopBounds(RewriterBase &rewriter, Location loc,
                                          Value lb, Value ub, Value step) {
  // For non-index types, generate `arith` instructions
  // Check if the loop is already known to have a constant zero lower bound or
  // a constant one step.
  bool isZeroBased = false;
  if (auto lbCst = getConstantIntValue(lb))
    isZeroBased = lbCst.value() == 0;

  bool isStepOne = false;
  if (auto stepCst = getConstantIntValue(step))
    isStepOne = stepCst.value() == 1;

  // Compute the number of iterations the loop executes: ceildiv(ub - lb, step)
  // assuming the step is strictly positive.  Update the bounds and the step
  // of the loop to go from 0 to the number of iterations, if necessary.
  if (isZeroBased && isStepOne)
    return {lb, ub, step};

  Value diff =
      isZeroBased ? ub : rewriter.createOrFold<arith::SubIOp>(loc, ub, lb);
  Value newUpperBound =
      isStepOne ? diff
                : rewriter.createOrFold<arith::CeilDivSIOp>(loc, diff, step);

  Value newLowerBound = isZeroBased
                            ? lb
                            : rewriter.create<arith::ConstantOp>(
                                  loc, rewriter.getZeroAttr(lb.getType()));
  Value newStep = isStepOne
                      ? step
                      : rewriter.create<arith::ConstantOp>(
                            loc, rewriter.getIntegerAttr(step.getType(), 1));

  return {newLowerBound, newUpperBound, newStep};
}
