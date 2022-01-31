//===- AlgebraicSimplification.cpp - Simplify algebraic expressions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewrites based on the basic rules of algebra
// (Commutativity, associativity, etc...) and strength reductions for math
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include <climits>

using namespace mlir;

//----------------------------------------------------------------------------//
// PowFOp strength reduction.
//----------------------------------------------------------------------------//

namespace {
struct PowFStrengthReduction : public OpRewritePattern<math::PowFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::PowFOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
PowFStrengthReduction::matchAndRewrite(math::PowFOp op,
                                       PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value x = op.getLhs();

  FloatAttr scalarExponent;
  DenseFPElementsAttr vectorExponent;

  bool isScalar = matchPattern(op.getRhs(), m_Constant(&scalarExponent));
  bool isVector = matchPattern(op.getRhs(), m_Constant(&vectorExponent));

  // Returns true if exponent is a constant equal to `value`.
  auto isExponentValue = [&](double value) -> bool {
    if (isScalar)
      return scalarExponent.getValue().isExactlyValue(value);

    if (isVector && vectorExponent.isSplat())
      return vectorExponent.getSplatValue<FloatAttr>()
          .getValue()
          .isExactlyValue(value);

    return false;
  };

  // Maybe broadcasts scalar value into vector type compatible with `op`.
  auto bcast = [&](Value value) -> Value {
    if (auto vec = op.getType().dyn_cast<VectorType>())
      return rewriter.create<vector::BroadcastOp>(op.getLoc(), vec, value);
    return value;
  };

  // Replace `pow(x, 1.0)` with `x`.
  if (isExponentValue(1.0)) {
    rewriter.replaceOp(op, x);
    return success();
  }

  // Replace `pow(x, 2.0)` with `x * x`.
  if (isExponentValue(2.0)) {
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, ValueRange({x, x}));
    return success();
  }

  // Replace `pow(x, 3.0)` with `x * x * x`.
  if (isExponentValue(3.0)) {
    Value square =
        rewriter.create<arith::MulFOp>(op.getLoc(), ValueRange({x, x}));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, ValueRange({x, square}));
    return success();
  }

  // Replace `pow(x, -1.0)` with `1.0 / x`.
  if (isExponentValue(-1.0)) {
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(getElementTypeOrSelf(op.getType()), 1.0));
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, ValueRange({bcast(one), x}));
    return success();
  }

  // Replace `pow(x, 0.5)` with `sqrt(x)`.
  if (isExponentValue(0.5)) {
    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, x);
    return success();
  }

  // Replace `pow(x, -0.5)` with `rsqrt(x)`.
  if (isExponentValue(-0.5)) {
    rewriter.replaceOpWithNewOp<math::RsqrtOp>(op, x);
    return success();
  }

  return failure();
}

//----------------------------------------------------------------------------//

void mlir::populateMathAlgebraicSimplificationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<PowFStrengthReduction>(patterns.getContext());
}
