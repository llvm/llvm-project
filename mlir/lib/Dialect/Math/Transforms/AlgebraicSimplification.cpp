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

#include "mlir/Dialect/Arith/IR/Arith.h"
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
    if (auto vec = dyn_cast<VectorType>(op.getType()))
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

  // Replace `pow(x, 0.75)` with `sqrt(sqrt(x)) * sqrt(x)`.
  if (isExponentValue(0.75)) {
    Value powHalf = rewriter.create<math::SqrtOp>(op.getLoc(), x);
    Value powQuarter = rewriter.create<math::SqrtOp>(op.getLoc(), powHalf);
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op,
                                               ValueRange{powHalf, powQuarter});
    return success();
  }

  return failure();
}

//----------------------------------------------------------------------------//
// FPowIOp/IPowIOp strength reduction.
//----------------------------------------------------------------------------//

namespace {
template <typename PowIOpTy, typename DivOpTy, typename MulOpTy>
struct PowIStrengthReduction : public OpRewritePattern<PowIOpTy> {

  unsigned exponentThreshold;

public:
  PowIStrengthReduction(MLIRContext *context, unsigned exponentThreshold = 3,
                        PatternBenefit benefit = 1,
                        ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<PowIOpTy>(context, benefit, generatedNames),
        exponentThreshold(exponentThreshold) {}

  LogicalResult matchAndRewrite(PowIOpTy op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

template <typename PowIOpTy, typename DivOpTy, typename MulOpTy>
LogicalResult
PowIStrengthReduction<PowIOpTy, DivOpTy, MulOpTy>::matchAndRewrite(
    PowIOpTy op, PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value base = op.getLhs();

  IntegerAttr scalarExponent;
  DenseIntElementsAttr vectorExponent;

  bool isScalar = matchPattern(op.getRhs(), m_Constant(&scalarExponent));
  bool isVector = matchPattern(op.getRhs(), m_Constant(&vectorExponent));

  // Simplify cases with known exponent value.
  int64_t exponentValue = 0;
  if (isScalar)
    exponentValue = scalarExponent.getInt();
  else if (isVector && vectorExponent.isSplat())
    exponentValue = vectorExponent.getSplatValue<IntegerAttr>().getInt();
  else
    return failure();

  // Maybe broadcasts scalar value into vector type compatible with `op`.
  auto bcast = [&loc, &op, &rewriter](Value value) -> Value {
    if (auto vec = dyn_cast<VectorType>(op.getType()))
      return rewriter.create<vector::BroadcastOp>(loc, vec, value);
    return value;
  };

  Value one;
  Type opType = getElementTypeOrSelf(op.getType());
  if constexpr (std::is_same_v<PowIOpTy, math::FPowIOp>)
    one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(opType, 1.0));
  else
    one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(opType, 1));

  // Replace `[fi]powi(x, 0)` with `1`.
  if (exponentValue == 0) {
    rewriter.replaceOp(op, bcast(one));
    return success();
  }

  bool exponentIsNegative = false;
  if (exponentValue < 0) {
    exponentIsNegative = true;
    exponentValue *= -1;
  }

  // Bail out if `abs(exponent)` exceeds the threshold.
  if (exponentValue > exponentThreshold)
    return failure();

  // Inverse the base for negative exponent, i.e. for
  // `[fi]powi(x, negative_exponent)` set `x` to `1 / x`.
  if (exponentIsNegative)
    base = rewriter.create<DivOpTy>(loc, bcast(one), base);

  Value result = base;
  // Transform to naive sequence of multiplications:
  //   * For positive exponent case replace:
  //       `[fi]powi(x, positive_exponent)`
  //     with:
  //       x * x * x * ...
  //   * For negative exponent case replace:
  //       `[fi]powi(x, negative_exponent)`
  //     with:
  //       (1 / x) * (1 / x) * (1 / x) * ...
  for (unsigned i = 1; i < exponentValue; ++i)
    result = rewriter.create<MulOpTy>(loc, result, base);

  rewriter.replaceOp(op, result);
  return success();
}

//----------------------------------------------------------------------------//

void mlir::populateMathAlgebraicSimplificationPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<PowFStrengthReduction,
           PowIStrengthReduction<math::IPowIOp, arith::DivSIOp, arith::MulIOp>,
           PowIStrengthReduction<math::FPowIOp, arith::DivFOp, arith::MulFOp>>(
          patterns.getContext());
}
