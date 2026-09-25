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
#include "mlir/Dialect/Complex/IR/Complex.h"
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
  arith::FastMathFlags fmf = op.getFastmathAttr().getValue();

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
      return vector::BroadcastOp::create(rewriter, loc, vec, value);
    return value;
  };

  // Replace `pow(x, 1.0)` with `x`.
  if (isExponentValue(1.0)) {
    rewriter.replaceOp(op, x);
    return success();
  }

  // Replace `pow(x, 2.0)` with `x * x`.
  if (isExponentValue(2.0)) {
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, x, x, fmf);
    return success();
  }

  // Replace `pow(x, 3.0)` with `x * x * x`.
  if (isExponentValue(3.0)) {
    Value square = arith::MulFOp::create(rewriter, loc, x, x, fmf);
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, x, square, fmf);
    return success();
  }

  // Replace `pow(x, -1.0)` with `1.0 / x`.
  if (isExponentValue(-1.0)) {
    Value one = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(getElementTypeOrSelf(op.getType()), 1.0));
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, bcast(one), x, fmf);
    return success();
  }

  // Replace `pow(x, 0.5)` with `sqrt(x)`.
  if (isExponentValue(0.5)) {
    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, x, fmf);
    return success();
  }

  // Replace `pow(x, -0.5)` with `rsqrt(x)`.
  if (isExponentValue(-0.5)) {
    rewriter.replaceOpWithNewOp<math::RsqrtOp>(op, x, fmf);
    return success();
  }

  // Replace `pow(x, 0.75)` with `sqrt(sqrt(x)) * sqrt(x)`.
  if (isExponentValue(0.75)) {
    Value powHalf = math::SqrtOp::create(rewriter, loc, x, fmf);
    Value powQuarter = math::SqrtOp::create(rewriter, loc, powHalf, fmf);
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, powHalf, powQuarter, fmf);
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

  // Compute abs(exponent) and check the threshold before creating any IR,
  // so that returning failure() here does not violate the pattern API contract.
  bool exponentIsNegative = false;
  if (exponentValue < 0) {
    exponentIsNegative = true;
    exponentValue *= -1;
  }

  // Bail out if `abs(exponent)` exceeds the threshold (exponent==0 is free).
  if (exponentValue != 0 && exponentValue > exponentThreshold)
    return failure();

  // Maybe broadcasts scalar value into vector type compatible with `op`.
  auto bcast = [&loc, &op, &rewriter](Value value) -> Value {
    if (auto vec = dyn_cast<VectorType>(op.getType()))
      return vector::BroadcastOp::create(rewriter, loc, vec, value);
    return value;
  };

  Value one;
  Type opType = getElementTypeOrSelf(op.getType());
  if constexpr (std::is_same_v<PowIOpTy, math::FPowIOp>) {
    one = arith::ConstantOp::create(rewriter, loc,
                                    rewriter.getFloatAttr(opType, 1.0));
  } else if constexpr (std::is_same_v<PowIOpTy, complex::PowiOp>) {
    auto complexTy = cast<ComplexType>(opType);
    Type elementType = complexTy.getElementType();
    auto realPart = rewriter.getFloatAttr(elementType, 1.0);
    auto imagPart = rewriter.getFloatAttr(elementType, 0.0);
    one = complex::ConstantOp::create(
        rewriter, loc, complexTy, rewriter.getArrayAttr({realPart, imagPart}));
  } else {
    one = arith::ConstantOp::create(rewriter, loc,
                                    rewriter.getIntegerAttr(opType, 1));
  }

  // Replace `[fi]powi(x, 0)` with `1`.
  if (exponentValue == 0) {
    rewriter.replaceOp(op, bcast(one));
    return success();
  }

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
  auto buildMul = [&](Value lhs, Value rhs) {
    if constexpr (std::is_same_v<PowIOpTy, complex::PowiOp>)
      return MulOpTy::create(rewriter, loc, op.getType(), lhs, rhs,
                             op.getFastmathAttr());
    else
      return MulOpTy::create(rewriter, loc, lhs, rhs);
  };
  for (unsigned i = 1; i < exponentValue; ++i)
    result = buildMul(result, base);

  // Inverse the base for negative exponent, i.e. for
  // `[fi]powi(x, negative_exponent)` set `x` to `1 / x`.
  if (exponentIsNegative) {
    if constexpr (std::is_same_v<PowIOpTy, complex::PowiOp>)
      result = DivOpTy::create(rewriter, loc, op.getType(), bcast(one), result,
                               op.getFastmathAttr());
    else
      result = DivOpTy::create(rewriter, loc, bcast(one), result);
  }

  rewriter.replaceOp(op, result);
  return success();
}

//----------------------------------------------------------------------------//
// ExpOp/Exp2Op quotient strength reduction.
//----------------------------------------------------------------------------//

namespace {
/// Replaces `exp(a) / exp(b)` with `exp(a - b)`, and likewise for `exp2`,
/// trading a division and an exponential for a subtraction.
template <typename ExpOpTy>
struct ExpQuotientStrengthReduction : public OpRewritePattern<arith::DivFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const final {
    auto numerator = op.getLhs().getDefiningOp<ExpOpTy>();
    auto denominator = op.getRhs().getDefiningOp<ExpOpTy>();
    if (!numerator || !denominator)
      return failure();

    // The rewrite is only valid when the division may be turned into a
    // reciprocal multiplication and then reassociated with the exponentials:
    //   exp(a) / exp(b) --> exp(a) * exp(-b) --> exp(a + -b)
    // This mirrors LLVM's InstCombine, which reaches the same result with
    // `arcp` for the first step and `reassoc` for the second. Note that the
    // rewrite also changes the overflow behaviour: for a large `a == b` the
    // original expression is `inf / inf`, i.e. NaN, while the folded one is
    // `exp(0.0)`, i.e. 1.0.
    arith::FastMathFlags fmf = op.getFastmath();
    if (!bitEnumContainsAll(fmf, arith::FastMathFlags::arcp |
                                     arith::FastMathFlags::reassoc))
      return failure();

    // The rewrite introduces a new exponential, so it is only profitable if at
    // least one of the two it feeds on dies with the division; the exponential
    // count then never grows while a division is traded for a subtraction.
    // This is the fused equivalent of the `isOnlyUserOfAnyOperand()` check
    // LLVM's InstCombine applies to `exp(X) * exp(Y) --> exp(X + Y)`.
    Operation *divOp = op;
    auto diesWithDivision = [divOp](Operation *exp) {
      return llvm::all_of(exp->getUsers(),
                          [divOp](Operation *user) { return user == divOp; });
    };
    if (!diesWithDivision(numerator) && !diesWithDivision(denominator))
      return failure();

    // `nnan` and `ninf` are assumptions about the values an operation sees, and
    // neither new operation sees the values its source did, so they cannot be
    // carried over:
    //  - the subtraction consumes the exponents instead of the exponentials.
    //    `ninf` holds for `exp(-inf) / exp(0.0)`, i.e. `0.0 / 1.0`, while the
    //    subtraction `-inf - 0.0` is infinite.
    //  - the new exponential may overflow where neither of the old ones does.
    //    For f32 `exp(80.0)` and `exp(-80.0)` are both finite while
    //    `exp(80.0 - -80.0)` is not.
    // All remaining flags only license *how* a value may be computed, so they
    // stay. Derive them separately for the two new operations.
    constexpr arith::FastMathFlags valueAssumptions =
        arith::FastMathFlags::nnan | arith::FastMathFlags::ninf;

    // The subtraction takes the place of the division.
    arith::FastMathFlags subFmf = bitEnumClear(fmf, valueAssumptions);

    // The new exponential may not be given a weaker accuracy contract than the
    // ones it replaces, so it only keeps the flags common to both of them.
    arith::FastMathFlags expFmf = bitEnumClear(
        numerator.getFastmath() & denominator.getFastmath(), valueAssumptions);

    Value exponent =
        arith::SubFOp::create(rewriter, op.getLoc(), numerator.getOperand(),
                              denominator.getOperand(), subFmf);
    rewriter.replaceOpWithNewOp<ExpOpTy>(op, exponent, expFmf);
    return success();
  }
};
} // namespace

//----------------------------------------------------------------------------//

void mlir::populateMathAlgebraicSimplificationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      PowFStrengthReduction,
      PowIStrengthReduction<math::IPowIOp, arith::DivSIOp, arith::MulIOp>,
      PowIStrengthReduction<math::FPowIOp, arith::DivFOp, arith::MulFOp>,
      PowIStrengthReduction<complex::PowiOp, complex::DivOp, complex::MulOp>>(
      patterns.getContext(), /*exponentThreshold=*/8);
  patterns.add<ExpQuotientStrengthReduction<math::ExpOp>,
               ExpQuotientStrengthReduction<math::Exp2Op>>(
      patterns.getContext());
}
