//===- ComplexToStandard.cpp - conversion from Complex to Standard dialect ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOSTANDARD
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

enum class AbsFn { abs, sqrt, rsqrt };

// Returns the absolute value, its square root or its reciprocal square root.
Value computeAbs(Value real, Value imag, arith::FastMathFlags fmf,
                 ImplicitLocOpBuilder &b, AbsFn fn = AbsFn::abs) {
  Value one = b.create<arith::ConstantOp>(real.getType(),
                                          b.getFloatAttr(real.getType(), 1.0));

  Value absReal = b.create<math::AbsFOp>(real, fmf);
  Value absImag = b.create<math::AbsFOp>(imag, fmf);

  Value max = b.create<arith::MaximumFOp>(absReal, absImag, fmf);
  Value min = b.create<arith::MinimumFOp>(absReal, absImag, fmf);
  Value ratio = b.create<arith::DivFOp>(min, max, fmf);
  Value ratioSq = b.create<arith::MulFOp>(ratio, ratio, fmf);
  Value ratioSqPlusOne = b.create<arith::AddFOp>(ratioSq, one, fmf);
  Value result;

  if (fn == AbsFn::rsqrt) {
    ratioSqPlusOne = b.create<math::RsqrtOp>(ratioSqPlusOne, fmf);
    min = b.create<math::RsqrtOp>(min, fmf);
    max = b.create<math::RsqrtOp>(max, fmf);
  }

  if (fn == AbsFn::sqrt) {
    Value quarter = b.create<arith::ConstantOp>(
        real.getType(), b.getFloatAttr(real.getType(), 0.25));
    // sqrt(sqrt(a*b)) would avoid the pow, but will overflow more easily.
    Value sqrt = b.create<math::SqrtOp>(max, fmf);
    Value p025 = b.create<math::PowFOp>(ratioSqPlusOne, quarter, fmf);
    result = b.create<arith::MulFOp>(sqrt, p025, fmf);
  } else {
    Value sqrt = b.create<math::SqrtOp>(ratioSqPlusOne, fmf);
    result = b.create<arith::MulFOp>(max, sqrt, fmf);
  }

  Value isNaN =
      b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, result, result, fmf);
  return b.create<arith::SelectOp>(isNaN, min, result);
}

struct AbsOpConversion : public OpConversionPattern<complex::AbsOp> {
  using OpConversionPattern<complex::AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    arith::FastMathFlags fmf = op.getFastMathFlagsAttr().getValue();

    Value real = b.create<complex::ReOp>(adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(adaptor.getComplex());
    rewriter.replaceOp(op, computeAbs(real, imag, fmf, b));

    return success();
  }
};

// atan2(y,x) = -i * log((x + i * y)/sqrt(x**2+y**2))
struct Atan2OpConversion : public OpConversionPattern<complex::Atan2Op> {
  using OpConversionPattern<complex::Atan2Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::Atan2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto type = cast<ComplexType>(op.getType());
    Type elementType = type.getElementType();
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value rhsSquared = b.create<complex::MulOp>(type, rhs, rhs, fmf);
    Value lhsSquared = b.create<complex::MulOp>(type, lhs, lhs, fmf);
    Value rhsSquaredPlusLhsSquared =
        b.create<complex::AddOp>(type, rhsSquared, lhsSquared, fmf);
    Value sqrtOfRhsSquaredPlusLhsSquared =
        b.create<complex::SqrtOp>(type, rhsSquaredPlusLhsSquared, fmf);

    Value zero =
        b.create<arith::ConstantOp>(elementType, b.getZeroAttr(elementType));
    Value one = b.create<arith::ConstantOp>(elementType,
                                            b.getFloatAttr(elementType, 1));
    Value i = b.create<complex::CreateOp>(type, zero, one);
    Value iTimesLhs = b.create<complex::MulOp>(i, lhs, fmf);
    Value rhsPlusILhs = b.create<complex::AddOp>(rhs, iTimesLhs, fmf);

    Value divResult = b.create<complex::DivOp>(
        rhsPlusILhs, sqrtOfRhsSquaredPlusLhsSquared, fmf);
    Value logResult = b.create<complex::LogOp>(divResult, fmf);

    Value negativeOne = b.create<arith::ConstantOp>(
        elementType, b.getFloatAttr(elementType, -1));
    Value negativeI = b.create<complex::CreateOp>(type, zero, negativeOne);

    rewriter.replaceOpWithNewOp<complex::MulOp>(op, negativeI, logResult, fmf);
    return success();
  }
};

template <typename ComparisonOp, arith::CmpFPredicate p>
struct ComparisonOpConversion : public OpConversionPattern<ComparisonOp> {
  using OpConversionPattern<ComparisonOp>::OpConversionPattern;
  using ResultCombiner =
      std::conditional_t<std::is_same<ComparisonOp, complex::EqualOp>::value,
                         arith::AndIOp, arith::OrIOp>;

  LogicalResult
  matchAndRewrite(ComparisonOp op, typename ComparisonOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getLhs().getType()).getElementType();

    Value realLhs = rewriter.create<complex::ReOp>(loc, type, adaptor.getLhs());
    Value imagLhs = rewriter.create<complex::ImOp>(loc, type, adaptor.getLhs());
    Value realRhs = rewriter.create<complex::ReOp>(loc, type, adaptor.getRhs());
    Value imagRhs = rewriter.create<complex::ImOp>(loc, type, adaptor.getRhs());
    Value realComparison =
        rewriter.create<arith::CmpFOp>(loc, p, realLhs, realRhs);
    Value imagComparison =
        rewriter.create<arith::CmpFOp>(loc, p, imagLhs, imagRhs);

    rewriter.replaceOpWithNewOp<ResultCombiner>(op, realComparison,
                                                imagComparison);
    return success();
  }
};

// Default conversion which applies the BinaryStandardOp separately on the real
// and imaginary parts. Can for example be used for complex::AddOp and
// complex::SubOp.
template <typename BinaryComplexOp, typename BinaryStandardOp>
struct BinaryComplexOpConversion : public OpConversionPattern<BinaryComplexOp> {
  using OpConversionPattern<BinaryComplexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BinaryComplexOp op, typename BinaryComplexOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<ComplexType>(adaptor.getLhs().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    Value realLhs = b.create<complex::ReOp>(elementType, adaptor.getLhs());
    Value realRhs = b.create<complex::ReOp>(elementType, adaptor.getRhs());
    Value resultReal = b.create<BinaryStandardOp>(elementType, realLhs, realRhs,
                                                  fmf.getValue());
    Value imagLhs = b.create<complex::ImOp>(elementType, adaptor.getLhs());
    Value imagRhs = b.create<complex::ImOp>(elementType, adaptor.getRhs());
    Value resultImag = b.create<BinaryStandardOp>(elementType, imagLhs, imagRhs,
                                                  fmf.getValue());
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

template <typename TrigonometricOp>
struct TrigonometricOpConversion : public OpConversionPattern<TrigonometricOp> {
  using OpAdaptor = typename OpConversionPattern<TrigonometricOp>::OpAdaptor;

  using OpConversionPattern<TrigonometricOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TrigonometricOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    Value real =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getComplex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getComplex());

    // Trigonometric ops use a set of common building blocks to convert to real
    // ops. Here we create these building blocks and call into an op-specific
    // implementation in the subclass to combine them.
    Value half = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0.5));
    Value exp = rewriter.create<math::ExpOp>(loc, imag, fmf);
    Value scaledExp = rewriter.create<arith::MulFOp>(loc, half, exp, fmf);
    Value reciprocalExp = rewriter.create<arith::DivFOp>(loc, half, exp, fmf);
    Value sin = rewriter.create<math::SinOp>(loc, real, fmf);
    Value cos = rewriter.create<math::CosOp>(loc, real, fmf);

    auto resultPair =
        combine(loc, scaledExp, reciprocalExp, sin, cos, rewriter, fmf);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultPair.first,
                                                   resultPair.second);
    return success();
  }

  virtual std::pair<Value, Value>
  combine(Location loc, Value scaledExp, Value reciprocalExp, Value sin,
          Value cos, ConversionPatternRewriter &rewriter,
          arith::FastMathFlagsAttr fmf) const = 0;
};

struct CosOpConversion : public TrigonometricOpConversion<complex::CosOp> {
  using TrigonometricOpConversion<complex::CosOp>::TrigonometricOpConversion;

  std::pair<Value, Value> combine(Location loc, Value scaledExp,
                                  Value reciprocalExp, Value sin, Value cos,
                                  ConversionPatternRewriter &rewriter,
                                  arith::FastMathFlagsAttr fmf) const override {
    // Complex cosine is defined as;
    //   cos(x + iy) = 0.5 * (exp(i(x + iy)) + exp(-i(x + iy)))
    // Plugging in:
    //   exp(i(x+iy)) = exp(-y + ix) = exp(-y)(cos(x) + i sin(x))
    //   exp(-i(x+iy)) = exp(y + i(-x)) = exp(y)(cos(x) + i (-sin(x)))
    // and defining t := exp(y)
    // We get:
    //   Re(cos(x + iy)) = (0.5/t + 0.5*t) * cos x
    //   Im(cos(x + iy)) = (0.5/t - 0.5*t) * sin x
    Value sum =
        rewriter.create<arith::AddFOp>(loc, reciprocalExp, scaledExp, fmf);
    Value resultReal = rewriter.create<arith::MulFOp>(loc, sum, cos, fmf);
    Value diff =
        rewriter.create<arith::SubFOp>(loc, reciprocalExp, scaledExp, fmf);
    Value resultImag = rewriter.create<arith::MulFOp>(loc, diff, sin, fmf);
    return {resultReal, resultImag};
  }
};

struct DivOpConversion : public OpConversionPattern<complex::DivOp> {
  using OpConversionPattern<complex::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getLhs().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    Value lhsReal =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getLhs());
    Value lhsImag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getLhs());
    Value rhsReal =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getRhs());
    Value rhsImag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getRhs());

    // Smith's algorithm to divide complex numbers. It is just a bit smarter
    // way to compute the following formula:
    //  (lhsReal + lhsImag * i) / (rhsReal + rhsImag * i)
    //    = (lhsReal + lhsImag * i) (rhsReal - rhsImag * i) /
    //          ((rhsReal + rhsImag * i)(rhsReal - rhsImag * i))
    //    = ((lhsReal * rhsReal + lhsImag * rhsImag) +
    //          (lhsImag * rhsReal - lhsReal * rhsImag) * i) / ||rhs||^2
    //
    // Depending on whether |rhsReal| < |rhsImag| we compute either
    //   rhsRealImagRatio = rhsReal / rhsImag
    //   rhsRealImagDenom = rhsImag + rhsReal * rhsRealImagRatio
    //   resultReal = (lhsReal * rhsRealImagRatio + lhsImag) / rhsRealImagDenom
    //   resultImag = (lhsImag * rhsRealImagRatio - lhsReal) / rhsRealImagDenom
    //
    // or
    //
    //   rhsImagRealRatio = rhsImag / rhsReal
    //   rhsImagRealDenom = rhsReal + rhsImag * rhsImagRealRatio
    //   resultReal = (lhsReal + lhsImag * rhsImagRealRatio) / rhsImagRealDenom
    //   resultImag = (lhsImag - lhsReal * rhsImagRealRatio) / rhsImagRealDenom
    //
    // See https://dl.acm.org/citation.cfm?id=368661 for more details.
    Value rhsRealImagRatio =
        rewriter.create<arith::DivFOp>(loc, rhsReal, rhsImag, fmf);
    Value rhsRealImagDenom = rewriter.create<arith::AddFOp>(
        loc, rhsImag,
        rewriter.create<arith::MulFOp>(loc, rhsRealImagRatio, rhsReal, fmf),
        fmf);
    Value realNumerator1 = rewriter.create<arith::AddFOp>(
        loc,
        rewriter.create<arith::MulFOp>(loc, lhsReal, rhsRealImagRatio, fmf),
        lhsImag, fmf);
    Value resultReal1 = rewriter.create<arith::DivFOp>(loc, realNumerator1,
                                                       rhsRealImagDenom, fmf);
    Value imagNumerator1 = rewriter.create<arith::SubFOp>(
        loc,
        rewriter.create<arith::MulFOp>(loc, lhsImag, rhsRealImagRatio, fmf),
        lhsReal, fmf);
    Value resultImag1 = rewriter.create<arith::DivFOp>(loc, imagNumerator1,
                                                       rhsRealImagDenom, fmf);

    Value rhsImagRealRatio =
        rewriter.create<arith::DivFOp>(loc, rhsImag, rhsReal, fmf);
    Value rhsImagRealDenom = rewriter.create<arith::AddFOp>(
        loc, rhsReal,
        rewriter.create<arith::MulFOp>(loc, rhsImagRealRatio, rhsImag, fmf),
        fmf);
    Value realNumerator2 = rewriter.create<arith::AddFOp>(
        loc, lhsReal,
        rewriter.create<arith::MulFOp>(loc, lhsImag, rhsImagRealRatio, fmf),
        fmf);
    Value resultReal2 = rewriter.create<arith::DivFOp>(loc, realNumerator2,
                                                       rhsImagRealDenom, fmf);
    Value imagNumerator2 = rewriter.create<arith::SubFOp>(
        loc, lhsImag,
        rewriter.create<arith::MulFOp>(loc, lhsReal, rhsImagRealRatio, fmf),
        fmf);
    Value resultImag2 = rewriter.create<arith::DivFOp>(loc, imagNumerator2,
                                                       rhsImagRealDenom, fmf);

    // Consider corner cases.
    // Case 1. Zero denominator, numerator contains at most one NaN value.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getZeroAttr(elementType));
    Value rhsRealAbs = rewriter.create<math::AbsFOp>(loc, rhsReal, fmf);
    Value rhsRealIsZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, rhsRealAbs, zero);
    Value rhsImagAbs = rewriter.create<math::AbsFOp>(loc, rhsImag, fmf);
    Value rhsImagIsZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, rhsImagAbs, zero);
    Value lhsRealIsNotNaN = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ORD, lhsReal, zero);
    Value lhsImagIsNotNaN = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ORD, lhsImag, zero);
    Value lhsContainsNotNaNValue =
        rewriter.create<arith::OrIOp>(loc, lhsRealIsNotNaN, lhsImagIsNotNaN);
    Value resultIsInfinity = rewriter.create<arith::AndIOp>(
        loc, lhsContainsNotNaNValue,
        rewriter.create<arith::AndIOp>(loc, rhsRealIsZero, rhsImagIsZero));
    Value inf = rewriter.create<arith::ConstantOp>(
        loc, elementType,
        rewriter.getFloatAttr(
            elementType, APFloat::getInf(elementType.getFloatSemantics())));
    Value infWithSignOfRhsReal =
        rewriter.create<math::CopySignOp>(loc, inf, rhsReal);
    Value infinityResultReal =
        rewriter.create<arith::MulFOp>(loc, infWithSignOfRhsReal, lhsReal, fmf);
    Value infinityResultImag =
        rewriter.create<arith::MulFOp>(loc, infWithSignOfRhsReal, lhsImag, fmf);

    // Case 2. Infinite numerator, finite denominator.
    Value rhsRealFinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, rhsRealAbs, inf);
    Value rhsImagFinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, rhsImagAbs, inf);
    Value rhsFinite =
        rewriter.create<arith::AndIOp>(loc, rhsRealFinite, rhsImagFinite);
    Value lhsRealAbs = rewriter.create<math::AbsFOp>(loc, lhsReal, fmf);
    Value lhsRealInfinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, lhsRealAbs, inf);
    Value lhsImagAbs = rewriter.create<math::AbsFOp>(loc, lhsImag, fmf);
    Value lhsImagInfinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, lhsImagAbs, inf);
    Value lhsInfinite =
        rewriter.create<arith::OrIOp>(loc, lhsRealInfinite, lhsImagInfinite);
    Value infNumFiniteDenom =
        rewriter.create<arith::AndIOp>(loc, lhsInfinite, rhsFinite);
    Value one = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1));
    Value lhsRealIsInfWithSign = rewriter.create<math::CopySignOp>(
        loc, rewriter.create<arith::SelectOp>(loc, lhsRealInfinite, one, zero),
        lhsReal);
    Value lhsImagIsInfWithSign = rewriter.create<math::CopySignOp>(
        loc, rewriter.create<arith::SelectOp>(loc, lhsImagInfinite, one, zero),
        lhsImag);
    Value lhsRealIsInfWithSignTimesRhsReal =
        rewriter.create<arith::MulFOp>(loc, lhsRealIsInfWithSign, rhsReal, fmf);
    Value lhsImagIsInfWithSignTimesRhsImag =
        rewriter.create<arith::MulFOp>(loc, lhsImagIsInfWithSign, rhsImag, fmf);
    Value resultReal3 = rewriter.create<arith::MulFOp>(
        loc, inf,
        rewriter.create<arith::AddFOp>(loc, lhsRealIsInfWithSignTimesRhsReal,
                                       lhsImagIsInfWithSignTimesRhsImag, fmf),
        fmf);
    Value lhsRealIsInfWithSignTimesRhsImag =
        rewriter.create<arith::MulFOp>(loc, lhsRealIsInfWithSign, rhsImag, fmf);
    Value lhsImagIsInfWithSignTimesRhsReal =
        rewriter.create<arith::MulFOp>(loc, lhsImagIsInfWithSign, rhsReal, fmf);
    Value resultImag3 = rewriter.create<arith::MulFOp>(
        loc, inf,
        rewriter.create<arith::SubFOp>(loc, lhsImagIsInfWithSignTimesRhsReal,
                                       lhsRealIsInfWithSignTimesRhsImag, fmf),
        fmf);

    // Case 3: Finite numerator, infinite denominator.
    Value lhsRealFinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, lhsRealAbs, inf);
    Value lhsImagFinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, lhsImagAbs, inf);
    Value lhsFinite =
        rewriter.create<arith::AndIOp>(loc, lhsRealFinite, lhsImagFinite);
    Value rhsRealInfinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, rhsRealAbs, inf);
    Value rhsImagInfinite = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, rhsImagAbs, inf);
    Value rhsInfinite =
        rewriter.create<arith::OrIOp>(loc, rhsRealInfinite, rhsImagInfinite);
    Value finiteNumInfiniteDenom =
        rewriter.create<arith::AndIOp>(loc, lhsFinite, rhsInfinite);
    Value rhsRealIsInfWithSign = rewriter.create<math::CopySignOp>(
        loc, rewriter.create<arith::SelectOp>(loc, rhsRealInfinite, one, zero),
        rhsReal);
    Value rhsImagIsInfWithSign = rewriter.create<math::CopySignOp>(
        loc, rewriter.create<arith::SelectOp>(loc, rhsImagInfinite, one, zero),
        rhsImag);
    Value rhsRealIsInfWithSignTimesLhsReal =
        rewriter.create<arith::MulFOp>(loc, lhsReal, rhsRealIsInfWithSign, fmf);
    Value rhsImagIsInfWithSignTimesLhsImag =
        rewriter.create<arith::MulFOp>(loc, lhsImag, rhsImagIsInfWithSign, fmf);
    Value resultReal4 = rewriter.create<arith::MulFOp>(
        loc, zero,
        rewriter.create<arith::AddFOp>(loc, rhsRealIsInfWithSignTimesLhsReal,
                                       rhsImagIsInfWithSignTimesLhsImag, fmf),
        fmf);
    Value rhsRealIsInfWithSignTimesLhsImag =
        rewriter.create<arith::MulFOp>(loc, lhsImag, rhsRealIsInfWithSign, fmf);
    Value rhsImagIsInfWithSignTimesLhsReal =
        rewriter.create<arith::MulFOp>(loc, lhsReal, rhsImagIsInfWithSign, fmf);
    Value resultImag4 = rewriter.create<arith::MulFOp>(
        loc, zero,
        rewriter.create<arith::SubFOp>(loc, rhsRealIsInfWithSignTimesLhsImag,
                                       rhsImagIsInfWithSignTimesLhsReal, fmf),
        fmf);

    Value realAbsSmallerThanImagAbs = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, rhsRealAbs, rhsImagAbs);
    Value resultReal = rewriter.create<arith::SelectOp>(
        loc, realAbsSmallerThanImagAbs, resultReal1, resultReal2);
    Value resultImag = rewriter.create<arith::SelectOp>(
        loc, realAbsSmallerThanImagAbs, resultImag1, resultImag2);
    Value resultRealSpecialCase3 = rewriter.create<arith::SelectOp>(
        loc, finiteNumInfiniteDenom, resultReal4, resultReal);
    Value resultImagSpecialCase3 = rewriter.create<arith::SelectOp>(
        loc, finiteNumInfiniteDenom, resultImag4, resultImag);
    Value resultRealSpecialCase2 = rewriter.create<arith::SelectOp>(
        loc, infNumFiniteDenom, resultReal3, resultRealSpecialCase3);
    Value resultImagSpecialCase2 = rewriter.create<arith::SelectOp>(
        loc, infNumFiniteDenom, resultImag3, resultImagSpecialCase3);
    Value resultRealSpecialCase1 = rewriter.create<arith::SelectOp>(
        loc, resultIsInfinity, infinityResultReal, resultRealSpecialCase2);
    Value resultImagSpecialCase1 = rewriter.create<arith::SelectOp>(
        loc, resultIsInfinity, infinityResultImag, resultImagSpecialCase2);

    Value resultRealIsNaN = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, resultReal, zero);
    Value resultImagIsNaN = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, resultImag, zero);
    Value resultIsNaN =
        rewriter.create<arith::AndIOp>(loc, resultRealIsNaN, resultImagIsNaN);
    Value resultRealWithSpecialCases = rewriter.create<arith::SelectOp>(
        loc, resultIsNaN, resultRealSpecialCase1, resultReal);
    Value resultImagWithSpecialCases = rewriter.create<arith::SelectOp>(
        loc, resultIsNaN, resultImagSpecialCase1, resultImag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(
        op, type, resultRealWithSpecialCases, resultImagWithSpecialCases);
    return success();
  }
};

struct ExpOpConversion : public OpConversionPattern<complex::ExpOp> {
  using OpConversionPattern<complex::ExpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    Value real =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getComplex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getComplex());
    Value expReal = rewriter.create<math::ExpOp>(loc, real, fmf.getValue());
    Value cosImag = rewriter.create<math::CosOp>(loc, imag, fmf.getValue());
    Value resultReal =
        rewriter.create<arith::MulFOp>(loc, expReal, cosImag, fmf.getValue());
    Value sinImag = rewriter.create<math::SinOp>(loc, imag, fmf.getValue());
    Value resultImag =
        rewriter.create<arith::MulFOp>(loc, expReal, sinImag, fmf.getValue());

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct Expm1OpConversion : public OpConversionPattern<complex::Expm1Op> {
  using OpConversionPattern<complex::Expm1Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::Expm1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value exp = b.create<complex::ExpOp>(adaptor.getComplex(), fmf.getValue());

    Value real = b.create<complex::ReOp>(elementType, exp);
    Value one = b.create<arith::ConstantOp>(elementType,
                                            b.getFloatAttr(elementType, 1));
    Value realMinusOne = b.create<arith::SubFOp>(real, one, fmf.getValue());
    Value imag = b.create<complex::ImOp>(elementType, exp);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, realMinusOne,
                                                   imag);
    return success();
  }
};

struct LogOpConversion : public OpConversionPattern<complex::LogOp> {
  using OpConversionPattern<complex::LogOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::LogOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value abs = b.create<complex::AbsOp>(elementType, adaptor.getComplex(),
                                         fmf.getValue());
    Value resultReal = b.create<math::LogOp>(elementType, abs, fmf.getValue());
    Value real = b.create<complex::ReOp>(elementType, adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(elementType, adaptor.getComplex());
    Value resultImag =
        b.create<math::Atan2Op>(elementType, imag, real, fmf.getValue());
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct Log1pOpConversion : public OpConversionPattern<complex::Log1pOp> {
  using OpConversionPattern<complex::Log1pOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::Log1pOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlags fmf = op.getFastMathFlagsAttr().getValue();
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value real = b.create<complex::ReOp>(adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(adaptor.getComplex());

    Value half = b.create<arith::ConstantOp>(elementType,
                                             b.getFloatAttr(elementType, 0.5));
    Value one = b.create<arith::ConstantOp>(elementType,
                                            b.getFloatAttr(elementType, 1));
    Value realPlusOne = b.create<arith::AddFOp>(real, one, fmf);
    Value absRealPlusOne = b.create<math::AbsFOp>(realPlusOne, fmf);
    Value absImag = b.create<math::AbsFOp>(imag, fmf);

    Value maxAbs = b.create<arith::MaximumFOp>(absRealPlusOne, absImag, fmf);
    Value minAbs = b.create<arith::MinimumFOp>(absRealPlusOne, absImag, fmf);

    Value useReal = b.create<arith::CmpFOp>(arith::CmpFPredicate::OGT,
                                            realPlusOne, absImag, fmf);
    Value maxMinusOne = b.create<arith::SubFOp>(maxAbs, one, fmf);
    Value maxAbsOfRealPlusOneAndImagMinusOne =
        b.create<arith::SelectOp>(useReal, real, maxMinusOne);
    Value minMaxRatio = b.create<arith::DivFOp>(minAbs, maxAbs, fmf);
    Value logOfMaxAbsOfRealPlusOneAndImag =
        b.create<math::Log1pOp>(maxAbsOfRealPlusOneAndImagMinusOne, fmf);
    Value logOfSqrtPart = b.create<math::Log1pOp>(
        b.create<arith::MulFOp>(minMaxRatio, minMaxRatio, fmf), fmf);
    Value r = b.create<arith::AddFOp>(
        b.create<arith::MulFOp>(half, logOfSqrtPart, fmf),
        logOfMaxAbsOfRealPlusOneAndImag, fmf);
    Value resultReal = b.create<arith::SelectOp>(
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, r, r, fmf), minAbs,
        r);
    Value resultImag = b.create<math::Atan2Op>(imag, realPlusOne, fmf);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct MulOpConversion : public OpConversionPattern<complex::MulOp> {
  using OpConversionPattern<complex::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto type = cast<ComplexType>(adaptor.getLhs().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();
    auto fmfValue = fmf.getValue();

    Value lhsReal = b.create<complex::ReOp>(elementType, adaptor.getLhs());
    Value lhsRealAbs = b.create<math::AbsFOp>(lhsReal, fmfValue);
    Value lhsImag = b.create<complex::ImOp>(elementType, adaptor.getLhs());
    Value lhsImagAbs = b.create<math::AbsFOp>(lhsImag, fmfValue);
    Value rhsReal = b.create<complex::ReOp>(elementType, adaptor.getRhs());
    Value rhsRealAbs = b.create<math::AbsFOp>(rhsReal, fmfValue);
    Value rhsImag = b.create<complex::ImOp>(elementType, adaptor.getRhs());
    Value rhsImagAbs = b.create<math::AbsFOp>(rhsImag, fmfValue);

    Value lhsRealTimesRhsReal =
        b.create<arith::MulFOp>(lhsReal, rhsReal, fmfValue);
    Value lhsRealTimesRhsRealAbs =
        b.create<math::AbsFOp>(lhsRealTimesRhsReal, fmfValue);
    Value lhsImagTimesRhsImag =
        b.create<arith::MulFOp>(lhsImag, rhsImag, fmfValue);
    Value lhsImagTimesRhsImagAbs =
        b.create<math::AbsFOp>(lhsImagTimesRhsImag, fmfValue);
    Value real = b.create<arith::SubFOp>(lhsRealTimesRhsReal,
                                         lhsImagTimesRhsImag, fmfValue);

    Value lhsImagTimesRhsReal =
        b.create<arith::MulFOp>(lhsImag, rhsReal, fmfValue);
    Value lhsImagTimesRhsRealAbs =
        b.create<math::AbsFOp>(lhsImagTimesRhsReal, fmfValue);
    Value lhsRealTimesRhsImag =
        b.create<arith::MulFOp>(lhsReal, rhsImag, fmfValue);
    Value lhsRealTimesRhsImagAbs =
        b.create<math::AbsFOp>(lhsRealTimesRhsImag, fmfValue);
    Value imag = b.create<arith::AddFOp>(lhsImagTimesRhsReal,
                                         lhsRealTimesRhsImag, fmfValue);

    // Handle cases where the "naive" calculation results in NaN values.
    Value realIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, real, real);
    Value imagIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, imag, imag);
    Value isNan = b.create<arith::AndIOp>(realIsNan, imagIsNan);

    Value inf = b.create<arith::ConstantOp>(
        elementType,
        b.getFloatAttr(elementType,
                       APFloat::getInf(elementType.getFloatSemantics())));

    // Case 1. `lhsReal` or `lhsImag` are infinite.
    Value lhsRealIsInf =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, lhsRealAbs, inf);
    Value lhsImagIsInf =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, lhsImagAbs, inf);
    Value lhsIsInf = b.create<arith::OrIOp>(lhsRealIsInf, lhsImagIsInf);
    Value rhsRealIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, rhsReal, rhsReal);
    Value rhsImagIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, rhsImag, rhsImag);
    Value zero =
        b.create<arith::ConstantOp>(elementType, b.getZeroAttr(elementType));
    Value one = b.create<arith::ConstantOp>(elementType,
                                            b.getFloatAttr(elementType, 1));
    Value lhsRealIsInfFloat =
        b.create<arith::SelectOp>(lhsRealIsInf, one, zero);
    lhsReal = b.create<arith::SelectOp>(
        lhsIsInf, b.create<math::CopySignOp>(lhsRealIsInfFloat, lhsReal),
        lhsReal);
    Value lhsImagIsInfFloat =
        b.create<arith::SelectOp>(lhsImagIsInf, one, zero);
    lhsImag = b.create<arith::SelectOp>(
        lhsIsInf, b.create<math::CopySignOp>(lhsImagIsInfFloat, lhsImag),
        lhsImag);
    Value lhsIsInfAndRhsRealIsNan =
        b.create<arith::AndIOp>(lhsIsInf, rhsRealIsNan);
    rhsReal = b.create<arith::SelectOp>(
        lhsIsInfAndRhsRealIsNan, b.create<math::CopySignOp>(zero, rhsReal),
        rhsReal);
    Value lhsIsInfAndRhsImagIsNan =
        b.create<arith::AndIOp>(lhsIsInf, rhsImagIsNan);
    rhsImag = b.create<arith::SelectOp>(
        lhsIsInfAndRhsImagIsNan, b.create<math::CopySignOp>(zero, rhsImag),
        rhsImag);

    // Case 2. `rhsReal` or `rhsImag` are infinite.
    Value rhsRealIsInf =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, rhsRealAbs, inf);
    Value rhsImagIsInf =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, rhsImagAbs, inf);
    Value rhsIsInf = b.create<arith::OrIOp>(rhsRealIsInf, rhsImagIsInf);
    Value lhsRealIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, lhsReal, lhsReal);
    Value lhsImagIsNan =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, lhsImag, lhsImag);
    Value rhsRealIsInfFloat =
        b.create<arith::SelectOp>(rhsRealIsInf, one, zero);
    rhsReal = b.create<arith::SelectOp>(
        rhsIsInf, b.create<math::CopySignOp>(rhsRealIsInfFloat, rhsReal),
        rhsReal);
    Value rhsImagIsInfFloat =
        b.create<arith::SelectOp>(rhsImagIsInf, one, zero);
    rhsImag = b.create<arith::SelectOp>(
        rhsIsInf, b.create<math::CopySignOp>(rhsImagIsInfFloat, rhsImag),
        rhsImag);
    Value rhsIsInfAndLhsRealIsNan =
        b.create<arith::AndIOp>(rhsIsInf, lhsRealIsNan);
    lhsReal = b.create<arith::SelectOp>(
        rhsIsInfAndLhsRealIsNan, b.create<math::CopySignOp>(zero, lhsReal),
        lhsReal);
    Value rhsIsInfAndLhsImagIsNan =
        b.create<arith::AndIOp>(rhsIsInf, lhsImagIsNan);
    lhsImag = b.create<arith::SelectOp>(
        rhsIsInfAndLhsImagIsNan, b.create<math::CopySignOp>(zero, lhsImag),
        lhsImag);
    Value recalc = b.create<arith::OrIOp>(lhsIsInf, rhsIsInf);

    // Case 3. One of the pairwise products of left hand side with right hand
    // side is infinite.
    Value lhsRealTimesRhsRealIsInf = b.create<arith::CmpFOp>(
        arith::CmpFPredicate::OEQ, lhsRealTimesRhsRealAbs, inf);
    Value lhsImagTimesRhsImagIsInf = b.create<arith::CmpFOp>(
        arith::CmpFPredicate::OEQ, lhsImagTimesRhsImagAbs, inf);
    Value isSpecialCase = b.create<arith::OrIOp>(lhsRealTimesRhsRealIsInf,
                                                 lhsImagTimesRhsImagIsInf);
    Value lhsRealTimesRhsImagIsInf = b.create<arith::CmpFOp>(
        arith::CmpFPredicate::OEQ, lhsRealTimesRhsImagAbs, inf);
    isSpecialCase =
        b.create<arith::OrIOp>(isSpecialCase, lhsRealTimesRhsImagIsInf);
    Value lhsImagTimesRhsRealIsInf = b.create<arith::CmpFOp>(
        arith::CmpFPredicate::OEQ, lhsImagTimesRhsRealAbs, inf);
    isSpecialCase =
        b.create<arith::OrIOp>(isSpecialCase, lhsImagTimesRhsRealIsInf);
    Type i1Type = b.getI1Type();
    Value notRecalc = b.create<arith::XOrIOp>(
        recalc,
        b.create<arith::ConstantOp>(i1Type, b.getIntegerAttr(i1Type, 1)));
    isSpecialCase = b.create<arith::AndIOp>(isSpecialCase, notRecalc);
    Value isSpecialCaseAndLhsRealIsNan =
        b.create<arith::AndIOp>(isSpecialCase, lhsRealIsNan);
    lhsReal = b.create<arith::SelectOp>(
        isSpecialCaseAndLhsRealIsNan, b.create<math::CopySignOp>(zero, lhsReal),
        lhsReal);
    Value isSpecialCaseAndLhsImagIsNan =
        b.create<arith::AndIOp>(isSpecialCase, lhsImagIsNan);
    lhsImag = b.create<arith::SelectOp>(
        isSpecialCaseAndLhsImagIsNan, b.create<math::CopySignOp>(zero, lhsImag),
        lhsImag);
    Value isSpecialCaseAndRhsRealIsNan =
        b.create<arith::AndIOp>(isSpecialCase, rhsRealIsNan);
    rhsReal = b.create<arith::SelectOp>(
        isSpecialCaseAndRhsRealIsNan, b.create<math::CopySignOp>(zero, rhsReal),
        rhsReal);
    Value isSpecialCaseAndRhsImagIsNan =
        b.create<arith::AndIOp>(isSpecialCase, rhsImagIsNan);
    rhsImag = b.create<arith::SelectOp>(
        isSpecialCaseAndRhsImagIsNan, b.create<math::CopySignOp>(zero, rhsImag),
        rhsImag);
    recalc = b.create<arith::OrIOp>(recalc, isSpecialCase);
    recalc = b.create<arith::AndIOp>(isNan, recalc);

    // Recalculate real part.
    lhsRealTimesRhsReal = b.create<arith::MulFOp>(lhsReal, rhsReal, fmfValue);
    lhsImagTimesRhsImag = b.create<arith::MulFOp>(lhsImag, rhsImag, fmfValue);
    Value newReal = b.create<arith::SubFOp>(lhsRealTimesRhsReal,
                                            lhsImagTimesRhsImag, fmfValue);
    real = b.create<arith::SelectOp>(
        recalc, b.create<arith::MulFOp>(inf, newReal, fmfValue), real);

    // Recalculate imag part.
    lhsImagTimesRhsReal = b.create<arith::MulFOp>(lhsImag, rhsReal, fmfValue);
    lhsRealTimesRhsImag = b.create<arith::MulFOp>(lhsReal, rhsImag, fmfValue);
    Value newImag = b.create<arith::AddFOp>(lhsImagTimesRhsReal,
                                            lhsRealTimesRhsImag, fmfValue);
    imag = b.create<arith::SelectOp>(
        recalc, b.create<arith::MulFOp>(inf, newImag, fmfValue), imag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, real, imag);
    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<complex::NegOp> {
  using OpConversionPattern<complex::NegOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());

    Value real =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getComplex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getComplex());
    Value negReal = rewriter.create<arith::NegFOp>(loc, real);
    Value negImag = rewriter.create<arith::NegFOp>(loc, imag);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, negReal, negImag);
    return success();
  }
};

struct SinOpConversion : public TrigonometricOpConversion<complex::SinOp> {
  using TrigonometricOpConversion<complex::SinOp>::TrigonometricOpConversion;

  std::pair<Value, Value> combine(Location loc, Value scaledExp,
                                  Value reciprocalExp, Value sin, Value cos,
                                  ConversionPatternRewriter &rewriter,
                                  arith::FastMathFlagsAttr fmf) const override {
    // Complex sine is defined as;
    //   sin(x + iy) = -0.5i * (exp(i(x + iy)) - exp(-i(x + iy)))
    // Plugging in:
    //   exp(i(x+iy)) = exp(-y + ix) = exp(-y)(cos(x) + i sin(x))
    //   exp(-i(x+iy)) = exp(y + i(-x)) = exp(y)(cos(x) + i (-sin(x)))
    // and defining t := exp(y)
    // We get:
    //   Re(sin(x + iy)) = (0.5*t + 0.5/t) * sin x
    //   Im(cos(x + iy)) = (0.5*t - 0.5/t) * cos x
    Value sum =
        rewriter.create<arith::AddFOp>(loc, scaledExp, reciprocalExp, fmf);
    Value resultReal = rewriter.create<arith::MulFOp>(loc, sum, sin, fmf);
    Value diff =
        rewriter.create<arith::SubFOp>(loc, scaledExp, reciprocalExp, fmf);
    Value resultImag = rewriter.create<arith::MulFOp>(loc, diff, cos, fmf);
    return {resultReal, resultImag};
  }
};

// The algorithm is listed in https://dl.acm.org/doi/pdf/10.1145/363717.363780.
struct SqrtOpConversion : public OpConversionPattern<complex::SqrtOp> {
  using OpConversionPattern<complex::SqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto type = cast<ComplexType>(op.getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlags fmf = op.getFastMathFlagsAttr().getValue();

    auto cst = [&](APFloat v) {
      return b.create<arith::ConstantOp>(elementType,
                                         b.getFloatAttr(elementType, v));
    };
    const auto &floatSemantics = elementType.getFloatSemantics();
    Value zero = cst(APFloat::getZero(floatSemantics));
    Value half = b.create<arith::ConstantOp>(elementType,
                                             b.getFloatAttr(elementType, 0.5));

    Value real = b.create<complex::ReOp>(elementType, adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(elementType, adaptor.getComplex());
    Value absSqrt = computeAbs(real, imag, fmf, b, AbsFn::sqrt);
    Value argArg = b.create<math::Atan2Op>(imag, real, fmf);
    Value sqrtArg = b.create<arith::MulFOp>(argArg, half, fmf);
    Value cos = b.create<math::CosOp>(sqrtArg, fmf);
    Value sin = b.create<math::SinOp>(sqrtArg, fmf);
    // sin(atan2(0, inf)) = 0, sqrt(abs(inf)) = inf, but we can't multiply
    // 0 * inf.
    Value sinIsZero =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, sin, zero, fmf);

    Value resultReal = b.create<arith::MulFOp>(absSqrt, cos, fmf);
    Value resultImag = b.create<arith::SelectOp>(
        sinIsZero, zero, b.create<arith::MulFOp>(absSqrt, sin, fmf));
    if (!arith::bitEnumContainsAll(fmf, arith::FastMathFlags::nnan |
                                            arith::FastMathFlags::ninf)) {
      Value inf = cst(APFloat::getInf(floatSemantics));
      Value negInf = cst(APFloat::getInf(floatSemantics, true));
      Value nan = cst(APFloat::getNaN(floatSemantics));
      Value absImag = b.create<math::AbsFOp>(elementType, imag, fmf);

      Value absImagIsInf =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, absImag, inf, fmf);
      Value absImagIsNotInf =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::ONE, absImag, inf, fmf);
      Value realIsInf =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, real, inf, fmf);
      Value realIsNegInf =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, real, negInf, fmf);

      resultReal = b.create<arith::SelectOp>(
          b.create<arith::AndIOp>(realIsNegInf, absImagIsNotInf), zero,
          resultReal);
      resultReal = b.create<arith::SelectOp>(
          b.create<arith::OrIOp>(absImagIsInf, realIsInf), inf, resultReal);

      Value imagSignInf = b.create<math::CopySignOp>(inf, imag, fmf);
      resultImag = b.create<arith::SelectOp>(
          b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, absSqrt, absSqrt),
          nan, resultImag);
      resultImag = b.create<arith::SelectOp>(
          b.create<arith::OrIOp>(absImagIsInf, realIsNegInf), imagSignInf,
          resultImag);
    }

    Value resultIsZero =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, absSqrt, zero, fmf);
    resultReal = b.create<arith::SelectOp>(resultIsZero, zero, resultReal);
    resultImag = b.create<arith::SelectOp>(resultIsZero, zero, resultImag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct SignOpConversion : public OpConversionPattern<complex::SignOp> {
  using OpConversionPattern<complex::SignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::SignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    Value real = b.create<complex::ReOp>(elementType, adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(elementType, adaptor.getComplex());
    Value zero =
        b.create<arith::ConstantOp>(elementType, b.getZeroAttr(elementType));
    Value realIsZero =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, real, zero);
    Value imagIsZero =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, imag, zero);
    Value isZero = b.create<arith::AndIOp>(realIsZero, imagIsZero);
    auto abs = b.create<complex::AbsOp>(elementType, adaptor.getComplex(), fmf);
    Value realSign = b.create<arith::DivFOp>(real, abs, fmf);
    Value imagSign = b.create<arith::DivFOp>(imag, abs, fmf);
    Value sign = b.create<complex::CreateOp>(type, realSign, imagSign);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isZero,
                                                 adaptor.getComplex(), sign);
    return success();
  }
};

template <typename Op>
struct TanTanhOpConversion : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlags fmf = op.getFastMathFlagsAttr().getValue();
    const auto &floatSemantics = elementType.getFloatSemantics();

    Value real =
        b.create<complex::ReOp>(loc, elementType, adaptor.getComplex());
    Value imag =
        b.create<complex::ImOp>(loc, elementType, adaptor.getComplex());
    Value negOne = b.create<arith::ConstantOp>(
        elementType, b.getFloatAttr(elementType, -1.0));

    if constexpr (std::is_same_v<Op, complex::TanOp>) {
      // tan(x+yi) = -i*tanh(-y + xi)
      std::swap(real, imag);
      real = b.create<arith::MulFOp>(real, negOne, fmf);
    }

    auto cst = [&](APFloat v) {
      return b.create<arith::ConstantOp>(elementType,
                                         b.getFloatAttr(elementType, v));
    };
    Value inf = cst(APFloat::getInf(floatSemantics));
    Value four = b.create<arith::ConstantOp>(elementType,
                                             b.getFloatAttr(elementType, 4.0));
    Value twoReal = b.create<arith::AddFOp>(real, real, fmf);
    Value negTwoReal = b.create<arith::MulFOp>(negOne, twoReal, fmf);

    Value expTwoRealMinusOne = b.create<math::ExpM1Op>(twoReal, fmf);
    Value expNegTwoRealMinusOne = b.create<math::ExpM1Op>(negTwoReal, fmf);
    Value realNum =
        b.create<arith::SubFOp>(expTwoRealMinusOne, expNegTwoRealMinusOne, fmf);

    Value cosImag = b.create<math::CosOp>(imag, fmf);
    Value cosImagSq = b.create<arith::MulFOp>(cosImag, cosImag, fmf);
    Value twoCosTwoImagPlusOne = b.create<arith::MulFOp>(cosImagSq, four, fmf);
    Value sinImag = b.create<math::SinOp>(imag, fmf);

    Value imagNum = b.create<arith::MulFOp>(
        four, b.create<arith::MulFOp>(cosImag, sinImag, fmf), fmf);

    Value expSumMinusTwo =
        b.create<arith::AddFOp>(expTwoRealMinusOne, expNegTwoRealMinusOne, fmf);
    Value denom =
        b.create<arith::AddFOp>(expSumMinusTwo, twoCosTwoImagPlusOne, fmf);

    Value isInf = b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ,
                                          expSumMinusTwo, inf, fmf);
    Value realLimit = b.create<math::CopySignOp>(negOne, real, fmf);

    Value resultReal = b.create<arith::SelectOp>(
        isInf, realLimit, b.create<arith::DivFOp>(realNum, denom, fmf));
    Value resultImag = b.create<arith::DivFOp>(imagNum, denom, fmf);

    if (!arith::bitEnumContainsAll(fmf, arith::FastMathFlags::nnan |
                                            arith::FastMathFlags::ninf)) {
      Value absReal = b.create<math::AbsFOp>(real, fmf);
      Value zero = b.create<arith::ConstantOp>(
          elementType, b.getFloatAttr(elementType, 0.0));
      Value nan = cst(APFloat::getNaN(floatSemantics));

      Value absRealIsInf =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, absReal, inf, fmf);
      Value imagIsZero =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, imag, zero, fmf);
      Value absRealIsNotInf = b.create<arith::XOrIOp>(
          absRealIsInf, b.create<arith::ConstantIntOp>(true, /*width=*/1));

      Value imagNumIsNaN = b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO,
                                                   imagNum, imagNum, fmf);
      Value resultRealIsNaN =
          b.create<arith::AndIOp>(imagNumIsNaN, absRealIsNotInf);
      Value resultImagIsZero = b.create<arith::OrIOp>(
          imagIsZero, b.create<arith::AndIOp>(absRealIsInf, imagNumIsNaN));

      resultReal = b.create<arith::SelectOp>(resultRealIsNaN, nan, resultReal);
      resultImag =
          b.create<arith::SelectOp>(resultImagIsZero, zero, resultImag);
    }

    if constexpr (std::is_same_v<Op, complex::TanOp>) {
      // tan(x+yi) = -i*tanh(-y + xi)
      std::swap(resultReal, resultImag);
      resultImag = b.create<arith::MulFOp>(resultImag, negOne, fmf);
    }

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct ConjOpConversion : public OpConversionPattern<complex::ConjOp> {
  using OpConversionPattern<complex::ConjOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::ConjOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    Value real =
        rewriter.create<complex::ReOp>(loc, elementType, adaptor.getComplex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, elementType, adaptor.getComplex());
    Value negImag = rewriter.create<arith::NegFOp>(loc, elementType, imag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, real, negImag);

    return success();
  }
};

/// Converts lhs^y = (a+bi)^(c+di) to
///    (a*a+b*b)^(0.5c) * exp(-d*atan2(b,a)) * (cos(q) + i*sin(q)),
///    where q = c*atan2(b,a)+0.5d*ln(a*a+b*b)
static Value powOpConversionImpl(mlir::ImplicitLocOpBuilder &builder,
                                 ComplexType type, Value lhs, Value c, Value d,
                                 arith::FastMathFlags fmf) {
  auto elementType = cast<FloatType>(type.getElementType());

  Value a = builder.create<complex::ReOp>(lhs);
  Value b = builder.create<complex::ImOp>(lhs);

  Value abs = builder.create<complex::AbsOp>(lhs, fmf);
  Value absToC = builder.create<math::PowFOp>(abs, c, fmf);

  Value negD = builder.create<arith::NegFOp>(d, fmf);
  Value argLhs = builder.create<math::Atan2Op>(b, a, fmf);
  Value negDArgLhs = builder.create<arith::MulFOp>(negD, argLhs, fmf);
  Value expNegDArgLhs = builder.create<math::ExpOp>(negDArgLhs, fmf);

  Value coeff = builder.create<arith::MulFOp>(absToC, expNegDArgLhs, fmf);
  Value lnAbs = builder.create<math::LogOp>(abs, fmf);
  Value cArgLhs = builder.create<arith::MulFOp>(c, argLhs, fmf);
  Value dLnAbs = builder.create<arith::MulFOp>(d, lnAbs, fmf);
  Value q = builder.create<arith::AddFOp>(cArgLhs, dLnAbs, fmf);
  Value cosQ = builder.create<math::CosOp>(q, fmf);
  Value sinQ = builder.create<math::SinOp>(q, fmf);

  Value inf = builder.create<arith::ConstantOp>(
      elementType,
      builder.getFloatAttr(elementType,
                           APFloat::getInf(elementType.getFloatSemantics())));
  Value zero = builder.create<arith::ConstantOp>(
      elementType, builder.getFloatAttr(elementType, 0.0));
  Value one = builder.create<arith::ConstantOp>(
      elementType, builder.getFloatAttr(elementType, 1.0));
  Value complexOne = builder.create<complex::CreateOp>(type, one, zero);
  Value complexZero = builder.create<complex::CreateOp>(type, zero, zero);
  Value complexInf = builder.create<complex::CreateOp>(type, inf, zero);

  // Case 0:
  // d^c is 0 if d is 0 and c > 0. 0^0 is defined to be 1.0, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  Value absEqZero =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, abs, zero, fmf);
  Value dEqZero =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, d, zero, fmf);
  Value cEqZero =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, c, zero, fmf);
  Value bEqZero =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, b, zero, fmf);

  Value zeroLeC =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLE, zero, c, fmf);
  Value coeffCosQ = builder.create<arith::MulFOp>(coeff, cosQ, fmf);
  Value coeffSinQ = builder.create<arith::MulFOp>(coeff, sinQ, fmf);
  Value complexOneOrZero =
      builder.create<arith::SelectOp>(cEqZero, complexOne, complexZero);
  Value coeffCosSin =
      builder.create<complex::CreateOp>(type, coeffCosQ, coeffSinQ);
  Value cutoff0 = builder.create<arith::SelectOp>(
      builder.create<arith::AndIOp>(
          builder.create<arith::AndIOp>(absEqZero, dEqZero), zeroLeC),
      complexOneOrZero, coeffCosSin);

  // Case 1:
  // x^0 is defined to be 1 for any x, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  Value rhsEqZero = builder.create<arith::AndIOp>(cEqZero, dEqZero);
  Value cutoff1 =
      builder.create<arith::SelectOp>(rhsEqZero, complexOne, cutoff0);

  // Case 2:
  // 1^(c + d*i) = 1 + 0*i
  Value lhsEqOne = builder.create<arith::AndIOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, a, one, fmf),
      bEqZero);
  Value cutoff2 =
      builder.create<arith::SelectOp>(lhsEqOne, complexOne, cutoff1);

  // Case 3:
  // inf^(c + 0*i) = inf + 0*i, c > 0
  Value lhsEqInf = builder.create<arith::AndIOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, a, inf, fmf),
      bEqZero);
  Value rhsGt0 = builder.create<arith::AndIOp>(
      dEqZero,
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, c, zero, fmf));
  Value cutoff3 = builder.create<arith::SelectOp>(
      builder.create<arith::AndIOp>(lhsEqInf, rhsGt0), complexInf, cutoff2);

  // Case 4:
  // inf^(c + 0*i) = 0 + 0*i, c < 0
  Value rhsLt0 = builder.create<arith::AndIOp>(
      dEqZero,
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, c, zero, fmf));
  Value cutoff4 = builder.create<arith::SelectOp>(
      builder.create<arith::AndIOp>(lhsEqInf, rhsLt0), complexZero, cutoff3);

  return cutoff4;
}

struct PowOpConversion : public OpConversionPattern<complex::PowOp> {
  using OpConversionPattern<complex::PowOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto type = cast<ComplexType>(adaptor.getLhs().getType());
    auto elementType = cast<FloatType>(type.getElementType());

    Value c = builder.create<complex::ReOp>(elementType, adaptor.getRhs());
    Value d = builder.create<complex::ImOp>(elementType, adaptor.getRhs());

    rewriter.replaceOp(op, {powOpConversionImpl(builder, type, adaptor.getLhs(),
                                                c, d, op.getFastmath())});
    return success();
  }
};

struct RsqrtOpConversion : public OpConversionPattern<complex::RsqrtOp> {
  using OpConversionPattern<complex::RsqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto elementType = cast<FloatType>(type.getElementType());

    arith::FastMathFlags fmf = op.getFastMathFlagsAttr().getValue();

    auto cst = [&](APFloat v) {
      return b.create<arith::ConstantOp>(elementType,
                                         b.getFloatAttr(elementType, v));
    };
    const auto &floatSemantics = elementType.getFloatSemantics();
    Value zero = cst(APFloat::getZero(floatSemantics));
    Value inf = cst(APFloat::getInf(floatSemantics));
    Value negHalf = b.create<arith::ConstantOp>(
        elementType, b.getFloatAttr(elementType, -0.5));
    Value nan = cst(APFloat::getNaN(floatSemantics));

    Value real = b.create<complex::ReOp>(elementType, adaptor.getComplex());
    Value imag = b.create<complex::ImOp>(elementType, adaptor.getComplex());
    Value absRsqrt = computeAbs(real, imag, fmf, b, AbsFn::rsqrt);
    Value argArg = b.create<math::Atan2Op>(imag, real, fmf);
    Value rsqrtArg = b.create<arith::MulFOp>(argArg, negHalf, fmf);
    Value cos = b.create<math::CosOp>(rsqrtArg, fmf);
    Value sin = b.create<math::SinOp>(rsqrtArg, fmf);

    Value resultReal = b.create<arith::MulFOp>(absRsqrt, cos, fmf);
    Value resultImag = b.create<arith::MulFOp>(absRsqrt, sin, fmf);

    if (!arith::bitEnumContainsAll(fmf, arith::FastMathFlags::nnan |
                                            arith::FastMathFlags::ninf)) {
      Value negOne = b.create<arith::ConstantOp>(
          elementType, b.getFloatAttr(elementType, -1));

      Value realSignedZero = b.create<math::CopySignOp>(zero, real, fmf);
      Value imagSignedZero = b.create<math::CopySignOp>(zero, imag, fmf);
      Value negImagSignedZero =
          b.create<arith::MulFOp>(negOne, imagSignedZero, fmf);

      Value absReal = b.create<math::AbsFOp>(real, fmf);
      Value absImag = b.create<math::AbsFOp>(imag, fmf);

      Value absImagIsInf =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, absImag, inf, fmf);
      Value realIsNan =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::UNO, real, real, fmf);
      Value realIsInf =
          b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, absReal, inf, fmf);
      Value inIsNanInf = b.create<arith::AndIOp>(absImagIsInf, realIsNan);

      Value resultIsZero = b.create<arith::OrIOp>(inIsNanInf, realIsInf);

      resultReal =
          b.create<arith::SelectOp>(resultIsZero, realSignedZero, resultReal);
      resultImag = b.create<arith::SelectOp>(resultIsZero, negImagSignedZero,
                                             resultImag);
    }

    Value isRealZero =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, real, zero, fmf);
    Value isImagZero =
        b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, imag, zero, fmf);
    Value isZero = b.create<arith::AndIOp>(isRealZero, isImagZero);

    resultReal = b.create<arith::SelectOp>(isZero, inf, resultReal);
    resultImag = b.create<arith::SelectOp>(isZero, nan, resultImag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct AngleOpConversion : public OpConversionPattern<complex::AngleOp> {
  using OpConversionPattern<complex::AngleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AngleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getType();
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    Value real =
        rewriter.create<complex::ReOp>(loc, type, adaptor.getComplex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, type, adaptor.getComplex());

    rewriter.replaceOpWithNewOp<math::Atan2Op>(op, imag, real, fmf);

    return success();
  }
};

} // namespace

void mlir::populateComplexToStandardConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AbsOpConversion,
      AngleOpConversion,
      Atan2OpConversion,
      BinaryComplexOpConversion<complex::AddOp, arith::AddFOp>,
      BinaryComplexOpConversion<complex::SubOp, arith::SubFOp>,
      ComparisonOpConversion<complex::EqualOp, arith::CmpFPredicate::OEQ>,
      ComparisonOpConversion<complex::NotEqualOp, arith::CmpFPredicate::UNE>,
      ConjOpConversion,
      CosOpConversion,
      DivOpConversion,
      ExpOpConversion,
      Expm1OpConversion,
      Log1pOpConversion,
      LogOpConversion,
      MulOpConversion,
      NegOpConversion,
      SignOpConversion,
      SinOpConversion,
      SqrtOpConversion,
      TanTanhOpConversion<complex::TanOp>,
      TanTanhOpConversion<complex::TanhOp>,
      PowOpConversion,
      RsqrtOpConversion
  >(patterns.getContext());
  // clang-format on
}

namespace {
struct ConvertComplexToStandardPass
    : public impl::ConvertComplexToStandardBase<ConvertComplexToStandardPass> {
  void runOnOperation() override;
};

void ConvertComplexToStandardPass::runOnOperation() {
  // Convert to the Standard dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  populateComplexToStandardConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, math::MathDialect>();
  target.addLegalOp<complex::CreateOp, complex::ImOp, complex::ReOp>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<Pass> mlir::createConvertComplexToStandardPass() {
  return std::make_unique<ConvertComplexToStandardPass>();
}
