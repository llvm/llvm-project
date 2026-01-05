//===- ComplexToStandard.cpp - conversion from Complex to Standard dialect ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"

#include "mlir/Conversion/ComplexCommon/DivisionConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOSTANDARDPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

enum class AbsFn { abs, sqrt, rsqrt };

// Returns the absolute value, its square root or its reciprocal square root.
Value computeAbs(Value real, Value imag, arith::FastMathFlags fmf,
                 ImplicitLocOpBuilder &b, AbsFn fn = AbsFn::abs) {
  Value one = arith::ConstantOp::create(b, real.getType(),
                                        b.getFloatAttr(real.getType(), 1.0));

  Value absReal = math::AbsFOp::create(b, real, fmf);
  Value absImag = math::AbsFOp::create(b, imag, fmf);

  Value max = arith::MaximumFOp::create(b, absReal, absImag, fmf);
  Value min = arith::MinimumFOp::create(b, absReal, absImag, fmf);

  // The lowering below requires NaNs and infinities to work correctly.
  arith::FastMathFlags fmfWithNaNInf = arith::bitEnumClear(
      fmf, arith::FastMathFlags::nnan | arith::FastMathFlags::ninf);
  Value ratio = arith::DivFOp::create(b, min, max, fmfWithNaNInf);
  Value ratioSq = arith::MulFOp::create(b, ratio, ratio, fmfWithNaNInf);
  Value ratioSqPlusOne = arith::AddFOp::create(b, ratioSq, one, fmfWithNaNInf);
  Value result;

  if (fn == AbsFn::rsqrt) {
    ratioSqPlusOne = math::RsqrtOp::create(b, ratioSqPlusOne, fmfWithNaNInf);
    min = math::RsqrtOp::create(b, min, fmfWithNaNInf);
    max = math::RsqrtOp::create(b, max, fmfWithNaNInf);
  }

  if (fn == AbsFn::sqrt) {
    Value quarter = arith::ConstantOp::create(
        b, real.getType(), b.getFloatAttr(real.getType(), 0.25));
    // sqrt(sqrt(a*b)) would avoid the pow, but will overflow more easily.
    Value sqrt = math::SqrtOp::create(b, max, fmfWithNaNInf);
    Value p025 =
        math::PowFOp::create(b, ratioSqPlusOne, quarter, fmfWithNaNInf);
    result = arith::MulFOp::create(b, sqrt, p025, fmfWithNaNInf);
  } else {
    Value sqrt = math::SqrtOp::create(b, ratioSqPlusOne, fmfWithNaNInf);
    result = arith::MulFOp::create(b, max, sqrt, fmfWithNaNInf);
  }

  Value isNaN = arith::CmpFOp::create(b, arith::CmpFPredicate::UNO, result,
                                      result, fmfWithNaNInf);
  return arith::SelectOp::create(b, isNaN, min, result);
}

struct AbsOpConversion : public OpConversionPattern<complex::AbsOp> {
  using OpConversionPattern<complex::AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    arith::FastMathFlags fmf = op.getFastMathFlagsAttr().getValue();

    Value real = complex::ReOp::create(b, adaptor.getComplex());
    Value imag = complex::ImOp::create(b, adaptor.getComplex());
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

    Value rhsSquared = complex::MulOp::create(b, type, rhs, rhs, fmf);
    Value lhsSquared = complex::MulOp::create(b, type, lhs, lhs, fmf);
    Value rhsSquaredPlusLhsSquared =
        complex::AddOp::create(b, type, rhsSquared, lhsSquared, fmf);
    Value sqrtOfRhsSquaredPlusLhsSquared =
        complex::SqrtOp::create(b, type, rhsSquaredPlusLhsSquared, fmf);

    Value zero =
        arith::ConstantOp::create(b, elementType, b.getZeroAttr(elementType));
    Value one = arith::ConstantOp::create(b, elementType,
                                          b.getFloatAttr(elementType, 1));
    Value i = complex::CreateOp::create(b, type, zero, one);
    Value iTimesLhs = complex::MulOp::create(b, i, lhs, fmf);
    Value rhsPlusILhs = complex::AddOp::create(b, rhs, iTimesLhs, fmf);

    Value divResult = complex::DivOp::create(
        b, rhsPlusILhs, sqrtOfRhsSquaredPlusLhsSquared, fmf);
    Value logResult = complex::LogOp::create(b, divResult, fmf);

    Value negativeOne = arith::ConstantOp::create(
        b, elementType, b.getFloatAttr(elementType, -1));
    Value negativeI = complex::CreateOp::create(b, type, zero, negativeOne);

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

    Value realLhs =
        complex::ReOp::create(rewriter, loc, type, adaptor.getLhs());
    Value imagLhs =
        complex::ImOp::create(rewriter, loc, type, adaptor.getLhs());
    Value realRhs =
        complex::ReOp::create(rewriter, loc, type, adaptor.getRhs());
    Value imagRhs =
        complex::ImOp::create(rewriter, loc, type, adaptor.getRhs());
    Value realComparison =
        arith::CmpFOp::create(rewriter, loc, p, realLhs, realRhs);
    Value imagComparison =
        arith::CmpFOp::create(rewriter, loc, p, imagLhs, imagRhs);

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

    Value realLhs = complex::ReOp::create(b, elementType, adaptor.getLhs());
    Value realRhs = complex::ReOp::create(b, elementType, adaptor.getRhs());
    Value resultReal = BinaryStandardOp::create(b, elementType, realLhs,
                                                realRhs, fmf.getValue());
    Value imagLhs = complex::ImOp::create(b, elementType, adaptor.getLhs());
    Value imagRhs = complex::ImOp::create(b, elementType, adaptor.getRhs());
    Value resultImag = BinaryStandardOp::create(b, elementType, imagLhs,
                                                imagRhs, fmf.getValue());
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
        complex::ReOp::create(rewriter, loc, elementType, adaptor.getComplex());
    Value imag =
        complex::ImOp::create(rewriter, loc, elementType, adaptor.getComplex());

    // Trigonometric ops use a set of common building blocks to convert to real
    // ops. Here we create these building blocks and call into an op-specific
    // implementation in the subclass to combine them.
    Value half = arith::ConstantOp::create(
        rewriter, loc, elementType, rewriter.getFloatAttr(elementType, 0.5));
    Value exp = math::ExpOp::create(rewriter, loc, imag, fmf);
    Value scaledExp = arith::MulFOp::create(rewriter, loc, half, exp, fmf);
    Value reciprocalExp = arith::DivFOp::create(rewriter, loc, half, exp, fmf);
    Value sin = math::SinOp::create(rewriter, loc, real, fmf);
    Value cos = math::CosOp::create(rewriter, loc, real, fmf);

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
        arith::AddFOp::create(rewriter, loc, reciprocalExp, scaledExp, fmf);
    Value resultReal = arith::MulFOp::create(rewriter, loc, sum, cos, fmf);
    Value diff =
        arith::SubFOp::create(rewriter, loc, reciprocalExp, scaledExp, fmf);
    Value resultImag = arith::MulFOp::create(rewriter, loc, diff, sin, fmf);
    return {resultReal, resultImag};
  }
};

struct DivOpConversion : public OpConversionPattern<complex::DivOp> {
  DivOpConversion(MLIRContext *context, complex::ComplexRangeFlags target)
      : OpConversionPattern<complex::DivOp>(context), complexRange(target) {}

  using OpConversionPattern<complex::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getLhs().getType());
    auto elementType = cast<FloatType>(type.getElementType());
    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();

    Value lhsReal =
        complex::ReOp::create(rewriter, loc, elementType, adaptor.getLhs());
    Value lhsImag =
        complex::ImOp::create(rewriter, loc, elementType, adaptor.getLhs());
    Value rhsReal =
        complex::ReOp::create(rewriter, loc, elementType, adaptor.getRhs());
    Value rhsImag =
        complex::ImOp::create(rewriter, loc, elementType, adaptor.getRhs());

    Value resultReal, resultImag;

    if (complexRange == complex::ComplexRangeFlags::basic ||
        complexRange == complex::ComplexRangeFlags::none) {
      mlir::complex::convertDivToStandardUsingAlgebraic(
          rewriter, loc, lhsReal, lhsImag, rhsReal, rhsImag, fmf, &resultReal,
          &resultImag);
    } else if (complexRange == complex::ComplexRangeFlags::improved) {
      mlir::complex::convertDivToStandardUsingRangeReduction(
          rewriter, loc, lhsReal, lhsImag, rhsReal, rhsImag, fmf, &resultReal,
          &resultImag);
    }

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);

    return success();
  }

private:
  complex::ComplexRangeFlags complexRange;
};

struct ExpOpConversion : public OpConversionPattern<complex::ExpOp> {
  using OpConversionPattern<complex::ExpOp>::OpConversionPattern;

  // exp(x+I*y) = exp(x)*(cos(y)+I*sin(y))
  // Handle special cases as StableHLO implementation does:
  // 1. When b == 0, set imag(exp(z)) = 0
  // 2. When exp(x) == inf, use exp(x/2)*(cos(y)+I*sin(y))*exp(x/2)
  LogicalResult
  matchAndRewrite(complex::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<ComplexType>(adaptor.getComplex().getType());
    auto ET = cast<FloatType>(type.getElementType());
    arith::FastMathFlags fmf = op.getFastMathFlagsAttr().getValue();
    const auto &floatSemantics = ET.getFloatSemantics();
    ImplicitLocOpBuilder b(loc, rewriter);

    Value x = complex::ReOp::create(b, ET, adaptor.getComplex());
    Value y = complex::ImOp::create(b, ET, adaptor.getComplex());
    Value zero = arith::ConstantOp::create(b, ET, b.getZeroAttr(ET));
    Value half = arith::ConstantOp::create(b, ET, b.getFloatAttr(ET, 0.5));
    Value inf = arith::ConstantOp::create(
        b, ET, b.getFloatAttr(ET, APFloat::getInf(floatSemantics)));

    Value exp = math::ExpOp::create(b, x, fmf);
    Value xHalf = arith::MulFOp::create(b, x, half, fmf);
    Value expHalf = math::ExpOp::create(b, xHalf, fmf);
    Value cos = math::CosOp::create(b, y, fmf);
    Value sin = math::SinOp::create(b, y, fmf);

    Value expIsInf =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, exp, inf, fmf);
    Value yIsZero =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, y, zero);

    // Real path: select between exp(x)*cos(y) and exp(x/2)*cos(y)*exp(x/2)
    Value realNormal = arith::MulFOp::create(b, exp, cos, fmf);
    Value expHalfCos = arith::MulFOp::create(b, expHalf, cos, fmf);
    Value realOverflow = arith::MulFOp::create(b, expHalfCos, expHalf, fmf);
    Value resultReal =
        arith::SelectOp::create(b, expIsInf, realOverflow, realNormal);

    // Imaginary part: if y == 0 return 0 else select between exp(x)*sin(y) and
    // exp(x/2)*sin(y)*exp(x/2)
    Value imagNormal = arith::MulFOp::create(b, exp, sin, fmf);
    Value expHalfSin = arith::MulFOp::create(b, expHalf, sin, fmf);
    Value imagOverflow = arith::MulFOp::create(b, expHalfSin, expHalf, fmf);
    Value imagNonZero =
        arith::SelectOp::create(b, expIsInf, imagOverflow, imagNormal);
    Value resultImag = arith::SelectOp::create(b, yIsZero, zero, imagNonZero);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

Value evaluatePolynomial(ImplicitLocOpBuilder &b, Value arg,
                         ArrayRef<double> coefficients,
                         arith::FastMathFlagsAttr fmf) {
  auto argType = mlir::cast<FloatType>(arg.getType());
  Value poly =
      arith::ConstantOp::create(b, b.getFloatAttr(argType, coefficients[0]));
  for (unsigned i = 1; i < coefficients.size(); ++i) {
    poly = math::FmaOp::create(
        b, poly, arg,
        arith::ConstantOp::create(b, b.getFloatAttr(argType, coefficients[i])),
        fmf);
  }
  return poly;
}

struct Expm1OpConversion : public OpConversionPattern<complex::Expm1Op> {
  using OpConversionPattern<complex::Expm1Op>::OpConversionPattern;

  // e^(a+bi)-1 = (e^a*cos(b)-1)+e^a*sin(b)i
  //            [handle inaccuracies when a and/or b are small]
  //            = ((e^a - 1) * cos(b) + cos(b) - 1) + e^a*sin(b)i
  //            = (expm1(a) * cos(b) + cosm1(b)) + e^a*sin(b)i
  LogicalResult
  matchAndRewrite(complex::Expm1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto elemType = mlir::cast<FloatType>(type.getElementType());

    arith::FastMathFlagsAttr fmf = op.getFastMathFlagsAttr();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value real = complex::ReOp::create(b, adaptor.getComplex());
    Value imag = complex::ImOp::create(b, adaptor.getComplex());

    Value zero = arith::ConstantOp::create(b, b.getFloatAttr(elemType, 0.0));
    Value one = arith::ConstantOp::create(b, b.getFloatAttr(elemType, 1.0));

    Value expm1Real = math::ExpM1Op::create(b, real, fmf);
    Value expReal = arith::AddFOp::create(b, expm1Real, one, fmf);

    Value sinImag = math::SinOp::create(b, imag, fmf);
    Value cosm1Imag = emitCosm1(imag, fmf, b);
    Value cosImag = arith::AddFOp::create(b, cosm1Imag, one, fmf);

    Value realResult = arith::AddFOp::create(
        b, arith::MulFOp::create(b, expm1Real, cosImag, fmf), cosm1Imag, fmf);

    Value imagIsZero = arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, imag,
                                             zero, fmf.getValue());
    Value imagResult = arith::SelectOp::create(
        b, imagIsZero, zero, arith::MulFOp::create(b, expReal, sinImag, fmf));

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, realResult,
                                                   imagResult);
    return success();
  }

private:
  Value emitCosm1(Value arg, arith::FastMathFlagsAttr fmf,
                  ImplicitLocOpBuilder &b) const {
    auto argType = mlir::cast<FloatType>(arg.getType());
    auto negHalf = arith::ConstantOp::create(b, b.getFloatAttr(argType, -0.5));
    auto negOne = arith::ConstantOp::create(b, b.getFloatAttr(argType, -1.0));

    // Algorithm copied from cephes cosm1.
    SmallVector<double, 7> kCoeffs{
        4.7377507964246204691685E-14, -1.1470284843425359765671E-11,
        2.0876754287081521758361E-9,  -2.7557319214999787979814E-7,
        2.4801587301570552304991E-5,  -1.3888888888888872993737E-3,
        4.1666666666666666609054E-2,
    };
    Value cos = math::CosOp::create(b, arg, fmf);
    Value forLargeArg = arith::AddFOp::create(b, cos, negOne, fmf);

    Value argPow2 = arith::MulFOp::create(b, arg, arg, fmf);
    Value argPow4 = arith::MulFOp::create(b, argPow2, argPow2, fmf);
    Value poly = evaluatePolynomial(b, argPow2, kCoeffs, fmf);

    auto forSmallArg =
        arith::AddFOp::create(b, arith::MulFOp::create(b, argPow4, poly, fmf),
                              arith::MulFOp::create(b, negHalf, argPow2, fmf));

    // (pi/4)^2 is approximately 0.61685
    Value piOver4Pow2 =
        arith::ConstantOp::create(b, b.getFloatAttr(argType, 0.61685));
    Value cond = arith::CmpFOp::create(b, arith::CmpFPredicate::OGE, argPow2,
                                       piOver4Pow2, fmf.getValue());
    return arith::SelectOp::create(b, cond, forLargeArg, forSmallArg);
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

    Value abs = complex::AbsOp::create(b, elementType, adaptor.getComplex(),
                                       fmf.getValue());
    Value resultReal = math::LogOp::create(b, elementType, abs, fmf.getValue());
    Value real = complex::ReOp::create(b, elementType, adaptor.getComplex());
    Value imag = complex::ImOp::create(b, elementType, adaptor.getComplex());
    Value resultImag =
        math::Atan2Op::create(b, elementType, imag, real, fmf.getValue());
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

    Value real = complex::ReOp::create(b, adaptor.getComplex());
    Value imag = complex::ImOp::create(b, adaptor.getComplex());

    Value half = arith::ConstantOp::create(b, elementType,
                                           b.getFloatAttr(elementType, 0.5));
    Value one = arith::ConstantOp::create(b, elementType,
                                          b.getFloatAttr(elementType, 1));
    Value realPlusOne = arith::AddFOp::create(b, real, one, fmf);
    Value absRealPlusOne = math::AbsFOp::create(b, realPlusOne, fmf);
    Value absImag = math::AbsFOp::create(b, imag, fmf);

    Value maxAbs = arith::MaximumFOp::create(b, absRealPlusOne, absImag, fmf);
    Value minAbs = arith::MinimumFOp::create(b, absRealPlusOne, absImag, fmf);

    Value useReal = arith::CmpFOp::create(b, arith::CmpFPredicate::OGT,
                                          realPlusOne, absImag, fmf);
    Value maxMinusOne = arith::SubFOp::create(b, maxAbs, one, fmf);
    Value maxAbsOfRealPlusOneAndImagMinusOne =
        arith::SelectOp::create(b, useReal, real, maxMinusOne);
    arith::FastMathFlags fmfWithNaNInf = arith::bitEnumClear(
        fmf, arith::FastMathFlags::nnan | arith::FastMathFlags::ninf);
    Value minMaxRatio = arith::DivFOp::create(b, minAbs, maxAbs, fmfWithNaNInf);
    Value logOfMaxAbsOfRealPlusOneAndImag =
        math::Log1pOp::create(b, maxAbsOfRealPlusOneAndImagMinusOne, fmf);
    Value logOfSqrtPart = math::Log1pOp::create(
        b, arith::MulFOp::create(b, minMaxRatio, minMaxRatio, fmfWithNaNInf),
        fmfWithNaNInf);
    Value r = arith::AddFOp::create(
        b, arith::MulFOp::create(b, half, logOfSqrtPart, fmfWithNaNInf),
        logOfMaxAbsOfRealPlusOneAndImag, fmfWithNaNInf);
    Value resultReal = arith::SelectOp::create(
        b,
        arith::CmpFOp::create(b, arith::CmpFPredicate::UNO, r, r,
                              fmfWithNaNInf),
        minAbs, r);
    Value resultImag = math::Atan2Op::create(b, imag, realPlusOne, fmf);
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
    Value lhsReal = complex::ReOp::create(b, elementType, adaptor.getLhs());
    Value lhsImag = complex::ImOp::create(b, elementType, adaptor.getLhs());
    Value rhsReal = complex::ReOp::create(b, elementType, adaptor.getRhs());
    Value rhsImag = complex::ImOp::create(b, elementType, adaptor.getRhs());
    Value lhsRealTimesRhsReal =
        arith::MulFOp::create(b, lhsReal, rhsReal, fmfValue);
    Value lhsImagTimesRhsImag =
        arith::MulFOp::create(b, lhsImag, rhsImag, fmfValue);
    Value real = arith::SubFOp::create(b, lhsRealTimesRhsReal,
                                       lhsImagTimesRhsImag, fmfValue);
    Value lhsImagTimesRhsReal =
        arith::MulFOp::create(b, lhsImag, rhsReal, fmfValue);
    Value lhsRealTimesRhsImag =
        arith::MulFOp::create(b, lhsReal, rhsImag, fmfValue);
    Value imag = arith::AddFOp::create(b, lhsImagTimesRhsReal,
                                       lhsRealTimesRhsImag, fmfValue);
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
        complex::ReOp::create(rewriter, loc, elementType, adaptor.getComplex());
    Value imag =
        complex::ImOp::create(rewriter, loc, elementType, adaptor.getComplex());
    Value negReal = arith::NegFOp::create(rewriter, loc, real);
    Value negImag = arith::NegFOp::create(rewriter, loc, imag);
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
        arith::AddFOp::create(rewriter, loc, scaledExp, reciprocalExp, fmf);
    Value resultReal = arith::MulFOp::create(rewriter, loc, sum, sin, fmf);
    Value diff =
        arith::SubFOp::create(rewriter, loc, scaledExp, reciprocalExp, fmf);
    Value resultImag = arith::MulFOp::create(rewriter, loc, diff, cos, fmf);
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
      return arith::ConstantOp::create(b, elementType,
                                       b.getFloatAttr(elementType, v));
    };
    const auto &floatSemantics = elementType.getFloatSemantics();
    Value zero = cst(APFloat::getZero(floatSemantics));
    Value half = arith::ConstantOp::create(b, elementType,
                                           b.getFloatAttr(elementType, 0.5));

    Value real = complex::ReOp::create(b, elementType, adaptor.getComplex());
    Value imag = complex::ImOp::create(b, elementType, adaptor.getComplex());
    Value absSqrt = computeAbs(real, imag, fmf, b, AbsFn::sqrt);
    Value argArg = math::Atan2Op::create(b, imag, real, fmf);
    Value sqrtArg = arith::MulFOp::create(b, argArg, half, fmf);
    Value cos = math::CosOp::create(b, sqrtArg, fmf);
    Value sin = math::SinOp::create(b, sqrtArg, fmf);
    // sin(atan2(0, inf)) = 0, sqrt(abs(inf)) = inf, but we can't multiply
    // 0 * inf.
    Value sinIsZero =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, sin, zero, fmf);

    Value resultReal = arith::MulFOp::create(b, absSqrt, cos, fmf);
    Value resultImag = arith::SelectOp::create(
        b, sinIsZero, zero, arith::MulFOp::create(b, absSqrt, sin, fmf));
    if (!arith::bitEnumContainsAll(fmf, arith::FastMathFlags::nnan |
                                            arith::FastMathFlags::ninf)) {
      Value inf = cst(APFloat::getInf(floatSemantics));
      Value negInf = cst(APFloat::getInf(floatSemantics, true));
      Value nan = cst(APFloat::getNaN(floatSemantics));
      Value absImag = math::AbsFOp::create(b, elementType, imag, fmf);

      Value absImagIsInf = arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ,
                                                 absImag, inf, fmf);
      Value absImagIsNotInf = arith::CmpFOp::create(
          b, arith::CmpFPredicate::ONE, absImag, inf, fmf);
      Value realIsInf =
          arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, real, inf, fmf);
      Value realIsNegInf = arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ,
                                                 real, negInf, fmf);

      resultReal = arith::SelectOp::create(
          b, arith::AndIOp::create(b, realIsNegInf, absImagIsNotInf), zero,
          resultReal);
      resultReal = arith::SelectOp::create(
          b, arith::OrIOp::create(b, absImagIsInf, realIsInf), inf, resultReal);

      Value imagSignInf = math::CopySignOp::create(b, inf, imag, fmf);
      resultImag = arith::SelectOp::create(
          b,
          arith::CmpFOp::create(b, arith::CmpFPredicate::UNO, absSqrt, absSqrt),
          nan, resultImag);
      resultImag = arith::SelectOp::create(
          b, arith::OrIOp::create(b, absImagIsInf, realIsNegInf), imagSignInf,
          resultImag);
    }

    Value resultIsZero =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, absSqrt, zero, fmf);
    resultReal = arith::SelectOp::create(b, resultIsZero, zero, resultReal);
    resultImag = arith::SelectOp::create(b, resultIsZero, zero, resultImag);

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

    Value real = complex::ReOp::create(b, elementType, adaptor.getComplex());
    Value imag = complex::ImOp::create(b, elementType, adaptor.getComplex());
    Value zero =
        arith::ConstantOp::create(b, elementType, b.getZeroAttr(elementType));
    Value realIsZero =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, real, zero);
    Value imagIsZero =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, imag, zero);
    Value isZero = arith::AndIOp::create(b, realIsZero, imagIsZero);
    auto abs =
        complex::AbsOp::create(b, elementType, adaptor.getComplex(), fmf);
    Value realSign = arith::DivFOp::create(b, real, abs, fmf);
    Value imagSign = arith::DivFOp::create(b, imag, abs, fmf);
    Value sign = complex::CreateOp::create(b, type, realSign, imagSign);
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
        complex::ReOp::create(b, loc, elementType, adaptor.getComplex());
    Value imag =
        complex::ImOp::create(b, loc, elementType, adaptor.getComplex());
    Value negOne = arith::ConstantOp::create(b, elementType,
                                             b.getFloatAttr(elementType, -1.0));

    if constexpr (std::is_same_v<Op, complex::TanOp>) {
      // tan(x+yi) = -i*tanh(-y + xi)
      std::swap(real, imag);
      real = arith::MulFOp::create(b, real, negOne, fmf);
    }

    auto cst = [&](APFloat v) {
      return arith::ConstantOp::create(b, elementType,
                                       b.getFloatAttr(elementType, v));
    };
    Value inf = cst(APFloat::getInf(floatSemantics));
    Value four = arith::ConstantOp::create(b, elementType,
                                           b.getFloatAttr(elementType, 4.0));
    Value twoReal = arith::AddFOp::create(b, real, real, fmf);
    Value negTwoReal = arith::MulFOp::create(b, negOne, twoReal, fmf);

    Value expTwoRealMinusOne = math::ExpM1Op::create(b, twoReal, fmf);
    Value expNegTwoRealMinusOne = math::ExpM1Op::create(b, negTwoReal, fmf);
    Value realNum = arith::SubFOp::create(b, expTwoRealMinusOne,
                                          expNegTwoRealMinusOne, fmf);

    Value cosImag = math::CosOp::create(b, imag, fmf);
    Value cosImagSq = arith::MulFOp::create(b, cosImag, cosImag, fmf);
    Value twoCosTwoImagPlusOne = arith::MulFOp::create(b, cosImagSq, four, fmf);
    Value sinImag = math::SinOp::create(b, imag, fmf);

    Value imagNum = arith::MulFOp::create(
        b, four, arith::MulFOp::create(b, cosImag, sinImag, fmf), fmf);

    Value expSumMinusTwo = arith::AddFOp::create(b, expTwoRealMinusOne,
                                                 expNegTwoRealMinusOne, fmf);
    Value denom =
        arith::AddFOp::create(b, expSumMinusTwo, twoCosTwoImagPlusOne, fmf);

    Value isInf = arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ,
                                        expSumMinusTwo, inf, fmf);
    Value realLimit = math::CopySignOp::create(b, negOne, real, fmf);

    Value resultReal = arith::SelectOp::create(
        b, isInf, realLimit, arith::DivFOp::create(b, realNum, denom, fmf));
    Value resultImag = arith::DivFOp::create(b, imagNum, denom, fmf);

    if (!arith::bitEnumContainsAll(fmf, arith::FastMathFlags::nnan |
                                            arith::FastMathFlags::ninf)) {
      Value absReal = math::AbsFOp::create(b, real, fmf);
      Value zero = arith::ConstantOp::create(b, elementType,
                                             b.getFloatAttr(elementType, 0.0));
      Value nan = cst(APFloat::getNaN(floatSemantics));

      Value absRealIsInf = arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ,
                                                 absReal, inf, fmf);
      Value imagIsZero =
          arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, imag, zero, fmf);
      Value absRealIsNotInf = arith::XOrIOp::create(
          b, absRealIsInf, arith::ConstantIntOp::create(b, true, /*width=*/1));

      Value imagNumIsNaN = arith::CmpFOp::create(b, arith::CmpFPredicate::UNO,
                                                 imagNum, imagNum, fmf);
      Value resultRealIsNaN =
          arith::AndIOp::create(b, imagNumIsNaN, absRealIsNotInf);
      Value resultImagIsZero = arith::OrIOp::create(
          b, imagIsZero, arith::AndIOp::create(b, absRealIsInf, imagNumIsNaN));

      resultReal = arith::SelectOp::create(b, resultRealIsNaN, nan, resultReal);
      resultImag =
          arith::SelectOp::create(b, resultImagIsZero, zero, resultImag);
    }

    if constexpr (std::is_same_v<Op, complex::TanOp>) {
      // tan(x+yi) = -i*tanh(-y + xi)
      std::swap(resultReal, resultImag);
      resultImag = arith::MulFOp::create(b, resultImag, negOne, fmf);
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
        complex::ReOp::create(rewriter, loc, elementType, adaptor.getComplex());
    Value imag =
        complex::ImOp::create(rewriter, loc, elementType, adaptor.getComplex());
    Value negImag = arith::NegFOp::create(rewriter, loc, elementType, imag);

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

  Value a = complex::ReOp::create(builder, lhs);
  Value b = complex::ImOp::create(builder, lhs);

  Value abs = complex::AbsOp::create(builder, lhs, fmf);
  Value absToC = math::PowFOp::create(builder, abs, c, fmf);

  Value negD = arith::NegFOp::create(builder, d, fmf);
  Value argLhs = math::Atan2Op::create(builder, b, a, fmf);
  Value negDArgLhs = arith::MulFOp::create(builder, negD, argLhs, fmf);
  Value expNegDArgLhs = math::ExpOp::create(builder, negDArgLhs, fmf);

  Value coeff = arith::MulFOp::create(builder, absToC, expNegDArgLhs, fmf);
  Value lnAbs = math::LogOp::create(builder, abs, fmf);
  Value cArgLhs = arith::MulFOp::create(builder, c, argLhs, fmf);
  Value dLnAbs = arith::MulFOp::create(builder, d, lnAbs, fmf);
  Value q = arith::AddFOp::create(builder, cArgLhs, dLnAbs, fmf);
  Value cosQ = math::CosOp::create(builder, q, fmf);
  Value sinQ = math::SinOp::create(builder, q, fmf);

  Value inf = arith::ConstantOp::create(
      builder, elementType,
      builder.getFloatAttr(elementType,
                           APFloat::getInf(elementType.getFloatSemantics())));
  Value zero = arith::ConstantOp::create(
      builder, elementType, builder.getFloatAttr(elementType, 0.0));
  Value one = arith::ConstantOp::create(builder, elementType,
                                        builder.getFloatAttr(elementType, 1.0));
  Value complexOne = complex::CreateOp::create(builder, type, one, zero);
  Value complexZero = complex::CreateOp::create(builder, type, zero, zero);
  Value complexInf = complex::CreateOp::create(builder, type, inf, zero);

  // Case 0:
  // d^c is 0 if d is 0 and c > 0. 0^0 is defined to be 1.0, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  Value absEqZero =
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OEQ, abs, zero, fmf);
  Value dEqZero =
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OEQ, d, zero, fmf);
  Value cEqZero =
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OEQ, c, zero, fmf);
  Value bEqZero =
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OEQ, b, zero, fmf);

  Value zeroLeC =
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OLE, zero, c, fmf);
  Value coeffCosQ = arith::MulFOp::create(builder, coeff, cosQ, fmf);
  Value coeffSinQ = arith::MulFOp::create(builder, coeff, sinQ, fmf);
  Value complexOneOrZero =
      arith::SelectOp::create(builder, cEqZero, complexOne, complexZero);
  Value coeffCosSin =
      complex::CreateOp::create(builder, type, coeffCosQ, coeffSinQ);
  Value cutoff0 = arith::SelectOp::create(
      builder,
      arith::AndIOp::create(
          builder, arith::AndIOp::create(builder, absEqZero, dEqZero), zeroLeC),
      complexOneOrZero, coeffCosSin);

  // Case 1:
  // x^0 is defined to be 1 for any x, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  Value rhsEqZero = arith::AndIOp::create(builder, cEqZero, dEqZero);
  Value cutoff1 =
      arith::SelectOp::create(builder, rhsEqZero, complexOne, cutoff0);

  // Case 2:
  // 1^(c + d*i) = 1 + 0*i
  Value lhsEqOne = arith::AndIOp::create(
      builder,
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OEQ, a, one, fmf),
      bEqZero);
  Value cutoff2 =
      arith::SelectOp::create(builder, lhsEqOne, complexOne, cutoff1);

  // Case 3:
  // inf^(c + 0*i) = inf + 0*i, c > 0
  Value lhsEqInf = arith::AndIOp::create(
      builder,
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OEQ, a, inf, fmf),
      bEqZero);
  Value rhsGt0 = arith::AndIOp::create(
      builder, dEqZero,
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OGT, c, zero, fmf));
  Value cutoff3 = arith::SelectOp::create(
      builder, arith::AndIOp::create(builder, lhsEqInf, rhsGt0), complexInf,
      cutoff2);

  // Case 4:
  // inf^(c + 0*i) = 0 + 0*i, c < 0
  Value rhsLt0 = arith::AndIOp::create(
      builder, dEqZero,
      arith::CmpFOp::create(builder, arith::CmpFPredicate::OLT, c, zero, fmf));
  Value cutoff4 = arith::SelectOp::create(
      builder, arith::AndIOp::create(builder, lhsEqInf, rhsLt0), complexZero,
      cutoff3);

  return cutoff4;
}

struct PowiOpConversion : public OpConversionPattern<complex::PowiOp> {
  using OpConversionPattern<complex::PowiOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::PowiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto type = cast<ComplexType>(op.getType());
    auto elementType = cast<FloatType>(type.getElementType());

    Value floatExponent =
        arith::SIToFPOp::create(builder, elementType, adaptor.getRhs());
    Value zero = arith::ConstantOp::create(
        builder, elementType, builder.getFloatAttr(elementType, 0.0));
    Value complexExponent =
        complex::CreateOp::create(builder, type, floatExponent, zero);

    auto pow = complex::PowOp::create(builder, type, adaptor.getLhs(),
                                      complexExponent, op.getFastmathAttr());
    rewriter.replaceOp(op, pow.getResult());
    return success();
  }
};

struct PowOpConversion : public OpConversionPattern<complex::PowOp> {
  using OpConversionPattern<complex::PowOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto type = cast<ComplexType>(adaptor.getLhs().getType());
    auto elementType = cast<FloatType>(type.getElementType());

    Value c = complex::ReOp::create(builder, elementType, adaptor.getRhs());
    Value d = complex::ImOp::create(builder, elementType, adaptor.getRhs());

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
      return arith::ConstantOp::create(b, elementType,
                                       b.getFloatAttr(elementType, v));
    };
    const auto &floatSemantics = elementType.getFloatSemantics();
    Value zero = cst(APFloat::getZero(floatSemantics));
    Value inf = cst(APFloat::getInf(floatSemantics));
    Value negHalf = arith::ConstantOp::create(
        b, elementType, b.getFloatAttr(elementType, -0.5));
    Value nan = cst(APFloat::getNaN(floatSemantics));

    Value real = complex::ReOp::create(b, elementType, adaptor.getComplex());
    Value imag = complex::ImOp::create(b, elementType, adaptor.getComplex());
    Value absRsqrt = computeAbs(real, imag, fmf, b, AbsFn::rsqrt);
    Value argArg = math::Atan2Op::create(b, imag, real, fmf);
    Value rsqrtArg = arith::MulFOp::create(b, argArg, negHalf, fmf);
    Value cos = math::CosOp::create(b, rsqrtArg, fmf);
    Value sin = math::SinOp::create(b, rsqrtArg, fmf);

    Value resultReal = arith::MulFOp::create(b, absRsqrt, cos, fmf);
    Value resultImag = arith::MulFOp::create(b, absRsqrt, sin, fmf);

    if (!arith::bitEnumContainsAll(fmf, arith::FastMathFlags::nnan |
                                            arith::FastMathFlags::ninf)) {
      Value negOne = arith::ConstantOp::create(b, elementType,
                                               b.getFloatAttr(elementType, -1));

      Value realSignedZero = math::CopySignOp::create(b, zero, real, fmf);
      Value imagSignedZero = math::CopySignOp::create(b, zero, imag, fmf);
      Value negImagSignedZero =
          arith::MulFOp::create(b, negOne, imagSignedZero, fmf);

      Value absReal = math::AbsFOp::create(b, real, fmf);
      Value absImag = math::AbsFOp::create(b, imag, fmf);

      Value absImagIsInf = arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ,
                                                 absImag, inf, fmf);
      Value realIsNan =
          arith::CmpFOp::create(b, arith::CmpFPredicate::UNO, real, real, fmf);
      Value realIsInf = arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ,
                                              absReal, inf, fmf);
      Value inIsNanInf = arith::AndIOp::create(b, absImagIsInf, realIsNan);

      Value resultIsZero = arith::OrIOp::create(b, inIsNanInf, realIsInf);

      resultReal =
          arith::SelectOp::create(b, resultIsZero, realSignedZero, resultReal);
      resultImag = arith::SelectOp::create(b, resultIsZero, negImagSignedZero,
                                           resultImag);
    }

    Value isRealZero =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, real, zero, fmf);
    Value isImagZero =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, imag, zero, fmf);
    Value isZero = arith::AndIOp::create(b, isRealZero, isImagZero);

    resultReal = arith::SelectOp::create(b, isZero, inf, resultReal);
    resultImag = arith::SelectOp::create(b, isZero, nan, resultImag);

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
        complex::ReOp::create(rewriter, loc, type, adaptor.getComplex());
    Value imag =
        complex::ImOp::create(rewriter, loc, type, adaptor.getComplex());

    rewriter.replaceOpWithNewOp<math::Atan2Op>(op, imag, real, fmf);

    return success();
  }
};

} // namespace

void mlir::populateComplexToStandardConversionPatterns(
    RewritePatternSet &patterns, complex::ComplexRangeFlags complexRange) {
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
      PowiOpConversion,
      PowOpConversion,
      RsqrtOpConversion
  >(patterns.getContext());

    patterns.add<DivOpConversion>(patterns.getContext(), complexRange);

  // clang-format on
}

namespace {
struct ConvertComplexToStandardPass
    : public impl::ConvertComplexToStandardPassBase<
          ConvertComplexToStandardPass> {
  using Base::Base;

  void runOnOperation() override;
};

void ConvertComplexToStandardPass::runOnOperation() {
  // Convert to the Standard dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  populateComplexToStandardConversionPatterns(patterns, complexRange);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, math::MathDialect>();
  target.addLegalOp<complex::CreateOp, complex::ImOp, complex::ReOp>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // namespace
