//===- ComplexToStandard.cpp - conversion from Complex to Standard dialect ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"

#include <memory>
#include <type_traits>

#include "../PassDetail.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct AbsOpConversion : public OpConversionPattern<complex::AbsOp> {
  using OpConversionPattern<complex::AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AbsOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::AbsOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    auto type = op.getType();

    Value real =
        rewriter.create<complex::ReOp>(loc, type, transformed.complex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, type, transformed.complex());
    Value realSqr = rewriter.create<MulFOp>(loc, real, real);
    Value imagSqr = rewriter.create<MulFOp>(loc, imag, imag);
    Value sqNorm = rewriter.create<AddFOp>(loc, realSqr, imagSqr);

    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, sqNorm);
    return success();
  }
};

template <typename ComparisonOp, CmpFPredicate p>
struct ComparisonOpConversion : public OpConversionPattern<ComparisonOp> {
  using OpConversionPattern<ComparisonOp>::OpConversionPattern;
  using ResultCombiner =
      std::conditional_t<std::is_same<ComparisonOp, complex::EqualOp>::value,
                         AndOp, OrOp>;

  LogicalResult
  matchAndRewrite(ComparisonOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename ComparisonOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    auto type = transformed.lhs()
                    .getType()
                    .template cast<ComplexType>()
                    .getElementType();

    Value realLhs =
        rewriter.create<complex::ReOp>(loc, type, transformed.lhs());
    Value imagLhs =
        rewriter.create<complex::ImOp>(loc, type, transformed.lhs());
    Value realRhs =
        rewriter.create<complex::ReOp>(loc, type, transformed.rhs());
    Value imagRhs =
        rewriter.create<complex::ImOp>(loc, type, transformed.rhs());
    Value realComparison = rewriter.create<CmpFOp>(loc, p, realLhs, realRhs);
    Value imagComparison = rewriter.create<CmpFOp>(loc, p, imagLhs, imagRhs);

    rewriter.replaceOpWithNewOp<ResultCombiner>(op, realComparison,
                                                imagComparison);
    return success();
  }
};

struct DivOpConversion : public OpConversionPattern<complex::DivOp> {
  using OpConversionPattern<complex::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::DivOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::DivOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    auto type = transformed.lhs().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();

    Value lhsReal =
        rewriter.create<complex::ReOp>(loc, elementType, transformed.lhs());
    Value lhsImag =
        rewriter.create<complex::ImOp>(loc, elementType, transformed.lhs());
    Value rhsReal =
        rewriter.create<complex::ReOp>(loc, elementType, transformed.rhs());
    Value rhsImag =
        rewriter.create<complex::ImOp>(loc, elementType, transformed.rhs());

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
    Value rhsRealImagRatio = rewriter.create<DivFOp>(loc, rhsReal, rhsImag);
    Value rhsRealImagDenom = rewriter.create<AddFOp>(
        loc, rhsImag, rewriter.create<MulFOp>(loc, rhsRealImagRatio, rhsReal));
    Value realNumerator1 = rewriter.create<AddFOp>(
        loc, rewriter.create<MulFOp>(loc, lhsReal, rhsRealImagRatio), lhsImag);
    Value resultReal1 =
        rewriter.create<DivFOp>(loc, realNumerator1, rhsRealImagDenom);
    Value imagNumerator1 = rewriter.create<SubFOp>(
        loc, rewriter.create<MulFOp>(loc, lhsImag, rhsRealImagRatio), lhsReal);
    Value resultImag1 =
        rewriter.create<DivFOp>(loc, imagNumerator1, rhsRealImagDenom);

    Value rhsImagRealRatio = rewriter.create<DivFOp>(loc, rhsImag, rhsReal);
    Value rhsImagRealDenom = rewriter.create<AddFOp>(
        loc, rhsReal, rewriter.create<MulFOp>(loc, rhsImagRealRatio, rhsImag));
    Value realNumerator2 = rewriter.create<AddFOp>(
        loc, lhsReal, rewriter.create<MulFOp>(loc, lhsImag, rhsImagRealRatio));
    Value resultReal2 =
        rewriter.create<DivFOp>(loc, realNumerator2, rhsImagRealDenom);
    Value imagNumerator2 = rewriter.create<SubFOp>(
        loc, lhsImag, rewriter.create<MulFOp>(loc, lhsReal, rhsImagRealRatio));
    Value resultImag2 =
        rewriter.create<DivFOp>(loc, imagNumerator2, rhsImagRealDenom);

    // Consider corner cases.
    // Case 1. Zero denominator, numerator contains at most one NaN value.
    Value zero = rewriter.create<ConstantOp>(loc, elementType,
                                             rewriter.getZeroAttr(elementType));
    Value rhsRealAbs = rewriter.create<AbsFOp>(loc, rhsReal);
    Value rhsRealIsZero =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, rhsRealAbs, zero);
    Value rhsImagAbs = rewriter.create<AbsFOp>(loc, rhsImag);
    Value rhsImagIsZero =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, rhsImagAbs, zero);
    Value lhsRealIsNotNaN =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::ORD, lhsReal, zero);
    Value lhsImagIsNotNaN =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::ORD, lhsImag, zero);
    Value lhsContainsNotNaNValue =
        rewriter.create<OrOp>(loc, lhsRealIsNotNaN, lhsImagIsNotNaN);
    Value resultIsInfinity = rewriter.create<AndOp>(
        loc, lhsContainsNotNaNValue,
        rewriter.create<AndOp>(loc, rhsRealIsZero, rhsImagIsZero));
    Value inf = rewriter.create<ConstantOp>(
        loc, elementType,
        rewriter.getFloatAttr(
            elementType, APFloat::getInf(elementType.getFloatSemantics())));
    Value infWithSignOfRhsReal = rewriter.create<CopySignOp>(loc, inf, rhsReal);
    Value infinityResultReal =
        rewriter.create<MulFOp>(loc, infWithSignOfRhsReal, lhsReal);
    Value infinityResultImag =
        rewriter.create<MulFOp>(loc, infWithSignOfRhsReal, lhsImag);

    // Case 2. Infinite numerator, finite denominator.
    Value rhsRealFinite =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::ONE, rhsRealAbs, inf);
    Value rhsImagFinite =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::ONE, rhsImagAbs, inf);
    Value rhsFinite = rewriter.create<AndOp>(loc, rhsRealFinite, rhsImagFinite);
    Value lhsRealAbs = rewriter.create<AbsFOp>(loc, lhsReal);
    Value lhsRealInfinite =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, lhsRealAbs, inf);
    Value lhsImagAbs = rewriter.create<AbsFOp>(loc, lhsImag);
    Value lhsImagInfinite =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, lhsImagAbs, inf);
    Value lhsInfinite =
        rewriter.create<OrOp>(loc, lhsRealInfinite, lhsImagInfinite);
    Value infNumFiniteDenom =
        rewriter.create<AndOp>(loc, lhsInfinite, rhsFinite);
    Value one = rewriter.create<ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1));
    Value lhsRealIsInfWithSign = rewriter.create<CopySignOp>(
        loc, rewriter.create<SelectOp>(loc, lhsRealInfinite, one, zero),
        lhsReal);
    Value lhsImagIsInfWithSign = rewriter.create<CopySignOp>(
        loc, rewriter.create<SelectOp>(loc, lhsImagInfinite, one, zero),
        lhsImag);
    Value lhsRealIsInfWithSignTimesRhsReal =
        rewriter.create<MulFOp>(loc, lhsRealIsInfWithSign, rhsReal);
    Value lhsImagIsInfWithSignTimesRhsImag =
        rewriter.create<MulFOp>(loc, lhsImagIsInfWithSign, rhsImag);
    Value resultReal3 = rewriter.create<MulFOp>(
        loc, inf,
        rewriter.create<AddFOp>(loc, lhsRealIsInfWithSignTimesRhsReal,
                                lhsImagIsInfWithSignTimesRhsImag));
    Value lhsRealIsInfWithSignTimesRhsImag =
        rewriter.create<MulFOp>(loc, lhsRealIsInfWithSign, rhsImag);
    Value lhsImagIsInfWithSignTimesRhsReal =
        rewriter.create<MulFOp>(loc, lhsImagIsInfWithSign, rhsReal);
    Value resultImag3 = rewriter.create<MulFOp>(
        loc, inf,
        rewriter.create<SubFOp>(loc, lhsImagIsInfWithSignTimesRhsReal,
                                lhsRealIsInfWithSignTimesRhsImag));

    // Case 3: Finite numerator, infinite denominator.
    Value lhsRealFinite =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::ONE, lhsRealAbs, inf);
    Value lhsImagFinite =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::ONE, lhsImagAbs, inf);
    Value lhsFinite = rewriter.create<AndOp>(loc, lhsRealFinite, lhsImagFinite);
    Value rhsRealInfinite =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, rhsRealAbs, inf);
    Value rhsImagInfinite =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, rhsImagAbs, inf);
    Value rhsInfinite =
        rewriter.create<OrOp>(loc, rhsRealInfinite, rhsImagInfinite);
    Value finiteNumInfiniteDenom =
        rewriter.create<AndOp>(loc, lhsFinite, rhsInfinite);
    Value rhsRealIsInfWithSign = rewriter.create<CopySignOp>(
        loc, rewriter.create<SelectOp>(loc, rhsRealInfinite, one, zero),
        rhsReal);
    Value rhsImagIsInfWithSign = rewriter.create<CopySignOp>(
        loc, rewriter.create<SelectOp>(loc, rhsImagInfinite, one, zero),
        rhsImag);
    Value rhsRealIsInfWithSignTimesLhsReal =
        rewriter.create<MulFOp>(loc, lhsReal, rhsRealIsInfWithSign);
    Value rhsImagIsInfWithSignTimesLhsImag =
        rewriter.create<MulFOp>(loc, lhsImag, rhsImagIsInfWithSign);
    Value resultReal4 = rewriter.create<MulFOp>(
        loc, zero,
        rewriter.create<AddFOp>(loc, rhsRealIsInfWithSignTimesLhsReal,
                                rhsImagIsInfWithSignTimesLhsImag));
    Value rhsRealIsInfWithSignTimesLhsImag =
        rewriter.create<MulFOp>(loc, lhsImag, rhsRealIsInfWithSign);
    Value rhsImagIsInfWithSignTimesLhsReal =
        rewriter.create<MulFOp>(loc, lhsReal, rhsImagIsInfWithSign);
    Value resultImag4 = rewriter.create<MulFOp>(
        loc, zero,
        rewriter.create<SubFOp>(loc, rhsRealIsInfWithSignTimesLhsImag,
                                rhsImagIsInfWithSignTimesLhsReal));

    Value realAbsSmallerThanImagAbs = rewriter.create<CmpFOp>(
        loc, CmpFPredicate::OLT, rhsRealAbs, rhsImagAbs);
    Value resultReal = rewriter.create<SelectOp>(loc, realAbsSmallerThanImagAbs,
                                                 resultReal1, resultReal2);
    Value resultImag = rewriter.create<SelectOp>(loc, realAbsSmallerThanImagAbs,
                                                 resultImag1, resultImag2);
    Value resultRealSpecialCase3 = rewriter.create<SelectOp>(
        loc, finiteNumInfiniteDenom, resultReal4, resultReal);
    Value resultImagSpecialCase3 = rewriter.create<SelectOp>(
        loc, finiteNumInfiniteDenom, resultImag4, resultImag);
    Value resultRealSpecialCase2 = rewriter.create<SelectOp>(
        loc, infNumFiniteDenom, resultReal3, resultRealSpecialCase3);
    Value resultImagSpecialCase2 = rewriter.create<SelectOp>(
        loc, infNumFiniteDenom, resultImag3, resultImagSpecialCase3);
    Value resultRealSpecialCase1 = rewriter.create<SelectOp>(
        loc, resultIsInfinity, infinityResultReal, resultRealSpecialCase2);
    Value resultImagSpecialCase1 = rewriter.create<SelectOp>(
        loc, resultIsInfinity, infinityResultImag, resultImagSpecialCase2);

    Value resultRealIsNaN =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::UNO, resultReal, zero);
    Value resultImagIsNaN =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::UNO, resultImag, zero);
    Value resultIsNaN =
        rewriter.create<AndOp>(loc, resultRealIsNaN, resultImagIsNaN);
    Value resultRealWithSpecialCases = rewriter.create<SelectOp>(
        loc, resultIsNaN, resultRealSpecialCase1, resultReal);
    Value resultImagWithSpecialCases = rewriter.create<SelectOp>(
        loc, resultIsNaN, resultImagSpecialCase1, resultImag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(
        op, type, resultRealWithSpecialCases, resultImagWithSpecialCases);
    return success();
  }
};

struct ExpOpConversion : public OpConversionPattern<complex::ExpOp> {
  using OpConversionPattern<complex::ExpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::ExpOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::ExpOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    auto type = transformed.complex().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();

    Value real =
        rewriter.create<complex::ReOp>(loc, elementType, transformed.complex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, elementType, transformed.complex());
    Value expReal = rewriter.create<math::ExpOp>(loc, real);
    Value cosImag = rewriter.create<math::CosOp>(loc, imag);
    Value resultReal = rewriter.create<MulFOp>(loc, expReal, cosImag);
    Value sinImag = rewriter.create<math::SinOp>(loc, imag);
    Value resultImag = rewriter.create<MulFOp>(loc, expReal, sinImag);

    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, resultReal,
                                                   resultImag);
    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<complex::NegOp> {
  using OpConversionPattern<complex::NegOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::NegOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::NegOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    auto type = transformed.complex().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();

    Value real =
        rewriter.create<complex::ReOp>(loc, elementType, transformed.complex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, elementType, transformed.complex());
    Value negReal = rewriter.create<NegFOp>(loc, real);
    Value negImag = rewriter.create<NegFOp>(loc, imag);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(op, type, negReal, negImag);
    return success();
  }
};

struct SignOpConversion : public OpConversionPattern<complex::SignOp> {
  using OpConversionPattern<complex::SignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::SignOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::SignOp::Adaptor transformed(operands);
    auto type = transformed.complex().getType().cast<ComplexType>();
    auto elementType = type.getElementType().cast<FloatType>();
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value real = b.create<complex::ReOp>(elementType, transformed.complex());
    Value imag = b.create<complex::ImOp>(elementType, transformed.complex());
    Value zero = b.create<ConstantOp>(elementType, b.getZeroAttr(elementType));
    Value realIsZero = b.create<CmpFOp>(CmpFPredicate::OEQ, real, zero);
    Value imagIsZero = b.create<CmpFOp>(CmpFPredicate::OEQ, imag, zero);
    Value isZero = b.create<AndOp>(realIsZero, imagIsZero);
    auto abs = b.create<complex::AbsOp>(elementType, transformed.complex());
    Value realSign = b.create<DivFOp>(real, abs);
    Value imagSign = b.create<DivFOp>(imag, abs);
    Value sign = b.create<complex::CreateOp>(type, realSign, imagSign);
    rewriter.replaceOpWithNewOp<SelectOp>(op, isZero, transformed.complex(),
                                          sign);
    return success();
  }
};
} // namespace

void mlir::populateComplexToStandardConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AbsOpConversion,
      ComparisonOpConversion<complex::EqualOp, CmpFPredicate::OEQ>,
      ComparisonOpConversion<complex::NotEqualOp, CmpFPredicate::UNE>,
      DivOpConversion,
      ExpOpConversion,
      NegOpConversion,
      SignOpConversion>(patterns.getContext());
  // clang-format on
}

namespace {
struct ConvertComplexToStandardPass
    : public ConvertComplexToStandardBase<ConvertComplexToStandardPass> {
  void runOnFunction() override;
};

void ConvertComplexToStandardPass::runOnFunction() {
  auto function = getFunction();

  // Convert to the Standard dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  populateComplexToStandardConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect, math::MathDialect,
                         complex::ComplexDialect>();
  target.addIllegalOp<complex::AbsOp, complex::DivOp, complex::EqualOp,
                      complex::ExpOp, complex::NotEqualOp, complex::NegOp,
                      complex::SignOp>();
  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertComplexToStandardPass() {
  return std::make_unique<ConvertComplexToStandardPass>();
}
