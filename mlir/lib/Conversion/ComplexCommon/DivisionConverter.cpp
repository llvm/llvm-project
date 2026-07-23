//===- DivisionConverter.cpp - Complex division conversion ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions for two different complex number division
// algorithms, the `algebraic formula` and `Smith's range reduction method`.
// These are used in two conversions: `ComplexToLLVM` and `ComplexToStandard`.
// When modifying the algorithms, both `ToLLVM` and `ToStandard` must be
// changed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexCommon/DivisionConverter.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;

void mlir::complex::convertDivToLLVMUsingAlgebraic(
    ConversionPatternRewriter &rewriter, Location loc, Value lhsRe, Value lhsIm,
    Value rhsRe, Value rhsIm, LLVM::FastmathFlagsAttr fmf, Value *resultRe,
    Value *resultIm) {
  Value rhsSqNorm = LLVM::FAddOp::create(
      rewriter, loc, LLVM::FMulOp::create(rewriter, loc, rhsRe, rhsRe, fmf),
      LLVM::FMulOp::create(rewriter, loc, rhsIm, rhsIm, fmf), fmf);

  Value realNumerator = LLVM::FAddOp::create(
      rewriter, loc, LLVM::FMulOp::create(rewriter, loc, lhsRe, rhsRe, fmf),
      LLVM::FMulOp::create(rewriter, loc, lhsIm, rhsIm, fmf), fmf);

  Value imagNumerator = LLVM::FSubOp::create(
      rewriter, loc, LLVM::FMulOp::create(rewriter, loc, lhsIm, rhsRe, fmf),
      LLVM::FMulOp::create(rewriter, loc, lhsRe, rhsIm, fmf), fmf);

  *resultRe =
      LLVM::FDivOp::create(rewriter, loc, realNumerator, rhsSqNorm, fmf);
  *resultIm =
      LLVM::FDivOp::create(rewriter, loc, imagNumerator, rhsSqNorm, fmf);
}

void mlir::complex::convertDivToStandardUsingAlgebraic(
    ConversionPatternRewriter &rewriter, Location loc, Value lhsRe, Value lhsIm,
    Value rhsRe, Value rhsIm, arith::FastMathFlagsAttr fmf, Value *resultRe,
    Value *resultIm) {
  Value rhsSqNorm = arith::AddFOp::create(
      rewriter, loc, arith::MulFOp::create(rewriter, loc, rhsRe, rhsRe, fmf),
      arith::MulFOp::create(rewriter, loc, rhsIm, rhsIm, fmf), fmf);

  Value realNumerator = arith::AddFOp::create(
      rewriter, loc, arith::MulFOp::create(rewriter, loc, lhsRe, rhsRe, fmf),
      arith::MulFOp::create(rewriter, loc, lhsIm, rhsIm, fmf), fmf);
  Value imagNumerator = arith::SubFOp::create(
      rewriter, loc, arith::MulFOp::create(rewriter, loc, lhsIm, rhsRe, fmf),
      arith::MulFOp::create(rewriter, loc, lhsRe, rhsIm, fmf), fmf);

  *resultRe =
      arith::DivFOp::create(rewriter, loc, realNumerator, rhsSqNorm, fmf);
  *resultIm =
      arith::DivFOp::create(rewriter, loc, imagNumerator, rhsSqNorm, fmf);
}

// Smith's algorithm to divide complex numbers. It is just a bit smarter
// way to compute the following algebraic formula:
//  (lhsRe + lhsIm * i) / (rhsRe + rhsIm * i)
//    = (lhsRe + lhsIm * i) (rhsRe - rhsIm * i) /
//          ((rhsRe + rhsIm * i)(rhsRe - rhsIm * i))
//    = ((lhsRe * rhsRe + lhsIm * rhsIm) +
//          (lhsIm * rhsRe - lhsRe * rhsIm) * i) / ||rhs||^2
//
// Depending on whether |rhsRe| < |rhsIm| we compute either
//   rhsRealImagRatio = rhsRe / rhsIm
//   rhsRealImagDenom = rhsIm + rhsRe * rhsRealImagRatio
//   resultRe = (lhsRe * rhsRealImagRatio + lhsIm) /
//                  rhsRealImagDenom
//   resultIm = (lhsIm * rhsRealImagRatio - lhsRe) /
//                  rhsRealImagDenom
//
// or
//
//   rhsImagRealRatio = rhsIm / rhsRe
//   rhsImagRealDenom = rhsRe + rhsIm * rhsImagRealRatio
//   resultRe = (lhsRe + lhsIm * rhsImagRealRatio) /
//                  rhsImagRealDenom
//   resultIm = (lhsIm - lhsRe * rhsImagRealRatio) /
//                  rhsImagRealDenom
//
// See https://dl.acm.org/citation.cfm?id=368661 for more details.

void mlir::complex::convertDivToLLVMUsingRangeReduction(
    ConversionPatternRewriter &rewriter, Location loc, Value lhsRe, Value lhsIm,
    Value rhsRe, Value rhsIm, LLVM::FastmathFlagsAttr fmf, Value *resultRe,
    Value *resultIm) {
  auto elementType = cast<FloatType>(rhsRe.getType());

  Value rhsRealImagRatio =
      LLVM::FDivOp::create(rewriter, loc, rhsRe, rhsIm, fmf);
  Value rhsRealImagDenom = LLVM::FAddOp::create(
      rewriter, loc, rhsIm,
      LLVM::FMulOp::create(rewriter, loc, rhsRealImagRatio, rhsRe, fmf), fmf);
  Value realNumerator1 = LLVM::FAddOp::create(
      rewriter, loc,
      LLVM::FMulOp::create(rewriter, loc, lhsRe, rhsRealImagRatio, fmf), lhsIm,
      fmf);
  Value resultReal1 = LLVM::FDivOp::create(rewriter, loc, realNumerator1,
                                           rhsRealImagDenom, fmf);
  Value imagNumerator1 = LLVM::FSubOp::create(
      rewriter, loc,
      LLVM::FMulOp::create(rewriter, loc, lhsIm, rhsRealImagRatio, fmf), lhsRe,
      fmf);
  Value resultImag1 = LLVM::FDivOp::create(rewriter, loc, imagNumerator1,
                                           rhsRealImagDenom, fmf);

  Value rhsImagRealRatio =
      LLVM::FDivOp::create(rewriter, loc, rhsIm, rhsRe, fmf);
  Value rhsImagRealDenom = LLVM::FAddOp::create(
      rewriter, loc, rhsRe,
      LLVM::FMulOp::create(rewriter, loc, rhsImagRealRatio, rhsIm, fmf), fmf);
  Value realNumerator2 = LLVM::FAddOp::create(
      rewriter, loc, lhsRe,
      LLVM::FMulOp::create(rewriter, loc, lhsIm, rhsImagRealRatio, fmf), fmf);
  Value resultReal2 = LLVM::FDivOp::create(rewriter, loc, realNumerator2,
                                           rhsImagRealDenom, fmf);
  Value imagNumerator2 = LLVM::FSubOp::create(
      rewriter, loc, lhsIm,
      LLVM::FMulOp::create(rewriter, loc, lhsRe, rhsImagRealRatio, fmf), fmf);
  Value resultImag2 = LLVM::FDivOp::create(rewriter, loc, imagNumerator2,
                                           rhsImagRealDenom, fmf);

  // Consider corner cases.
  // Case 1. Zero denominator, numerator contains at most one NaN value.
  Value zero = LLVM::ConstantOp::create(rewriter, loc, elementType,
                                        rewriter.getZeroAttr(elementType));
  Value rhsRealAbs = LLVM::FAbsOp::create(rewriter, loc, rhsRe, fmf);
  Value rhsRealIsZero = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::oeq, rhsRealAbs, zero);
  Value rhsImagAbs = LLVM::FAbsOp::create(rewriter, loc, rhsIm, fmf);
  Value rhsImagIsZero = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::oeq, rhsImagAbs, zero);
  Value lhsRealIsNotNaN = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::ord, lhsRe, zero);
  Value lhsImagIsNotNaN = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::ord, lhsIm, zero);
  Value lhsContainsNotNaNValue =
      LLVM::OrOp::create(rewriter, loc, lhsRealIsNotNaN, lhsImagIsNotNaN);
  Value resultIsInfinity = LLVM::AndOp::create(
      rewriter, loc, lhsContainsNotNaNValue,
      LLVM::AndOp::create(rewriter, loc, rhsRealIsZero, rhsImagIsZero));
  Value inf = LLVM::ConstantOp::create(
      rewriter, loc, elementType,
      rewriter.getFloatAttr(elementType,
                            APFloat::getInf(elementType.getFloatSemantics())));
  Value infWithSignOfrhsReal =
      LLVM::CopySignOp::create(rewriter, loc, inf, rhsRe);
  Value infinityResultReal =
      LLVM::FMulOp::create(rewriter, loc, infWithSignOfrhsReal, lhsRe, fmf);
  Value infinityResultImag =
      LLVM::FMulOp::create(rewriter, loc, infWithSignOfrhsReal, lhsIm, fmf);

  // Case 2. Infinite numerator, finite denominator.
  Value rhsRealFinite = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::one, rhsRealAbs, inf);
  Value rhsImagFinite = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::one, rhsImagAbs, inf);
  Value rhsFinite =
      LLVM::AndOp::create(rewriter, loc, rhsRealFinite, rhsImagFinite);
  Value lhsRealAbs = LLVM::FAbsOp::create(rewriter, loc, lhsRe, fmf);
  Value lhsRealInfinite = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::oeq, lhsRealAbs, inf);
  Value lhsImagAbs = LLVM::FAbsOp::create(rewriter, loc, lhsIm, fmf);
  Value lhsImagInfinite = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::oeq, lhsImagAbs, inf);
  Value lhsInfinite =
      LLVM::OrOp::create(rewriter, loc, lhsRealInfinite, lhsImagInfinite);
  Value infNumFiniteDenom =
      LLVM::AndOp::create(rewriter, loc, lhsInfinite, rhsFinite);
  Value one = LLVM::ConstantOp::create(rewriter, loc, elementType,
                                       rewriter.getFloatAttr(elementType, 1));
  Value lhsRealIsInfWithSign = LLVM::CopySignOp::create(
      rewriter, loc,
      LLVM::SelectOp::create(rewriter, loc, lhsRealInfinite, one, zero), lhsRe);
  Value lhsImagIsInfWithSign = LLVM::CopySignOp::create(
      rewriter, loc,
      LLVM::SelectOp::create(rewriter, loc, lhsImagInfinite, one, zero), lhsIm);
  Value lhsRealIsInfWithSignTimesrhsReal =
      LLVM::FMulOp::create(rewriter, loc, lhsRealIsInfWithSign, rhsRe, fmf);
  Value lhsImagIsInfWithSignTimesrhsImag =
      LLVM::FMulOp::create(rewriter, loc, lhsImagIsInfWithSign, rhsIm, fmf);
  Value resultReal3 = LLVM::FMulOp::create(
      rewriter, loc, inf,
      LLVM::FAddOp::create(rewriter, loc, lhsRealIsInfWithSignTimesrhsReal,
                           lhsImagIsInfWithSignTimesrhsImag, fmf),
      fmf);
  Value lhsRealIsInfWithSignTimesrhsImag =
      LLVM::FMulOp::create(rewriter, loc, lhsRealIsInfWithSign, rhsIm, fmf);
  Value lhsImagIsInfWithSignTimesrhsReal =
      LLVM::FMulOp::create(rewriter, loc, lhsImagIsInfWithSign, rhsRe, fmf);
  Value resultImag3 = LLVM::FMulOp::create(
      rewriter, loc, inf,
      LLVM::FSubOp::create(rewriter, loc, lhsImagIsInfWithSignTimesrhsReal,
                           lhsRealIsInfWithSignTimesrhsImag, fmf),
      fmf);

  // Case 3: Finite numerator, infinite denominator.
  Value lhsRealFinite = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::one, lhsRealAbs, inf);
  Value lhsImagFinite = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::one, lhsImagAbs, inf);
  Value lhsFinite =
      LLVM::AndOp::create(rewriter, loc, lhsRealFinite, lhsImagFinite);
  Value rhsRealInfinite = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::oeq, rhsRealAbs, inf);
  Value rhsImagInfinite = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::oeq, rhsImagAbs, inf);
  Value rhsInfinite =
      LLVM::OrOp::create(rewriter, loc, rhsRealInfinite, rhsImagInfinite);
  Value finiteNumInfiniteDenom =
      LLVM::AndOp::create(rewriter, loc, lhsFinite, rhsInfinite);
  Value rhsRealIsInfWithSign = LLVM::CopySignOp::create(
      rewriter, loc,
      LLVM::SelectOp::create(rewriter, loc, rhsRealInfinite, one, zero), rhsRe);
  Value rhsImagIsInfWithSign = LLVM::CopySignOp::create(
      rewriter, loc,
      LLVM::SelectOp::create(rewriter, loc, rhsImagInfinite, one, zero), rhsIm);
  Value rhsRealIsInfWithSignTimeslhsReal =
      LLVM::FMulOp::create(rewriter, loc, lhsRe, rhsRealIsInfWithSign, fmf);
  Value rhsImagIsInfWithSignTimeslhsImag =
      LLVM::FMulOp::create(rewriter, loc, lhsIm, rhsImagIsInfWithSign, fmf);
  Value resultReal4 = LLVM::FMulOp::create(
      rewriter, loc, zero,
      LLVM::FAddOp::create(rewriter, loc, rhsRealIsInfWithSignTimeslhsReal,
                           rhsImagIsInfWithSignTimeslhsImag, fmf),
      fmf);
  Value rhsRealIsInfWithSignTimeslhsImag =
      LLVM::FMulOp::create(rewriter, loc, lhsIm, rhsRealIsInfWithSign, fmf);
  Value rhsImagIsInfWithSignTimeslhsReal =
      LLVM::FMulOp::create(rewriter, loc, lhsRe, rhsImagIsInfWithSign, fmf);
  Value resultImag4 = LLVM::FMulOp::create(
      rewriter, loc, zero,
      LLVM::FSubOp::create(rewriter, loc, rhsRealIsInfWithSignTimeslhsImag,
                           rhsImagIsInfWithSignTimeslhsReal, fmf),
      fmf);

  Value realAbsSmallerThanImagAbs = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::olt, rhsRealAbs, rhsImagAbs);
  Value resultReal5 = LLVM::SelectOp::create(
      rewriter, loc, realAbsSmallerThanImagAbs, resultReal1, resultReal2);
  Value resultImag5 = LLVM::SelectOp::create(
      rewriter, loc, realAbsSmallerThanImagAbs, resultImag1, resultImag2);
  Value resultRealSpecialCase3 = LLVM::SelectOp::create(
      rewriter, loc, finiteNumInfiniteDenom, resultReal4, resultReal5);
  Value resultImagSpecialCase3 = LLVM::SelectOp::create(
      rewriter, loc, finiteNumInfiniteDenom, resultImag4, resultImag5);
  Value resultRealSpecialCase2 = LLVM::SelectOp::create(
      rewriter, loc, infNumFiniteDenom, resultReal3, resultRealSpecialCase3);
  Value resultImagSpecialCase2 = LLVM::SelectOp::create(
      rewriter, loc, infNumFiniteDenom, resultImag3, resultImagSpecialCase3);
  Value resultRealSpecialCase1 =
      LLVM::SelectOp::create(rewriter, loc, resultIsInfinity,
                             infinityResultReal, resultRealSpecialCase2);
  Value resultImagSpecialCase1 =
      LLVM::SelectOp::create(rewriter, loc, resultIsInfinity,
                             infinityResultImag, resultImagSpecialCase2);

  Value resultRealIsNaN = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::uno, resultReal5, zero);
  Value resultImagIsNaN = LLVM::FCmpOp::create(
      rewriter, loc, LLVM::FCmpPredicate::uno, resultImag5, zero);
  Value resultIsNaN =
      LLVM::AndOp::create(rewriter, loc, resultRealIsNaN, resultImagIsNaN);

  *resultRe = LLVM::SelectOp::create(rewriter, loc, resultIsNaN,
                                     resultRealSpecialCase1, resultReal5);
  *resultIm = LLVM::SelectOp::create(rewriter, loc, resultIsNaN,
                                     resultImagSpecialCase1, resultImag5);
}

void mlir::complex::convertDivToStandardUsingRangeReduction(
    ConversionPatternRewriter &rewriter, Location loc, Value lhsRe, Value lhsIm,
    Value rhsRe, Value rhsIm, arith::FastMathFlagsAttr fmf, Value *resultRe,
    Value *resultIm) {
  auto elementType = cast<FloatType>(rhsRe.getType());

  Value rhsRealImagRatio =
      arith::DivFOp::create(rewriter, loc, rhsRe, rhsIm, fmf);
  Value rhsRealImagDenom = arith::AddFOp::create(
      rewriter, loc, rhsIm,
      arith::MulFOp::create(rewriter, loc, rhsRealImagRatio, rhsRe, fmf), fmf);
  Value realNumerator1 = arith::AddFOp::create(
      rewriter, loc,
      arith::MulFOp::create(rewriter, loc, lhsRe, rhsRealImagRatio, fmf), lhsIm,
      fmf);
  Value resultReal1 = arith::DivFOp::create(rewriter, loc, realNumerator1,
                                            rhsRealImagDenom, fmf);
  Value imagNumerator1 = arith::SubFOp::create(
      rewriter, loc,
      arith::MulFOp::create(rewriter, loc, lhsIm, rhsRealImagRatio, fmf), lhsRe,
      fmf);
  Value resultImag1 = arith::DivFOp::create(rewriter, loc, imagNumerator1,
                                            rhsRealImagDenom, fmf);

  Value rhsImagRealRatio =
      arith::DivFOp::create(rewriter, loc, rhsIm, rhsRe, fmf);
  Value rhsImagRealDenom = arith::AddFOp::create(
      rewriter, loc, rhsRe,
      arith::MulFOp::create(rewriter, loc, rhsImagRealRatio, rhsIm, fmf), fmf);
  Value realNumerator2 = arith::AddFOp::create(
      rewriter, loc, lhsRe,
      arith::MulFOp::create(rewriter, loc, lhsIm, rhsImagRealRatio, fmf), fmf);
  Value resultReal2 = arith::DivFOp::create(rewriter, loc, realNumerator2,
                                            rhsImagRealDenom, fmf);
  Value imagNumerator2 = arith::SubFOp::create(
      rewriter, loc, lhsIm,
      arith::MulFOp::create(rewriter, loc, lhsRe, rhsImagRealRatio, fmf), fmf);
  Value resultImag2 = arith::DivFOp::create(rewriter, loc, imagNumerator2,
                                            rhsImagRealDenom, fmf);

  // Consider corner cases.
  // Case 1. Zero denominator, numerator contains at most one NaN value.
  Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                         rewriter.getZeroAttr(elementType));
  Value rhsRealAbs = math::AbsFOp::create(rewriter, loc, rhsRe, fmf);
  Value rhsRealIsZero = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OEQ, rhsRealAbs, zero);
  Value rhsImagAbs = math::AbsFOp::create(rewriter, loc, rhsIm, fmf);
  Value rhsImagIsZero = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OEQ, rhsImagAbs, zero);
  Value lhsRealIsNotNaN = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::ORD, lhsRe, zero);
  Value lhsImagIsNotNaN = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::ORD, lhsIm, zero);
  Value lhsContainsNotNaNValue =
      arith::OrIOp::create(rewriter, loc, lhsRealIsNotNaN, lhsImagIsNotNaN);
  Value resultIsInfinity = arith::AndIOp::create(
      rewriter, loc, lhsContainsNotNaNValue,
      arith::AndIOp::create(rewriter, loc, rhsRealIsZero, rhsImagIsZero));
  Value inf = arith::ConstantOp::create(
      rewriter, loc, elementType,
      rewriter.getFloatAttr(elementType,
                            APFloat::getInf(elementType.getFloatSemantics())));
  Value infWithSignOfRhsReal =
      math::CopySignOp::create(rewriter, loc, inf, rhsRe);
  Value infinityResultReal =
      arith::MulFOp::create(rewriter, loc, infWithSignOfRhsReal, lhsRe, fmf);
  Value infinityResultImag =
      arith::MulFOp::create(rewriter, loc, infWithSignOfRhsReal, lhsIm, fmf);

  // Case 2. Infinite numerator, finite denominator.
  Value rhsRealFinite = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::ONE, rhsRealAbs, inf);
  Value rhsImagFinite = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::ONE, rhsImagAbs, inf);
  Value rhsFinite =
      arith::AndIOp::create(rewriter, loc, rhsRealFinite, rhsImagFinite);
  Value lhsRealAbs = math::AbsFOp::create(rewriter, loc, lhsRe, fmf);
  Value lhsRealInfinite = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OEQ, lhsRealAbs, inf);
  Value lhsImagAbs = math::AbsFOp::create(rewriter, loc, lhsIm, fmf);
  Value lhsImagInfinite = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OEQ, lhsImagAbs, inf);
  Value lhsInfinite =
      arith::OrIOp::create(rewriter, loc, lhsRealInfinite, lhsImagInfinite);
  Value infNumFiniteDenom =
      arith::AndIOp::create(rewriter, loc, lhsInfinite, rhsFinite);
  Value one = arith::ConstantOp::create(rewriter, loc, elementType,
                                        rewriter.getFloatAttr(elementType, 1));
  Value lhsRealIsInfWithSign = math::CopySignOp::create(
      rewriter, loc,
      arith::SelectOp::create(rewriter, loc, lhsRealInfinite, one, zero),
      lhsRe);
  Value lhsImagIsInfWithSign = math::CopySignOp::create(
      rewriter, loc,
      arith::SelectOp::create(rewriter, loc, lhsImagInfinite, one, zero),
      lhsIm);
  Value lhsRealIsInfWithSignTimesRhsReal =
      arith::MulFOp::create(rewriter, loc, lhsRealIsInfWithSign, rhsRe, fmf);
  Value lhsImagIsInfWithSignTimesRhsImag =
      arith::MulFOp::create(rewriter, loc, lhsImagIsInfWithSign, rhsIm, fmf);
  Value resultReal3 = arith::MulFOp::create(
      rewriter, loc, inf,
      arith::AddFOp::create(rewriter, loc, lhsRealIsInfWithSignTimesRhsReal,
                            lhsImagIsInfWithSignTimesRhsImag, fmf),
      fmf);
  Value lhsRealIsInfWithSignTimesRhsImag =
      arith::MulFOp::create(rewriter, loc, lhsRealIsInfWithSign, rhsIm, fmf);
  Value lhsImagIsInfWithSignTimesRhsReal =
      arith::MulFOp::create(rewriter, loc, lhsImagIsInfWithSign, rhsRe, fmf);
  Value resultImag3 = arith::MulFOp::create(
      rewriter, loc, inf,
      arith::SubFOp::create(rewriter, loc, lhsImagIsInfWithSignTimesRhsReal,
                            lhsRealIsInfWithSignTimesRhsImag, fmf),
      fmf);

  // Case 3: Finite numerator, infinite denominator.
  Value lhsRealFinite = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::ONE, lhsRealAbs, inf);
  Value lhsImagFinite = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::ONE, lhsImagAbs, inf);
  Value lhsFinite =
      arith::AndIOp::create(rewriter, loc, lhsRealFinite, lhsImagFinite);
  Value rhsRealInfinite = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OEQ, rhsRealAbs, inf);
  Value rhsImagInfinite = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OEQ, rhsImagAbs, inf);
  Value rhsInfinite =
      arith::OrIOp::create(rewriter, loc, rhsRealInfinite, rhsImagInfinite);
  Value finiteNumInfiniteDenom =
      arith::AndIOp::create(rewriter, loc, lhsFinite, rhsInfinite);
  Value rhsRealIsInfWithSign = math::CopySignOp::create(
      rewriter, loc,
      arith::SelectOp::create(rewriter, loc, rhsRealInfinite, one, zero),
      rhsRe);
  Value rhsImagIsInfWithSign = math::CopySignOp::create(
      rewriter, loc,
      arith::SelectOp::create(rewriter, loc, rhsImagInfinite, one, zero),
      rhsIm);
  Value rhsRealIsInfWithSignTimesLhsReal =
      arith::MulFOp::create(rewriter, loc, lhsRe, rhsRealIsInfWithSign, fmf);
  Value rhsImagIsInfWithSignTimesLhsImag =
      arith::MulFOp::create(rewriter, loc, lhsIm, rhsImagIsInfWithSign, fmf);
  Value resultReal4 = arith::MulFOp::create(
      rewriter, loc, zero,
      arith::AddFOp::create(rewriter, loc, rhsRealIsInfWithSignTimesLhsReal,
                            rhsImagIsInfWithSignTimesLhsImag, fmf),
      fmf);
  Value rhsRealIsInfWithSignTimesLhsImag =
      arith::MulFOp::create(rewriter, loc, lhsIm, rhsRealIsInfWithSign, fmf);
  Value rhsImagIsInfWithSignTimesLhsReal =
      arith::MulFOp::create(rewriter, loc, lhsRe, rhsImagIsInfWithSign, fmf);
  Value resultImag4 = arith::MulFOp::create(
      rewriter, loc, zero,
      arith::SubFOp::create(rewriter, loc, rhsRealIsInfWithSignTimesLhsImag,
                            rhsImagIsInfWithSignTimesLhsReal, fmf),
      fmf);

  Value realAbsSmallerThanImagAbs = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OLT, rhsRealAbs, rhsImagAbs);
  Value resultReal5 = arith::SelectOp::create(
      rewriter, loc, realAbsSmallerThanImagAbs, resultReal1, resultReal2);
  Value resultImag5 = arith::SelectOp::create(
      rewriter, loc, realAbsSmallerThanImagAbs, resultImag1, resultImag2);
  Value resultRealSpecialCase3 = arith::SelectOp::create(
      rewriter, loc, finiteNumInfiniteDenom, resultReal4, resultReal5);
  Value resultImagSpecialCase3 = arith::SelectOp::create(
      rewriter, loc, finiteNumInfiniteDenom, resultImag4, resultImag5);
  Value resultRealSpecialCase2 = arith::SelectOp::create(
      rewriter, loc, infNumFiniteDenom, resultReal3, resultRealSpecialCase3);
  Value resultImagSpecialCase2 = arith::SelectOp::create(
      rewriter, loc, infNumFiniteDenom, resultImag3, resultImagSpecialCase3);
  Value resultRealSpecialCase1 =
      arith::SelectOp::create(rewriter, loc, resultIsInfinity,
                              infinityResultReal, resultRealSpecialCase2);
  Value resultImagSpecialCase1 =
      arith::SelectOp::create(rewriter, loc, resultIsInfinity,
                              infinityResultImag, resultImagSpecialCase2);

  Value resultRealIsNaN = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::UNO, resultReal5, zero);
  Value resultImagIsNaN = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::UNO, resultImag5, zero);
  Value resultIsNaN =
      arith::AndIOp::create(rewriter, loc, resultRealIsNaN, resultImagIsNaN);

  *resultRe = arith::SelectOp::create(rewriter, loc, resultIsNaN,
                                      resultRealSpecialCase1, resultReal5);
  *resultIm = arith::SelectOp::create(rewriter, loc, resultIsNaN,
                                      resultImagSpecialCase1, resultImag5);
}
