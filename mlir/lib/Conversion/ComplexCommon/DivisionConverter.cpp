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
  Value rhsSqNorm = rewriter.create<LLVM::FAddOp>(
      loc, rewriter.create<LLVM::FMulOp>(loc, rhsRe, rhsRe, fmf),
      rewriter.create<LLVM::FMulOp>(loc, rhsIm, rhsIm, fmf), fmf);

  Value realNumerator = rewriter.create<LLVM::FAddOp>(
      loc, rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsRe, fmf),
      rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsIm, fmf), fmf);

  Value imagNumerator = rewriter.create<LLVM::FSubOp>(
      loc, rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsRe, fmf),
      rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsIm, fmf), fmf);

  *resultRe = rewriter.create<LLVM::FDivOp>(loc, realNumerator, rhsSqNorm, fmf);
  *resultIm = rewriter.create<LLVM::FDivOp>(loc, imagNumerator, rhsSqNorm, fmf);
}

void mlir::complex::convertDivToStandardUsingAlgebraic(
    ConversionPatternRewriter &rewriter, Location loc, Value lhsRe, Value lhsIm,
    Value rhsRe, Value rhsIm, arith::FastMathFlagsAttr fmf, Value *resultRe,
    Value *resultIm) {
  Value rhsSqNorm = rewriter.create<arith::AddFOp>(
      loc, rewriter.create<arith::MulFOp>(loc, rhsRe, rhsRe, fmf),
      rewriter.create<arith::MulFOp>(loc, rhsIm, rhsIm, fmf), fmf);

  Value realNumerator = rewriter.create<arith::AddFOp>(
      loc, rewriter.create<arith::MulFOp>(loc, lhsRe, rhsRe, fmf),
      rewriter.create<arith::MulFOp>(loc, lhsIm, rhsIm, fmf), fmf);
  Value imagNumerator = rewriter.create<arith::SubFOp>(
      loc, rewriter.create<arith::MulFOp>(loc, lhsIm, rhsRe, fmf),
      rewriter.create<arith::MulFOp>(loc, lhsRe, rhsIm, fmf), fmf);

  *resultRe =
      rewriter.create<arith::DivFOp>(loc, realNumerator, rhsSqNorm, fmf);
  *resultIm =
      rewriter.create<arith::DivFOp>(loc, imagNumerator, rhsSqNorm, fmf);
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
      rewriter.create<LLVM::FDivOp>(loc, rhsRe, rhsIm, fmf);
  Value rhsRealImagDenom = rewriter.create<LLVM::FAddOp>(
      loc, rhsIm,
      rewriter.create<LLVM::FMulOp>(loc, rhsRealImagRatio, rhsRe, fmf), fmf);
  Value realNumerator1 = rewriter.create<LLVM::FAddOp>(
      loc, rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsRealImagRatio, fmf),
      lhsIm, fmf);
  Value resultReal1 =
      rewriter.create<LLVM::FDivOp>(loc, realNumerator1, rhsRealImagDenom, fmf);
  Value imagNumerator1 = rewriter.create<LLVM::FSubOp>(
      loc, rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsRealImagRatio, fmf),
      lhsRe, fmf);
  Value resultImag1 =
      rewriter.create<LLVM::FDivOp>(loc, imagNumerator1, rhsRealImagDenom, fmf);

  Value rhsImagRealRatio =
      rewriter.create<LLVM::FDivOp>(loc, rhsIm, rhsRe, fmf);
  Value rhsImagRealDenom = rewriter.create<LLVM::FAddOp>(
      loc, rhsRe,
      rewriter.create<LLVM::FMulOp>(loc, rhsImagRealRatio, rhsIm, fmf), fmf);
  Value realNumerator2 = rewriter.create<LLVM::FAddOp>(
      loc, lhsRe,
      rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsImagRealRatio, fmf), fmf);
  Value resultReal2 =
      rewriter.create<LLVM::FDivOp>(loc, realNumerator2, rhsImagRealDenom, fmf);
  Value imagNumerator2 = rewriter.create<LLVM::FSubOp>(
      loc, lhsIm,
      rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsImagRealRatio, fmf), fmf);
  Value resultImag2 =
      rewriter.create<LLVM::FDivOp>(loc, imagNumerator2, rhsImagRealDenom, fmf);

  // Consider corner cases.
  // Case 1. Zero denominator, numerator contains at most one NaN value.
  Value zero = rewriter.create<LLVM::ConstantOp>(
      loc, elementType, rewriter.getZeroAttr(elementType));
  Value rhsRealAbs = rewriter.create<LLVM::FAbsOp>(loc, rhsRe, fmf);
  Value rhsRealIsZero = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::oeq, rhsRealAbs, zero);
  Value rhsImagAbs = rewriter.create<LLVM::FAbsOp>(loc, rhsIm, fmf);
  Value rhsImagIsZero = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::oeq, rhsImagAbs, zero);
  Value lhsRealIsNotNaN =
      rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::ord, lhsRe, zero);
  Value lhsImagIsNotNaN =
      rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::ord, lhsIm, zero);
  Value lhsContainsNotNaNValue =
      rewriter.create<LLVM::OrOp>(loc, lhsRealIsNotNaN, lhsImagIsNotNaN);
  Value resultIsInfinity = rewriter.create<LLVM::AndOp>(
      loc, lhsContainsNotNaNValue,
      rewriter.create<LLVM::AndOp>(loc, rhsRealIsZero, rhsImagIsZero));
  Value inf = rewriter.create<LLVM::ConstantOp>(
      loc, elementType,
      rewriter.getFloatAttr(elementType,
                            APFloat::getInf(elementType.getFloatSemantics())));
  Value infWithSignOfrhsReal =
      rewriter.create<LLVM::CopySignOp>(loc, inf, rhsRe);
  Value infinityResultReal =
      rewriter.create<LLVM::FMulOp>(loc, infWithSignOfrhsReal, lhsRe, fmf);
  Value infinityResultImag =
      rewriter.create<LLVM::FMulOp>(loc, infWithSignOfrhsReal, lhsIm, fmf);

  // Case 2. Infinite numerator, finite denominator.
  Value rhsRealFinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::one, rhsRealAbs, inf);
  Value rhsImagFinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::one, rhsImagAbs, inf);
  Value rhsFinite =
      rewriter.create<LLVM::AndOp>(loc, rhsRealFinite, rhsImagFinite);
  Value lhsRealAbs = rewriter.create<LLVM::FAbsOp>(loc, lhsRe, fmf);
  Value lhsRealInfinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::oeq, lhsRealAbs, inf);
  Value lhsImagAbs = rewriter.create<LLVM::FAbsOp>(loc, lhsIm, fmf);
  Value lhsImagInfinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::oeq, lhsImagAbs, inf);
  Value lhsInfinite =
      rewriter.create<LLVM::OrOp>(loc, lhsRealInfinite, lhsImagInfinite);
  Value infNumFiniteDenom =
      rewriter.create<LLVM::AndOp>(loc, lhsInfinite, rhsFinite);
  Value one = rewriter.create<LLVM::ConstantOp>(
      loc, elementType, rewriter.getFloatAttr(elementType, 1));
  Value lhsRealIsInfWithSign = rewriter.create<LLVM::CopySignOp>(
      loc, rewriter.create<LLVM::SelectOp>(loc, lhsRealInfinite, one, zero),
      lhsRe);
  Value lhsImagIsInfWithSign = rewriter.create<LLVM::CopySignOp>(
      loc, rewriter.create<LLVM::SelectOp>(loc, lhsImagInfinite, one, zero),
      lhsIm);
  Value lhsRealIsInfWithSignTimesrhsReal =
      rewriter.create<LLVM::FMulOp>(loc, lhsRealIsInfWithSign, rhsRe, fmf);
  Value lhsImagIsInfWithSignTimesrhsImag =
      rewriter.create<LLVM::FMulOp>(loc, lhsImagIsInfWithSign, rhsIm, fmf);
  Value resultReal3 = rewriter.create<LLVM::FMulOp>(
      loc, inf,
      rewriter.create<LLVM::FAddOp>(loc, lhsRealIsInfWithSignTimesrhsReal,
                                    lhsImagIsInfWithSignTimesrhsImag, fmf),
      fmf);
  Value lhsRealIsInfWithSignTimesrhsImag =
      rewriter.create<LLVM::FMulOp>(loc, lhsRealIsInfWithSign, rhsIm, fmf);
  Value lhsImagIsInfWithSignTimesrhsReal =
      rewriter.create<LLVM::FMulOp>(loc, lhsImagIsInfWithSign, rhsRe, fmf);
  Value resultImag3 = rewriter.create<LLVM::FMulOp>(
      loc, inf,
      rewriter.create<LLVM::FSubOp>(loc, lhsImagIsInfWithSignTimesrhsReal,
                                    lhsRealIsInfWithSignTimesrhsImag, fmf),
      fmf);

  // Case 3: Finite numerator, infinite denominator.
  Value lhsRealFinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::one, lhsRealAbs, inf);
  Value lhsImagFinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::one, lhsImagAbs, inf);
  Value lhsFinite =
      rewriter.create<LLVM::AndOp>(loc, lhsRealFinite, lhsImagFinite);
  Value rhsRealInfinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::oeq, rhsRealAbs, inf);
  Value rhsImagInfinite = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::oeq, rhsImagAbs, inf);
  Value rhsInfinite =
      rewriter.create<LLVM::OrOp>(loc, rhsRealInfinite, rhsImagInfinite);
  Value finiteNumInfiniteDenom =
      rewriter.create<LLVM::AndOp>(loc, lhsFinite, rhsInfinite);
  Value rhsRealIsInfWithSign = rewriter.create<LLVM::CopySignOp>(
      loc, rewriter.create<LLVM::SelectOp>(loc, rhsRealInfinite, one, zero),
      rhsRe);
  Value rhsImagIsInfWithSign = rewriter.create<LLVM::CopySignOp>(
      loc, rewriter.create<LLVM::SelectOp>(loc, rhsImagInfinite, one, zero),
      rhsIm);
  Value rhsRealIsInfWithSignTimeslhsReal =
      rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsRealIsInfWithSign, fmf);
  Value rhsImagIsInfWithSignTimeslhsImag =
      rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsImagIsInfWithSign, fmf);
  Value resultReal4 = rewriter.create<LLVM::FMulOp>(
      loc, zero,
      rewriter.create<LLVM::FAddOp>(loc, rhsRealIsInfWithSignTimeslhsReal,
                                    rhsImagIsInfWithSignTimeslhsImag, fmf),
      fmf);
  Value rhsRealIsInfWithSignTimeslhsImag =
      rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsRealIsInfWithSign, fmf);
  Value rhsImagIsInfWithSignTimeslhsReal =
      rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsImagIsInfWithSign, fmf);
  Value resultImag4 = rewriter.create<LLVM::FMulOp>(
      loc, zero,
      rewriter.create<LLVM::FSubOp>(loc, rhsRealIsInfWithSignTimeslhsImag,
                                    rhsImagIsInfWithSignTimeslhsReal, fmf),
      fmf);

  Value realAbsSmallerThanImagAbs = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::olt, rhsRealAbs, rhsImagAbs);
  Value resultReal5 = rewriter.create<LLVM::SelectOp>(
      loc, realAbsSmallerThanImagAbs, resultReal1, resultReal2);
  Value resultImag5 = rewriter.create<LLVM::SelectOp>(
      loc, realAbsSmallerThanImagAbs, resultImag1, resultImag2);
  Value resultRealSpecialCase3 = rewriter.create<LLVM::SelectOp>(
      loc, finiteNumInfiniteDenom, resultReal4, resultReal5);
  Value resultImagSpecialCase3 = rewriter.create<LLVM::SelectOp>(
      loc, finiteNumInfiniteDenom, resultImag4, resultImag5);
  Value resultRealSpecialCase2 = rewriter.create<LLVM::SelectOp>(
      loc, infNumFiniteDenom, resultReal3, resultRealSpecialCase3);
  Value resultImagSpecialCase2 = rewriter.create<LLVM::SelectOp>(
      loc, infNumFiniteDenom, resultImag3, resultImagSpecialCase3);
  Value resultRealSpecialCase1 = rewriter.create<LLVM::SelectOp>(
      loc, resultIsInfinity, infinityResultReal, resultRealSpecialCase2);
  Value resultImagSpecialCase1 = rewriter.create<LLVM::SelectOp>(
      loc, resultIsInfinity, infinityResultImag, resultImagSpecialCase2);

  Value resultRealIsNaN = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::uno, resultReal5, zero);
  Value resultImagIsNaN = rewriter.create<LLVM::FCmpOp>(
      loc, LLVM::FCmpPredicate::uno, resultImag5, zero);
  Value resultIsNaN =
      rewriter.create<LLVM::AndOp>(loc, resultRealIsNaN, resultImagIsNaN);

  *resultRe = rewriter.create<LLVM::SelectOp>(
      loc, resultIsNaN, resultRealSpecialCase1, resultReal5);
  *resultIm = rewriter.create<LLVM::SelectOp>(
      loc, resultIsNaN, resultImagSpecialCase1, resultImag5);
}

void mlir::complex::convertDivToStandardUsingRangeReduction(
    ConversionPatternRewriter &rewriter, Location loc, Value lhsRe, Value lhsIm,
    Value rhsRe, Value rhsIm, arith::FastMathFlagsAttr fmf, Value *resultRe,
    Value *resultIm) {
  auto elementType = cast<FloatType>(rhsRe.getType());

  Value rhsRealImagRatio =
      rewriter.create<arith::DivFOp>(loc, rhsRe, rhsIm, fmf);
  Value rhsRealImagDenom = rewriter.create<arith::AddFOp>(
      loc, rhsIm,
      rewriter.create<arith::MulFOp>(loc, rhsRealImagRatio, rhsRe, fmf), fmf);
  Value realNumerator1 = rewriter.create<arith::AddFOp>(
      loc, rewriter.create<arith::MulFOp>(loc, lhsRe, rhsRealImagRatio, fmf),
      lhsIm, fmf);
  Value resultReal1 = rewriter.create<arith::DivFOp>(loc, realNumerator1,
                                                     rhsRealImagDenom, fmf);
  Value imagNumerator1 = rewriter.create<arith::SubFOp>(
      loc, rewriter.create<arith::MulFOp>(loc, lhsIm, rhsRealImagRatio, fmf),
      lhsRe, fmf);
  Value resultImag1 = rewriter.create<arith::DivFOp>(loc, imagNumerator1,
                                                     rhsRealImagDenom, fmf);

  Value rhsImagRealRatio =
      rewriter.create<arith::DivFOp>(loc, rhsIm, rhsRe, fmf);
  Value rhsImagRealDenom = rewriter.create<arith::AddFOp>(
      loc, rhsRe,
      rewriter.create<arith::MulFOp>(loc, rhsImagRealRatio, rhsIm, fmf), fmf);
  Value realNumerator2 = rewriter.create<arith::AddFOp>(
      loc, lhsRe,
      rewriter.create<arith::MulFOp>(loc, lhsIm, rhsImagRealRatio, fmf), fmf);
  Value resultReal2 = rewriter.create<arith::DivFOp>(loc, realNumerator2,
                                                     rhsImagRealDenom, fmf);
  Value imagNumerator2 = rewriter.create<arith::SubFOp>(
      loc, lhsIm,
      rewriter.create<arith::MulFOp>(loc, lhsRe, rhsImagRealRatio, fmf), fmf);
  Value resultImag2 = rewriter.create<arith::DivFOp>(loc, imagNumerator2,
                                                     rhsImagRealDenom, fmf);

  // Consider corner cases.
  // Case 1. Zero denominator, numerator contains at most one NaN value.
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getZeroAttr(elementType));
  Value rhsRealAbs = rewriter.create<math::AbsFOp>(loc, rhsRe, fmf);
  Value rhsRealIsZero = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, rhsRealAbs, zero);
  Value rhsImagAbs = rewriter.create<math::AbsFOp>(loc, rhsIm, fmf);
  Value rhsImagIsZero = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, rhsImagAbs, zero);
  Value lhsRealIsNotNaN = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::ORD, lhsRe, zero);
  Value lhsImagIsNotNaN = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::ORD, lhsIm, zero);
  Value lhsContainsNotNaNValue =
      rewriter.create<arith::OrIOp>(loc, lhsRealIsNotNaN, lhsImagIsNotNaN);
  Value resultIsInfinity = rewriter.create<arith::AndIOp>(
      loc, lhsContainsNotNaNValue,
      rewriter.create<arith::AndIOp>(loc, rhsRealIsZero, rhsImagIsZero));
  Value inf = rewriter.create<arith::ConstantOp>(
      loc, elementType,
      rewriter.getFloatAttr(elementType,
                            APFloat::getInf(elementType.getFloatSemantics())));
  Value infWithSignOfRhsReal =
      rewriter.create<math::CopySignOp>(loc, inf, rhsRe);
  Value infinityResultReal =
      rewriter.create<arith::MulFOp>(loc, infWithSignOfRhsReal, lhsRe, fmf);
  Value infinityResultImag =
      rewriter.create<arith::MulFOp>(loc, infWithSignOfRhsReal, lhsIm, fmf);

  // Case 2. Infinite numerator, finite denominator.
  Value rhsRealFinite = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::ONE, rhsRealAbs, inf);
  Value rhsImagFinite = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::ONE, rhsImagAbs, inf);
  Value rhsFinite =
      rewriter.create<arith::AndIOp>(loc, rhsRealFinite, rhsImagFinite);
  Value lhsRealAbs = rewriter.create<math::AbsFOp>(loc, lhsRe, fmf);
  Value lhsRealInfinite = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, lhsRealAbs, inf);
  Value lhsImagAbs = rewriter.create<math::AbsFOp>(loc, lhsIm, fmf);
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
      lhsRe);
  Value lhsImagIsInfWithSign = rewriter.create<math::CopySignOp>(
      loc, rewriter.create<arith::SelectOp>(loc, lhsImagInfinite, one, zero),
      lhsIm);
  Value lhsRealIsInfWithSignTimesRhsReal =
      rewriter.create<arith::MulFOp>(loc, lhsRealIsInfWithSign, rhsRe, fmf);
  Value lhsImagIsInfWithSignTimesRhsImag =
      rewriter.create<arith::MulFOp>(loc, lhsImagIsInfWithSign, rhsIm, fmf);
  Value resultReal3 = rewriter.create<arith::MulFOp>(
      loc, inf,
      rewriter.create<arith::AddFOp>(loc, lhsRealIsInfWithSignTimesRhsReal,
                                     lhsImagIsInfWithSignTimesRhsImag, fmf),
      fmf);
  Value lhsRealIsInfWithSignTimesRhsImag =
      rewriter.create<arith::MulFOp>(loc, lhsRealIsInfWithSign, rhsIm, fmf);
  Value lhsImagIsInfWithSignTimesRhsReal =
      rewriter.create<arith::MulFOp>(loc, lhsImagIsInfWithSign, rhsRe, fmf);
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
      rhsRe);
  Value rhsImagIsInfWithSign = rewriter.create<math::CopySignOp>(
      loc, rewriter.create<arith::SelectOp>(loc, rhsImagInfinite, one, zero),
      rhsIm);
  Value rhsRealIsInfWithSignTimesLhsReal =
      rewriter.create<arith::MulFOp>(loc, lhsRe, rhsRealIsInfWithSign, fmf);
  Value rhsImagIsInfWithSignTimesLhsImag =
      rewriter.create<arith::MulFOp>(loc, lhsIm, rhsImagIsInfWithSign, fmf);
  Value resultReal4 = rewriter.create<arith::MulFOp>(
      loc, zero,
      rewriter.create<arith::AddFOp>(loc, rhsRealIsInfWithSignTimesLhsReal,
                                     rhsImagIsInfWithSignTimesLhsImag, fmf),
      fmf);
  Value rhsRealIsInfWithSignTimesLhsImag =
      rewriter.create<arith::MulFOp>(loc, lhsIm, rhsRealIsInfWithSign, fmf);
  Value rhsImagIsInfWithSignTimesLhsReal =
      rewriter.create<arith::MulFOp>(loc, lhsRe, rhsImagIsInfWithSign, fmf);
  Value resultImag4 = rewriter.create<arith::MulFOp>(
      loc, zero,
      rewriter.create<arith::SubFOp>(loc, rhsRealIsInfWithSignTimesLhsImag,
                                     rhsImagIsInfWithSignTimesLhsReal, fmf),
      fmf);

  Value realAbsSmallerThanImagAbs = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, rhsRealAbs, rhsImagAbs);
  Value resultReal5 = rewriter.create<arith::SelectOp>(
      loc, realAbsSmallerThanImagAbs, resultReal1, resultReal2);
  Value resultImag5 = rewriter.create<arith::SelectOp>(
      loc, realAbsSmallerThanImagAbs, resultImag1, resultImag2);
  Value resultRealSpecialCase3 = rewriter.create<arith::SelectOp>(
      loc, finiteNumInfiniteDenom, resultReal4, resultReal5);
  Value resultImagSpecialCase3 = rewriter.create<arith::SelectOp>(
      loc, finiteNumInfiniteDenom, resultImag4, resultImag5);
  Value resultRealSpecialCase2 = rewriter.create<arith::SelectOp>(
      loc, infNumFiniteDenom, resultReal3, resultRealSpecialCase3);
  Value resultImagSpecialCase2 = rewriter.create<arith::SelectOp>(
      loc, infNumFiniteDenom, resultImag3, resultImagSpecialCase3);
  Value resultRealSpecialCase1 = rewriter.create<arith::SelectOp>(
      loc, resultIsInfinity, infinityResultReal, resultRealSpecialCase2);
  Value resultImagSpecialCase1 = rewriter.create<arith::SelectOp>(
      loc, resultIsInfinity, infinityResultImag, resultImagSpecialCase2);

  Value resultRealIsNaN = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::UNO, resultReal5, zero);
  Value resultImagIsNaN = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::UNO, resultImag5, zero);
  Value resultIsNaN =
      rewriter.create<arith::AndIOp>(loc, resultRealIsNaN, resultImagIsNaN);

  *resultRe = rewriter.create<arith::SelectOp>(
      loc, resultIsNaN, resultRealSpecialCase1, resultReal5);
  *resultIm = rewriter.create<arith::SelectOp>(
      loc, resultIsNaN, resultImagSpecialCase1, resultImag5);
}
