//===----- DivisionByConstantInfo.cpp - division by constant -*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements support for optimizing divisions by a constant
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/DivisionByConstantInfo.h"

using namespace llvm;

/// Find M = ceil(2^S / D) and S such that
///   trunc(srl(mul(zext(x, W), M), S)) == udiv(x, D)
/// for all x in [0, MaxX], where the multiply stays within W bits (no MULHU).
///
/// This gives a fixup-free alternative to the Hacker's Delight add-and-shift
/// for narrow types (i8/i16) widened into a larger integer.  The HD algorithm
/// in wide space produces MULHU-style magic (≈2^W/D), which overflows a plain
/// W-bit multiply; this routine instead finds the smallest S ≥ ceil(log2(D))
/// for which the product MaxX * ceil(2^S/D) fits in W bits and the rounding
/// error is harmless.
static bool findSimpleWideMagic(const APInt &D, const APInt &MaxX, unsigned W,
                                APInt &Magic, unsigned &Shift) {
  APInt DivW = D.zext(W);
  APInt MaxW = MaxX.zext(W);
  for (unsigned S = D.ceilLogBase2(); S < W; ++S) {
    APInt TwoToS = APInt::getOneBitSet(W, S);
    APInt M = APIntOps::RoundingUDiv(TwoToS, DivW, APInt::Rounding::UP);
    bool Overflow = false;
    (void)MaxW.umul_ov(M, Overflow);
    if (Overflow)
      break; // M grows monotonically; no larger S can succeed.
    APInt Error = M * DivW - TwoToS;
    APInt MaxError = MaxW.umul_ov(Error, Overflow);
    if (Overflow || MaxError.uge(TwoToS))
      continue;
    Magic = M;
    Shift = S;
    return true;
  }
  return false;
}

/// Calculate the magic numbers required to implement a signed integer division
/// by a constant as a sequence of multiplies, adds and shifts.  Requires that
/// the divisor not be 0, 1, or -1.  Taken from "Hacker's Delight", Henry S.
/// Warren, Jr., Chapter 10.
SignedDivisionByConstantInfo SignedDivisionByConstantInfo::get(const APInt &D) {
  assert(!D.isZero() && "Precondition violation.");

  // We'd be endlessly stuck in the loop.
  assert(D.getBitWidth() >= 3 && "Does not work at smaller bitwidths.");

  APInt Delta;
  APInt SignedMin = APInt::getSignedMinValue(D.getBitWidth());
  struct SignedDivisionByConstantInfo Retval;

  APInt AD = D.abs();
  APInt T = SignedMin + (D.lshr(D.getBitWidth() - 1));
  APInt ANC = T - 1 - T.urem(AD);   // absolute value of NC
  unsigned P = D.getBitWidth() - 1; // initialize P
  APInt Q1, R1, Q2, R2;
  // initialize Q1 = 2P/abs(NC); R1 = rem(2P,abs(NC))
  APInt::udivrem(SignedMin, ANC, Q1, R1);
  // initialize Q2 = 2P/abs(D); R2 = rem(2P,abs(D))
  APInt::udivrem(SignedMin, AD, Q2, R2);
  do {
    P = P + 1;
    Q1 <<= 1;      // update Q1 = 2P/abs(NC)
    R1 <<= 1;      // update R1 = rem(2P/abs(NC))
    if (R1.uge(ANC)) { // must be unsigned comparison
      ++Q1;
      R1 -= ANC;
    }
    Q2 <<= 1;     // update Q2 = 2P/abs(D)
    R2 <<= 1;     // update R2 = rem(2P/abs(D))
    if (R2.uge(AD)) { // must be unsigned comparison
      ++Q2;
      R2 -= AD;
    }
    // Delta = AD - R2
    Delta = AD;
    Delta -= R2;
  } while (Q1.ult(Delta) || (Q1 == Delta && R1.isZero()));

  Retval.Magic = std::move(Q2);
  ++Retval.Magic;
  if (D.isNegative())
    Retval.Magic.negate();                  // resulting magic number
  Retval.ShiftAmount = P - D.getBitWidth(); // resulting shift
  return Retval;
}

/// Calculate the magic numbers required to implement an unsigned integer
/// division by a constant as a sequence of multiplies, adds and shifts.
/// Requires that the divisor not be 0.  Taken from "Hacker's Delight", Henry
/// S. Warren, Jr., chapter 10.
/// LeadingZeros can be used to simplify the calculation if the upper bits
/// of the divided value are known zero.
UnsignedDivisionByConstantInfo
UnsignedDivisionByConstantInfo::get(const APInt &D, unsigned LeadingZeros,
                                    bool AllowEvenDivisorOptimization,
                                    IntegerBitWidth MaxBitWidth) {
  unsigned WideningBitWidth = static_cast<unsigned>(MaxBitWidth);
  assert(!D.isZero() && !D.isOne() && "Precondition violation.");
  assert(D.getBitWidth() > 1 && "Does not work at smaller bitwidths.");

  APInt Delta;
  struct UnsignedDivisionByConstantInfo Retval;
  Retval.IsAdd = false; // initialize "add" indicator
  Retval.Widening = UnsignedDivisionByConstantWidening::None;
  APInt AllOnes =
      APInt::getLowBitsSet(D.getBitWidth(), D.getBitWidth() - LeadingZeros);
  APInt SignedMin = APInt::getSignedMinValue(D.getBitWidth());
  APInt SignedMax = APInt::getSignedMaxValue(D.getBitWidth());

  // Calculate NC, the largest dividend such that NC.urem(D) == D-1.
  APInt NC = AllOnes - (AllOnes + 1 - D).urem(D);
  assert(NC.urem(D) == D - 1 && "Unexpected NC value");
  unsigned P = D.getBitWidth() - 1; // initialize P
  APInt Q1, R1, Q2, R2;
  // initialize Q1 = 2P/NC; R1 = rem(2P,NC)
  APInt::udivrem(SignedMin, NC, Q1, R1);
  // initialize Q2 = (2P-1)/D; R2 = rem((2P-1),D)
  APInt::udivrem(SignedMax, D, Q2, R2);
  do {
    P = P + 1;
    if (R1.uge(NC - R1)) {
      // update Q1
      Q1 <<= 1;
      ++Q1;
      // update R1
      R1 <<= 1;
      R1 -= NC;
    } else {
      Q1 <<= 1; // update Q1
      R1 <<= 1; // update R1
    }
    if ((R2 + 1).uge(D - R2)) {
      if (Q2.uge(SignedMax))
        Retval.IsAdd = true;
      // update Q2
      Q2 <<= 1;
      ++Q2;
      // update R2
      R2 <<= 1;
      ++R2;
      R2 -= D;
    } else {
      if (Q2.uge(SignedMin))
        Retval.IsAdd = true;
      // update Q2
      Q2 <<= 1;
      // update R2
      R2 <<= 1;
      ++R2;
    }
    // Delta = D - 1 - R2
    Delta = D;
    --Delta;
    Delta -= R2;
  } while (P < D.getBitWidth() * 2 &&
           (Q1.ult(Delta) || (Q1 == Delta && R1.isZero())));

  if (Retval.IsAdd && !D[0] && AllowEvenDivisorOptimization) {
    unsigned PreShift = D.countr_zero();
    APInt ShiftedD = D.lshr(PreShift);
    Retval =
        UnsignedDivisionByConstantInfo::get(ShiftedD, LeadingZeros + PreShift);
    assert(!Retval.IsAdd && Retval.PreShift == 0);
    Retval.PreShift = PreShift;
    return Retval;
  }

  Retval.Magic = std::move(Q2);             // resulting magic number
  ++Retval.Magic;
  Retval.PostShift = P - D.getBitWidth(); // resulting shift
  // Reduce shift amount for IsAdd.
  if (Retval.IsAdd) {
    assert(Retval.PostShift > 0 && "Unexpected shift");
    Retval.PostShift -= 1;
  }
  Retval.PreShift = 0;

  if (Retval.IsAdd && WideningBitWidth) {
    unsigned W = D.getBitWidth();
    if (WideningBitWidth == W * 2) {
      // MULHU-style widen: pre-shift the (W+1)-bit magic into a W*2-bit value
      // so the high W bits of the wide multiply give the quotient directly.
      unsigned OriginalShift = Retval.PostShift + W + 1;
      // Since PostShift >= 1, shift amount is at most W-2, so W*2 bits suffice.
      Retval.Magic = (APInt::getOneBitSet(W * 2, W) + Retval.Magic.zext(W * 2))
                         .shl(W * 2 - OriginalShift);
      Retval.IsAdd = false;
      Retval.PostShift = 0;
      Retval.Widening = UnsignedDivisionByConstantWidening::MulHigh;
    } else if (WideningBitWidth > W * 2) {
      // Simple wide magic: trunc(srl(mul(zext(x, W), ceil(2^S/D)), S)).
      // The HD algorithm in wide space produces MULHU-style magic (≈2^W/D)
      // whose full product overflows W bits; findSimpleWideMagic instead finds
      // the smallest ceil(2^S/D) whose W-bit product with MaxX stays in bounds.
      APInt Magic;
      unsigned Shift;
      if (findSimpleWideMagic(D, AllOnes, WideningBitWidth, Magic, Shift)) {
        Retval.Magic = std::move(Magic);
        Retval.PostShift = Shift;
        Retval.IsAdd = false;
        Retval.Widening = UnsignedDivisionByConstantWidening::FullMultiply;
      }
    }
  }

  return Retval;
}
