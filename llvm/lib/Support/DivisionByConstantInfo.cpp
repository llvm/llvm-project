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

static UnsignedDivisionByConstantInfo PowerOf2Magic(const APInt &D,
                                                    unsigned LeadingZeros) {
  assert(!D.isZero() && D.isPowerOf2() && "Precondition violation.");
  assert(D.getBitWidth() > 1 && "Does not work at smaller bitwidths.");

  APInt Delta;
  struct UnsignedDivisionByConstantInfo Retval;
  Retval.IsAdd = false; // initialize "add" indicator
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

  if (Retval.IsAdd && !D[0]) {
    unsigned PreShift = D.countr_zero();
    // D is always 1 after removing trailing zeros from a power of 2
    APInt ShiftedD = APInt(1, D.getBitWidth());
    Retval = PowerOf2Magic(ShiftedD, LeadingZeros + PreShift);
    assert(Retval.IsAdd == 0 && Retval.PreShift == 0);
    Retval.PreShift = PreShift;
    return Retval;
  }

  Retval.Magic = std::move(Q2); // resulting magic number
  ++Retval.Magic;
  Retval.PostShift = P - D.getBitWidth(); // resulting shift

  Retval.PreShift = 0;
  return Retval;
}

/// Calculate the magic numbers required to implement an unsigned integer
/// division by a constant as a sequence of multiplies, adds and shifts.
/// Requires that the divisor not be 0.

UnsignedDivisionByConstantInfo
UnsignedDivisionByConstantInfo::get(const APInt &D, unsigned LeadingZeros) {
  assert(!D.isZero() && !D.isOne() && "Precondition violation.");
  assert(D.getBitWidth() > 1 && "Does not work at smaller bitwidths.");

  if (D.isPowerOf2())
    return PowerOf2Magic(D, LeadingZeros);
  struct UnsignedDivisionByConstantInfo Retval;
  APInt SignedMax = APInt::getSignedMaxValue(D.getBitWidth());

  // Calculate NC, the largest dividend such that NC.urem(D) == D-1.
  APInt Q, R;
  // initialize Q = (2P-1)/D; R2 = rem((2P-1),D)
  APInt::udivrem(SignedMax, D, Q, R);

  APInt MultiplierRoundDown = APInt::getZero(D.getBitWidth());
  unsigned ExponentRoundDown = 0;
  bool HasMagicDown = false;

  unsigned Log2D = D.ceilLogBase2();
  unsigned Exponent = 0;

  for (;; Exponent++) {
    if (R.uge(D - R)) {
      Q <<= 1;
      ++Q;
      R <<= 1;
      R -= D;
    } else {
      Q <<= 1;
      R <<= 1;
    }

    if (Exponent + LeadingZeros >= Log2D)
      break;

    APInt Ule = APInt::getOneBitSet(D.getBitWidth(), Exponent + LeadingZeros);

    if ((D - R).ule(Ule))
      break;

    // Set HasMagicDown if we have not set it yet and this exponent works for
    // the round_down algorithm
    if (!HasMagicDown && R.ule(Ule)) {
      HasMagicDown = true;
      MultiplierRoundDown = Q;
      ExponentRoundDown = Exponent;
    }
  }

  if (Exponent < Log2D) {
    // Do the normal values
    Retval.Magic = std::move(Q);
    ++Retval.Magic;
    Retval.PreShift = 0;
    Retval.PostShift = Exponent;
    Retval.IsAdd = false;
  } else if (D[0]) {
    assert(HasMagicDown && "Expected to round down but it was not set!");
    Retval.Magic = std::move(MultiplierRoundDown);
    Retval.PreShift = 0;
    Retval.PostShift = ExponentRoundDown;
    Retval.IsAdd = true;
  } else {
    unsigned PreShift = D.countr_zero();
    APInt ShiftedD = D.lshr(PreShift);
    Retval =
        UnsignedDivisionByConstantInfo::get(ShiftedD, LeadingZeros + PreShift);
    assert(!Retval.IsAdd && Retval.PreShift == 0);
    Retval.PreShift = PreShift;
  }

  return Retval;
}
