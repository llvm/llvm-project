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
UnsignedDivisionByConstantInfo
UnsignedDivisionByConstantInfo::get(const APInt &D, unsigned LeadingZeros,
                                    bool AllowEvenDivisorOptimization) {
  assert(!D.isZero() && !D.isOne() && "Precondition violation.");
  assert(D.getBitWidth() > 1 && "Does not work at smaller bitwidths.");
  assert(D.getBitWidth() >= LeadingZeros &&
         "Leading zeros more than bitwidth.");

  APInt SignedMin = APInt::getSignedMinValue(D.getBitWidth());
  APInt Quotient, Remainder;

  // initialize Q1 = 2P/NC; R1 = rem(2P,NC)
  APInt::udivrem(SignedMin, D, Quotient, Remainder);

  APInt DownMultiplier = APInt::getZero(D.getBitWidth());
  unsigned DownExponent = 0;

  bool hasMagicDown = false;

  unsigned CeilLog2 = D.ceilLogBase2();
  struct UnsignedDivisionByConstantInfo Retval;

  // Begin a loop that increments the exponent, until we find a power of 2 that
  // works.
  unsigned exponent;
  for (exponent = 0;; exponent++) {
    // Calculate the multiplier for the current exponent.
    // Quotient and remainder is from previous exponent; compute it for this
    // exponent.
    if (Remainder.uge(D - Remainder)) {
      // Doubling remainder will wrap around D
      Quotient <<= 1;
      ++Quotient;

      Remainder <<= 1;
      Remainder -= D;
    } else {
      Quotient <<= 1;
      Remainder <<= 1;
    }

    if (exponent + LeadingZeros >= CeilLog2) {
      // If we have reached the point where the multiplier is larger than
      // the divisor, we can stop.
      break;
    }

    APInt PowerOf2 =
        APInt::getOneBitSet(D.getBitWidth(), exponent + LeadingZeros);

    if ((D - Remainder).ule(PowerOf2)) {
      break;
    }

    if (!hasMagicDown && Remainder.ule(PowerOf2)) {
      // If we have not found a magic number yet, and the remainder is less
      // than or equal to D - Remainder, we can use the current quotient as
      // a magic number.
      hasMagicDown = true;
      DownMultiplier = Quotient;
      DownExponent = exponent;
    }
  }

  if (exponent < CeilLog2) {
    // magic_up is efficient
    Retval.Magic = std::move(Quotient);
    ++Retval.Magic;
    Retval.PreShift = 0;
    Retval.PostShift = exponent;
    Retval.IsAdd = false;
  } else if (D[0]) {
    assert(hasMagicDown && "Expected a magic number for down multiplier");
    Retval.Magic = std::move(DownMultiplier);
    Retval.PreShift = 0;
    Retval.PostShift = DownExponent;
    Retval.IsAdd = true;
  } else {
    unsigned PreShift = D.countr_zero();
    APInt ShiftedD = D.lshr(PreShift);
    Retval =
        UnsignedDivisionByConstantInfo::get(ShiftedD, LeadingZeros + PreShift);
    assert(Retval.IsAdd == 0 && Retval.PreShift == 0);
    Retval.PreShift = PreShift;
  }
  return Retval;
}
