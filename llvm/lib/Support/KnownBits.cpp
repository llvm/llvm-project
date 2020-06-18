//===-- KnownBits.cpp - Stores known zeros/ones ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a class for representing known zeros and ones used by
// computeKnownBits.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/KnownBits.h"
#include <cassert>

using namespace llvm;

static KnownBits computeForAddCarry(
    const KnownBits &LHS, const KnownBits &RHS,
    bool CarryZero, bool CarryOne) {
  assert(!(CarryZero && CarryOne) &&
         "Carry can't be zero and one at the same time");

  APInt PossibleSumZero = LHS.getMaxValue() + RHS.getMaxValue() + !CarryZero;
  APInt PossibleSumOne = LHS.getMinValue() + RHS.getMinValue() + CarryOne;

  // Compute known bits of the carry.
  APInt CarryKnownZero = ~(PossibleSumZero ^ LHS.Zero ^ RHS.Zero);
  APInt CarryKnownOne = PossibleSumOne ^ LHS.One ^ RHS.One;

  // Compute set of known bits (where all three relevant bits are known).
  APInt LHSKnownUnion = LHS.Zero | LHS.One;
  APInt RHSKnownUnion = RHS.Zero | RHS.One;
  APInt CarryKnownUnion = std::move(CarryKnownZero) | CarryKnownOne;
  APInt Known = std::move(LHSKnownUnion) & RHSKnownUnion & CarryKnownUnion;

  assert((PossibleSumZero & Known) == (PossibleSumOne & Known) &&
         "known bits of sum differ");

  // Compute known bits of the result.
  KnownBits KnownOut;
  KnownOut.Zero = ~std::move(PossibleSumZero) & Known;
  KnownOut.One = std::move(PossibleSumOne) & Known;
  return KnownOut;
}

KnownBits KnownBits::computeForAddCarry(
    const KnownBits &LHS, const KnownBits &RHS, const KnownBits &Carry) {
  assert(Carry.getBitWidth() == 1 && "Carry must be 1-bit");
  return ::computeForAddCarry(
      LHS, RHS, Carry.Zero.getBoolValue(), Carry.One.getBoolValue());
}

KnownBits KnownBits::computeForAddSub(bool Add, bool NSW,
                                      const KnownBits &LHS, KnownBits RHS) {
  KnownBits KnownOut;
  if (Add) {
    // Sum = LHS + RHS + 0
    KnownOut = ::computeForAddCarry(
        LHS, RHS, /*CarryZero*/true, /*CarryOne*/false);
  } else {
    // Sum = LHS + ~RHS + 1
    std::swap(RHS.Zero, RHS.One);
    KnownOut = ::computeForAddCarry(
        LHS, RHS, /*CarryZero*/false, /*CarryOne*/true);
  }

  // Are we still trying to solve for the sign bit?
  if (!KnownOut.isNegative() && !KnownOut.isNonNegative()) {
    if (NSW) {
      // Adding two non-negative numbers, or subtracting a negative number from
      // a non-negative one, can't wrap into negative.
      if (LHS.isNonNegative() && RHS.isNonNegative())
        KnownOut.makeNonNegative();
      // Adding two negative numbers, or subtracting a non-negative number from
      // a negative one, can't wrap into non-negative.
      else if (LHS.isNegative() && RHS.isNegative())
        KnownOut.makeNegative();
    }
  }

  return KnownOut;
}

KnownBits &KnownBits::operator&=(const KnownBits &RHS) {
  // Result bit is 0 if either operand bit is 0.
  Zero |= RHS.Zero;
  // Result bit is 1 if both operand bits are 1.
  One &= RHS.One;
  return *this;
}

KnownBits &KnownBits::operator|=(const KnownBits &RHS) {
  // Result bit is 0 if both operand bits are 0.
  Zero &= RHS.Zero;
  // Result bit is 1 if either operand bit is 1.
  One |= RHS.One;
  return *this;
}

KnownBits &KnownBits::operator^=(const KnownBits &RHS) {
  // Result bit is 0 if both operand bits are 0 or both are 1.
  APInt Z = (Zero & RHS.Zero) | (One & RHS.One);
  // Result bit is 1 if one operand bit is 0 and the other is 1.
  One = (Zero & RHS.One) | (One & RHS.Zero);
  Zero = std::move(Z);
  return *this;
}
