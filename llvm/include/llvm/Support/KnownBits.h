//===- llvm/Support/KnownBits.h - Stores known zeros/ones -------*- C++ -*-===//
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

#ifndef LLVM_SUPPORT_KNOWNBITS_H
#define LLVM_SUPPORT_KNOWNBITS_H

#include "llvm/ADT/APInt.h"

namespace llvm {

// Struct for tracking the known zeros and ones of a value.
struct KnownBits {
  APInt Zero;
  APInt One;

private:
  // Internal constructor for creating a KnownBits from two APInts.
  KnownBits(APInt Zero, APInt One)
      : Zero(std::move(Zero)), One(std::move(One)) {}

public:
  // Default construct Zero and One.
  KnownBits() {}

  /// Create a known bits object of BitWidth bits initialized to unknown.
  KnownBits(unsigned BitWidth) : Zero(BitWidth, 0), One(BitWidth, 0) {}

  /// Get the bit width of this value.
  unsigned getBitWidth() const {
    assert(Zero.getBitWidth() == One.getBitWidth() &&
           "Zero and One should have the same width!");
    return Zero.getBitWidth();
  }

  /// Returns true if there is conflicting information.
  bool hasConflict() const { return Zero.intersects(One); }

  /// Returns true if we know the value of all bits.
  bool isConstant() const {
    assert(!hasConflict() && "KnownBits conflict!");
    return Zero.countPopulation() + One.countPopulation() == getBitWidth();
  }

  /// Returns the value when all bits have a known value. This just returns One
  /// with a protective assertion.
  const APInt &getConstant() const {
    assert(isConstant() && "Can only get value when all bits are known");
    return One;
  }

  /// Returns true if we don't know any bits.
  bool isUnknown() const { return Zero.isNullValue() && One.isNullValue(); }

  /// Resets the known state of all bits.
  void resetAll() {
    Zero.clearAllBits();
    One.clearAllBits();
  }

  /// Returns true if value is all zero.
  bool isZero() const {
    assert(!hasConflict() && "KnownBits conflict!");
    return Zero.isAllOnesValue();
  }

  /// Returns true if value is all one bits.
  bool isAllOnes() const {
    assert(!hasConflict() && "KnownBits conflict!");
    return One.isAllOnesValue();
  }

  /// Make all bits known to be zero and discard any previous information.
  void setAllZero() {
    Zero.setAllBits();
    One.clearAllBits();
  }

  /// Make all bits known to be one and discard any previous information.
  void setAllOnes() {
    Zero.clearAllBits();
    One.setAllBits();
  }

  /// Returns true if this value is known to be negative.
  bool isNegative() const { return One.isSignBitSet(); }

  /// Returns true if this value is known to be non-negative.
  bool isNonNegative() const { return Zero.isSignBitSet(); }

  /// Returns true if this value is known to be positive.
  bool isStrictlyPositive() const { return Zero.isSignBitSet() && !One.isNullValue(); }

  /// Make this value negative.
  void makeNegative() {
    One.setSignBit();
  }

  /// Make this value non-negative.
  void makeNonNegative() {
    Zero.setSignBit();
  }

  /// Return the minimal value possible given these KnownBits.
  APInt getMinValue() const {
    // Assume that all bits that aren't known-ones are zeros.
    return One;
  }

  /// Return the maximal value possible given these KnownBits.
  APInt getMaxValue() const {
    // Assume that all bits that aren't known-zeros are ones.
    return ~Zero;
  }

  /// Truncate the underlying known Zero and One bits. This is equivalent
  /// to truncating the value we're tracking.
  KnownBits trunc(unsigned BitWidth) const {
    return KnownBits(Zero.trunc(BitWidth), One.trunc(BitWidth));
  }

  /// Extends the underlying known Zero and One bits.
  /// By setting ExtendedBitsAreKnownZero=true this will be equivalent to
  /// zero extending the value we're tracking.
  /// With ExtendedBitsAreKnownZero=false the extended bits are set to unknown.
  KnownBits zext(unsigned BitWidth, bool ExtendedBitsAreKnownZero) const {
    unsigned OldBitWidth = getBitWidth();
    APInt NewZero = Zero.zext(BitWidth);
    if (ExtendedBitsAreKnownZero)
      NewZero.setBitsFrom(OldBitWidth);
    return KnownBits(NewZero, One.zext(BitWidth));
  }

  /// Sign extends the underlying known Zero and One bits. This is equivalent
  /// to sign extending the value we're tracking.
  KnownBits sext(unsigned BitWidth) const {
    return KnownBits(Zero.sext(BitWidth), One.sext(BitWidth));
  }

  /// Extends or truncates the underlying known Zero and One bits. When
  /// extending the extended bits can either be set as known zero (if
  /// ExtendedBitsAreKnownZero=true) or as unknown (if
  /// ExtendedBitsAreKnownZero=false).
  KnownBits zextOrTrunc(unsigned BitWidth,
                        bool ExtendedBitsAreKnownZero) const {
    if (BitWidth > getBitWidth())
      return zext(BitWidth, ExtendedBitsAreKnownZero);
    return KnownBits(Zero.zextOrTrunc(BitWidth), One.zextOrTrunc(BitWidth));
  }

  /// Returns the minimum number of trailing zero bits.
  unsigned countMinTrailingZeros() const {
    return Zero.countTrailingOnes();
  }

  /// Returns the minimum number of trailing one bits.
  unsigned countMinTrailingOnes() const {
    return One.countTrailingOnes();
  }

  /// Returns the minimum number of leading zero bits.
  unsigned countMinLeadingZeros() const {
    return Zero.countLeadingOnes();
  }

  /// Returns the minimum number of leading one bits.
  unsigned countMinLeadingOnes() const {
    return One.countLeadingOnes();
  }

  /// Returns the number of times the sign bit is replicated into the other
  /// bits.
  unsigned countMinSignBits() const {
    if (isNonNegative())
      return countMinLeadingZeros();
    if (isNegative())
      return countMinLeadingOnes();
    return 0;
  }

  /// Returns the maximum number of trailing zero bits possible.
  unsigned countMaxTrailingZeros() const {
    return One.countTrailingZeros();
  }

  /// Returns the maximum number of trailing one bits possible.
  unsigned countMaxTrailingOnes() const {
    return Zero.countTrailingZeros();
  }

  /// Returns the maximum number of leading zero bits possible.
  unsigned countMaxLeadingZeros() const {
    return One.countLeadingZeros();
  }

  /// Returns the maximum number of leading one bits possible.
  unsigned countMaxLeadingOnes() const {
    return Zero.countLeadingZeros();
  }

  /// Returns the number of bits known to be one.
  unsigned countMinPopulation() const {
    return One.countPopulation();
  }

  /// Returns the maximum number of bits that could be one.
  unsigned countMaxPopulation() const {
    return getBitWidth() - Zero.countPopulation();
  }

  /// Compute known bits resulting from adding LHS, RHS and a 1-bit Carry.
  static KnownBits computeForAddCarry(
      const KnownBits &LHS, const KnownBits &RHS, const KnownBits &Carry);

  /// Compute known bits resulting from adding LHS and RHS.
  static KnownBits computeForAddSub(bool Add, bool NSW, const KnownBits &LHS,
                                    KnownBits RHS);
};

} // end namespace llvm

#endif
