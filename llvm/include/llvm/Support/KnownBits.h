//===- llvm/Support/KnownBits.h - Stores known zeros/ones -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  // Internal constructor for creating a ConstantRange from two APInts.
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

  /// Make this value negative.
  void makeNegative() {
    assert(!isNonNegative() && "Can't make a non-negative value negative");
    One.setSignBit();
  }

  /// Make this value negative.
  void makeNonNegative() {
    assert(!isNegative() && "Can't make a negative value non-negative");
    Zero.setSignBit();
  }

  /// Truncate the underlying known Zero and One bits. This is equivalent
  /// to truncating the value we're tracking.
  KnownBits trunc(unsigned BitWidth) {
    return KnownBits(Zero.trunc(BitWidth), One.trunc(BitWidth));
  }

  /// Zero extends the underlying known Zero and One bits. This is equivalent
  /// to zero extending the value we're tracking.
  KnownBits zext(unsigned BitWidth) {
    return KnownBits(Zero.zext(BitWidth), One.zext(BitWidth));
  }

  /// Sign extends the underlying known Zero and One bits. This is equivalent
  /// to sign extending the value we're tracking.
  KnownBits sext(unsigned BitWidth) {
    return KnownBits(Zero.sext(BitWidth), One.sext(BitWidth));
  }

  /// Zero extends or truncates the underlying known Zero and One bits. This is
  /// equivalent to zero extending or truncating the value we're tracking.
  KnownBits zextOrTrunc(unsigned BitWidth) {
    return KnownBits(Zero.zextOrTrunc(BitWidth), One.zextOrTrunc(BitWidth));
  }
};

} // end namespace llvm

#endif
