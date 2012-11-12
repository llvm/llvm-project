//===-- llvm/ADT/APSInt.h - Arbitrary Precision Signed Int -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the APSInt class, which is a simple class that
// represents an arbitrary sized integer that knows its signedness.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_APSINT_H
#define LLVM_APSINT_H

#include "llvm/ADT/APInt.h"

namespace llvm {

class APSInt : public APInt {
  bool IsUnsigned;
public:
  /// Default constructor that creates an uninitialized APInt.
  explicit APSInt() {}

  /// APSInt ctor - Create an APSInt with the specified width, default to
  /// unsigned.
  explicit APSInt(uint32_t BitWidth, bool isUnsigned = true)
   : APInt(BitWidth, 0), IsUnsigned(isUnsigned) {}

  explicit APSInt(const APInt &I, bool isUnsigned = true)
   : APInt(I), IsUnsigned(isUnsigned) {}

  APSInt &operator=(const APSInt &RHS) {
    APInt::operator=(RHS);
    IsUnsigned = RHS.IsUnsigned;
    return *this;
  }

  APSInt &operator=(const APInt &RHS) {
    // Retain our current sign.
    APInt::operator=(RHS);
    return *this;
  }

  APSInt &operator=(uint64_t RHS) {
    // Retain our current sign.
    APInt::operator=(RHS);
    return *this;
  }

  // Query sign information.
  bool isSigned() const { return !IsUnsigned; }
  bool isUnsigned() const { return IsUnsigned; }
  void setIsUnsigned(bool Val) { IsUnsigned = Val; }
  void setIsSigned(bool Val) { IsUnsigned = !Val; }

  /// toString - Append this APSInt to the specified SmallString.
  void toString(SmallVectorImpl<char> &Str, unsigned Radix = 10) const {
    APInt::toString(Str, Radix, isSigned());
  }
  /// toString - Converts an APInt to a std::string.  This is an inefficient
  /// method, your should prefer passing in a SmallString instead.
  std::string toString(unsigned Radix) const {
    return APInt::toString(Radix, isSigned());
  }
  using APInt::toString;

  APSInt trunc(uint32_t width) const {
    return APSInt(APInt::trunc(width), IsUnsigned);
  }

  APSInt extend(uint32_t width) const {
    if (IsUnsigned)
      return APSInt(zext(width), IsUnsigned);
    else
      return APSInt(sext(width), IsUnsigned);
  }

  APSInt extOrTrunc(uint32_t width) const {
      if (IsUnsigned)
        return APSInt(zextOrTrunc(width), IsUnsigned);
      else
        return APSInt(sextOrTrunc(width), IsUnsigned);
  }

  const APSInt &operator%=(const APSInt &RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    if (IsUnsigned)
      *this = urem(RHS);
    else
      *this = srem(RHS);
    return *this;
  }
  const APSInt &operator/=(const APSInt &RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    if (IsUnsigned)
      *this = udiv(RHS);
    else
      *this = sdiv(RHS);
    return *this;
  }
  APSInt operator%(const APSInt &RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? APSInt(urem(RHS), true) : APSInt(srem(RHS), false);
  }
  APSInt operator/(const APSInt &RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? APSInt(udiv(RHS), true) : APSInt(sdiv(RHS), false);
  }

  APSInt operator>>(unsigned Amt) const {
    return IsUnsigned ? APSInt(lshr(Amt), true) : APSInt(ashr(Amt), false);
  }
  APSInt& operator>>=(unsigned Amt) {
    *this = *this >> Amt;
    return *this;
  }

  inline bool operator<(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? ult(RHS) : slt(RHS);
  }
  inline bool operator>(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? ugt(RHS) : sgt(RHS);
  }
  inline bool operator<=(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? ule(RHS) : sle(RHS);
  }
  inline bool operator>=(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return IsUnsigned ? uge(RHS) : sge(RHS);
  }
  inline bool operator==(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return eq(RHS);
  }
  inline bool operator==(int64_t RHS) const {
    return isSameValue(*this, APSInt(APInt(64, RHS), true));
  }
  inline bool operator!=(const APSInt& RHS) const {
    return !((*this) == RHS);
  }
  inline bool operator!=(int64_t RHS) const {
    return !((*this) == RHS);
  }

  // The remaining operators just wrap the logic of APInt, but retain the
  // signedness information.

  APSInt operator<<(unsigned Bits) const {
    return APSInt(static_cast<const APInt&>(*this) << Bits, IsUnsigned);
  }
  APSInt& operator<<=(unsigned Amt) {
    *this = *this << Amt;
    return *this;
  }

  APSInt& operator++() {
    static_cast<APInt&>(*this)++;
    return *this;
  }
  APSInt& operator--() {
    static_cast<APInt&>(*this)--;
    return *this;
  }
  APSInt operator++(int) {
    return APSInt(++static_cast<APInt&>(*this), IsUnsigned);
  }
  APSInt operator--(int) {
    return APSInt(--static_cast<APInt&>(*this), IsUnsigned);
  }
  APSInt operator-() const {
    return APSInt(-static_cast<const APInt&>(*this), IsUnsigned);
  }
  APSInt& operator+=(const APSInt& RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    static_cast<APInt&>(*this) += RHS;
    return *this;
  }
  APSInt& operator-=(const APSInt& RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    static_cast<APInt&>(*this) -= RHS;
    return *this;
  }
  APSInt& operator*=(const APSInt& RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    static_cast<APInt&>(*this) *= RHS;
    return *this;
  }
  APSInt& operator&=(const APSInt& RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    static_cast<APInt&>(*this) &= RHS;
    return *this;
  }
  APSInt& operator|=(const APSInt& RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    static_cast<APInt&>(*this) |= RHS;
    return *this;
  }
  APSInt& operator^=(const APSInt& RHS) {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    static_cast<APInt&>(*this) ^= RHS;
    return *this;
  }

  APSInt operator&(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return APSInt(static_cast<const APInt&>(*this) & RHS, IsUnsigned);
  }
  APSInt And(const APSInt& RHS) const {
    return this->operator&(RHS);
  }

  APSInt operator|(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return APSInt(static_cast<const APInt&>(*this) | RHS, IsUnsigned);
  }
  APSInt Or(const APSInt& RHS) const {
    return this->operator|(RHS);
  }


  APSInt operator^(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return APSInt(static_cast<const APInt&>(*this) ^ RHS, IsUnsigned);
  }
  APSInt Xor(const APSInt& RHS) const {
    return this->operator^(RHS);
  }

  APSInt operator*(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return APSInt(static_cast<const APInt&>(*this) * RHS, IsUnsigned);
  }
  APSInt operator+(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return APSInt(static_cast<const APInt&>(*this) + RHS, IsUnsigned);
  }
  APSInt operator-(const APSInt& RHS) const {
    assert(IsUnsigned == RHS.IsUnsigned && "Signedness mismatch!");
    return APSInt(static_cast<const APInt&>(*this) - RHS, IsUnsigned);
  }
  APSInt operator~() const {
    return APSInt(~static_cast<const APInt&>(*this), IsUnsigned);
  }

  /// getMaxValue - Return the APSInt representing the maximum integer value
  ///  with the given bit width and signedness.
  static APSInt getMaxValue(uint32_t numBits, bool Unsigned) {
    return APSInt(Unsigned ? APInt::getMaxValue(numBits)
                           : APInt::getSignedMaxValue(numBits), Unsigned);
  }

  /// getMinValue - Return the APSInt representing the minimum integer value
  ///  with the given bit width and signedness.
  static APSInt getMinValue(uint32_t numBits, bool Unsigned) {
    return APSInt(Unsigned ? APInt::getMinValue(numBits)
                           : APInt::getSignedMinValue(numBits), Unsigned);
  }

  /// \brief Determine if two APSInts have the same value, zero- or
  /// sign-extending as needed.  
  static bool isSameValue(const APSInt &I1, const APSInt &I2) {
    if (I1.getBitWidth() == I2.getBitWidth() && I1.isSigned() == I2.isSigned())
      return I1 == I2;

    // Check for a bit-width mismatch.
    if (I1.getBitWidth() > I2.getBitWidth())
      return isSameValue(I1, I2.extend(I1.getBitWidth()));
    else if (I2.getBitWidth() > I1.getBitWidth())
      return isSameValue(I1.extend(I2.getBitWidth()), I2);

    // We have a signedness mismatch. Turn the signed value into an unsigned
    // value.
    if (I1.isSigned()) {
      if (I1.isNegative())
        return false;

      return APSInt(I1, true) == I2;
    }

    if (I2.isNegative())
      return false;

    return I1 == APSInt(I2, true);
  }

  /// Profile - Used to insert APSInt objects, or objects that contain APSInt
  ///  objects, into FoldingSets.
  void Profile(FoldingSetNodeID& ID) const;
};

inline bool operator==(int64_t V1, const APSInt& V2) {
  return V2 == V1;
}
inline bool operator!=(int64_t V1, const APSInt& V2) {
  return V2 != V1;
}

inline raw_ostream &operator<<(raw_ostream &OS, const APSInt &I) {
  I.print(OS, I.isSigned());
  return OS;
}

} // end namespace llvm

#endif
