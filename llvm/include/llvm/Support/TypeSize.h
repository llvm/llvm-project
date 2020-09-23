//===- TypeSize.h - Wrapper around type sizes -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a struct that can be used to query the size of IR types
// which may be scalable vectors. It provides convenience operators so that
// it can be used in much the same way as a single scalar value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TYPESIZE_H
#define LLVM_SUPPORT_TYPESIZE_H

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/WithColor.h"

#include <cstdint>
#include <cassert>

namespace llvm {

template <typename T> struct DenseMapInfo;

// TODO: This class will be redesigned in a later patch that introduces full
// polynomial behaviour, i.e. the ability to have composites made up of both
// fixed and scalable sizes.
template <typename T> class PolySize {
protected:
  T MinVal;        // The minimum value that it could be.
  bool IsScalable; // If true, the total value is determined by multiplying
                   // 'MinVal' by a runtime determinded quantity, 'vscale'.

  constexpr PolySize(T MinVal, bool IsScalable)
      : MinVal(MinVal), IsScalable(IsScalable) {}

public:

  static constexpr PolySize getFixed(T MinVal) { return {MinVal, false}; }
  static constexpr PolySize getScalable(T MinVal) { return {MinVal, true}; }
  static constexpr PolySize get(T MinVal, bool IsScalable) {
    return {MinVal, IsScalable};
  }

  static constexpr PolySize getNull() { return {0, false}; }

  /// Counting predicates.
  ///
  ///@{ No elements..
  bool isZero() const { return MinVal == 0; }
  /// At least one element.
  bool isNonZero() const { return !isZero(); }
  /// A return value of true indicates we know at compile time that the number
  /// of elements (vscale * Min) is definitely even. However, returning false
  /// does not guarantee that the total number of elements is odd.
  bool isKnownEven() const { return (MinVal & 0x1) == 0; }
  ///@}

  T getKnownMinValue() const { return MinVal; }

  // Return the minimum value with the assumption that the count is exact.
  // Use in places where a scalable count doesn't make sense (e.g. non-vector
  // types, or vectors in backends which don't support scalable vectors).
  T getFixedValue() const {
    assert(!IsScalable &&
           "Request for a fixed element count on a scalable object");
    return MinVal;
  }

  bool isScalable() const { return IsScalable; }

  bool operator==(const PolySize &RHS) const {
    return MinVal == RHS.MinVal && IsScalable == RHS.IsScalable;
  }

  bool operator!=(const PolySize &RHS) const { return !(*this == RHS); }

  // For some cases, size ordering between scalable and fixed size types cannot
  // be determined at compile time, so such comparisons aren't allowed.
  //
  // e.g. <vscale x 2 x i16> could be bigger than <4 x i32> with a runtime
  // vscale >= 5, equal sized with a vscale of 4, and smaller with
  // a vscale <= 3.
  //
  // All the functions below make use of the fact vscale is always >= 1, which
  // means that <vscale x 4 x i32> is guaranteed to be >= <4 x i32>, etc.

  static bool isKnownLT(const PolySize &LHS, const PolySize &RHS) {
    if (!LHS.IsScalable || RHS.IsScalable)
      return LHS.MinVal < RHS.MinVal;

    // LHS.IsScalable = true, RHS.IsScalable = false
    return false;
  }

  static bool isKnownGT(const PolySize &LHS, const PolySize &RHS) {
    if (LHS.IsScalable || !RHS.IsScalable)
      return LHS.MinVal > RHS.MinVal;

    // LHS.IsScalable = false, RHS.IsScalable = true
    return false;
  }

  static bool isKnownLE(const PolySize &LHS, const PolySize &RHS) {
    if (!LHS.IsScalable || RHS.IsScalable)
      return LHS.MinVal <= RHS.MinVal;

    // LHS.IsScalable = true, RHS.IsScalable = false
    return false;
  }

  static bool isKnownGE(const PolySize &LHS, const PolySize &RHS) {
    if (LHS.IsScalable || !RHS.IsScalable)
      return LHS.MinVal >= RHS.MinVal;

    // LHS.IsScalable = false, RHS.IsScalable = true
    return false;
  }

  PolySize operator*(T RHS) { return {MinVal * RHS, IsScalable}; }

  PolySize &operator*=(T RHS) {
    MinVal *= RHS;
    return *this;
  }

  friend PolySize operator-(const PolySize &LHS, const PolySize &RHS) {
    assert(LHS.IsScalable == RHS.IsScalable &&
           "Arithmetic using mixed scalable and fixed types");
    return {LHS.MinVal - RHS.MinVal, LHS.IsScalable};
  }

  /// This function tells the caller whether the element count is known at
  /// compile time to be a multiple of the scalar value RHS.
  bool isKnownMultipleOf(T RHS) const { return MinVal % RHS == 0; }

  /// We do not provide the '/' operator here because division for polynomial
  /// types does not work in the same way as for normal integer types. We can
  /// only divide the minimum value (or coefficient) by RHS, which is not the
  /// same as
  ///   (Min * Vscale) / RHS
  /// The caller is recommended to use this function in combination with
  /// isKnownMultipleOf(RHS), which lets the caller know if it's possible to
  /// perform a lossless divide by RHS.
  PolySize divideCoefficientBy(T RHS) const {
    return PolySize(MinVal / RHS, IsScalable);
  }

  PolySize coefficientNextPowerOf2() const {
    return PolySize(static_cast<T>(llvm::NextPowerOf2(MinVal)), IsScalable);
  }

  /// Printing function.
  void print(raw_ostream &OS) const {
    if (IsScalable)
      OS << "vscale x ";
    OS << MinVal;
  }
};

/// Stream operator function for `PolySize`.
template <typename T>
inline raw_ostream &operator<<(raw_ostream &OS, const PolySize<T> &PS) {
  PS.print(OS);
  return OS;
}

class ElementCount : public PolySize<unsigned> {
public:

  constexpr ElementCount(PolySize<unsigned> V) : PolySize(V) {}

  /// Counting predicates.
  ///
  /// Notice that MinVal = 1 and IsScalable = true is considered more than
  /// one element.
  ///
  ///@{ No elements..
  /// Exactly one element.
  bool isScalar() const { return !IsScalable && MinVal == 1; }
  /// One or more elements.
  bool isVector() const { return (IsScalable && MinVal != 0) || MinVal > 1; }
  ///@}
};

// This class is used to represent the size of types. If the type is of fixed
// size, it will represent the exact size. If the type is a scalable vector,
// it will represent the known minimum size.
class TypeSize : public PolySize<uint64_t> {
public:
  constexpr TypeSize(PolySize<uint64_t> V) : PolySize(V) {}

  constexpr TypeSize(uint64_t MinVal, bool IsScalable)
      : PolySize(MinVal, IsScalable) {}

  static constexpr TypeSize Fixed(uint64_t MinVal) {
    return TypeSize(MinVal, false);
  }
  static constexpr TypeSize Scalable(uint64_t MinVal) {
    return TypeSize(MinVal, true);
  }

  uint64_t getFixedSize() const { return getFixedValue(); }
  uint64_t getKnownMinSize() const { return getKnownMinValue(); }

  friend bool operator<(const TypeSize &LHS, const TypeSize &RHS) {
    assert(LHS.IsScalable == RHS.IsScalable &&
           "Ordering comparison of scalable and fixed types");

    return LHS.MinVal < RHS.MinVal;
  }

  friend bool operator>(const TypeSize &LHS, const TypeSize &RHS) {
    return RHS < LHS;
  }

  friend bool operator<=(const TypeSize &LHS, const TypeSize &RHS) {
    return !(RHS < LHS);
  }

  friend bool operator>=(const TypeSize &LHS, const TypeSize& RHS) {
    return !(LHS < RHS);
  }

  TypeSize &operator-=(TypeSize RHS) {
    assert(IsScalable == RHS.IsScalable &&
           "Subtraction using mixed scalable and fixed types");
    MinVal -= RHS.MinVal;
    return *this;
  }

  TypeSize &operator+=(TypeSize RHS) {
    assert(IsScalable == RHS.IsScalable &&
           "Addition using mixed scalable and fixed types");
    MinVal += RHS.MinVal;
    return *this;
  }

  friend TypeSize operator-(const TypeSize &LHS, const TypeSize &RHS) {
    assert(LHS.IsScalable == RHS.IsScalable &&
           "Arithmetic using mixed scalable and fixed types");
    return {LHS.MinVal - RHS.MinVal, LHS.IsScalable};
  }

  // Casts to a uint64_t if this is a fixed-width size.
  //
  // This interface is deprecated and will be removed in a future version
  // of LLVM in favour of upgrading uses that rely on this implicit conversion
  // to uint64_t. Calls to functions that return a TypeSize should use the
  // proper interfaces to TypeSize.
  // In practice this is mostly calls to MVT/EVT::getSizeInBits().
  //
  // To determine how to upgrade the code:
  //
  //   if (<algorithm works for both scalable and fixed-width vectors>)
  //     use getKnownMinValue()
  //   else if (<algorithm works only for fixed-width vectors>) {
  //     if <algorithm can be adapted for both scalable and fixed-width vectors>
  //       update the algorithm and use getKnownMinValue()
  //     else
  //       bail out early for scalable vectors and use getFixedValue()
  //   }
  operator uint64_t() const {
#ifdef STRICT_FIXED_SIZE_VECTORS
    return getFixedValue();
#else
    if (isScalable())
      WithColor::warning() << "Compiler has made implicit assumption that "
                              "TypeSize is not scalable. This may or may not "
                              "lead to broken code.\n";
    return getKnownMinValue();
#endif
  }

  // Convenience operators to obtain relative sizes independently of
  // the scalable flag.
  TypeSize operator*(unsigned RHS) const { return {MinVal * RHS, IsScalable}; }

  friend TypeSize operator*(const unsigned LHS, const TypeSize &RHS) {
    return {LHS * RHS.MinVal, RHS.IsScalable};
  }

  // Additional convenience operators needed to avoid ambiguous parses.
  // TODO: Make uint64_t the default operator?
  TypeSize operator*(uint64_t RHS) const { return {MinVal * RHS, IsScalable}; }

  TypeSize operator*(int RHS) const { return {MinVal * RHS, IsScalable}; }

  TypeSize operator*(int64_t RHS) const { return {MinVal * RHS, IsScalable}; }

  friend TypeSize operator*(const uint64_t LHS, const TypeSize &RHS) {
    return {LHS * RHS.MinVal, RHS.IsScalable};
  }

  friend TypeSize operator*(const int LHS, const TypeSize &RHS) {
    return {LHS * RHS.MinVal, RHS.IsScalable};
  }

  friend TypeSize operator*(const int64_t LHS, const TypeSize &RHS) {
    return {LHS * RHS.MinVal, RHS.IsScalable};
  }
};

/// Returns a TypeSize with a known minimum size that is the next integer
/// (mod 2**64) that is greater than or equal to \p Value and is a multiple
/// of \p Align. \p Align must be non-zero.
///
/// Similar to the alignTo functions in MathExtras.h
inline TypeSize alignTo(TypeSize Size, uint64_t Align) {
  assert(Align != 0u && "Align must be non-zero");
  return {(Size.getKnownMinValue() + Align - 1) / Align * Align,
          Size.isScalable()};
}

template <> struct DenseMapInfo<ElementCount> {
  static inline ElementCount getEmptyKey() {
    return ElementCount::getScalable(~0U);
  }
  static inline ElementCount getTombstoneKey() {
    return ElementCount::getFixed(~0U - 1);
  }
  static unsigned getHashValue(const ElementCount& EltCnt) {
    unsigned HashVal = EltCnt.getKnownMinValue() * 37U;
    if (EltCnt.isScalable())
      return (HashVal - 1U);

    return HashVal;
  }

  static bool isEqual(const ElementCount& LHS, const ElementCount& RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_TypeSize_H
