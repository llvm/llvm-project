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

#include "llvm/Support/WithColor.h"

#include <cstdint>
#include <cassert>

namespace llvm {

template <typename T> struct DenseMapInfo;

class ElementCount {
public:
  unsigned Min;  // Minimum number of vector elements.
  bool Scalable; // If true, NumElements is a multiple of 'Min' determined
                 // at runtime rather than compile time.

  ElementCount(unsigned Min, bool Scalable)
  : Min(Min), Scalable(Scalable) {}

  ElementCount operator*(unsigned RHS) {
    return { Min * RHS, Scalable };
  }
  ElementCount operator/(unsigned RHS) {
    return { Min / RHS, Scalable };
  }

  bool operator==(const ElementCount& RHS) const {
    return Min == RHS.Min && Scalable == RHS.Scalable;
  }
  bool operator!=(const ElementCount& RHS) const {
    return !(*this == RHS);
  }
};

// This class is used to represent the size of types. If the type is of fixed
// size, it will represent the exact size. If the type is a scalable vector,
// it will represent the known minimum size.
class TypeSize {
  uint64_t MinSize;   // The known minimum size.
  bool IsScalable;    // If true, then the runtime size is an integer multiple
                      // of MinSize.

public:
  constexpr TypeSize(uint64_t MinSize, bool Scalable)
    : MinSize(MinSize), IsScalable(Scalable) {}

  static constexpr TypeSize Fixed(uint64_t Size) {
    return TypeSize(Size, /*IsScalable=*/false);
  }

  static constexpr TypeSize Scalable(uint64_t MinSize) {
    return TypeSize(MinSize, /*IsScalable=*/true);
  }

  // Scalable vector types with the same minimum size as a fixed size type are
  // not guaranteed to be the same size at runtime, so they are never
  // considered to be equal.
  friend bool operator==(const TypeSize &LHS, const TypeSize &RHS) {
    return LHS.MinSize == RHS.MinSize && LHS.IsScalable == RHS.IsScalable;
  }

  friend bool operator!=(const TypeSize &LHS, const TypeSize &RHS) {
    return !(LHS == RHS);
  }

  // For many cases, size ordering between scalable and fixed size types cannot
  // be determined at compile time, so such comparisons aren't allowed.
  //
  // e.g. <vscale x 2 x i16> could be bigger than <4 x i32> with a runtime
  // vscale >= 5, equal sized with a vscale of 4, and smaller with
  // a vscale <= 3.
  //
  // If the scalable flags match, just perform the requested comparison
  // between the minimum sizes.
  friend bool operator<(const TypeSize &LHS, const TypeSize &RHS) {
    assert(LHS.IsScalable == RHS.IsScalable &&
           "Ordering comparison of scalable and fixed types");

    return LHS.MinSize < RHS.MinSize;
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

  // Convenience operators to obtain relative sizes independently of
  // the scalable flag.
  TypeSize operator*(unsigned RHS) const {
    return { MinSize * RHS, IsScalable };
  }

  friend TypeSize operator*(const unsigned LHS, const TypeSize &RHS) {
    return { LHS * RHS.MinSize, RHS.IsScalable };
  }

  TypeSize operator/(unsigned RHS) const {
    return { MinSize / RHS, IsScalable };
  }

  // Return the minimum size with the assumption that the size is exact.
  // Use in places where a scalable size doesn't make sense (e.g. non-vector
  // types, or vectors in backends which don't support scalable vectors).
  uint64_t getFixedSize() const {
    assert(!IsScalable && "Request for a fixed size on a scalable object");
    return MinSize;
  }

  // Return the known minimum size. Use in places where the scalable property
  // doesn't matter (e.g. determining alignment) or in conjunction with the
  // isScalable method below.
  uint64_t getKnownMinSize() const {
    return MinSize;
  }

  // Return whether or not the size is scalable.
  bool isScalable() const {
    return IsScalable;
  }

  // Returns true if the number of bits is a multiple of an 8-bit byte.
  bool isByteSized() const {
    return (MinSize & 7) == 0;
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
  //     use getKnownMinSize()
  //   else if (<algorithm works only for fixed-width vectors>) {
  //     if <algorithm can be adapted for both scalable and fixed-width vectors>
  //       update the algorithm and use getKnownMinSize()
  //     else
  //       bail out early for scalable vectors and use getFixedSize()
  //   }
  operator uint64_t() const {
#ifdef STRICT_FIXED_SIZE_VECTORS
    return getFixedSize();
#else
    if (isScalable())
      WithColor::warning() << "Compiler has made implicit assumption that "
                              "TypeSize is not scalable. This may or may not "
                              "lead to broken code.\n";
    return getKnownMinSize();
#endif
  }

  // Additional convenience operators needed to avoid ambiguous parses.
  // TODO: Make uint64_t the default operator?
  TypeSize operator*(uint64_t RHS) const {
    return { MinSize * RHS, IsScalable };
  }

  TypeSize operator*(int RHS) const {
    return { MinSize * RHS, IsScalable };
  }

  TypeSize operator*(int64_t RHS) const {
    return { MinSize * RHS, IsScalable };
  }

  friend TypeSize operator*(const uint64_t LHS, const TypeSize &RHS) {
    return { LHS * RHS.MinSize, RHS.IsScalable };
  }

  friend TypeSize operator*(const int LHS, const TypeSize &RHS) {
    return { LHS * RHS.MinSize, RHS.IsScalable };
  }

  friend TypeSize operator*(const int64_t LHS, const TypeSize &RHS) {
    return { LHS * RHS.MinSize, RHS.IsScalable };
  }

  TypeSize operator/(uint64_t RHS) const {
    return { MinSize / RHS, IsScalable };
  }

  TypeSize operator/(int RHS) const {
    return { MinSize / RHS, IsScalable };
  }

  TypeSize operator/(int64_t RHS) const {
    return { MinSize / RHS, IsScalable };
  }
};

/// Returns a TypeSize with a known minimum size that is the next integer
/// (mod 2**64) that is greater than or equal to \p Value and is a multiple
/// of \p Align. \p Align must be non-zero.
///
/// Similar to the alignTo functions in MathExtras.h
inline TypeSize alignTo(TypeSize Size, uint64_t Align) {
  assert(Align != 0u && "Align must be non-zero");
  return {(Size.getKnownMinSize() + Align - 1) / Align * Align,
          Size.isScalable()};
}

template <> struct DenseMapInfo<ElementCount> {
  static inline ElementCount getEmptyKey() { return {~0U, true}; }
  static inline ElementCount getTombstoneKey() { return {~0U - 1, false}; }
  static unsigned getHashValue(const ElementCount& EltCnt) {
    if (EltCnt.Scalable)
      return (EltCnt.Min * 37U) - 1U;

    return EltCnt.Min * 37U;
  }

  static bool isEqual(const ElementCount& LHS, const ElementCount& RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_TypeSize_H
