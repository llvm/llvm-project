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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>

namespace llvm {

/// Reports a diagnostic message to indicate an invalid size request has been
/// done on a scalable vector. This function may not return.
void reportInvalidSizeRequest(const char *Msg);

template <typename LeafTy> struct LinearPolyBaseTypeTraits {};

//===----------------------------------------------------------------------===//
// LinearPolyBase - a base class for linear polynomials with multiple
// dimensions. This can e.g. be used to describe offsets that are have both a
// fixed and scalable component.
//===----------------------------------------------------------------------===//

/// LinearPolyBase describes a linear polynomial:
///  c0 * scale0 + c1 * scale1 + ... + cK * scaleK
/// where the scale is implicit, so only the coefficients are encoded.
template <typename LeafTy>
class LinearPolyBase {
public:
  using ScalarTy = typename LinearPolyBaseTypeTraits<LeafTy>::ScalarTy;
  static constexpr auto Dimensions = LinearPolyBaseTypeTraits<LeafTy>::Dimensions;
  static_assert(Dimensions != std::numeric_limits<unsigned>::max(),
                "Dimensions out of range");

private:
  std::array<ScalarTy, Dimensions> Coefficients;

protected:
  LinearPolyBase(ArrayRef<ScalarTy> Values) {
    std::copy(Values.begin(), Values.end(), Coefficients.begin());
  }

public:
  friend LeafTy &operator+=(LeafTy &LHS, const LeafTy &RHS) {
    for (unsigned I=0; I<Dimensions; ++I)
      LHS.Coefficients[I] += RHS.Coefficients[I];
    return LHS;
  }

  friend LeafTy &operator-=(LeafTy &LHS, const LeafTy &RHS) {
    for (unsigned I=0; I<Dimensions; ++I)
      LHS.Coefficients[I] -= RHS.Coefficients[I];
    return LHS;
  }

  friend LeafTy &operator*=(LeafTy &LHS, ScalarTy RHS) {
    for (auto &C : LHS.Coefficients)
      C *= RHS;
    return LHS;
  }

  friend LeafTy operator+(const LeafTy &LHS, const LeafTy &RHS) {
    LeafTy Copy = LHS;
    return Copy += RHS;
  }

  friend LeafTy operator-(const LeafTy &LHS, const LeafTy &RHS) {
    LeafTy Copy = LHS;
    return Copy -= RHS;
  }

  friend LeafTy operator*(const LeafTy &LHS, ScalarTy RHS) {
    LeafTy Copy = LHS;
    return Copy *= RHS;
  }

  template <typename U = ScalarTy>
  friend typename std::enable_if_t<std::is_signed<U>::value, LeafTy>
  operator-(const LeafTy &LHS) {
    LeafTy Copy = LHS;
    return Copy *= -1;
  }

  bool operator==(const LinearPolyBase &RHS) const {
    return std::equal(Coefficients.begin(), Coefficients.end(),
                      RHS.Coefficients.begin());
  }

  bool operator!=(const LinearPolyBase &RHS) const {
    return !(*this == RHS);
  }

  bool isZero() const {
    return all_of(Coefficients, [](const ScalarTy &C) { return C == 0; });
  }
  bool isNonZero() const { return !isZero(); }
  explicit operator bool() const { return isNonZero(); }

  ScalarTy getValue(unsigned Dim) const { return Coefficients[Dim]; }
};

//===----------------------------------------------------------------------===//
// StackOffset - Represent an offset with named fixed and scalable components.
//===----------------------------------------------------------------------===//

class StackOffset;
template <> struct LinearPolyBaseTypeTraits<StackOffset> {
  using ScalarTy = int64_t;
  static constexpr unsigned Dimensions = 2;
};

/// StackOffset is a class to represent an offset with 2 dimensions,
/// named fixed and scalable, respectively. This class allows a value for both
/// dimensions to depict e.g. "8 bytes and 16 scalable bytes", which is needed
/// to represent stack offsets.
class StackOffset : public LinearPolyBase<StackOffset> {
protected:
  StackOffset(ScalarTy Fixed, ScalarTy Scalable)
      : LinearPolyBase<StackOffset>({Fixed, Scalable}) {}

public:
  StackOffset() : StackOffset({0, 0}) {}
  StackOffset(const LinearPolyBase<StackOffset> &Other)
      : LinearPolyBase<StackOffset>(Other) {}
  static StackOffset getFixed(ScalarTy Fixed) { return {Fixed, 0}; }
  static StackOffset getScalable(ScalarTy Scalable) { return {0, Scalable}; }
  static StackOffset get(ScalarTy Fixed, ScalarTy Scalable) {
    return {Fixed, Scalable};
  }

  ScalarTy getFixed() const { return this->getValue(0); }
  ScalarTy getScalable() const { return this->getValue(1); }
};

//===----------------------------------------------------------------------===//
// UnivariateLinearPolyBase - a base class for linear polynomials with multiple
// dimensions, but where only one dimension can be set at any time.
// This can e.g. be used to describe sizes that are either fixed or scalable.
//===----------------------------------------------------------------------===//

/// UnivariateLinearPolyBase is a base class for ElementCount and TypeSize.
/// Like LinearPolyBase it tries to represent a linear polynomial
/// where only one dimension can be set at any time, e.g.
///   0 * scale0 + 0 * scale1 + ... + cJ * scaleJ + ... + 0 * scaleK
/// The dimension that is set is the univariate dimension.
template <typename LeafTy>
class UnivariateLinearPolyBase {
public:
  using ScalarTy = typename LinearPolyBaseTypeTraits<LeafTy>::ScalarTy;
  static constexpr auto Dimensions = LinearPolyBaseTypeTraits<LeafTy>::Dimensions;
  static_assert(Dimensions != std::numeric_limits<unsigned>::max(),
                "Dimensions out of range");

protected:
  ScalarTy Value;         // The value at the univeriate dimension.
  unsigned UnivariateDim; // The univeriate dimension.

  UnivariateLinearPolyBase(ScalarTy Val, unsigned UnivariateDim)
      : Value(Val), UnivariateDim(UnivariateDim) {
    assert(UnivariateDim < Dimensions && "Dimension out of range");
  }

  friend LeafTy &operator+=(LeafTy &LHS, const LeafTy &RHS) {
    assert(LHS.UnivariateDim == RHS.UnivariateDim && "Invalid dimensions");
    LHS.Value += RHS.Value;
    return LHS;
  }

  friend LeafTy &operator-=(LeafTy &LHS, const LeafTy &RHS) {
    assert(LHS.UnivariateDim == RHS.UnivariateDim && "Invalid dimensions");
    LHS.Value -= RHS.Value;
    return LHS;
  }

  friend LeafTy &operator*=(LeafTy &LHS, ScalarTy RHS) {
    LHS.Value *= RHS;
    return LHS;
  }

  friend LeafTy operator+(const LeafTy &LHS, const LeafTy &RHS) {
    LeafTy Copy = LHS;
    return Copy += RHS;
  }

  friend LeafTy operator-(const LeafTy &LHS, const LeafTy &RHS) {
    LeafTy Copy = LHS;
    return Copy -= RHS;
  }

  friend LeafTy operator*(const LeafTy &LHS, ScalarTy RHS) {
    LeafTy Copy = LHS;
    return Copy *= RHS;
  }

  template <typename U = ScalarTy>
  friend typename std::enable_if<std::is_signed<U>::value, LeafTy>::type
  operator-(const LeafTy &LHS) {
    LeafTy Copy = LHS;
    return Copy *= -1;
  }

public:
  bool operator==(const UnivariateLinearPolyBase &RHS) const {
    return Value == RHS.Value && UnivariateDim == RHS.UnivariateDim;
  }

  bool operator!=(const UnivariateLinearPolyBase &RHS) const {
    return !(*this == RHS);
  }

  bool isZero() const { return !Value; }
  bool isNonZero() const { return !isZero(); }
  explicit operator bool() const { return isNonZero(); }
  ScalarTy getValue(unsigned Dim) const {
    return Dim == UnivariateDim ? Value : 0;
  }

  /// Add \p RHS to the value at the univariate dimension.
  LeafTy getWithIncrement(ScalarTy RHS) const {
    return static_cast<LeafTy>(
        UnivariateLinearPolyBase(Value + RHS, UnivariateDim));
  }

  /// Subtract \p RHS from the value at the univariate dimension.
  LeafTy getWithDecrement(ScalarTy RHS) const {
    return static_cast<LeafTy>(
        UnivariateLinearPolyBase(Value - RHS, UnivariateDim));
  }
};


//===----------------------------------------------------------------------===//
// LinearPolySize - base class for fixed- or scalable sizes.
//  ^  ^
//  |  |
//  |  +----- ElementCount - Leaf class to represent an element count
//  |                        (vscale x unsigned)
//  |
//  +-------- TypeSize - Leaf class to represent a type size
//                       (vscale x uint64_t)
//===----------------------------------------------------------------------===//

/// LinearPolySize is a base class to represent sizes. It is either
/// fixed-sized or it is scalable-sized, but it cannot be both.
template <typename LeafTy>
class LinearPolySize : public UnivariateLinearPolyBase<LeafTy> {
  // Make the parent class a friend, so that it can access the protected
  // conversion/copy-constructor for UnivariatePolyBase<LeafTy> ->
  // LinearPolySize<LeafTy>.
  friend class UnivariateLinearPolyBase<LeafTy>;

public:
  using ScalarTy = typename UnivariateLinearPolyBase<LeafTy>::ScalarTy;
  enum Dims : unsigned { FixedDim = 0, ScalableDim = 1 };

protected:
  LinearPolySize(ScalarTy MinVal, Dims D)
      : UnivariateLinearPolyBase<LeafTy>(MinVal, D) {}

  LinearPolySize(const UnivariateLinearPolyBase<LeafTy> &V)
      : UnivariateLinearPolyBase<LeafTy>(V) {}

public:

  static LeafTy getFixed(ScalarTy MinVal) {
    return static_cast<LeafTy>(LinearPolySize(MinVal, FixedDim));
  }
  static LeafTy getScalable(ScalarTy MinVal) {
    return static_cast<LeafTy>(LinearPolySize(MinVal, ScalableDim));
  }
  static LeafTy get(ScalarTy MinVal, bool Scalable) {
    return static_cast<LeafTy>(
        LinearPolySize(MinVal, Scalable ? ScalableDim : FixedDim));
  }
  static LeafTy getNull() { return get(0, false); }

  /// Returns the minimum value this size can represent.
  ScalarTy getKnownMinValue() const { return this->Value; }
  /// Returns whether the size is scaled by a runtime quantity (vscale).
  bool isScalable() const { return this->UnivariateDim == ScalableDim; }
  /// A return value of true indicates we know at compile time that the number
  /// of elements (vscale * Min) is definitely even. However, returning false
  /// does not guarantee that the total number of elements is odd.
  bool isKnownEven() const { return (getKnownMinValue() & 0x1) == 0; }
  /// This function tells the caller whether the element count is known at
  /// compile time to be a multiple of the scalar value RHS.
  bool isKnownMultipleOf(ScalarTy RHS) const {
    return getKnownMinValue() % RHS == 0;
  }

  // Return the minimum value with the assumption that the count is exact.
  // Use in places where a scalable count doesn't make sense (e.g. non-vector
  // types, or vectors in backends which don't support scalable vectors).
  ScalarTy getFixedValue() const {
    assert(!isScalable() &&
           "Request for a fixed element count on a scalable object");
    return getKnownMinValue();
  }

  // For some cases, size ordering between scalable and fixed size types cannot
  // be determined at compile time, so such comparisons aren't allowed.
  //
  // e.g. <vscale x 2 x i16> could be bigger than <4 x i32> with a runtime
  // vscale >= 5, equal sized with a vscale of 4, and smaller with
  // a vscale <= 3.
  //
  // All the functions below make use of the fact vscale is always >= 1, which
  // means that <vscale x 4 x i32> is guaranteed to be >= <4 x i32>, etc.

  static bool isKnownLT(const LinearPolySize &LHS, const LinearPolySize &RHS) {
    if (!LHS.isScalable() || RHS.isScalable())
      return LHS.getKnownMinValue() < RHS.getKnownMinValue();
    return false;
  }

  static bool isKnownGT(const LinearPolySize &LHS, const LinearPolySize &RHS) {
    if (LHS.isScalable() || !RHS.isScalable())
      return LHS.getKnownMinValue() > RHS.getKnownMinValue();
    return false;
  }

  static bool isKnownLE(const LinearPolySize &LHS, const LinearPolySize &RHS) {
    if (!LHS.isScalable() || RHS.isScalable())
      return LHS.getKnownMinValue() <= RHS.getKnownMinValue();
    return false;
  }

  static bool isKnownGE(const LinearPolySize &LHS, const LinearPolySize &RHS) {
    if (LHS.isScalable() || !RHS.isScalable())
      return LHS.getKnownMinValue() >= RHS.getKnownMinValue();
    return false;
  }

  /// We do not provide the '/' operator here because division for polynomial
  /// types does not work in the same way as for normal integer types. We can
  /// only divide the minimum value (or coefficient) by RHS, which is not the
  /// same as
  ///   (Min * Vscale) / RHS
  /// The caller is recommended to use this function in combination with
  /// isKnownMultipleOf(RHS), which lets the caller know if it's possible to
  /// perform a lossless divide by RHS.
  LeafTy divideCoefficientBy(ScalarTy RHS) const {
    return static_cast<LeafTy>(
        LinearPolySize::get(getKnownMinValue() / RHS, isScalable()));
  }

  LeafTy multiplyCoefficientBy(ScalarTy RHS) const {
    return static_cast<LeafTy>(
        LinearPolySize::get(getKnownMinValue() * RHS, isScalable()));
  }

  LeafTy coefficientNextPowerOf2() const {
    return static_cast<LeafTy>(LinearPolySize::get(
        static_cast<ScalarTy>(llvm::NextPowerOf2(getKnownMinValue())),
        isScalable()));
  }

  /// Printing function.
  void print(raw_ostream &OS) const {
    if (isScalable())
      OS << "vscale x ";
    OS << getKnownMinValue();
  }
};

class ElementCount;
template <> struct LinearPolyBaseTypeTraits<ElementCount> {
  using ScalarTy = unsigned;
  static constexpr unsigned Dimensions = 2;
};

class ElementCount : public LinearPolySize<ElementCount> {
public:
  ElementCount() : LinearPolySize(LinearPolySize::getNull()) {}

  ElementCount(const LinearPolySize<ElementCount> &V) : LinearPolySize(V) {}

  /// Counting predicates.
  ///
  ///@{ Number of elements..
  /// Exactly one element.
  bool isScalar() const { return !isScalable() && getKnownMinValue() == 1; }
  /// One or more elements.
  bool isVector() const {
    return (isScalable() && getKnownMinValue() != 0) || getKnownMinValue() > 1;
  }
  ///@}
};

// This class is used to represent the size of types. If the type is of fixed
class TypeSize;
template <> struct LinearPolyBaseTypeTraits<TypeSize> {
  using ScalarTy = uint64_t;
  static constexpr unsigned Dimensions = 2;
};

// TODO: Most functionality in this class will gradually be phased out
// so it will resemble LinearPolySize as much as possible.
//
// TypeSize is used to represent the size of types. If the type is of fixed
// size, it will represent the exact size. If the type is a scalable vector,
// it will represent the known minimum size.
class TypeSize : public LinearPolySize<TypeSize> {
public:
  TypeSize(const LinearPolySize<TypeSize> &V) : LinearPolySize(V) {}
  TypeSize(ScalarTy MinVal, bool IsScalable)
      : LinearPolySize(LinearPolySize::get(MinVal, IsScalable)) {}

  static TypeSize Fixed(ScalarTy MinVal) { return TypeSize(MinVal, false); }
  static TypeSize Scalable(ScalarTy MinVal) { return TypeSize(MinVal, true); }

  ScalarTy getFixedSize() const { return getFixedValue(); }
  ScalarTy getKnownMinSize() const { return getKnownMinValue(); }

  // All code for this class below this point is needed because of the
  // temporary implicit conversion to uint64_t. The operator overloads are
  // needed because otherwise the conversion of the parent class
  // UnivariateLinearPolyBase -> TypeSize is ambiguous.
  // TODO: Remove the implicit conversion.

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
  operator ScalarTy() const;

  // Additional operators needed to avoid ambiguous parses
  // because of the implicit conversion hack.
  friend TypeSize operator*(const TypeSize &LHS, const int RHS) {
    return LHS * (ScalarTy)RHS;
  }
  friend TypeSize operator*(const TypeSize &LHS, const unsigned RHS) {
    return LHS * (ScalarTy)RHS;
  }
  friend TypeSize operator*(const TypeSize &LHS, const int64_t RHS) {
    return LHS * (ScalarTy)RHS;
  }
  friend TypeSize operator*(const int LHS, const TypeSize &RHS) {
    return RHS * LHS;
  }
  friend TypeSize operator*(const unsigned LHS, const TypeSize &RHS) {
    return RHS * LHS;
  }
  friend TypeSize operator*(const int64_t LHS, const TypeSize &RHS) {
    return RHS * LHS;
  }
  friend TypeSize operator*(const uint64_t LHS, const TypeSize &RHS) {
    return RHS * LHS;
  }
};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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

/// Stream operator function for `LinearPolySize`.
template <typename LeafTy>
inline raw_ostream &operator<<(raw_ostream &OS,
                               const LinearPolySize<LeafTy> &PS) {
  PS.print(OS);
  return OS;
}

template <> struct DenseMapInfo<ElementCount, void> {
  static inline ElementCount getEmptyKey() {
    return ElementCount::getScalable(~0U);
  }
  static inline ElementCount getTombstoneKey() {
    return ElementCount::getFixed(~0U - 1);
  }
  static unsigned getHashValue(const ElementCount &EltCnt) {
    unsigned HashVal = EltCnt.getKnownMinValue() * 37U;
    if (EltCnt.isScalable())
      return (HashVal - 1U);

    return HashVal;
  }

  static bool isEqual(const ElementCount &LHS, const ElementCount &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_TYPESIZE_H
