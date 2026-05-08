//===- llvm/ADT/PointerUnion.h - Pointer Type Union -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the PointerUnion class, which is a discriminated union of
/// pointer types.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_POINTERUNION_H
#define LLVM_ADT_POINTERUNION_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace llvm {

namespace pointer_union_detail {

/// Determine the number of bits required to store values in [0, NumValues).
/// This is ceil(log2(NumValues)).
constexpr int bitsRequired(unsigned NumValues) {
  return NumValues == 0 ? 0 : llvm::bit_width_constexpr(NumValues - 1);
}

template <typename... Ts> constexpr int lowBitsAvailable() {
  return std::min(
      {static_cast<int>(PointerLikeTypeTraits<Ts>::NumLowBitsAvailable)...});
}

/// True if all types have enough low bits for a fixed-width tag.
template <typename... PTs> constexpr bool useFixedWidthTags() {
  return lowBitsAvailable<PTs...>() >= bitsRequired(sizeof...(PTs));
}

/// True if types are in non-decreasing NumLowBitsAvailable order.
// TODO: Switch to llvm::is_sorted when it becomes constexpr.
template <typename... PTs> constexpr bool typesInNonDecreasingBitOrder() {
  int Bits[] = {PointerLikeTypeTraits<PTs>::NumLowBitsAvailable...};
  for (size_t I = 1; I < sizeof...(PTs); ++I)
    if (Bits[I] < Bits[I - 1])
      return false;
  return true;
}

/// Tag descriptor for one type in the union.
struct TagEntry {
  uintptr_t Value; // Bit pattern stored in the low bits.
  uintptr_t Mask;  // Mask covering all tag bits for this entry.
};

/// Compute fixed-width tag table (all types have enough bits for the tag).
/// For example, with 4 types and 3 available bits, the tag is 2 bits wide
/// (values 0-3) and each entry has the same mask of 0x3.
template <typename... PTs>
constexpr std::array<TagEntry, sizeof...(PTs)> computeFixedTags() {
  constexpr size_t N = sizeof...(PTs);
  constexpr uintptr_t TagMask = (uintptr_t(1) << bitsRequired(N)) - 1;
  std::array<TagEntry, N> Result = {};
  for (size_t I = 0; I < N; ++I) {
    Result[I].Value = uintptr_t(I);
    Result[I].Mask = TagMask;
  }
  return Result;
}

/// Compute variable-width tag table, or return std::nullopt if the types
/// don't fit. Types must be in non-decreasing NumLowBitsAvailable order.
/// Groups types by available bits into tiers; each non-final tier reserves
/// its highest code as an escape prefix.
///
/// Example with 3 tiers (2-bit, 3-bit, 5-bit types):
///   Tier 0 (2 bits): codes 0b00, 0b01, 0b10; escape = 0b11
///   Tier 1 (3 bits): codes 0b011, escape = 0b111
///   Tier 2 (5 bits): codes 0b00111, 0b01111, 0b10111, 0b11111
template <typename... PTs>
constexpr std::optional<std::array<TagEntry, sizeof...(PTs)>>
computeExtendedTags() {
  constexpr size_t N = sizeof...(PTs);
  std::array<TagEntry, N> Result = {};
  int Bits[] = {PointerLikeTypeTraits<PTs>::NumLowBitsAvailable...};
  uintptr_t EscapePrefix = 0;
  int PrevBits = 0;
  size_t I = 0;
  // Walk tiers (groups of types with the same NumLowBitsAvailable). For each
  // tier, assign tag values using the new bits introduced by this tier,
  // prefixed by the accumulated escape codes from previous tiers. Non-final
  // tiers reserve their highest code as an escape to the next tier.
  while (I < N) {
    int TierBits = Bits[I];
    if (TierBits < PrevBits)
      return std::nullopt;
    int NewBits = TierBits - PrevBits;
    size_t TierEnd = I;
    while (TierEnd < N && Bits[TierEnd] == TierBits)
      ++TierEnd;
    bool IsLastTier = (TierEnd == N);
    size_t TypesInTier = TierEnd - I;
    size_t Capacity =
        IsLastTier ? (size_t(1) << NewBits) : ((size_t(1) << NewBits) - 1);
    if (TypesInTier > Capacity)
      return std::nullopt;
    for (size_t J = 0; J < TypesInTier; ++J) {
      Result[I + J].Value = EscapePrefix | (uintptr_t(J) << PrevBits);
      Result[I + J].Mask = (uintptr_t(1) << TierBits) - 1;
    }
    uintptr_t EscapeCode = (uintptr_t(1) << NewBits) - 1;
    EscapePrefix |= EscapeCode << PrevBits;
    PrevBits = TierBits;
    I = TierEnd;
  }
  return Result;
}

/// CRTP base that generates non-template constructors and assignment operators
/// for each type in the union. Non-template constructors allow implicit
/// conversions (derived-to-base, non-const-to-const).
template <typename Derived, int Idx, typename... Types>
class PointerUnionMembers;

template <typename Derived, int Idx> class PointerUnionMembers<Derived, Idx> {
protected:
  detail::PunnedPointer<void *> Val;
  PointerUnionMembers() : Val(uintptr_t(0)) {}

  template <typename To, typename From, typename Enable>
  friend struct ::llvm::CastInfo;
  template <typename> friend struct ::llvm::PointerLikeTypeTraits;
};

template <typename Derived, int Idx, typename Type, typename... Types>
class PointerUnionMembers<Derived, Idx, Type, Types...>
    : public PointerUnionMembers<Derived, Idx + 1, Types...> {
  using Base = PointerUnionMembers<Derived, Idx + 1, Types...>;

public:
  using Base::Base;
  PointerUnionMembers() = default;

  PointerUnionMembers(Type V) { this->Val = Derived::encode(V); }

  using Base::operator=;
  Derived &operator=(Type V) {
    this->Val = Derived::encode(V);
    return static_cast<Derived &>(*this);
  }
};

} // end namespace pointer_union_detail

/// A discriminated union of two or more pointer types, with the discriminator
/// in the low bits of the pointer.
///
/// This implementation is extremely efficient in space due to leveraging the
/// low bits of the pointer, while exposing a natural and type-safe API.
///
/// When all types have enough alignment for a fixed-width tag,
/// the tag is placed in the high end of the available low bits, leaving spare
/// low bits for nesting in PointerIntPair or SmallPtrSet. When types have
/// heterogeneous alignment, a variable-length escape-encoded tag
/// is used; in that case, types must be listed in non-decreasing
/// NumLowBitsAvailable order.
///
/// Common use patterns would be something like this:
///    PointerUnion<int*, float*> P;
///    P = (int*)0;
///    printf("%d %d", P.is<int*>(), P.is<float*>());  // prints "1 0"
///    X = P.get<int*>();     // ok.
///    Y = P.get<float*>();   // runtime assertion failure.
///    Z = P.get<double*>();  // compile time failure.
///    P = (float*)0;
///    Y = P.get<float*>();   // ok.
///    X = P.get<int*>();     // runtime assertion failure.
///    PointerUnion<int*, int*> Q; // compile time failure.
template <typename... PTs>
class PointerUnion
    : public pointer_union_detail::PointerUnionMembers<PointerUnion<PTs...>, 0,
                                                       PTs...> {
  static_assert(sizeof...(PTs) > 0, "PointerUnion must have at least one type");
  static_assert(TypesAreDistinct<PTs...>::value,
                "PointerUnion alternative types cannot be repeated");

  using Base = typename PointerUnion::PointerUnionMembers;
  using First = TypeAtIndex<0, PTs...>;

  template <typename, int, typename...>
  friend class pointer_union_detail::PointerUnionMembers;
  template <typename To, typename From, typename Enable> friend struct CastInfo;
  template <typename> friend struct PointerLikeTypeTraits;

  // These are constexpr functions rather than static constexpr data members
  // so that alignof() on potentially incomplete types is not evaluated at
  // class-definition time.

  static constexpr bool useFixedWidthTags() {
    return pointer_union_detail::useFixedWidthTags<PTs...>();
  }

  static constexpr int minLowBitsAvailable() {
    return pointer_union_detail::lowBitsAvailable<PTs...>();
  }

  static constexpr int tagBits() {
    return pointer_union_detail::bitsRequired(sizeof...(PTs));
  }

  /// When using fixed-width tags, the tag is shifted to the high end of the
  /// available low bits so that the lowest bits remain free for nesting. With
  /// variable-width encoding mode, the tag starts at bit 0.
  static constexpr int tagShift() {
    return useFixedWidthTags() ? (minLowBitsAvailable() - tagBits()) : 0;
  }

  using TagTable = std::array<pointer_union_detail::TagEntry, sizeof...(PTs)>;

  /// Returns the tag lookup table for this union's encoding scheme.
  static constexpr TagTable getTagTable() {
    if constexpr (useFixedWidthTags()) {
      return pointer_union_detail::computeFixedTags<PTs...>();
    } else {
      static_assert(
          pointer_union_detail::typesInNonDecreasingBitOrder<PTs...>(),
          "Variable-width PointerUnion types must be in non-decreasing "
          "NumLowBitsAvailable order");
      constexpr auto Table =
          pointer_union_detail::computeExtendedTags<PTs...>();
      static_assert(Table.has_value(),
                    "Too many types for the available low bits");
      return *Table;
    }
  }

  // Variable-width isNull: check membership in the sparse set of tag values.
  // A single threshold comparison does not work here because lower-tier
  // non-null pointers can encode to values below higher-tier thresholds.
  template <size_t... Is>
  static constexpr bool isNullVariableImpl(uintptr_t V,
                                           std::index_sequence<Is...>) {
    constexpr TagTable Table = getTagTable();
    static_assert(tagShift() == 0,
                  "isNullVariableImpl assumes tag starts at bit 0");
    return ((V == Table[Is].Value) || ...);
  }

  template <typename T> static uintptr_t encode(T V) {
    constexpr TagTable Table = getTagTable();
    constexpr int Shift = tagShift();
    constexpr size_t Idx = FirstIndexOfType<T, PTs...>::value;
    static_assert(Table[0].Value == 0,
                  "First type must have tag value 0 for getAddrOfPtr1");
    uintptr_t PtrInt = reinterpret_cast<uintptr_t>(
        PointerLikeTypeTraits<T>::getAsVoidPointer(V));
    assert((PtrInt & (Table[Idx].Mask << Shift)) == 0 &&
           "Pointer low bits collide with tag");
    return PtrInt | (Table[Idx].Value << Shift);
  }

public:
  PointerUnion() = default;
  PointerUnion(std::nullptr_t) : PointerUnion() {}
  using Base::Base;
  using Base::operator=;

  /// Assignment from nullptr clears the union, resetting to the first type.
  const PointerUnion &operator=(std::nullptr_t) {
    this->Val = uintptr_t(0);
    return *this;
  }

  /// Test if the pointer held in the union is null, regardless of
  /// which type it is.
  bool isNull() const {
    if constexpr (useFixedWidthTags()) {
      return (static_cast<uintptr_t>(this->Val.asInt()) >>
              minLowBitsAvailable()) == 0;
    } else {
      return isNullVariableImpl(static_cast<uintptr_t>(this->Val.asInt()),
                                std::index_sequence_for<PTs...>{});
    }
  }

  explicit operator bool() const { return !isNull(); }

  // FIXME: Replace the uses of is(), get() and dyn_cast() with
  //        isa<T>, cast<T> and the llvm::dyn_cast<T>

  /// Test if the Union currently holds the type matching T.
  template <typename T> [[deprecated("Use isa instead")]] bool is() const {
    return isa<T>(*this);
  }

  /// Returns the value of the specified pointer type.
  ///
  /// If the specified pointer type is incorrect, assert.
  template <typename T> [[deprecated("Use cast instead")]] T get() const {
    assert(isa<T>(*this) && "Invalid accessor called");
    return cast<T>(*this);
  }

  /// Returns the current pointer if it is of the specified pointer type,
  /// otherwise returns null.
  template <typename T> inline T dyn_cast() const {
    return llvm::dyn_cast_if_present<T>(*this);
  }

  /// If the union is set to the first pointer type get an address pointing to
  /// it.
  First const *getAddrOfPtr1() const {
    return const_cast<PointerUnion *>(this)->getAddrOfPtr1();
  }

  /// If the union is set to the first pointer type get an address pointing to
  /// it.
  First *getAddrOfPtr1() {
    static_assert(FirstIndexOfType<First, PTs...>::value == 0,
                  "First type must have tag value 0 for getAddrOfPtr1");
    assert(isa<First>(*this) && "Val is not the first pointer");
    // tag == 0 for first type, so asInt() is the raw pointer value.
    assert(
        PointerLikeTypeTraits<First>::getAsVoidPointer(cast<First>(*this)) ==
            reinterpret_cast<void *>(this->Val.asInt()) &&
        "Can't get the address because PointerLikeTypeTraits changes the ptr");
    return const_cast<First *>(
        reinterpret_cast<const First *>(this->Val.getPointerAddress()));
  }

  void *getOpaqueValue() const {
    return reinterpret_cast<void *>(this->Val.asInt());
  }

  static inline PointerUnion getFromOpaqueValue(void *VP) {
    PointerUnion V;
    V.Val = reinterpret_cast<intptr_t>(VP);
    return V;
  }

  friend bool operator==(PointerUnion lhs, PointerUnion rhs) {
    return lhs.getOpaqueValue() == rhs.getOpaqueValue();
  }

  friend bool operator!=(PointerUnion lhs, PointerUnion rhs) {
    return lhs.getOpaqueValue() != rhs.getOpaqueValue();
  }

  friend bool operator<(PointerUnion lhs, PointerUnion rhs) {
    return lhs.getOpaqueValue() < rhs.getOpaqueValue();
  }
};

// Specialization of CastInfo for PointerUnion.
template <typename To, typename... PTs>
struct CastInfo<To, PointerUnion<PTs...>>
    : public DefaultDoCastIfPossible<To, PointerUnion<PTs...>,
                                     CastInfo<To, PointerUnion<PTs...>>> {
  using From = PointerUnion<PTs...>;

  static inline bool isPossible(From &F) {
    constexpr std::array<pointer_union_detail::TagEntry, sizeof...(PTs)> Table =
        From::getTagTable();
    constexpr int Shift = From::tagShift();
    constexpr size_t Idx = FirstIndexOfType<To, PTs...>::value;
    auto V = reinterpret_cast<uintptr_t>(F.getOpaqueValue());
    constexpr uintptr_t TagMask = Table[Idx].Mask << Shift;
    constexpr uintptr_t TagValue = Table[Idx].Value << Shift;
    return (V & TagMask) == TagValue;
  }

  static To doCast(From &F) {
    assert(isPossible(F) && "cast to an incompatible type!");
    constexpr std::array<pointer_union_detail::TagEntry, sizeof...(PTs)> Table =
        From::getTagTable();
    constexpr int Shift = From::tagShift();
    constexpr size_t Idx = FirstIndexOfType<To, PTs...>::value;
    constexpr uintptr_t PtrMask = ~(uintptr_t(Table[Idx].Mask) << Shift);
    void *Ptr = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(F.getOpaqueValue()) & PtrMask);
    return PointerLikeTypeTraits<To>::getFromVoidPointer(Ptr);
  }

  static inline To castFailed() { return To(); }
};

template <typename To, typename... PTs>
struct CastInfo<To, const PointerUnion<PTs...>>
    : public ConstStrippingForwardingCast<To, const PointerUnion<PTs...>,
                                          CastInfo<To, PointerUnion<PTs...>>> {
};

// Teach SmallPtrSet that PointerUnion is "basically a pointer".
// Spare low bits below the tag are available for nesting.
// This specialization is only instantiated when used (lazy), so
// PointerLikeTypeTraits<PTs> / alignof() are not evaluated for
// incomplete types.
template <typename... PTs> struct PointerLikeTypeTraits<PointerUnion<PTs...>> {
  using Union = PointerUnion<PTs...>;

  static inline void *getAsVoidPointer(const Union &P) {
    return P.getOpaqueValue();
  }

  static inline Union getFromVoidPointer(void *P) {
    return Union::getFromOpaqueValue(P);
  }

  // The number of bits available are the min of the pointer types minus the
  // bits needed for the discriminator.
  static constexpr int NumLowBitsAvailable = Union::tagShift();
};

// Teach DenseMap how to use PointerUnions as keys.
template <typename... PTs> struct DenseMapInfo<PointerUnion<PTs...>> {
  using Union = PointerUnion<PTs...>;
  using FirstInfo = DenseMapInfo<TypeAtIndex<0, PTs...>>;

  static inline Union getEmptyKey() { return Union(FirstInfo::getEmptyKey()); }

  static inline Union getTombstoneKey() {
    return Union(FirstInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Union &UnionVal) {
    auto Key = reinterpret_cast<uintptr_t>(UnionVal.getOpaqueValue());
    return DenseMapInfo<uintptr_t>::getHashValue(Key);
  }

  static bool isEqual(const Union &LHS, const Union &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_ADT_POINTERUNION_H
