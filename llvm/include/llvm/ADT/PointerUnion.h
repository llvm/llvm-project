//===- llvm/ADT/PointerUnion.h - Discriminated Union of 2 Ptrs --*- C++ -*-===//
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
#include <cassert>
#include <cstddef>
#include <cstdint>

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
  static constexpr int minLowBitsAvailable() {
    return pointer_union_detail::lowBitsAvailable<PTs...>();
  }

  static constexpr int tagBits() {
    return pointer_union_detail::bitsRequired(sizeof...(PTs));
  }

  /// The tag is shifted to the high end of the available low bits so that
  /// the lowest bits remain free for nesting in PointerIntPair or SmallPtrSet.
  static constexpr int tagShift() { return minLowBitsAvailable() - tagBits(); }

  static constexpr uintptr_t tagMask() {
    return (uintptr_t(1) << tagBits()) - 1;
  }

  template <typename T> static uintptr_t encode(T V) {
    constexpr int Shift = tagShift();
    constexpr auto Tag = uintptr_t(FirstIndexOfType<T, PTs...>::value);
    uintptr_t PtrInt = reinterpret_cast<uintptr_t>(
        PointerLikeTypeTraits<T>::getAsVoidPointer(V));
    assert((PtrInt & (tagMask() << Shift)) == 0 &&
           "Pointer low bits collide with tag");
    return PtrInt | (Tag << Shift);
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
    return (static_cast<uintptr_t>(this->Val.asInt()) >>
            minLowBitsAvailable()) == 0;
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
    constexpr int Shift = From::tagShift();
    constexpr auto Tag = uintptr_t(FirstIndexOfType<To, PTs...>::value);
    auto V = reinterpret_cast<uintptr_t>(F.getOpaqueValue());
    return ((V >> Shift) & From::tagMask()) == Tag;
  }

  static To doCast(From &F) {
    assert(isPossible(F) && "cast to an incompatible type!");
    constexpr uintptr_t PtrMask =
        ~((uintptr_t(1) << PointerLikeTypeTraits<To>::NumLowBitsAvailable) - 1);
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
