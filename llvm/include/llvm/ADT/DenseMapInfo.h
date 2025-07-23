//===- llvm/ADT/DenseMapInfo.h - Type traits for DenseMap -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines DenseMapInfo traits for DenseMap.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSEMAPINFO_H
#define LLVM_ADT_DENSEMAPINFO_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace llvm {

namespace densemap::detail {
// A bit mixer with very low latency using one multiplications and one
// xor-shift. The constant is from splitmix64.
inline uint64_t mix(uint64_t x) {
  x *= 0xbf58476d1ce4e5b9u;
  x ^= x >> 31;
  return x;
}
} // namespace densemap::detail

namespace detail {

/// Simplistic combination of 32-bit hash values into 32-bit hash values.
inline unsigned combineHashValue(unsigned a, unsigned b) {
  uint64_t x = (uint64_t)a << 32 | (uint64_t)b;
  return (unsigned)densemap::detail::mix(x);
}

} // end namespace detail

/// An information struct used to provide DenseMap with the various necessary
/// components for a given value type `T`. `Enable` is an optional additional
/// parameter that is used to support SFINAE (generally using std::enable_if_t)
/// in derived DenseMapInfo specializations; in non-SFINAE use cases this should
/// just be `void`.
template<typename T, typename Enable = void>
struct DenseMapInfo {
  // static constexpr T getEmptyKey();
  // static constexpr T getTombstoneKey();
  // static unsigned getHashValue(const T &Val);
  // static bool isEqual(const T &LHS, const T &RHS);
};

// Provide DenseMapInfo for all pointers. Come up with sentinel pointer values
// that are aligned to alignof(T) bytes, but try to avoid requiring T to be
// complete. This allows clients to instantiate DenseMap<T*, ...> with forward
// declared key types. Assume that no pointer key type requires more than 4096
// bytes of alignment.
template<typename T>
struct DenseMapInfo<T*> {
  // The following should hold, but it would require T to be complete:
  // static_assert(alignof(T) <= (1 << Log2MaxAlign),
  //               "DenseMap does not support pointer keys requiring more than "
  //               "Log2MaxAlign bits of alignment");
  static constexpr uintptr_t Log2MaxAlign = 12;

  static constexpr T *getEmptyKey() {
    uintptr_t Val = static_cast<uintptr_t>(-1);
    Val <<= Log2MaxAlign;
    return reinterpret_cast<T*>(Val);
  }

  static constexpr T *getTombstoneKey() {
    uintptr_t Val = static_cast<uintptr_t>(-2);
    Val <<= Log2MaxAlign;
    return reinterpret_cast<T*>(Val);
  }

  static unsigned getHashValue(const T *PtrVal) {
    return (unsigned((uintptr_t)PtrVal) >> 4) ^
           (unsigned((uintptr_t)PtrVal) >> 9);
  }

  static bool isEqual(const T *LHS, const T *RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for chars.
template<> struct DenseMapInfo<char> {
  static constexpr char getEmptyKey() { return ~0; }
  static constexpr char getTombstoneKey() { return ~0 - 1; }
  static unsigned getHashValue(const char& Val) { return Val * 37U; }

  static bool isEqual(const char &LHS, const char &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned chars.
template <> struct DenseMapInfo<unsigned char> {
  static constexpr unsigned char getEmptyKey() { return ~0; }
  static constexpr unsigned char getTombstoneKey() { return ~0 - 1; }
  static unsigned getHashValue(const unsigned char &Val) { return Val * 37U; }

  static bool isEqual(const unsigned char &LHS, const unsigned char &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned shorts.
template <> struct DenseMapInfo<unsigned short> {
  static constexpr unsigned short getEmptyKey() { return 0xFFFF; }
  static constexpr unsigned short getTombstoneKey() { return 0xFFFF - 1; }
  static unsigned getHashValue(const unsigned short &Val) { return Val * 37U; }

  static bool isEqual(const unsigned short &LHS, const unsigned short &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned ints.
template<> struct DenseMapInfo<unsigned> {
  static constexpr unsigned getEmptyKey() { return ~0U; }
  static constexpr unsigned getTombstoneKey() { return ~0U - 1; }
  static unsigned getHashValue(const unsigned& Val) { return Val * 37U; }

  static bool isEqual(const unsigned& LHS, const unsigned& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned longs.
template<> struct DenseMapInfo<unsigned long> {
  static constexpr unsigned long getEmptyKey() { return ~0UL; }
  static constexpr unsigned long getTombstoneKey() { return ~0UL - 1L; }

  static unsigned getHashValue(const unsigned long& Val) {
    if constexpr (sizeof(Val) == 4)
      return DenseMapInfo<unsigned>::getHashValue(Val);
    else
      return densemap::detail::mix(Val);
  }

  static bool isEqual(const unsigned long& LHS, const unsigned long& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned long longs.
template<> struct DenseMapInfo<unsigned long long> {
  static constexpr unsigned long long getEmptyKey() { return ~0ULL; }
  static constexpr unsigned long long getTombstoneKey() { return ~0ULL - 1ULL; }

  static unsigned getHashValue(const unsigned long long& Val) {
    return densemap::detail::mix(Val);
  }

  static bool isEqual(const unsigned long long& LHS,
                      const unsigned long long& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for shorts.
template <> struct DenseMapInfo<short> {
  static constexpr short getEmptyKey() { return 0x7FFF; }
  static constexpr short getTombstoneKey() { return -0x7FFF - 1; }
  static unsigned getHashValue(const short &Val) { return Val * 37U; }
  static bool isEqual(const short &LHS, const short &RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for ints.
template<> struct DenseMapInfo<int> {
  static constexpr int getEmptyKey() { return 0x7fffffff; }
  static constexpr int getTombstoneKey() { return -0x7fffffff - 1; }
  static unsigned getHashValue(const int& Val) { return (unsigned)(Val * 37U); }

  static bool isEqual(const int& LHS, const int& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for longs.
template<> struct DenseMapInfo<long> {
  static constexpr long getEmptyKey() {
    return (1UL << (sizeof(long) * 8 - 1)) - 1UL;
  }

  static constexpr long getTombstoneKey() { return getEmptyKey() - 1L; }

  static unsigned getHashValue(const long& Val) {
    return (unsigned)(Val * 37UL);
  }

  static bool isEqual(const long& LHS, const long& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for long longs.
template<> struct DenseMapInfo<long long> {
  static constexpr long long getEmptyKey() { return 0x7fffffffffffffffLL; }
  static constexpr long long getTombstoneKey() {
    return -0x7fffffffffffffffLL - 1;
  }

  static unsigned getHashValue(const long long& Val) {
    return (unsigned)(Val * 37ULL);
  }

  static bool isEqual(const long long& LHS,
                      const long long& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for all pairs whose members have info.
template<typename T, typename U>
struct DenseMapInfo<std::pair<T, U>> {
  using Pair = std::pair<T, U>;
  using FirstInfo = DenseMapInfo<T>;
  using SecondInfo = DenseMapInfo<U>;

  static constexpr Pair getEmptyKey() {
    return std::make_pair(FirstInfo::getEmptyKey(),
                          SecondInfo::getEmptyKey());
  }

  static constexpr Pair getTombstoneKey() {
    return std::make_pair(FirstInfo::getTombstoneKey(),
                          SecondInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Pair& PairVal) {
    return detail::combineHashValue(FirstInfo::getHashValue(PairVal.first),
                                    SecondInfo::getHashValue(PairVal.second));
  }

  // Expose an additional function intended to be used by other
  // specializations of DenseMapInfo without needing to know how
  // to combine hash values manually
  static unsigned getHashValuePiecewise(const T &First, const U &Second) {
    return detail::combineHashValue(FirstInfo::getHashValue(First),
                                    SecondInfo::getHashValue(Second));
  }

  static bool isEqual(const Pair &LHS, const Pair &RHS) {
    return FirstInfo::isEqual(LHS.first, RHS.first) &&
           SecondInfo::isEqual(LHS.second, RHS.second);
  }
};

// Provide DenseMapInfo for all tuples whose members have info.
template <typename... Ts> struct DenseMapInfo<std::tuple<Ts...>> {
  using Tuple = std::tuple<Ts...>;

  static constexpr Tuple getEmptyKey() {
    return Tuple(DenseMapInfo<Ts>::getEmptyKey()...);
  }

  static constexpr Tuple getTombstoneKey() {
    return Tuple(DenseMapInfo<Ts>::getTombstoneKey()...);
  }

  template <unsigned I>
  static unsigned getHashValueImpl(const Tuple &values, std::false_type) {
    using EltType = std::tuple_element_t<I, Tuple>;
    std::integral_constant<bool, I + 1 == sizeof...(Ts)> atEnd;
    return detail::combineHashValue(
        DenseMapInfo<EltType>::getHashValue(std::get<I>(values)),
        getHashValueImpl<I + 1>(values, atEnd));
  }

  template <unsigned I>
  static unsigned getHashValueImpl(const Tuple &, std::true_type) {
    return 0;
  }

  static unsigned getHashValue(const std::tuple<Ts...> &values) {
    std::integral_constant<bool, 0 == sizeof...(Ts)> atEnd;
    return getHashValueImpl<0>(values, atEnd);
  }

  template <unsigned I>
  static bool isEqualImpl(const Tuple &lhs, const Tuple &rhs, std::false_type) {
    using EltType = std::tuple_element_t<I, Tuple>;
    std::integral_constant<bool, I + 1 == sizeof...(Ts)> atEnd;
    return DenseMapInfo<EltType>::isEqual(std::get<I>(lhs), std::get<I>(rhs)) &&
           isEqualImpl<I + 1>(lhs, rhs, atEnd);
  }

  template <unsigned I>
  static bool isEqualImpl(const Tuple &, const Tuple &, std::true_type) {
    return true;
  }

  static bool isEqual(const Tuple &lhs, const Tuple &rhs) {
    std::integral_constant<bool, 0 == sizeof...(Ts)> atEnd;
    return isEqualImpl<0>(lhs, rhs, atEnd);
  }
};

// Provide DenseMapInfo for enum classes.
template <typename Enum>
struct DenseMapInfo<Enum, std::enable_if_t<std::is_enum_v<Enum>>> {
  using UnderlyingType = std::underlying_type_t<Enum>;
  using Info = DenseMapInfo<UnderlyingType>;

  // If an enum does not have a "fixed" underlying type, it may be UB to cast
  // some values of the underlying type to the enum. We use an "extra" constexpr
  // local to ensure that such UB would trigger "static assertion expression is
  // not an integral constant expression", rather than runtime UB.
  //
  // If you hit this error, you can fix by switching to `enum class`, or adding
  // an explicit underlying type (e.g. `enum X : int`) to the enum's definition.

  static constexpr Enum getEmptyKey() {
    constexpr Enum V = static_cast<Enum>(Info::getEmptyKey());
    return V;
  }

  static constexpr Enum getTombstoneKey() {
    constexpr Enum V = static_cast<Enum>(Info::getTombstoneKey());
    return V;
  }

  static unsigned getHashValue(const Enum &Val) {
    return Info::getHashValue(static_cast<UnderlyingType>(Val));
  }

  static bool isEqual(const Enum &LHS, const Enum &RHS) { return LHS == RHS; }
};

template <typename T> struct DenseMapInfo<std::optional<T>> {
  using Optional = std::optional<T>;
  using Info = DenseMapInfo<T>;

  static constexpr Optional getEmptyKey() { return {Info::getEmptyKey()}; }

  static constexpr Optional getTombstoneKey() {
    return {Info::getTombstoneKey()};
  }

  static unsigned getHashValue(const Optional &OptionalVal) {
    return detail::combineHashValue(
        OptionalVal.has_value(),
        Info::getHashValue(OptionalVal.value_or(Info::getEmptyKey())));
  }

  static bool isEqual(const Optional &LHS, const Optional &RHS) {
    if (LHS && RHS) {
      return Info::isEqual(LHS.value(), RHS.value());
    }
    return !LHS && !RHS;
  }
};
} // end namespace llvm

#endif // LLVM_ADT_DENSEMAPINFO_H
