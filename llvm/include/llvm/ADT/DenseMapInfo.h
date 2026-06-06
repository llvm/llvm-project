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
#include <limits>
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

  static unsigned getHashValue(const T *PtrVal) {
    return densemap::detail::mix(reinterpret_cast<uintptr_t>(PtrVal));
  }

  static bool isEqual(const T *LHS, const T *RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for all integral types.
template <typename T>
struct DenseMapInfo<T, std::enable_if_t<std::is_integral_v<T>>> {
  static unsigned getHashValue(const T &Val) {
    if constexpr (std::is_unsigned_v<T> && sizeof(T) > sizeof(unsigned))
      return densemap::detail::mix(Val);
    else
      return static_cast<unsigned>(Val *
                                   static_cast<std::make_unsigned_t<T>>(37U));
  }

  static bool isEqual(const T &LHS, const T &RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for all pairs whose members have info.
template<typename T, typename U>
struct DenseMapInfo<std::pair<T, U>> {
  using Pair = std::pair<T, U>;
  using FirstInfo = DenseMapInfo<T>;
  using SecondInfo = DenseMapInfo<U>;

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

  template <unsigned I> static unsigned getHashValueImpl(const Tuple &values) {
    if constexpr (I == sizeof...(Ts)) {
      return 0;
    } else {
      using EltType = std::tuple_element_t<I, Tuple>;
      return detail::combineHashValue(
          DenseMapInfo<EltType>::getHashValue(std::get<I>(values)),
          getHashValueImpl<I + 1>(values));
    }
  }

  static unsigned getHashValue(const std::tuple<Ts...> &values) {
    return getHashValueImpl<0>(values);
  }

  template <std::size_t... Is>
  static bool isEqualImpl(const Tuple &lhs, const Tuple &rhs,
                          std::index_sequence<Is...>) {
    return (DenseMapInfo<std::tuple_element_t<Is, Tuple>>::isEqual(
                std::get<Is>(lhs), std::get<Is>(rhs)) &&
            ...);
  }

  static bool isEqual(const Tuple &lhs, const Tuple &rhs) {
    return isEqualImpl(lhs, rhs, std::index_sequence_for<Ts...>{});
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

  static unsigned getHashValue(const Enum &Val) {
    return Info::getHashValue(static_cast<UnderlyingType>(Val));
  }

  static bool isEqual(const Enum &LHS, const Enum &RHS) { return LHS == RHS; }
};

template <typename T> struct DenseMapInfo<std::optional<T>> {
  using Optional = std::optional<T>;
  using Info = DenseMapInfo<T>;

  static unsigned getHashValue(const Optional &OptionalVal) {
    if (OptionalVal)
      return detail::combineHashValue(1, Info::getHashValue(*OptionalVal));
    return 0;
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
