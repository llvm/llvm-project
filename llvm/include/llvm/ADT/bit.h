//===-- llvm/ADT/bit.h - C++20 <bit> ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the C++20 <bit> header.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_BIT_H
#define LLVM_ADT_BIT_H

#include <cstring>
#include <type_traits>

#include <stdint.h>

namespace llvm {

// This implementation of bit_cast is different from the C++20 one in two ways:
//  - It isn't constexpr because that requires compiler support.
//  - It requires trivially-constructible To, to avoid UB in the implementation.
template <
    typename To, typename From,
    typename = std::enable_if_t<sizeof(To) == sizeof(From)>,
    typename = std::enable_if_t<std::is_trivially_constructible<To>::value>,
    typename = std::enable_if_t<std::is_trivially_copyable<To>::value>,
    typename = std::enable_if_t<std::is_trivially_copyable<From>::value>>
inline To bit_cast(const From &from) noexcept {
  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
constexpr inline bool has_single_bit(T Value) noexcept {
  return (Value != 0) && ((Value & (Value - 1)) == 0);
}

namespace detail {
template <typename T, std::size_t SizeOfT> struct PopulationCounter {
  static int count(T Value) {
    // Generic version, forward to 32 bits.
    static_assert(SizeOfT <= 4, "Not implemented!");
#if defined(__GNUC__)
    return (int)__builtin_popcount(Value);
#else
    uint32_t v = Value;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return int(((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24);
#endif
  }
};

template <typename T> struct PopulationCounter<T, 8> {
  static int count(T Value) {
#if defined(__GNUC__)
    return (int)__builtin_popcountll(Value);
#else
    uint64_t v = Value;
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return int((uint64_t)(v * 0x0101010101010101ULL) >> 56);
#endif
  }
};
} // namespace detail

/// Count the number of set bits in a value.
/// Ex. popcount(0xF000F000) = 8
/// Returns 0 if the word is zero.
template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
inline int popcount(T Value) noexcept {
  return detail::PopulationCounter<T, sizeof(T)>::count(Value);
}

} // namespace llvm

#endif
