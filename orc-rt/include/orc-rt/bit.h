//===-------- bit.h - Substitute for future STL bit APIs -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Substitutes for STL <bit> APIs that aren't available to the ORC runtime yet.
//
// TODO: Replace all uses once the respective APIs are available.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BIT_H
#define ORC_RT_BIT_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#if defined(_MSC_VER) && !defined(_DEBUG)
#include <stdlib.h>
#endif

#if defined(__linux__) || defined(__GNU__) || defined(__HAIKU__) ||            \
    defined(__Fuchsia__) || defined(__EMSCRIPTEN__)
#include <endian.h>
#elif defined(_AIX)
#include <sys/machine.h>
#elif defined(__sun)
/* Solaris provides _BIG_ENDIAN/_LITTLE_ENDIAN selector in sys/types.h */
#include <sys/types.h>
#define BIG_ENDIAN 4321
#define LITTLE_ENDIAN 1234
#if defined(_BIG_ENDIAN)
#define BYTE_ORDER BIG_ENDIAN
#else
#define BYTE_ORDER LITTLE_ENDIAN
#endif
#elif defined(__MVS__)
#define BIG_ENDIAN 4321
#define LITTLE_ENDIAN 1234
#define BYTE_ORDER BIG_ENDIAN
#else
#if !defined(BYTE_ORDER) && !defined(_WIN32)
#include <machine/endian.h>
#endif
#endif

namespace orc_rt {

enum class endian {
  big,
  little,
#if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && BYTE_ORDER == BIG_ENDIAN
  native = big
#else
  native = little
#endif
};

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
[[nodiscard]] constexpr T byteswap(T V) noexcept {
  // Implementation taken from llvm/include/ADT/bit.h.
  if constexpr (sizeof(T) == 1) {
    return V;
  } else if constexpr (sizeof(T) == 2) {
    uint16_t UV = V;
#if defined(_MSC_VER) && !defined(_DEBUG)
    // The DLL version of the runtime lacks these functions (bug!?), but in a
    // release build they're replaced with BSWAP instructions anyway.
    return _byteswap_ushort(UV);
#else
    uint16_t Hi = UV << 8;
    uint16_t Lo = UV >> 8;
    return Hi | Lo;
#endif
  } else if constexpr (sizeof(T) == 4) {
    uint32_t UV = V;
#if __has_builtin(__builtin_bswap32)
    return __builtin_bswap32(UV);
#elif defined(_MSC_VER) && !defined(_DEBUG)
    return _byteswap_ulong(UV);
#else
    uint32_t Byte0 = UV & 0x000000FF;
    uint32_t Byte1 = UV & 0x0000FF00;
    uint32_t Byte2 = UV & 0x00FF0000;
    uint32_t Byte3 = UV & 0xFF000000;
    return (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24);
#endif
  } else if constexpr (sizeof(T) == 8) {
    uint64_t UV = V;
#if __has_builtin(__builtin_bswap64)
    return __builtin_bswap64(UV);
#elif defined(_MSC_VER) && !defined(_DEBUG)
    return _byteswap_uint64(UV);
#else
    uint64_t Hi = byteswap<uint32_t>(UV);
    uint32_t Lo = byteswap<uint32_t>(UV >> 32);
    return (Hi << 32) | Lo;
#endif
  } else {
    static_assert(!sizeof(T *), "Don't know how to handle the given type.");
    return 0;
  }
}

/// Calculates the number of leading zeros.
template <typename T, typename _ = std::enable_if_t<std::is_unsigned_v<T>>>
[[nodiscard]] constexpr int countl_zero(T Val) noexcept {
  if (!Val)
    return std::numeric_limits<T>::digits;

  unsigned ZeroBits = 0;
  for (T Shift = std::numeric_limits<T>::digits >> 1; Shift; Shift >>= 1) {
    T Tmp = Val >> Shift;
    if (Tmp)
      Val = Tmp;
    else
      ZeroBits |= Shift;
  }
  return ZeroBits;
}

template <typename T, typename _ = std::enable_if_t<std::is_unsigned_v<T>>>
[[nodiscard]] constexpr int bit_width(T x) noexcept {
  return std::numeric_limits<T>::digits - countl_zero(x);
}

} // namespace orc_rt

#endif // ORC_RT_BIT_H
