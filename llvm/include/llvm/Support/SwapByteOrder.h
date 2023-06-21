//===- SwapByteOrder.h - Generic and optimized byte swaps -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares generic and optimized functions to swap the byte order of
// an integral type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SWAPBYTEORDER_H
#define LLVM_SUPPORT_SWAPBYTEORDER_H

#include "llvm/ADT/bit.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

namespace llvm {

namespace sys {

#if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && BYTE_ORDER == BIG_ENDIAN
constexpr bool IsBigEndianHost = true;
#else
constexpr bool IsBigEndianHost = false;
#endif

static const bool IsLittleEndianHost = !IsBigEndianHost;

inline unsigned char      getSwappedBytes(unsigned char      C) { return llvm::byteswap(C); }
inline   signed char      getSwappedBytes( signed  char      C) { return llvm::byteswap(C); }
inline          char      getSwappedBytes(         char      C) { return llvm::byteswap(C); }

inline unsigned short     getSwappedBytes(unsigned short     C) { return llvm::byteswap(C); }
inline   signed short     getSwappedBytes(  signed short     C) { return llvm::byteswap(C); }

inline unsigned int       getSwappedBytes(unsigned int       C) { return llvm::byteswap(C); }
inline   signed int       getSwappedBytes(  signed int       C) { return llvm::byteswap(C); }

inline unsigned long      getSwappedBytes(unsigned long      C) { return llvm::byteswap(C); }
inline   signed long      getSwappedBytes(  signed long      C) { return llvm::byteswap(C); }

inline unsigned long long getSwappedBytes(unsigned long long C) { return llvm::byteswap(C); }
inline   signed long long getSwappedBytes(  signed long long C) { return llvm::byteswap(C); }

inline float getSwappedBytes(float C) {
  union {
    uint32_t i;
    float f;
  } in, out;
  in.f = C;
  out.i = llvm::byteswap(in.i);
  return out.f;
}

inline double getSwappedBytes(double C) {
  union {
    uint64_t i;
    double d;
  } in, out;
  in.d = C;
  out.i = llvm::byteswap(in.i);
  return out.d;
}

template <typename T>
inline std::enable_if_t<std::is_enum_v<T>, T> getSwappedBytes(T C) {
  return static_cast<T>(
      llvm::byteswap(static_cast<std::underlying_type_t<T>>(C)));
}

template<typename T>
inline void swapByteOrder(T &Value) {
  Value = getSwappedBytes(Value);
}

} // end namespace sys
} // end namespace llvm

#endif
