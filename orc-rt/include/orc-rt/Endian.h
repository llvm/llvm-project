//===----- Endian.h - Endianness helpers for the ORC runtime ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Endianness helper functions for the ORC runtime.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_ENDIAN_H
#define ORC_RT_ENDIAN_H

#include "bit.h"
#include <cstring>
#include <type_traits>

namespace orc_rt {

/// Read a value with the given endianness from memory.
template <typename T>
[[nodiscard]] inline std::enable_if_t<std::is_integral_v<T>, T>
endian_read(const void *Src, orc_rt::endian E) noexcept {
  T Val;
  memcpy(&Val, Src, sizeof(T));
  if (E != orc_rt::endian::native)
    Val = orc_rt::byteswap(Val);
  return Val;
}

/// Write a value with the given endianness to memory.
template <typename T>
inline std::enable_if_t<std::is_integral_v<T>>
endian_write(void *Dst, T Val, orc_rt::endian E) noexcept {
  if (E != orc_rt::endian::native)
    Val = orc_rt::byteswap(Val);
  memcpy(Dst, &Val, sizeof(T));
}

} // namespace orc_rt

#endif // ORC_RT_ENDIAN_H
