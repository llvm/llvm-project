//===--------- Math.h - Math helpers for the ORC runtime --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Math helper functions for the ORC runtime.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_MATH_H
#define ORC_RT_MATH_H

#include <cstdint>
#include <limits>

namespace orc_rt {

/// Test whether the given value is a power of 2.
template <typename T> constexpr bool isPowerOf2(T Val) noexcept {
  return Val != 0 && (Val & (Val - 1)) == 0;
}

/// Calculates the next power of 2.
template <typename T> constexpr T nextPowerOf2(T Val) noexcept {
  for (size_t I = 1; I < std::numeric_limits<T>::digits; I <<= 1)
    Val |= (Val >> I);
  return Val + 1;
}

} // namespace orc_rt

#endif // ORC_RT_MATH_H
