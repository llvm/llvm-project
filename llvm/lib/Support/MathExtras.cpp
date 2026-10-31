//===-- MathExtras.cpp - Implement the MathExtras header --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MathExtras.h header
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/APInt.h"

#ifdef _MSC_VER
#include <limits>
#else
#include <cmath>
#endif

namespace llvm {

#if defined(_MSC_VER)
  // Visual Studio defines the HUGE_VAL class of macros using purposeful
  // constant arithmetic overflow, which it then warns on when encountered.
  const float huge_valf = std::numeric_limits<float>::infinity();
#else
  const float huge_valf = HUGE_VALF;
#endif

  /// Returns the number of digits in the given integer.
  int NumDigitsBase10(uint64_t X) {
    static constexpr struct ConstexprData {
      uint8_t AtLeast[65] = {};
      uint64_t Boundaries[20] = {};
      static constexpr int NumDigitsConstexpr(uint64_t N) {
        int res = 1;
        while (N >= 10) {
          res++;
          N /= 10;
        }
        return res;
      }
      constexpr ConstexprData() {
        uint64_t Val = ~0ull;
        for (uint64_t i = 0; i <= 64; i++) {
          uint64_t Digits = NumDigitsConstexpr(Val) - 1;
          AtLeast[i] = Digits;
          Val >>= 1;
        }
        // Special case because X=0 should return 1 and not 0
        Boundaries[0] = 0;
        Val = 10;
        for (uint64_t i = 1; i < 20; i++) {
          Boundaries[i] = Val;
          Val *= 10;
        }
      }
    } Data;

    uint64_t Base2 = X ? countl_zero(X) : 64;
    uint64_t Digits = Data.AtLeast[Base2];
    return Digits + (X >= Data.Boundaries[Digits]);
  }

} // namespace llvm

uint64_t llvm::mulDivNearest(uint64_t X, uint64_t Y, uint64_t Z) {
  assert(Z && "Division by zero");
  bool Overflowed = false;
  uint64_t Product = SaturatingMultiply(X, Y, &Overflowed);
  if (!Overflowed)
    return divideNearest(Product, Z);
  // Adding Z / 2 before the division rounds half up, like divideNearest.
  APInt WideProduct = APInt(128, X) * APInt(128, Y);
  APInt Divisor(128, Z);
  return (WideProduct + Divisor.lshr(1)).udiv(Divisor).getLimitedValue();
}
