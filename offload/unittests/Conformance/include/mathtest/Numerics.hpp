//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of numeric utilities, including functions
/// to compute ULP distance and traits for floating-point types.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_NUMERICS_HPP
#define MATHTEST_NUMERICS_HPP

#include "mathtest/Support.hpp"
#include "mathtest/TypeExtras.hpp"

// These headers are in the shared LLVM-libc header library
#include "shared/fp_bits.h"
#include "shared/sign.h"

#include <climits>
#include <cstdint>
#include <limits>
#include <math.h>
#include <type_traits>

namespace mathtest {

template <typename FloatType>
using FPBits = LIBC_NAMESPACE::shared::FPBits<FloatType>;

using Sign = LIBC_NAMESPACE::shared::Sign;

//===----------------------------------------------------------------------===//
// Type Traits
//===----------------------------------------------------------------------===//

template <typename T> struct IsFloatingPoint : std::is_floating_point<T> {};

template <> struct IsFloatingPoint<float16> : std::true_type {};

template <typename T>
inline constexpr bool IsFloatingPoint_v // NOLINT(readability-identifier-naming)
    = IsFloatingPoint<T>::value;

template <typename T> struct StorageTypeOf {
private:
  static constexpr auto getStorageType() noexcept {
    if constexpr (IsFloatingPoint_v<T>)
      return TypeIdentityOf<typename FPBits<T>::StorageType>{};
    else if constexpr (std::is_unsigned_v<T>)
      return TypeIdentityOf<T>{};
    else if constexpr (std::is_signed_v<T>)
      return TypeIdentityOf<std::make_unsigned_t<T>>{};
    else
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }

public:
  using type = typename decltype(getStorageType())::type;
};

template <typename T> using StorageTypeOf_t = typename StorageTypeOf<T>::type;

//===----------------------------------------------------------------------===//
// Numeric Functions
//===----------------------------------------------------------------------===//

template <typename T> [[nodiscard]] constexpr T getMinOrNegInf() noexcept {
  if constexpr (IsFloatingPoint_v<T>) {
    // All currently supported floating-point types have infinity
    return FPBits<T>::inf(Sign::NEG).get_val();
  } else {
    static_assert(std::is_integral_v<T>,
                  "Type T must be an integral or floating-point type");

    return std::numeric_limits<T>::lowest();
  }
}

template <typename T> [[nodiscard]] constexpr T getMaxOrInf() noexcept {
  if constexpr (IsFloatingPoint_v<T>) {
    // All currently supported floating-point types have infinity
    return FPBits<T>::inf(Sign::POS).get_val();
  } else {
    static_assert(std::is_integral_v<T>,
                  "Type T must be an integral or floating-point type");

    return std::numeric_limits<T>::max();
  }
}

template <typename FloatType>
[[nodiscard]] uint64_t computeUlpDistance(FloatType X, FloatType Y) noexcept {
  static_assert(IsFloatingPoint_v<FloatType>,
                "FloatType must be a floating-point type");
  using FPBits = FPBits<FloatType>;
  using StorageType = typename FPBits::StorageType;

  const FPBits XBits(X);
  const FPBits YBits(Y);

  if (X == Y) {
    if (XBits.sign() != YBits.sign()) [[unlikely]] {
      // When X == Y, different sign bits imply that X and Y are +0.0 and -0.0
      // (in any order). Since we want to treat them as unequal in the context
      // of accuracy testing of mathematical functions, we return the smallest
      // non-zero value.
      return 1;
    }
    return 0;
  }

  const bool XIsNaN = XBits.is_nan();
  const bool YIsNaN = YBits.is_nan();

  if (XIsNaN && YIsNaN)
    return 0;

  if (XIsNaN || YIsNaN)
    return std::numeric_limits<uint64_t>::max();

  constexpr StorageType SignMask = FPBits::SIGN_MASK;

  // Linearise FloatType values into an ordered unsigned space. Let a and b
  // be bits(x), bits(y), respectively, where x and y are FloatType values.
  //  * The mapping is monotonic: x >= y if, and only if, map(a) >= map(b).
  //  * The difference |map(a) − map(b)| equals the number of std::nextafter
  //    steps between a and b within the same type.
  auto MapToOrderedUnsigned = [](FPBits Bits) {
    const StorageType Unsigned = Bits.uintval();
    return (Unsigned & SignMask) ? SignMask - (Unsigned - SignMask)
                                 : SignMask + Unsigned;
  };

  const StorageType MappedX = MapToOrderedUnsigned(XBits);
  const StorageType MappedY = MapToOrderedUnsigned(YBits);
  return static_cast<uint64_t>(MappedX > MappedY ? MappedX - MappedY
                                                 : MappedY - MappedX);
}
} // namespace mathtest

#endif // MATHTEST_NUMERICS_HPP
