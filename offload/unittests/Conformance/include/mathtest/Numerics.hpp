#ifndef MATHTEST_NUMERICS_HPP
#define MATHTEST_NUMERICS_HPP

#include "mathtest/Support.hpp"
#include "mathtest/TypeExtras.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <math.h>
#include <type_traits>

namespace mathtest {

//===----------------------------------------------------------------------===//
// Type Traits
//===----------------------------------------------------------------------===//

template <typename T> struct StorageTypeOf {
private:
  static constexpr auto getStorageType() noexcept {
    if constexpr (std::is_unsigned_v<T>) {
      return TypeIdentityOf<T>{};
    } else if constexpr (std::is_signed_v<T>) {
      return TypeIdentityOf<std::make_unsigned_t<T>>{};
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
    }
  }

public:
  using type = typename decltype(getStorageType())::type;
};

#ifdef MATHTEST_HAS_FLOAT16
template <> struct StorageTypeOf<float16> {
  using type = uint16_t;
};
#endif // MATHTEST_HAS_FLOAT16

template <> struct StorageTypeOf<float> {
  using type = uint32_t;
};

template <> struct StorageTypeOf<double> {
  using type = uint64_t;
};

template <typename T> using StorageTypeOf_t = typename StorageTypeOf<T>::type;

template <typename T> struct IsFloatingPoint : std::is_floating_point<T> {};

#ifdef MATHTEST_HAS_FLOAT16
template <> struct IsFloatingPoint<float16> : std::true_type {};
#endif // MATHTEST_HAS_FLOAT16

template <typename T>
inline constexpr bool IsFloatingPoint_v // NOLINT(readability-identifier-naming)
    = IsFloatingPoint<T>::value;

//===----------------------------------------------------------------------===//
// Bitmask Utilities
//===----------------------------------------------------------------------===//

template <typename UIntType, std::size_t Count>
[[nodiscard]] constexpr UIntType maskLeadingOnes() noexcept {
  static_assert(std::is_unsigned_v<UIntType>,
                "UIntType must be an unsigned integer type");

  constexpr unsigned TotalBits = CHAR_BIT * sizeof(UIntType);
  static_assert(
      Count <= TotalBits,
      "Count must be less than or equal to the bit width of UIntType");

  return Count == 0 ? UIntType(0) : (~UIntType(0) << (TotalBits - Count));
  ;
}

template <typename UIntType, std::size_t Count>
[[nodiscard]] constexpr UIntType maskTrailingOnes() noexcept {
  static_assert(std::is_unsigned_v<UIntType>,
                "UIntType must be an unsigned integer type");

  constexpr unsigned TotalBits = CHAR_BIT * sizeof(UIntType);
  static_assert(
      Count <= TotalBits,
      "Count must be less than or equal to the bit width of UIntType");

  return Count == 0 ? UIntType(0) : (~UIntType(0) >> (TotalBits - Count));
}

//===----------------------------------------------------------------------===//
// Floating-Point Utilities
//===----------------------------------------------------------------------===//

template <typename FloatType> struct FPLayout;

#ifdef MATHTEST_HAS_FLOAT16
template <> struct FPLayout<float16> {
  static constexpr std::size_t SignLen = 1;
  static constexpr std::size_t ExponentLen = 5;
  static constexpr std::size_t FractionLen = 10;
};
#endif // MATHTEST_HAS_FLOAT16

template <> struct FPLayout<float> {
  static constexpr std::size_t SignLen = 1;
  static constexpr std::size_t ExponentLen = 8;
  static constexpr std::size_t FractionLen = 23;
};

template <> struct FPLayout<double> {
  static constexpr std::size_t SignLen = 1;
  static constexpr std::size_t ExponentLen = 11;
  static constexpr std::size_t FractionLen = 52;
};

template <typename FloatType> struct FPUtils : public FPLayout<FloatType> {
  using FPLayout = FPLayout<FloatType>;
  using StorageType = StorageTypeOf_t<FloatType>;
  using FPLayout::ExponentLen;
  using FPLayout::FractionLen;
  using FPLayout::SignLen;

  static constexpr StorageType SignMask =
      maskTrailingOnes<StorageType, SignLen>() << (ExponentLen + FractionLen);

  FPUtils() = delete;

  [[nodiscard]] static constexpr FloatType
  createFromBits(StorageType Bits) noexcept {
    return __builtin_bit_cast(FloatType, Bits);
  }

  [[nodiscard]] static constexpr StorageType
  getAsBits(FloatType Value) noexcept {
    return __builtin_bit_cast(StorageType, Value);
  }

  [[nodiscard]] static constexpr bool isNaN(FloatType Value) noexcept {
    return __builtin_isnan(Value);
  }

  [[nodiscard]] static constexpr bool getSignBit(FloatType Value) noexcept {
    return getAsBits(Value) & SignMask;
  }
};

//===----------------------------------------------------------------------===//
// Numeric Functions
//===----------------------------------------------------------------------===//

template <typename T> [[nodiscard]] constexpr T getMinOrNegInf() noexcept {
  static_assert(std::is_arithmetic_v<T>, "Type T must be an arithmetic type");

  if constexpr (std::is_floating_point_v<T> &&
                std::numeric_limits<T>::has_infinity) {
    return -std::numeric_limits<T>::infinity();
  }

  return std::numeric_limits<T>::lowest();
}

#ifdef MATHTEST_HAS_FLOAT16
template <> [[nodiscard]] constexpr float16 getMinOrNegInf<float16>() noexcept {
  using StorageType = StorageTypeOf_t<float16>;

  return __builtin_bit_cast(float16, static_cast<StorageType>(0xFC00U));
}
#endif // MATHTEST_HAS_FLOAT16

template <typename T> [[nodiscard]] constexpr T getMaxOrInf() noexcept {
  static_assert(std::is_arithmetic_v<T>, "Type T must be an arithmetic type");

  if constexpr (std::is_floating_point_v<T> &&
                std::numeric_limits<T>::has_infinity) {
    return std::numeric_limits<T>::infinity();
  }

  return std::numeric_limits<T>::max();
}

#ifdef MATHTEST_HAS_FLOAT16
template <> [[nodiscard]] constexpr float16 getMaxOrInf<float16>() noexcept {
  using StorageType = StorageTypeOf_t<float16>;

  return __builtin_bit_cast(float16, static_cast<StorageType>(0x7C00U));
}
#endif // MATHTEST_HAS_FLOAT16

template <typename FloatType>
[[nodiscard]] uint64_t computeUlpDistance(FloatType X, FloatType Y) noexcept {
  static_assert(IsFloatingPoint_v<FloatType>,
                "FloatType must be a floating-point type");
  using FPUtils = FPUtils<FloatType>;
  using StorageType = typename FPUtils::StorageType;

  if (X == Y) {
    if (FPUtils::getSignBit(X) != FPUtils::getSignBit(Y)) [[unlikely]] {
      // When X == Y, different sign bits imply that X and Y are +0.0 and -0.0
      // (in any order). Since we want to treat them as unequal in the context
      // of accuracy testing of mathematical functions, we return the smallest
      // non-zero value
      return 1;
    }
    return 0;
  }

  const bool XIsNaN = FPUtils::isNaN(X);
  const bool YIsNaN = FPUtils::isNaN(Y);

  if (XIsNaN && YIsNaN) {
    return 0;
  }
  if (XIsNaN || YIsNaN) {
    return std::numeric_limits<uint64_t>::max();
  }

  constexpr StorageType SignMask = FPUtils::SignMask;

  // Linearise FloatType values into an ordered unsigned space:
  //  * The mapping is monotonic: a >= b if, and only if, map(a) >= map(b)
  //  * The difference |map(a) − map(b)| equals the number of std::nextafter
  //    steps between a and b within the same type
  auto MapToOrderedUnsigned = [](FloatType Value) {
    const StorageType Bits = FPUtils::getAsBits(Value);
    return (Bits & SignMask) ? SignMask - (Bits - SignMask) : SignMask + Bits;
  };

  const StorageType MappedX = MapToOrderedUnsigned(X);
  const StorageType MappedY = MapToOrderedUnsigned(Y);
  return static_cast<uint64_t>(MappedX > MappedY ? MappedX - MappedY
                                                 : MappedY - MappedX);
}
} // namespace mathtest

#endif // MATHTEST_NUMERICS_HPP
