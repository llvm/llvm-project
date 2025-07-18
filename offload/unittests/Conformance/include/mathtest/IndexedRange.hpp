#ifndef MATHTEST_INDEXEDRANGE_HPP
#define MATHTEST_INDEXEDRANGE_HPP

#include "mathtest/Numerics.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mathtest {

template <typename T> class [[nodiscard]] IndexedRange {
  static_assert(IsFloatingPoint_v<T> || std::is_integral_v<T>,
                "Type T must be an integral or floating-point type");
  static_assert(sizeof(T) <= sizeof(uint64_t),
                "Type T must be no wider than uint64_t");

public:
  constexpr IndexedRange() noexcept
      : IndexedRange(getMinOrNegInf<T>(), getMaxOrInf<T>(), true) {}

  explicit constexpr IndexedRange(T Begin, T End, bool Inclusive) noexcept
      : MappedBegin(mapToOrderedUnsigned(Begin)),
        MappedEnd(mapToOrderedUnsigned(End)) {
    if (Inclusive) {
      assert((Begin <= End) && "Begin must be less than or equal to End");
    } else {
      assert((Begin < End) && "Begin must be less than End");
      --MappedEnd;
    }

    assert(((MappedEnd - MappedBegin) < std::numeric_limits<uint64_t>::max()) &&
           "The range is too large to index");
  }

  [[nodiscard]] constexpr uint64_t getSize() const noexcept {
    return static_cast<uint64_t>(MappedEnd) - MappedBegin + 1;
  }

  [[nodiscard]] constexpr T operator[](uint64_t Index) const noexcept {
    assert((Index < getSize()) && "Index is out of range");

    StorageType MappedValue = MappedBegin + Index;
    return mapFromOrderedUnsigned(MappedValue);
  }

private:
  using StorageType = StorageTypeOf_t<T>;

  // Linearise T values into an ordered unsigned space:
  //  * The mapping is monotonic: a >= b if, and only if, map(a) >= map(b)
  //  * The difference |map(a) − map(b)| equals the number of representable
  //    values between a and b within the same type
  static constexpr StorageType mapToOrderedUnsigned(T Value) {
    if constexpr (IsFloatingPoint_v<T>) {
      StorageType SignMask = FPUtils<T>::SignMask;
      StorageType Bits = FPUtils<T>::getAsBits(Value);
      return (Bits & SignMask) ? SignMask - (Bits - SignMask) - 1
                               : SignMask + Bits;
    }

    if constexpr (std::is_signed_v<T>) {
      StorageType SignMask = maskLeadingOnes<StorageType, 1>();
      return __builtin_bit_cast(StorageType, Value) ^ SignMask;
    }

    return Value;
  }

  static constexpr T mapFromOrderedUnsigned(StorageType MappedValue) {
    if constexpr (IsFloatingPoint_v<T>) {
      StorageType SignMask = FPUtils<T>::SignMask;
      StorageType Bits = (MappedValue < SignMask)
                             ? (SignMask - MappedValue) + SignMask - 1
                             : MappedValue - SignMask;

      return FPUtils<T>::createFromBits(Bits);
    }

    if constexpr (std::is_signed_v<T>) {
      StorageType SignMask = maskLeadingOnes<StorageType, 1>();
      return __builtin_bit_cast(T, MappedValue ^ SignMask);
    }

    return MappedValue;
  }

  StorageType MappedBegin;
  StorageType MappedEnd;
};
} // namespace mathtest

#endif // MATHTEST_INDEXEDRANGE_HPP
