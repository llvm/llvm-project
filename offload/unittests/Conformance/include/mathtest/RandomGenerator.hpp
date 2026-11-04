//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the RandomGenerator class, a concrete
/// range-based generator that randomly creates inputs from a given sequence of
/// ranges.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_RANDOMGENERATOR_HPP
#define MATHTEST_RANDOMGENERATOR_HPP

#include "mathtest/IndexedRange.hpp"
#include "mathtest/RandomState.hpp"
#include "mathtest/RangeBasedGenerator.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace mathtest {

template <typename... InTypes>
class [[nodiscard]] RandomGenerator final
    : public RangeBasedGenerator<RandomGenerator<InTypes...>, InTypes...> {

  friend class RangeBasedGenerator<RandomGenerator<InTypes...>, InTypes...>;

  using Base = RangeBasedGenerator<RandomGenerator<InTypes...>, InTypes...>;

  using Base::RangesTuple;
  using Base::Size;

public:
  explicit constexpr RandomGenerator(
      SeedTy BaseSeed, uint64_t Size,
      const IndexedRange<InTypes> &...Ranges) noexcept
      : Base(Size, Ranges...), BaseSeed(BaseSeed) {}

private:
  [[nodiscard]] static uint64_t getRandomIndex(RandomState &RNG,
                                               uint64_t RangeSize) noexcept {
    if (RangeSize == 0)
      return 0;

    const uint64_t Threshold = (-RangeSize) % RangeSize;

    uint64_t RandomNumber;
    do {
      RandomNumber = RNG.next();
    } while (RandomNumber < Threshold);

    return RandomNumber % RangeSize;
  }

  template <typename BufferPtrsTupleType>
  void writeInputs(uint64_t CurrentFlatIndex, uint64_t Offset,
                   BufferPtrsTupleType BufferPtrsTuple) const noexcept {

    RandomState RNG(SeedTy{BaseSeed.Value ^ (CurrentFlatIndex + Offset)});
    writeInputsImpl<0>(RNG, Offset, BufferPtrsTuple);
  }

  template <std::size_t Index, typename BufferPtrsTupleType>
  void writeInputsImpl(RandomState &RNG, uint64_t Offset,
                       BufferPtrsTupleType BufferPtrsTuple) const noexcept {
    if constexpr (Index < Base::NumInputs) {
      const auto &Range = std::get<Index>(RangesTuple);
      const auto RandomIndex = getRandomIndex(RNG, Range.getSize());
      std::get<Index>(BufferPtrsTuple)[Offset] = Range[RandomIndex];

      writeInputsImpl<Index + 1>(RNG, Offset, BufferPtrsTuple);
    }
  }

  SeedTy BaseSeed;
};
} // namespace mathtest

#endif // MATHTEST_RANDOMGENERATOR_HPP
