//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the RangeBasedGenerator class, a base
/// class for input generators that operate on a sequence of ranges.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_RANGEBASEDGENERATOR_HPP
#define MATHTEST_RANGEBASEDGENERATOR_HPP

#include "mathtest/IndexedRange.hpp"
#include "mathtest/InputGenerator.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Parallel.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace mathtest {

template <typename Derived, typename... InTypes>
class [[nodiscard]] RangeBasedGenerator : public InputGenerator<InTypes...> {
public:
  void reset() noexcept override { NextFlatIndex = 0; }

  [[nodiscard]] std::size_t
  fill(llvm::MutableArrayRef<InTypes>... Buffers) noexcept override {
    const std::array<std::size_t, NumInputs> BufferSizes = {Buffers.size()...};
    const std::size_t BufferSize = BufferSizes[0];
    assert((BufferSize != 0) && "Buffer size cannot be zero");
    assert(std::all_of(BufferSizes.begin(), BufferSizes.end(),
                       [&](std::size_t Size) { return Size == BufferSize; }) &&
           "All input buffers must have the same size");

    if (NextFlatIndex >= Size)
      return 0;

    const auto BatchSize = std::min<uint64_t>(BufferSize, Size - NextFlatIndex);
    const auto CurrentFlatIndex = NextFlatIndex;
    NextFlatIndex += BatchSize;

    auto BufferPtrsTuple = std::make_tuple(Buffers.data()...);

    llvm::parallelFor(0, BatchSize, [&](std::size_t Offset) {
      static_cast<Derived *>(this)->writeInputs(CurrentFlatIndex, Offset,
                                                BufferPtrsTuple);
    });

    return static_cast<std::size_t>(BatchSize);
  }

protected:
  using RangesTupleType = std::tuple<IndexedRange<InTypes>...>;

  static constexpr std::size_t NumInputs = sizeof...(InTypes);
  static_assert(NumInputs > 0, "The number of inputs must be at least 1");

  explicit constexpr RangeBasedGenerator(
      const IndexedRange<InTypes> &...Ranges) noexcept
      : RangesTuple(Ranges...) {}

  explicit constexpr RangeBasedGenerator(
      uint64_t Size, const IndexedRange<InTypes> &...Ranges) noexcept
      : RangesTuple(Ranges...), Size(Size) {}

  RangesTupleType RangesTuple;
  uint64_t Size = 0;

private:
  uint64_t NextFlatIndex = 0;
};
} // namespace mathtest

#endif // MATHTEST_RANGEBASEDGENERATOR_HPP
