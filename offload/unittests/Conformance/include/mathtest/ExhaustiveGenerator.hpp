//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the ExhaustiveGenerator class, a
/// concrete input generator that exhaustively creates inputs from a given
/// sequence of ranges.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_EXHAUSTIVEGENERATOR_HPP
#define MATHTEST_EXHAUSTIVEGENERATOR_HPP

#include "mathtest/IndexedRange.hpp"
#include "mathtest/InputGenerator.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Parallel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace mathtest {

template <typename... InTypes>
class [[nodiscard]] ExhaustiveGenerator final
    : public InputGenerator<InTypes...> {
  static constexpr std::size_t NumInputs = sizeof...(InTypes);
  static_assert(NumInputs > 0, "The number of inputs must be at least 1");

public:
  explicit constexpr ExhaustiveGenerator(
      const IndexedRange<InTypes> &...Ranges) noexcept
      : RangesTuple(Ranges...) {
    bool Overflowed = getSizeWithOverflow(Ranges..., Size);

    assert(!Overflowed && "The input space size is too large");
    assert((Size > 0) && "The input space size must be at least 1");

    IndexArrayType DimSizes = {};
    std::size_t DimIndex = 0;
    ((DimSizes[DimIndex++] = Ranges.getSize()), ...);

    Strides[NumInputs - 1] = 1;
    if constexpr (NumInputs > 1)
      for (int Index = static_cast<int>(NumInputs) - 2; Index >= 0; --Index)
        Strides[Index] = Strides[Index + 1] * DimSizes[Index + 1];
  }

  [[nodiscard]] std::size_t
  fill(llvm::MutableArrayRef<InTypes>... Buffers) noexcept override {
    const std::array<std::size_t, NumInputs> BufferSizes = {Buffers.size()...};
    const std::size_t BufferSize = BufferSizes[0];
    assert((BufferSize != 0) && "Buffer size cannot be zero");
    assert(std::all_of(BufferSizes.begin(), BufferSizes.end(),
                       [&](std::size_t Size) { return Size == BufferSize; }) &&
           "All input buffers must have the same size");

    uint64_t StartFlatIndex, BatchSize;
    while (true) {
      uint64_t CurrentFlatIndex =
          FlatIndexGenerator.load(std::memory_order_relaxed);
      if (CurrentFlatIndex >= Size)
        return 0;

      BatchSize = std::min<uint64_t>(BufferSize, Size - CurrentFlatIndex);
      uint64_t NextFlatIndex = CurrentFlatIndex + BatchSize;

      if (FlatIndexGenerator.compare_exchange_weak(
              CurrentFlatIndex, NextFlatIndex,
              std::memory_order_acq_rel, // Success
              std::memory_order_acquire  // Failure
              )) {
        StartFlatIndex = CurrentFlatIndex;
        break;
      }
    }

    auto BufferPtrsTuple = std::make_tuple(Buffers.data()...);

    llvm::parallelFor(0, BatchSize, [&](std::size_t Offset) {
      writeInputs(StartFlatIndex, Offset, BufferPtrsTuple);
    });

    return static_cast<std::size_t>(BatchSize);
  }

private:
  using RangesTupleType = std::tuple<IndexedRange<InTypes>...>;
  using IndexArrayType = std::array<uint64_t, NumInputs>;

  static bool getSizeWithOverflow(const IndexedRange<InTypes> &...Ranges,
                                  uint64_t &Size) noexcept {
    Size = 1;
    bool Overflowed = false;

    auto Multiplier = [&](const uint64_t RangeSize) {
      if (!Overflowed)
        Overflowed = __builtin_mul_overflow(Size, RangeSize, &Size);
    };

    (Multiplier(Ranges.getSize()), ...);

    return Overflowed;
  }

  template <typename BufferPtrsTupleType>
  void writeInputs(uint64_t StartFlatIndex, uint64_t Offset,
                   BufferPtrsTupleType BufferPtrsTuple) const noexcept {
    auto NDIndex = getNDIndex(StartFlatIndex + Offset);
    writeInputsImpl<0>(NDIndex, Offset, BufferPtrsTuple);
  }

  constexpr IndexArrayType getNDIndex(uint64_t FlatIndex) const noexcept {
    IndexArrayType NDIndex;

    for (std::size_t Index = 0; Index < NumInputs; ++Index) {
      NDIndex[Index] = FlatIndex / Strides[Index];
      FlatIndex -= NDIndex[Index] * Strides[Index];
    }

    return NDIndex;
  }

  template <std::size_t Index, typename BufferPtrsTupleType>
  void writeInputsImpl(IndexArrayType NDIndex, uint64_t Offset,
                       BufferPtrsTupleType BufferPtrsTuple) const noexcept {
    if constexpr (Index < NumInputs) {
      const auto &Range = std::get<Index>(RangesTuple);
      std::get<Index>(BufferPtrsTuple)[Offset] = Range[NDIndex[Index]];
      writeInputsImpl<Index + 1>(NDIndex, Offset, BufferPtrsTuple);
    }
  }

  uint64_t Size = 1;
  RangesTupleType RangesTuple;
  IndexArrayType Strides = {};
  std::atomic<uint64_t> FlatIndexGenerator = 0;
};
} // namespace mathtest

#endif // MATHTEST_EXHAUSTIVEGENERATOR_HPP
