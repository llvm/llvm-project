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
/// concrete range-based generator that exhaustively creates inputs from a
/// given sequence of ranges.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_EXHAUSTIVEGENERATOR_HPP
#define MATHTEST_EXHAUSTIVEGENERATOR_HPP

#include "mathtest/IndexedRange.hpp"
#include "mathtest/RangeBasedGenerator.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace mathtest {

template <typename... InTypes>
class [[nodiscard]] ExhaustiveGenerator final
    : public RangeBasedGenerator<ExhaustiveGenerator<InTypes...>, InTypes...> {

  friend class RangeBasedGenerator<ExhaustiveGenerator<InTypes...>, InTypes...>;

  using Base = RangeBasedGenerator<ExhaustiveGenerator<InTypes...>, InTypes...>;
  using IndexArrayType = std::array<uint64_t, Base::NumInputs>;

  using Base::InputSpaceSize;
  using Base::RangesTuple;
  using Base::SizeToGenerate;

public:
  explicit constexpr ExhaustiveGenerator(
      const IndexedRange<InTypes> &...Ranges) noexcept
      : Base(Ranges...) {
    SizeToGenerate = InputSpaceSize;

    IndexArrayType DimSizes = {};
    std::size_t DimIndex = 0;
    ((DimSizes[DimIndex++] = Ranges.getSize()), ...);

    Strides[Base::NumInputs - 1] = 1;
    if constexpr (Base::NumInputs > 1)
      for (int Index = static_cast<int>(Base::NumInputs) - 2; Index >= 0;
           --Index)
        Strides[Index] = Strides[Index + 1] * DimSizes[Index + 1];
  }

private:
  constexpr IndexArrayType getNDIndex(uint64_t FlatIndex) const noexcept {
    IndexArrayType NDIndex;

    for (std::size_t Index = 0; Index < Base::NumInputs; ++Index) {
      NDIndex[Index] = FlatIndex / Strides[Index];
      FlatIndex -= NDIndex[Index] * Strides[Index];
    }

    return NDIndex;
  }

  template <typename BufferPtrsTupleType>
  void writeInputs(uint64_t CurrentFlatIndex, uint64_t Offset,
                   BufferPtrsTupleType BufferPtrsTuple) const noexcept {
    auto NDIndex = getNDIndex(CurrentFlatIndex + Offset);
    writeInputsImpl<0>(NDIndex, Offset, BufferPtrsTuple);
  }

  template <std::size_t Index, typename BufferPtrsTupleType>
  void writeInputsImpl(IndexArrayType NDIndex, uint64_t Offset,
                       BufferPtrsTupleType BufferPtrsTuple) const noexcept {
    if constexpr (Index < Base::NumInputs) {
      const auto &Range = std::get<Index>(RangesTuple);
      std::get<Index>(BufferPtrsTuple)[Offset] = Range[NDIndex[Index]];

      writeInputsImpl<Index + 1>(NDIndex, Offset, BufferPtrsTuple);
    }
  }

  IndexArrayType Strides = {};
};
} // namespace mathtest

#endif // MATHTEST_EXHAUSTIVEGENERATOR_HPP
