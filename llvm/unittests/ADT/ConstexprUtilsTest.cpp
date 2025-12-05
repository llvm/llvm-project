//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains testing for constexpr utilities.
///
/// Utilities include sorting, various other algorithms and integer sequence
/// manipulation.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ConstexprUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <type_traits>
#include <utility>

using namespace llvm;

static constexpr auto Greater = [](auto L, auto R) constexpr { return L > R; };

template <typename T> struct Type {
  using type = T;
};

// clang-format off
using IntTypes =
    ::testing::Types<
      signed char,
      short,
      int,
      long,
      long long,
      unsigned char,
      unsigned short,
      unsigned,
      unsigned long,
      unsigned long long
    >;
// clang-format on

template <typename T> class ConstexprUtilsTest : public testing::Test {
public:
  using IntT = T;

  enum class Number : IntT {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
  };

  static constexpr IntT min() { return std::numeric_limits<IntT>::min(); }
  static constexpr IntT max() { return std::numeric_limits<IntT>::max(); }
};

#define EXPOSE_TYPE(type)                                                      \
  using type = typename remove_cvref_t<decltype(*this)>::type

TYPED_TEST_SUITE(ConstexprUtilsTest, IntTypes, );

TYPED_TEST(ConstexprUtilsTest, CompileTimeSwap) {
  EXPOSE_TYPE(IntT);
  EXPOSE_TYPE(Number);

  // Swap pair elements.
  {
    constexpr auto P1 = []() {
      auto Ret = std::pair(IntT(1), IntT(2));
      ce_swap(Ret.first, Ret.second);
      return Ret;
    }();

    static_assert(P1.first == 2);
    static_assert(P1.second == 1);

    constexpr auto P2 = [](auto PIn) {
      ce_swap(PIn.first, PIn.second);
      return PIn;
    }(P1);

    static_assert(P2.first == 1);
    static_assert(P2.second == 2);
  }

  // Swap pair elements (enum).
  {
    constexpr auto P1 = []() {
      auto Ret = std::pair(Number::One, Number::Two);
      ce_swap(Ret.first, Ret.second);
      return Ret;
    }();

    static_assert(P1.first == Number::Two);
    static_assert(P1.second == Number::One);

    constexpr auto P2 = [](auto PIn) {
      ce_swap(PIn.first, PIn.second);
      return PIn;
    }(P1);

    static_assert(P2.first == Number::One);
    static_assert(P2.second == Number::Two);
  }
}

TYPED_TEST(ConstexprUtilsTest, MinMaxCompiletime) {
  EXPOSE_TYPE(IntT);
  EXPOSE_TYPE(Number);
  using BaseT = remove_cvref_t<decltype(*this)>;

  if constexpr (BaseT::max() > 4723) {
    static_assert(ce_max<IntT>(192, 273, 4723, 20, 3709) == IntT(4723));
    static_assert(ce_min<IntT>(192, 273, 4723, 20, 3709) == IntT(20));
  }

  static_assert(ce_max<IntT>(20, 120, 124, 57) == IntT(124));
  static_assert(ce_min<IntT>(20, 120, 124, 57) == IntT(20));

  if constexpr (std::is_signed_v<IntT>) {
    if constexpr (BaseT::max() > 4723 && BaseT::min() < -2873) {
      static_assert(ce_max<IntT>(192, 273, -2873, 4723, 20, 3709) ==
                    IntT(4723));
      static_assert(ce_min<IntT>(192, 273, -2873, 4723, 20, 3709) ==
                    IntT(-2873));
    }
  }

  if constexpr (std::is_signed_v<IntT>) {
    static_assert(ce_max<IntT>(127, 0, 20, -20, -128, -90, 103) == IntT(127));
    static_assert(ce_min<IntT>(127, 0, 20, -20, -128, -90, 103) == IntT(-128));
  }

  static_assert(ce_max<Number>(Number::Four, Number::Two, Number::Six,
                               Number::Three, Number::Four) == Number::Six);

  static_assert(ce_min<Number>(Number::Four, Number::Two, Number::Six,
                               Number::Three, Number::Four) == Number::Two);
}

TYPED_TEST(ConstexprUtilsTest, ToArray) {
  EXPOSE_TYPE(IntT);
  EXPOSE_TYPE(Number);

  // int array conversion.
  {
    static constexpr IntT CArr[]{1, 3, 5, 7, 9};
    static constexpr auto Arr = to_array(CArr);

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Arr)>, std::array<IntT, 5>>);

    static_assert(Arr.size() == std::size(CArr));
    static_assert(Arr[0] == IntT(1));
    static_assert(Arr[1] == IntT(3));
    static_assert(Arr[2] == IntT(5));
    static_assert(Arr[3] == IntT(7));
    static_assert(Arr[4] == IntT(9));

    EXPECT_TRUE(equal(CArr, Arr));
  }

  // enum array conversion.
  {
    static constexpr Number CArr[]{Number::One, Number::Three, Number::Five,
                                   Number::Seven, Number::Nine};
    static constexpr auto Arr = to_array(CArr);

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Arr)>, std::array<Number, 5>>);

    static_assert(Arr.size() == std::size(CArr));
    static_assert(Arr[0] == Number::One);
    static_assert(Arr[1] == Number::Three);
    static_assert(Arr[2] == Number::Five);
    static_assert(Arr[3] == Number::Seven);
    static_assert(Arr[4] == Number::Nine);

    EXPECT_TRUE(equal(CArr, Arr));
  }
}

TYPED_TEST(ConstexprUtilsTest, SliceArray) {
  EXPOSE_TYPE(IntT);
  EXPOSE_TYPE(Number);

  // int array slice.
  {
    static constexpr std::array<IntT, 5> Arr{{1, 3, 5, 7, 9}};

    // Front slice.
    {
      static constexpr auto Slice = ce_slice<0, 2>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<IntT, 2>>);

      static_assert(Slice[0] == 1);
      static_assert(Slice[1] == 3);
    }

    // Back slice.
    {
      static constexpr auto Slice = ce_slice<1, 4>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<IntT, 4>>);

      static_assert(Slice[0] == 3);
      static_assert(Slice[1] == 5);
      static_assert(Slice[2] == 7);
      static_assert(Slice[3] == 9);
    }

    // Middle slice.
    {
      static constexpr auto Slice = ce_slice<2, 2>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<IntT, 2>>);

      static_assert(Slice[0] == 5);
      static_assert(Slice[1] == 7);
    }

    // Zero-length slice (begin).
    {
      static constexpr auto Slice = ce_slice<0, 0>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<IntT, 0>>);
    }

    // Zero-length slice (last)
    {
      static constexpr auto Slice = ce_slice<4, 0>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<IntT, 0>>);
    }

    // Zero-length slice (end)
    {
      static constexpr auto Slice = ce_slice<5, 0>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<IntT, 0>>);
    }

    // Complete slice.
    {
      static constexpr auto Slice = ce_slice<0, 5>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<IntT, 5>>);

      static_assert(Slice[0] == 1);
      static_assert(Slice[1] == 3);
      static_assert(Slice[2] == 5);
      static_assert(Slice[3] == 7);
      static_assert(Slice[4] == 9);
    }
  }

  // enum array slice.
  {
    static constexpr std::array<Number, 5> Arr{{Number::One, Number::Three,
                                                Number::Five, Number::Seven,
                                                Number::Nine}};

    // Front slice.
    {
      static constexpr auto Slice = ce_slice<0, 2>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<Number, 2>>);

      static_assert(Slice[0] == Number::One);
      static_assert(Slice[1] == Number::Three);
    }

    // Back slice.
    {
      static constexpr auto Slice = ce_slice<1, 4>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<Number, 4>>);

      static_assert(Slice[0] == Number::Three);
      static_assert(Slice[1] == Number::Five);
      static_assert(Slice[2] == Number::Seven);
      static_assert(Slice[3] == Number::Nine);
    }

    // Middle slice.
    {
      static constexpr auto Slice = ce_slice<2, 2>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<Number, 2>>);

      static_assert(Slice[0] == Number::Five);
      static_assert(Slice[1] == Number::Seven);
    }

    // Zero-length slice (begin).
    {
      static constexpr auto Slice = ce_slice<0, 0>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<Number, 0>>);
    }

    // Zero-length slice (last)
    {
      static constexpr auto Slice = ce_slice<4, 0>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<Number, 0>>);
    }

    // Zero-length slice (end)
    {
      static constexpr auto Slice = ce_slice<5, 0>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<Number, 0>>);
    }

    // Complete slice.
    {
      static constexpr auto Slice = ce_slice<0, 5>(Arr);

      static_assert(std::is_same_v<std::remove_cv_t<decltype(Slice)>,
                                   std::array<Number, 5>>);

      static_assert(Slice[0] == Number::One);
      static_assert(Slice[1] == Number::Three);
      static_assert(Slice[2] == Number::Five);
      static_assert(Slice[3] == Number::Seven);
      static_assert(Slice[4] == Number::Nine);
    }
  }
}

TYPED_TEST(ConstexprUtilsTest, SliceLiterals) {
  EXPOSE_TYPE(IntT);

  // Front slice.
  {
    using Slice = SliceLiterals<0, 2, IntT, 5, 10, 15, 20, 25, 30, 35, 40>;
    static_assert(std::is_same_v<Slice, std::integer_sequence<IntT, 5, 10>>);
  }

  // Back slice.
  {
    using Slice = SliceLiterals<5, 3, IntT, 5, 10, 15, 20, 25, 30, 35, 40>;
    static_assert(
        std::is_same_v<Slice, std::integer_sequence<IntT, 30, 35, 40>>);
  }

  // Middle slice.
  {
    using Slice = SliceLiterals<2, 4, IntT, 5, 10, 15, 20, 25, 30, 35, 40>;
    static_assert(
        std::is_same_v<Slice, std::integer_sequence<IntT, 15, 20, 25, 30>>);
  }

  // Zero-length slice (begin).
  {
    using Slice = SliceLiterals<0, 0, IntT, 5, 10, 15, 20, 25, 30, 35, 40>;
    static_assert(std::is_same_v<Slice, std::integer_sequence<IntT>>);
  }

  // Zero-length slice (last)
  {
    using Slice = SliceLiterals<7, 0, IntT, 5, 10, 15, 20, 25, 30, 35, 40>;
    static_assert(std::is_same_v<Slice, std::integer_sequence<IntT>>);
  }

  // Zero-length slice (end)
  {
    using Slice = SliceLiterals<8, 0, IntT, 5, 10, 15, 20, 25, 30, 35, 40>;
    static_assert(std::is_same_v<Slice, std::integer_sequence<IntT>>);
  }

  // Complete slice.
  {
    using Slice = SliceLiterals<0, 8, IntT, 5, 10, 15, 20, 25, 30, 35, 40>;
    static_assert(
        std::is_same_v<
            Slice, std::integer_sequence<IntT, 5, 10, 15, 20, 25, 30, 35, 40>>);
  }
}

TYPED_TEST(ConstexprUtilsTest, SliceSequence) {
  EXPOSE_TYPE(IntT);

  using Seq = std::integer_sequence<IntT, 5, 10, 15, 20, 25, 30, 35, 40>;

  // Front slice.
  {
    using Slice = SliceSequence<0, 2, Seq>;
    static_assert(std::is_same_v<Slice, std::integer_sequence<IntT, 5, 10>>);
  }

  // Back slice.
  {
    using Slice = SliceSequence<5, 3, Seq>;
    static_assert(
        std::is_same_v<Slice, std::integer_sequence<IntT, 30, 35, 40>>);
  }

  // Middle slice.
  {
    using Slice = SliceSequence<2, 4, Seq>;
    static_assert(
        std::is_same_v<Slice, std::integer_sequence<IntT, 15, 20, 25, 30>>);
  }

  // Zero-length slice (begin).
  {
    using Slice = SliceSequence<0, 0, Seq>;
    static_assert(std::is_same_v<Slice, std::integer_sequence<IntT>>);
  }

  // Zero-length slice (last)
  {
    using Slice = SliceSequence<Seq().size() - 1, 0, Seq>;
    static_assert(std::is_same_v<Slice, std::integer_sequence<IntT>>);
  }

  // Zero-length slice (end)
  {
    using Slice = SliceSequence<Seq().size(), 0, Seq>;
    static_assert(std::is_same_v<Slice, std::integer_sequence<IntT>>);
  }

  // Complete slice.
  {
    using Slice = SliceSequence<0, Seq().size(), Seq>;
    static_assert(
        std::is_same_v<
            Slice, std::integer_sequence<IntT, 5, 10, 15, 20, 25, 30, 35, 40>>);
  }
}

TYPED_TEST(ConstexprUtilsTest, PushBackSequence) {
  EXPOSE_TYPE(IntT);

  using Empty = std::integer_sequence<IntT>;

  using Seq1 = PushBackSequence<Empty, 10>;
  static_assert(std::is_same_v<Seq1, std::integer_sequence<IntT, 10>>);

  using Seq2 = PushBackSequence<Seq1, 20>;
  static_assert(std::is_same_v<Seq2, std::integer_sequence<IntT, 10, 20>>);

  using Seq3 = PushBackSequence<Seq2, 30>;
  static_assert(std::is_same_v<Seq3, std::integer_sequence<IntT, 10, 20, 30>>);

  using Seq4 = PushBackSequence<Seq3, 30>;
  static_assert(
      std::is_same_v<Seq4, std::integer_sequence<IntT, 10, 20, 30, 30>>);

  using Seq5 = PushBackSequence<Seq4, 20>;
  static_assert(
      std::is_same_v<Seq5, std::integer_sequence<IntT, 10, 20, 30, 30, 20>>);

  using Seq6 = PushBackSequence<Seq5, 10>;
  static_assert(
      std::is_same_v<Seq6,
                     std::integer_sequence<IntT, 10, 20, 30, 30, 20, 10>>);
}

TYPED_TEST(ConstexprUtilsTest, CompileTimeBubbleSortLiterals) {
  EXPOSE_TYPE(IntT);

  // Pre-sorted
  {
    using Sorted = SortLiterals<IntT, 1, 2, 3, 4, 5, 6, 7, 8>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, 1, 2, 3, 4, 5, 6, 7, 8>>);
  }

  // Pre-sorted + Duplicate elements
  {
    using Sorted = SortLiterals<IntT, 1, 1, 2, 2, 3, 3, 4, 4>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, 1, 1, 2, 2, 3, 3, 4, 4>>);
  }

  // Scrambled
  {
    using Sorted = SortLiterals<IntT, 4, 1, 3, 2>;

    static_assert(
        std::is_same_v<Sorted, std::integer_sequence<IntT, 1, 2, 3, 4>>);
  }

  // Scrambled + Duplicate elements
  {
    using Sorted = SortLiterals<IntT, 1, 4, 1, 2, 4, 3, 2>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, 1, 1, 2, 2, 3, 4, 4>>);
  }

  // Negative values.
  if constexpr (std::is_signed_v<IntT>) {
    using Sorted = SortLiterals<IntT, -1, 4, 1, 2, -4, 3, -2>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, -4, -2, -1, 1, 2, 3, 4>>);
  }
}

TYPED_TEST(ConstexprUtilsTest, CompileTimeQuickSortLiterals) {
  EXPOSE_TYPE(IntT);

  // Quick Sort engages when the number of elements is >= 20

  // Pre-sorted
  {
    using Sorted = SortLiterals<IntT, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16, 17, 18, 19, 20>;

    static_assert(
        std::is_same_v<Sorted, std::integer_sequence<IntT, 1, 2, 3, 4, 5, 6, 7,
                                                     8, 9, 10, 11, 12, 13, 14,
                                                     15, 16, 17, 18, 19, 20>>);
  }

  // Pre-sorted + Duplicate elements
  {
    using Sorted = SortLiterals<IntT, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                                8, 8, 9, 9, 10, 10>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                                             6, 6, 7, 7, 8, 8, 9, 9, 10, 10>>);
  }

  // Scrambled
  {
    using Sorted = SortLiterals<IntT, 1, 20, 3, 18, 5, 16, 7, 14, 9, 12, 11, 10,
                                13, 8, 15, 6, 17, 4, 19, 2>;

    static_assert(
        std::is_same_v<Sorted, std::integer_sequence<IntT, 1, 2, 3, 4, 5, 6, 7,
                                                     8, 9, 10, 11, 12, 13, 14,
                                                     15, 16, 17, 18, 19, 20>>);
  }

  // Scrambled + Duplicate elements
  {
    using Sorted = SortLiterals<IntT, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6, 6, 5, 7, 4,
                                8, 3, 9, 2, 10, 1>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                                             6, 6, 7, 7, 8, 8, 9, 9, 10, 10>>);
  }

  // Negative values
  if constexpr (std::is_signed_v<IntT>) {
    using Sorted = SortLiterals<IntT, 1, -10, 2, -9, 3, -8, 4, -7, 5, -6, 6, -5,
                                7, -4, 8, -3, 9, -2, 10, -1, 0>;

    static_assert(
        std::is_same_v<Sorted, std::integer_sequence<
                                   IntT, -10, -9, -8, -7, -6, -5, -4, -3, -2,
                                   -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>>);
  }
}

TYPED_TEST(ConstexprUtilsTest, CompileTimeBubbleSortSequences) {
  EXPOSE_TYPE(IntT);

  // Pre-sorted
  {
    using Seq = std::integer_sequence<IntT, 1, 2, 3, 4, 5, 6, 7, 8>;
    using Sorted = SortSequence<Seq>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, 1, 2, 3, 4, 5, 6, 7, 8>>);
  }

  // Pre-sorted + Duplicate elements
  {
    using Seq = std::integer_sequence<IntT, 1, 1, 2, 2, 3, 3, 4, 4>;
    using Sorted = SortSequence<Seq>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, 1, 1, 2, 2, 3, 3, 4, 4>>);
  }

  // Scrambled
  {
    using Seq = std::integer_sequence<IntT, 4, 1, 3, 2>;
    using Sorted = SortSequence<Seq>;

    static_assert(
        std::is_same_v<Sorted, std::integer_sequence<IntT, 1, 2, 3, 4>>);
  }

  // Scrambled + Duplicate elements
  {
    using Seq = std::integer_sequence<IntT, 1, 4, 1, 2, 4, 3, 2>;
    using Sorted = SortSequence<Seq>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, 1, 1, 2, 2, 3, 4, 4>>);
  }

  // Negative values.
  if constexpr (std::is_signed_v<IntT>) {
    using Seq = std::integer_sequence<IntT, -1, 4, 1, 2, -4, 3, -2>;
    using Sorted = SortSequence<Seq>;

    static_assert(
        std::is_same_v<Sorted,
                       std::integer_sequence<IntT, -4, -2, -1, 1, 2, 3, 4>>);
  }
}

TYPED_TEST(ConstexprUtilsTest, BubbleSortArrays) {
  EXPOSE_TYPE(IntT);
  EXPOSE_TYPE(Number);

  // Pre-sorted
  {
    static constexpr IntT Seq[]{1, 2, 3, 4, 5, 6, 7, 8};
    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);

    EXPECT_TRUE(equal(Seq, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Seq, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{8, 7, 6, 5, 4, 3, 2, 1};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Pre-sorted + Duplicate elements
  {
    static constexpr IntT Seq[]{1, 1, 2, 2, 3, 3, 4, 4};
    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);

    EXPECT_TRUE(equal(Seq, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Seq, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{4, 4, 3, 3, 2, 2, 1, 1};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Pre-sorted (enum)
  {
    constexpr Number Seq[]{Number::One, Number::Two, Number::Three,
                           Number::Four};
    static constexpr std::array<Number, std::size(Seq)> Sorted = ce_sort(Seq);

    EXPECT_TRUE(equal(Seq, Sorted));

    // Run-time sort in-place.
    std::array<Number, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Seq, Arr));

    // Custom comparison object.
    static constexpr std::array<Number, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const Number ExpectedG[std::size(Seq)]{Number::Four, Number::Three,
                                                  Number::Two, Number::One};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Pre-sorted (enum) + Duplicate elements
  {
    constexpr Number Seq[]{Number::One, Number::One,   Number::Two,
                           Number::Two, Number::Three, Number::Four,
                           Number::Four};
    static constexpr std::array<Number, std::size(Seq)> Sorted = ce_sort(Seq);

    EXPECT_TRUE(equal(Seq, Sorted));

    // Run-time sort in-place.
    std::array<Number, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Seq, Arr));

    // Custom comparison object.
    static constexpr std::array<Number, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const Number ExpectedG[std::size(Seq)]{
        Number::Four, Number::Four, Number::Three, Number::Two,
        Number::Two,  Number::One,  Number::One};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Scrambled
  {
    static constexpr IntT Seq[]{8, 4, 1, 6, 2, 3, 7, 5};

    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);
    static const IntT Expected[std::size(Seq)]{1, 2, 3, 4, 5, 6, 7, 8};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{8, 7, 6, 5, 4, 3, 2, 1};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Scrambled + Duplicate elements
  {
    static constexpr IntT Seq[]{2, 4, 1, 3, 1, 3, 4, 2};
    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);
    static constexpr IntT Expected[]{1, 1, 2, 2, 3, 3, 4, 4};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{4, 4, 3, 3, 2, 2, 1, 1};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Scrambled (enum)
  {
    static constexpr Number Seq[]{Number::Four, Number::One, Number::Three,
                                  Number::Two};
    static constexpr std::array<Number, std::size(Seq)> Sorted = ce_sort(Seq);
    static const Number Expected[std::size(Seq)]{Number::One, Number::Two,
                                                 Number::Three, Number::Four};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<Number, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<Number, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const Number ExpectedG[std::size(Seq)]{Number::Four, Number::Three,
                                                  Number::Two, Number::One};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Scrambled (enum) + Duplicate elements
  {
    static constexpr Number Seq[]{Number::One, Number::Four, Number::One,
                                  Number::Two, Number::Four, Number::Three,
                                  Number::Two};
    static constexpr std::array<Number, std::size(Seq)> Sorted = ce_sort(Seq);
    static const Number Expected[std::size(Seq)]{
        Number::One,   Number::One,  Number::Two, Number::Two,
        Number::Three, Number::Four, Number::Four};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<Number, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<Number, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const Number ExpectedG[std::size(Seq)]{
        Number::Four, Number::Four, Number::Three, Number::Two,
        Number::Two,  Number::One,  Number::One};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Negative values
  if constexpr (std::is_signed_v<IntT>) {
    static constexpr IntT Seq[]{-2, 4, 1, -3, -1, 3, -4, 2};
    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);
    static constexpr IntT Expected[]{-4, -3, -2, -1, 1, 2, 3, 4};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{4, 3, 2, 1, -1, -2, -3, -4};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }
}

TYPED_TEST(ConstexprUtilsTest, QuickSortArrays) {
  EXPOSE_TYPE(IntT);

  // Quick Sort engages when the number of elements is >= 20

  // Pre-sorted
  {
    static constexpr IntT Seq[]{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);
    static const IntT Expected[std::size(Seq)]{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{
        20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Pre-sorted + Duplicate elements
  {
    static constexpr IntT Seq[]{1, 1, 2, 2, 3, 3, 4, 4, 5,  5,
                                6, 6, 7, 7, 8, 8, 9, 9, 10, 10};

    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);
    static const IntT Expected[std::size(Seq)]{1, 1, 2, 2, 3, 3, 4, 4, 5,  5,
                                               6, 6, 7, 7, 8, 8, 9, 9, 10, 10};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{10, 10, 9, 9, 8, 8, 7, 7, 6, 6,
                                                5,  5,  4, 4, 3, 3, 2, 2, 1, 1};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Scrambled
  {
    static constexpr IntT Seq[]{1,  20, 3,  18, 5,  16, 7,  14, 9,  12,
                                11, 10, 13, 8,  15, 6,  17, 4,  19, 2};

    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);
    static const IntT Expected[std::size(Seq)]{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{
        20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }

  // Scrambled + Duplicate elements
  {
    static constexpr IntT Seq[]{1, 10, 2, 9, 3, 8, 4, 7, 5,  6,
                                6, 5,  7, 4, 8, 3, 9, 2, 10, 1};

    static constexpr std::array<IntT, std::size(Seq)> Sorted = ce_sort(Seq);
    static const IntT Expected[std::size(Seq)]{1, 1, 2, 2, 3, 3, 4, 4, 5,  5,
                                               6, 6, 7, 7, 8, 8, 9, 9, 10, 10};

    EXPECT_TRUE(equal(Expected, Sorted));

    // Run-time sort in-place.
    std::array<IntT, std::size(Seq)> Arr = to_array(Seq);
    ce_sort_inplace(Arr);

    EXPECT_TRUE(equal(Expected, Arr));

    // Custom comparison object.
    static constexpr std::array<IntT, std::size(Seq)> SortedG =
        ce_sort(Seq, Greater);
    static const IntT ExpectedG[std::size(Seq)]{10, 10, 9, 9, 8, 8, 7, 7, 6, 6,
                                                5,  5,  4, 4, 3, 3, 2, 2, 1, 1};

    EXPECT_TRUE(equal(ExpectedG, SortedG));
  }
}