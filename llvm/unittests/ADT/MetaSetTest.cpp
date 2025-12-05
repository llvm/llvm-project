//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains testing for the MetaSet infrastructure.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/MetaSet.h"
#include "llvm/ADT/ConstexprUtils.h"

#include "gtest/gtest.h"
#include <limits>
#include <type_traits>
#include <utility>

using namespace llvm;

namespace {

template <typename Iter0, typename Iter1>
constexpr bool ce_equal( // NOLINT (readability-identifier-naming)
    Iter0 B0, Iter0 E0, Iter1 B1, Iter1 E1) {
  for (; B0 != E0; ++B0, ++B1) {
    if (B1 == E1 || *B0 != *B1)
      return false;
  }
  return B1 == E1;
}

template <typename R1, typename R2>
constexpr bool ce_equal( // NOLINT (readability-identifier-naming)
    R1 const &L, R2 const &R) {
  return ce_equal(adl_begin(L), adl_end(L), adl_begin(R), adl_end(R));
}

template <typename Range1, typename Range2>
constexpr bool ce_requal( // NOLINT (readability-identifier-naming)
    Range1 const &L, Range2 const &R) {
  return ce_equal(adl_rbegin(L), adl_rend(L), adl_rbegin(R), adl_rend(R));
}

template <typename PosT, auto... Values>
inline constexpr size_t ComputeNumWords =
    MetaBitsetNumWordsDetailed<PosT, ce_min<PosT>(Values...),
                               ce_max<PosT>(Values...)>;

enum class Kind { Bitset, Sequence, SparseBitset };
template <typename T> struct BuildBitset {
  using PosT = T;

  static constexpr bool IsBitset = true;
  static constexpr bool IsSparseBitset = false;

  template <auto... Vals> static auto makeSet() {
    return MakeMetaBitset<PosT, Vals...>();
  }

  template <Kind K, typename QueryT, typename ExpectedT> struct CheckType {
    static_assert(K != Kind::Bitset || std::is_same_v<QueryT, ExpectedT>);
  };
};

template <typename T> struct BuildBitsetSorted {
  using PosT = T;

  static constexpr bool IsBitset = true;
  static constexpr bool IsSparseBitset = false;

  template <auto... Vals> static auto makeSet() {
    using Sorted = SortLiterals<PosT, PosT(Vals)...>;
    return MakeMetaBitsetFromSortedSequence<Sorted>();
  }

  template <Kind K, typename QueryT, typename ExpectedT> struct CheckType {
    static_assert(K != Kind::Bitset || std::is_same_v<QueryT, ExpectedT>);
  };
};

template <typename T> struct BuildSequenceSet {
  using PosT = T;

  static constexpr bool IsBitset = false;
  static constexpr bool IsSparseBitset = false;

  template <auto... Vals> static auto makeSet() {
    return MakeMetaSequenceSet<PosT, Vals...>();
  }

  template <Kind K, typename QueryT, typename ExpectedT> struct CheckType {
    static_assert(K != Kind::Sequence || std::is_same_v<QueryT, ExpectedT>);
  };
};

template <typename T> struct BuildSparseBitset {
  using PosT = T;

  static constexpr bool IsBitset = false;
  static constexpr bool IsSparseBitset = true;

  template <auto... Vals> static auto makeSet() {
    return MakeMetaSparseBitset<PosT, 512, Vals...>();
  }

  template <Kind K, typename QueryT, typename ExpectedT> struct CheckType {
    static_assert(K != Kind::SparseBitset || std::is_same_v<QueryT, ExpectedT>);
  };
};

template <typename BuilderT, auto... Vals>
using MakeMetaSet = decltype(BuilderT::template makeSet<Vals...>());

template <typename T> class MetaSetTest : public testing::Test {
protected:
  using Builder = T;
  using PosT = typename T::PosT;
  using ThisT = MetaSetTest;

  enum class E : PosT { A, B, C, D, E };

  static constexpr int64_t min() { return std::numeric_limits<PosT>::min(); }
  static constexpr uint64_t max() { return std::numeric_limits<PosT>::max(); }
};

#define EXPOSE_TYPE(type)                                                      \
  using type = typename remove_cvref_t<decltype(*this)>::type

// clang-format off
using TestTypes =
    ::testing::Types<
      // MetaBitset
      BuildBitset<signed char>,
      BuildBitset<short>,
      BuildBitset<int>,
      BuildBitset<long>,
      BuildBitset<long long>,
      BuildBitset<unsigned char>,
      BuildBitset<unsigned short>,
      BuildBitset<unsigned>,
      BuildBitset<unsigned long>,
      BuildBitset<unsigned long long>,
      // MetaBitset (sorted first)
      BuildBitsetSorted<signed char>,
      BuildBitsetSorted<short>,
      BuildBitsetSorted<int>,
      BuildBitsetSorted<long>,
      BuildBitsetSorted<long long>,
      BuildBitsetSorted<unsigned char>,
      BuildBitsetSorted<unsigned short>,
      BuildBitsetSorted<unsigned>,
      BuildBitsetSorted<unsigned long>,
      BuildBitsetSorted<unsigned long long>,
      // MetaSequenceSet
      BuildSequenceSet<signed char>,
      BuildSequenceSet<short>,
      BuildSequenceSet<int>,
      BuildSequenceSet<long>,
      BuildSequenceSet<long long>,
      BuildSequenceSet<unsigned char>,
      BuildSequenceSet<unsigned short>,
      BuildSequenceSet<unsigned>,
      BuildSequenceSet<unsigned long>,
      BuildSequenceSet<unsigned long long>,
      // MetaSparseBitset
      BuildSparseBitset<signed char>,
      BuildSparseBitset<short>,
      BuildSparseBitset<int>,
      BuildSparseBitset<long>,
      BuildSparseBitset<long long>,
      BuildSparseBitset<unsigned char>,
      BuildSparseBitset<unsigned short>,
      BuildSparseBitset<unsigned>,
      BuildSparseBitset<unsigned long>,
      BuildSparseBitset<unsigned long long>
    >;
// clang-format on

TYPED_TEST_SUITE(MetaSetTest, TestTypes, );

TYPED_TEST(MetaSetTest, BitsetBuilderCompiletime) {
  EXPOSE_TYPE(PosT);
  EXPOSE_TYPE(Builder);

  if constexpr (!Builder::IsBitset)
    return;

  static constexpr PosT MIN = std::numeric_limits<PosT>::min();
  static constexpr PosT MAX = std::numeric_limits<PosT>::max();

  using Empty = MakeMetaBitset<PosT>;
  static_assert(MetaBitsetNumWords<Empty> == 0);
  static_assert(std::is_same_v<Empty, MetaBitset<PosT, 0>>);

  using ZeroOffset = MakeMetaBitset<PosT, 0, 1, 2, 3>;
  using ZeroOffsetD = MakeMetaBitsetDetailed<PosT, 0, 3, 0, 1, 2, 3>;
  static_assert(MetaBitsetNumWords<ZeroOffset> == 1);
  static_assert(ComputeNumWords<PosT, 0, 1, 2, 3> == 1);
  static_assert(std::is_same_v<ZeroOffset, ZeroOffsetD>);
  static_assert(std::is_same_v<ZeroOffset, MetaBitset<PosT, 0, 0xf>>);

  using PositiveOffset = MakeMetaBitset<PosT, 6, 5, 7>;
  using PositiveOffsetD = MakeMetaBitsetDetailed<PosT, 5, 7, 6, 5, 7>;
  static_assert(MetaBitsetNumWords<PositiveOffset> == 1);
  static_assert(ComputeNumWords<PosT, 6, 5, 7> == 1);
  static_assert(std::is_same_v<PositiveOffset, PositiveOffsetD>);
  static_assert(std::is_same_v<PositiveOffset, MetaBitset<PosT, 5, 0x7>>);

  if constexpr (std::is_signed_v<PosT>) {
    using NegativeOffset = MakeMetaBitset<PosT, 0, -1, -2, -3, 1>;
    using NegativeOffsetD =
        MakeMetaBitsetDetailed<PosT, -3, 1, 0, -1, -2, -3, 1>;
    static_assert(MetaBitsetNumWords<NegativeOffset> == 1);
    static_assert(ComputeNumWords<PosT, -3, 0> == 1);
    static_assert(std::is_same_v<NegativeOffset, NegativeOffsetD>);
    static_assert(std::is_same_v<NegativeOffset, MetaBitset<PosT, -3, 0x1f>>);

    using NegativeOffset2 = MakeMetaBitset<PosT, -128, 127>;
    using NegativeOffset2D = MakeMetaBitset<PosT, -128, 127, -128, 127>;
    static_assert(MetaBitsetNumWords<NegativeOffset2> == 4);
    static_assert(ComputeNumWords<PosT, -128, 127> == 4);
    static_assert(std::is_same_v<NegativeOffset2, NegativeOffset2D>);
    static_assert(
        std::is_same_v<NegativeOffset2,
                       MetaBitset<PosT, -128, 1ULL, 0ULL, 0ULL, (1ULL << 63)>>);
  }

  using DuplicateElements = MakeMetaBitset<PosT, 0, 1, 1, 2, 2, 3, 3>;
  using DuplicateElementsD =
      MakeMetaBitsetDetailed<PosT, 0, 3, 0, 1, 1, 2, 2, 3, 3>;
  static_assert(MetaBitsetNumWords<DuplicateElements> == 1);
  static_assert(ComputeNumWords<PosT, 0, 3> == 1);
  static_assert(std::is_same_v<DuplicateElements, DuplicateElementsD>);
  static_assert(std::is_same_v<DuplicateElements, MetaBitset<PosT, 0, 0xf>>);

  using WordBoundary1 = MakeMetaBitset<PosT, 0, 63>;
  using WordBoundary1D = MakeMetaBitsetDetailed<PosT, 0, 63, 0, 63>;
  static_assert(ComputeNumWords<PosT, 0, 63> == 1);
  static_assert(MetaBitsetNumWords<WordBoundary1> == 1);
  static_assert(std::is_same_v<WordBoundary1, WordBoundary1D>);
  static_assert(std::is_same_v<WordBoundary1,
                               MetaBitset<PosT, 0, (1ULL << 0 | 1ULL << 63)>>);

  using WordBoundary2 = MakeMetaBitset<PosT, 0, 64>;
  using WordBoundary2D = MakeMetaBitsetDetailed<PosT, 0, 64, 0, 64>;
  static_assert(ComputeNumWords<PosT, 0, 64> == 2);
  static_assert(MetaBitsetNumWords<WordBoundary2> == 2);
  static_assert(std::is_same_v<WordBoundary2, WordBoundary2D>);
  static_assert(std::is_same_v<WordBoundary2, MetaBitset<PosT, 0, 1ULL, 1ULL>>);

  using WordBoundary3 = MakeMetaBitset<PosT, 0, 127>;
  using WordBoundary3D = MakeMetaBitset<PosT, 0, 127>;
  static_assert(ComputeNumWords<PosT, 0, 127> == 2);
  static_assert(MetaBitsetNumWords<WordBoundary3> == 2);
  static_assert(std::is_same_v<WordBoundary3, WordBoundary3D>);
  static_assert(
      std::is_same_v<WordBoundary3, MetaBitset<PosT, 0, 1ULL, (1ULL << 63)>>);

  if constexpr (MAX > 128) {
    using WordBoundary4 = MakeMetaBitset<PosT, 0, 128>;
    using WordBoundary4D = MakeMetaBitsetDetailed<PosT, 0, 128, 0, 128>;
    static_assert(ComputeNumWords<PosT, 0, 128> == 3);
    static_assert(MetaBitsetNumWords<WordBoundary4> == 3);
    static_assert(std::is_same_v<WordBoundary4, WordBoundary4D>);
    static_assert(
        std::is_same_v<WordBoundary4, MetaBitset<PosT, 0, 1ULL, 0ULL, 1ULL>>);
  }

  if constexpr (MAX > 511) {
    using BigGap = MakeMetaBitset<PosT, 0, 511>;
    using BigGapD = MakeMetaBitsetDetailed<PosT, 0, 511, 0, 511>;
    static_assert(ComputeNumWords<PosT, 0, 511> == 8);
    static_assert(MetaBitsetNumWords<BigGap> == 8);
    static_assert(std::is_same_v<BigGap, BigGapD>);
    using BigGapExp = MetaBitset<PosT, 0, 1ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
                                 0ULL, (1ULL << 63)>;
    static_assert(std::is_same_v<BigGap, BigGapExp>);
  } else {
    // 8-bit integer types.
    using BigGap = MakeMetaBitset<PosT, 0, MAX>;
    using BigGapD = MakeMetaBitsetDetailed<PosT, 0, MAX, 0, MAX>;
    static_assert(std::is_same_v<BigGap, BigGapD>);
    if constexpr (std::is_unsigned_v<PosT>) {
      static_assert(ComputeNumWords<PosT, 0, MAX> == 4);
      static_assert(MetaBitsetNumWords<BigGap> == 4);
      using BigGapExp = MetaBitset<PosT, 0, 1ULL, 0ULL, 0ULL, (1ULL << 63)>;
      static_assert(std::is_same_v<BigGap, BigGapExp>);
    } else {
      static_assert(ComputeNumWords<PosT, 0, MAX> == 2);
      static_assert(MetaBitsetNumWords<BigGap> == 2);
      using BigGapExp = MetaBitset<PosT, 0, 1ULL, (1ULL << 63)>;
      static_assert(std::is_same_v<BigGap, BigGapExp>);
    }
  }

  using LargeOffset = MakeMetaBitset<PosT, MAX - 63, MAX>;
  using LargeOffsetD = MakeMetaBitset<PosT, MAX - 63, MAX, (MAX - 63), MAX>;
  static_assert(ComputeNumWords<PosT, MAX - 63, MAX> == 1);
  static_assert(MetaBitsetNumWords<LargeOffset> == 1);
  static_assert(std::is_same_v<LargeOffset, LargeOffsetD>);
  static_assert(
      std::is_same_v<LargeOffset,
                     MetaBitset<PosT, MAX - 63, (1ULL << 0 | 1ULL << 63)>>);

  if constexpr (std::is_signed_v<PosT>) {
    if constexpr (MIN < -36729) {
      using LargeNegativeOffset = MakeMetaBitset<PosT, -36729, -36729 + 64>;
      using LargeNegativeOffsetD =
          MakeMetaBitset<PosT, -36729, -36729 + 64, -36729, -36729 + 64>;
      static_assert(ComputeNumWords<PosT, -36729, -36729 + 64> == 2);
      static_assert(MetaBitsetNumWords<LargeNegativeOffset> == 2);
      static_assert(std::is_same_v<LargeNegativeOffset, LargeNegativeOffsetD>);
      static_assert(std::is_same_v<LargeNegativeOffset,
                                   MetaBitset<PosT, -36729, 1ULL, 1ULL>>);
    }
  }
}

template <typename BuilderT> class CheckMetaTypes {
public:
  template <typename QueryT, typename ExpectedBitsetT,
            typename ExpectedSequenceT, typename ExpectedSparseBitsetT>
  static constexpr void check() {
    typename BuilderT::template CheckType<Kind::Bitset, QueryT,
                                          ExpectedBitsetT>();

    typename BuilderT::template CheckType<Kind::Sequence, QueryT,
                                          ExpectedSequenceT>();

    typename BuilderT::template CheckType<Kind::SparseBitset, QueryT,
                                          ExpectedSparseBitsetT>();
  }
};

TYPED_TEST(MetaSetTest, BuiltTypeVerify) {
  EXPOSE_TYPE(PosT);
  // EXPOSE_TYPE(E);
  EXPOSE_TYPE(Builder);
  using ThisT = std::remove_cv_t<std::remove_reference_t<decltype(*this)>>;

  using Checker = CheckMetaTypes<Builder>;

  // All Set types.
  {
    // Sorted values, small range
    {
      using BS = MakeMetaSet<Builder, PosT(1), PosT(2), PosT(3), PosT(4)>;

      using ExpectedBitset = MetaBitset<PosT, 1, 0xf>;

      using ExpectedSequence =
          MetaSequenceSet<std::integer_sequence<PosT, 1, 2, 3, 4>>;

      using ExpectedSparseBitset = MetaSparseBitset<ExpectedBitset>;

      Checker::template check<BS, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }

    // Reverse sorted values, small range.
    {
      using BS = MakeMetaSet<Builder, PosT(4), PosT(3), PosT(2), PosT(1)>;

      using ExpectedBitset = MetaBitset<PosT, 1, 0xf>;

      using ExpectedSequence =
          MetaSequenceSet<std::integer_sequence<PosT, 4, 3, 2, 1>>;

      using ExpectedSparseBitset = MetaSparseBitset<ExpectedBitset>;

      Checker::template check<BS, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }

    // Duplicate values, small range.
    {
      using BS = MakeMetaSet<Builder, 1, 3, 5, 7, 9, 3, 5, 7>;

      using ExpectedBitset = MetaBitset<PosT, 1, 0x155>;

      using ExpectedSequence =
          MetaSequenceSet<std::integer_sequence<PosT, 1, 3, 5, 7, 9, 3, 5, 7>>;

      using ExpectedSparseBitset = MetaSparseBitset<ExpectedBitset>;

      Checker::template check<BS, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }

    // Large offset from 0.
    {
      using BS = MakeMetaSet<Builder, 101, 103, 105, 107, 109>;

      using ExpectedBitset = MetaBitset<PosT, 101, 0x155>;

      using ExpectedSequence =
          MetaSequenceSet<std::integer_sequence<PosT, 101, 103, 105, 107, 109>>;

      using ExpectedSparseBitset = MetaSparseBitset<ExpectedBitset>;

      Checker::template check<BS, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }

    // Larger offset from 0.
    if constexpr (ThisT::max() > 10009u) {
      using BS = MakeMetaSet<Builder, 10001, 10003, 10005, 10007, 10009>;

      using ExpectedBitset = MetaBitset<PosT, 10001, 0x155>;

      using ExpectedSequence = MetaSequenceSet<
          std::integer_sequence<PosT, 10001, 10003, 10005, 10007, 10009>>;

      using ExpectedSparseBitset = MetaSparseBitset<ExpectedBitset>;

      Checker::template check<BS, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }

    // Negative offset.
    if constexpr (ThisT::min() < -1) {
      using BS = MakeMetaSet<Builder, -101, -103, -105, -107, -109>;

      using ExpectedBitset = MetaBitset<PosT, -109, 0x155>;

      using ExpectedSequence = MetaSequenceSet<
          std::integer_sequence<PosT, -101, -103, -105, -107, -109>>;

      using ExpectedSparseBitset = MetaSparseBitset<ExpectedBitset>;

      Checker::template check<BS, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }
  }

  // MetaSparseBitset + MetaSequenceSet validation.
  if constexpr (!Builder::IsBitset) {
    using ExpectedBitset = void;

    // Single cluster; no singletons
    {
      using Fives = MakeMetaSet<Builder, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95,
                                10, 20, 30, 40, 50, 60, 70, 80, 90, 100>;

      using ExpectedSequence = MetaSequenceSet<
          std::integer_sequence<PosT, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 10,
                                20, 30, 40, 50, 60, 70, 80, 90, 100>>;
      using ExpectedSparseBitset =
          MetaSparseBitset<MetaBitset<PosT, 5, 0x1084210842108421, 0x84210842>>;

      Checker::template check<Fives, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }

    // Multiple clusters; no singletons.
    if constexpr (ThisT::max() >= 10008u) {
      using Clusters = MakeMetaSet<Builder, 10000, 10002, 10004, 10006, 10008,
                                   1000, 1002, 1004, 1006, 1008>;

      using ExpectedSequence = MetaSequenceSet<
          std::integer_sequence<PosT, 10000, 10002, 10004, 10006, 10008, 1000,
                                1002, 1004, 1006, 1008>>;

      using ExpectedSparseBitset =
          MetaSparseBitset<MetaBitset<PosT, 1000, 0x155>,
                           MetaBitset<PosT, 10000, 0x155>>;

      Checker::template check<Clusters, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }

    // Multiple clusters; with singletons.
    if constexpr (ThisT::max() >= 10008u) {
      using ClustersAndSingletons =
          MakeMetaSet<Builder, 10000, 10002, 10004, 10006, 10008, 1000, 1002,
                      1004, 1006, 1008, 8000, 5000>;

      using ExpectedSequence = MetaSequenceSet<
          std::integer_sequence<PosT, 10000, 10002, 10004, 10006, 10008, 1000,
                                1002, 1004, 1006, 1008, 8000, 5000>>;
      using ExpectedSparseBitset =
          MetaSparseBitset<MetaBitset<PosT, 1000, 0x155>,
                           MetaBitset<PosT, 10000, 0x155>,
                           MakeMetaSequenceSet<PosT, 5000, 8000>>;

      Checker::template check<ClustersAndSingletons, ExpectedBitset,
                              ExpectedSequence, ExpectedSparseBitset>();
    }

    // No clusters; all singletons.
    if constexpr (ThisT::max() >= 90000u) {
      using Singletons = MakeMetaSet<Builder, 10000, 5000, 8000, 20000, 90000>;

      using ExpectedSequence = MetaSequenceSet<
          std::integer_sequence<PosT, 10000, 5000, 8000, 20000, 90000>>;

      using ExpectedSparseBitset = MetaSparseBitset<
          MakeMetaSequenceSet<PosT, 5000, 8000, 10000, 20000, 90000>>;

      Checker::template check<Singletons, ExpectedBitset, ExpectedSequence,
                              ExpectedSparseBitset>();
    }
  }

  // MetaSparseBitset validation (varied word sizes)
  if constexpr (Builder::IsSparseBitset) {
    // Single cluster; no singletons
    {
      using Fives =
          MakeMetaSparseBitset<PosT, 128, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95,
                               10, 20, 30, 40, 50, 60, 70, 80, 90, 100>;

      using Expected =
          MetaSparseBitset<MetaBitset<PosT, 5, 0x1084210842108421, 0x84210842>>;

      static_assert(std::is_same_v<Fives, Expected>);
    }

    // Multiple clusters; no singletons.
    if constexpr (ThisT::max() >= 10008u) {
      using Clusters =
          MakeMetaSparseBitset<PosT, 256, 10000, 10002, 10004, 10006, 10008,
                               1000, 1002, 1004, 1006, 1008>;

      using Expected = MetaSparseBitset<MetaBitset<PosT, 1000, 0x155>,
                                        MetaBitset<PosT, 10000, 0x155>>;

      static_assert(std::is_same_v<Clusters, Expected>);
    }

    // Multiple clusters; with singletons.
    if constexpr (ThisT::max() >= 10008u) {
      using ClustersAndSingletons =
          MakeMetaSparseBitset<PosT, 64, 10000, 10002, 10004, 10006, 10008,
                               1000, 1002, 1004, 1006, 1008, 8000, 5000>;

      using Expected = MetaSparseBitset<MetaBitset<PosT, 1000, 0x155>,
                                        MetaBitset<PosT, 10000, 0x155>,
                                        MakeMetaSequenceSet<PosT, 5000, 8000>>;

      static_assert(std::is_same_v<ClustersAndSingletons, Expected>);
    }

    // No clusters; all singletons.
    if constexpr (ThisT::max() >= 90000u) {
      using Singletons =
          MakeMetaSparseBitset<PosT, 64, 10000, 5000, 8000, 20000, 90000>;

      using Expected = MetaSparseBitset<
          MakeMetaSequenceSet<PosT, 5000, 8000, 10000, 20000, 90000>>;

      static_assert(std::is_same_v<Singletons, Expected>);
    }

    // Word size too small resulting in singletons
    if constexpr (ThisT::max() > 192u) {
      using WordSizeTooSmall = MakeMetaSparseBitset<PosT, 64, 5, 4, 3, 2, 1, 0,
                                                    64, 65, 130, 129, 128, 192>;

      using Expected = MetaSparseBitset<
          MetaBitset<PosT, 0, 0x3f>, MetaBitset<PosT, 128, 0x7>,
          MetaSequenceSet<std::integer_sequence<PosT, 64, 65, 192>>>;

      static_assert(std::is_same_v<WordSizeTooSmall, Expected>);
    }

    // Larger word size; no singletons
    if constexpr (ThisT::max() > 192u) {
      using WordSizeTooSmall = MakeMetaSparseBitset<PosT, 128, 5, 4, 3, 2, 1, 0,
                                                    64, 65, 130, 129, 128, 192>;

      using Expected = MetaSparseBitset<MetaBitset<PosT, 0, 0x3f, 0x3>,
                                        MetaBitset<PosT, 128, 0x7, 0x1>>;

      static_assert(std::is_same_v<WordSizeTooSmall, Expected>);
    }

    // Even larger word size; coalesce MetaBitsets
    if constexpr (ThisT::max() > 192u) {
      using WordSizeTooSmall = MakeMetaSparseBitset<PosT, 256, 5, 4, 3, 2, 1, 0,
                                                    64, 65, 130, 129, 128, 192>;

      using Expected =
          MetaSparseBitset<MetaBitset<PosT, 0, 0x3f, 0x3, 0x7, 0x1>>;

      static_assert(std::is_same_v<WordSizeTooSmall, Expected>);
    }
  }
}

TYPED_TEST(MetaSetTest, SortedSequence) {
  EXPOSE_TYPE(PosT);
  EXPOSE_TYPE(Builder);
  EXPOSE_TYPE(ThisT);

  {
    using Empty = MakeMetaSet<Builder>;

    using Expected = std::integer_sequence<PosT>;

    static_assert(
        std::is_same_v<typename Empty::sorted_sequence_type, Expected>);
  }

  {
    using ZeroOffset = MakeMetaSet<Builder, 0, 1, 2, 3>;

    using Expected = std::integer_sequence<PosT, 0, 1, 2, 3>;
    static_assert(
        std::is_same_v<typename ZeroOffset::sorted_sequence_type, Expected>);
  }

  {
    using PositiveOffset = MakeMetaSet<Builder, 6, 5, 7>;

    using Expected = std::integer_sequence<PosT, 5, 6, 7>;
    static_assert(std::is_same_v<typename PositiveOffset::sorted_sequence_type,
                                 Expected>);
  }

  if constexpr (std::is_signed_v<PosT>) {
    using NegativeOffset = MakeMetaSet<Builder, 0, -1, -2, -3, 1>;

    using Expected = std::integer_sequence<PosT, -3, -2, -1, 0, 1>;
    static_assert(std::is_same_v<typename NegativeOffset::sorted_sequence_type,
                                 Expected>);
  }

  if constexpr (std::is_signed_v<PosT>) {
    using NegativeOffset2 = MakeMetaSet<Builder, -128, 127>;

    using Expected = std::integer_sequence<PosT, -128, 127>;
    static_assert(std::is_same_v<typename NegativeOffset2::sorted_sequence_type,
                                 Expected>);
  }

  {
    using DuplicateElements = MakeMetaSet<Builder, 0, 1, 1, 2, 2, 3, 3>;

    using Expected =
        std::conditional_t<IsMetaSetWithDuplicateElements<DuplicateElements>,
                           std::integer_sequence<PosT, 0, 1, 1, 2, 2, 3, 3>,
                           std::integer_sequence<PosT, 0, 1, 2, 3>>;
    static_assert(
        std::is_same_v<typename DuplicateElements::sorted_sequence_type,
                       Expected>);
  }

  {
    using WordBoundary1 = MakeMetaSet<Builder, 0, 63>;

    using Expected = std::integer_sequence<PosT, 0, 63>;
    static_assert(
        std::is_same_v<typename WordBoundary1::sorted_sequence_type, Expected>);
  }

  {
    using WordBoundary2 = MakeMetaSet<Builder, 0, 64>;

    using Expected = std::integer_sequence<PosT, 0, 64>;
    static_assert(
        std::is_same_v<typename WordBoundary2::sorted_sequence_type, Expected>);
  }

  {
    using WordBoundary3 = MakeMetaSet<Builder, 0, 127>;

    using Expected = std::integer_sequence<PosT, 0, 127>;
    static_assert(
        std::is_same_v<typename WordBoundary3::sorted_sequence_type, Expected>);
  }

  if constexpr (ThisT::max() > 128) {
    using WordBoundary4 = MakeMetaSet<Builder, 0, 128>;

    using Expected = std::integer_sequence<PosT, 0, 128>;
    static_assert(
        std::is_same_v<typename WordBoundary4::sorted_sequence_type, Expected>);
  }

  if constexpr (ThisT::max() > 511) {
    using BigGap = MakeMetaSet<Builder, 0, 511>;

    using Expected = std::integer_sequence<PosT, 0, 511>;
    static_assert(
        std::is_same_v<typename BigGap::sorted_sequence_type, Expected>);
  } else {
    // 8-bit integer types.
    using BigGap = MakeMetaSet<Builder, 0, ThisT::max()>;

    using Expected = std::integer_sequence<PosT, 0, ThisT::max()>;
    static_assert(
        std::is_same_v<typename BigGap::sorted_sequence_type, Expected>);
  }

  {
    using LargeOffset = MakeMetaSet<Builder, ThisT::max() - 63, ThisT::max()>;

    using Expected =
        std::integer_sequence<PosT, ThisT::max() - 63, ThisT::max()>;
    static_assert(
        std::is_same_v<typename LargeOffset::sorted_sequence_type, Expected>);
  }

  if constexpr (std::is_signed_v<PosT> && ThisT::min() < -36729) {
    using LargeNegativeOffset = MakeMetaSet<Builder, -36729, -36729 + 64>;

    using Expected = std::integer_sequence<PosT, -36729, -36729 + 64>;
    static_assert(
        std::is_same_v<typename LargeNegativeOffset::sorted_sequence_type,
                       Expected>);
  }
}

TYPED_TEST(MetaSetTest, Container) {
  EXPOSE_TYPE(PosT);
  EXPOSE_TYPE(Builder);
  EXPOSE_TYPE(ThisT);

  {
    using Empty = MakeMetaSet<Builder>;

    MetaSetSortedContainer<Empty> Test;
    static_assert(Test.size() == 0);
    static_assert(Test.empty() == true);
  }

  {
    using OneElem = MakeMetaSet<Builder, 127>;

    MetaSetSortedContainer<OneElem> Test;
    static constexpr PosT Expected[]{127};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[0] == Expected[0]);

    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }

  {
    using ZeroOffset = MakeMetaSet<Builder, 0, 1, 2, 3>;

    MetaSetSortedContainer<ZeroOffset> Test;
    static constexpr PosT Expected[]{0, 1, 2, 3};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  }

  {
    using PositiveOffset = MakeMetaSet<Builder, 6, 5, 7>;

    MetaSetSortedContainer<PositiveOffset> Test;
    static constexpr PosT Expected[]{5, 6, 7};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  }

  if constexpr (std::is_signed_v<PosT>) {
    using NegativeOffset = MakeMetaSet<Builder, 0, -1, -2, -3, 1>;

    MetaSetSortedContainer<NegativeOffset> Test;
    static constexpr PosT Expected[]{-3, -2, -1, 0, 1};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  }

  if constexpr (std::is_signed_v<PosT>) {
    using NegativeOffset2 = MakeMetaSet<Builder, -128, 127>;

    MetaSetSortedContainer<NegativeOffset2> Test;
    static constexpr PosT Expected[]{-128, 127};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  }

  {
    using DuplicateElements = MakeMetaSet<Builder, 0, 1, 1, 2, 2, 3, 3>;

    MetaSetSortedContainer<DuplicateElements> Test;

    auto MakeExpected = []() constexpr {
      if constexpr (IsMetaSetWithDuplicateElements<DuplicateElements>)
        return std::integer_sequence<PosT, 0, 1, 1, 2, 2, 3, 3>();
      else
        return std::integer_sequence<PosT, 0, 1, 2, 3>();
    };

    static constexpr auto Expected = to_array(MakeExpected());

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  }

  if constexpr (ThisT::max() > 511) {
    using BigGap = MakeMetaSet<Builder, 0, 511>;

    MetaSetSortedContainer<BigGap> Test;
    static constexpr PosT Expected[]{0, 511};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  } else {
    // 8-bit integer types.
    using BigGap = MakeMetaSet<Builder, 0, ThisT::max()>;

    MetaSetSortedContainer<BigGap> Test;
    static constexpr PosT Expected[]{0, ThisT::max()};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  }

  {
    using LargeOffset = MakeMetaSet<Builder, ThisT::max() - 63, ThisT::max()>;

    MetaSetSortedContainer<LargeOffset> Test;
    static constexpr PosT Expected[]{ThisT::max() - 63, ThisT::max()};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  }

  if constexpr (std::is_signed_v<PosT> && ThisT::min() < -36729) {
    using LargeNegativeOffset = MakeMetaSet<Builder, -36729, -36729 + 64>;

    MetaSetSortedContainer<LargeNegativeOffset> Test;
    static constexpr PosT Expected[]{-36729, -36729 + 64};

    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.empty() == false);
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);

    EXPECT_TRUE(ce_equal(Test, Expected));
    EXPECT_TRUE(ce_requal(Test, Expected));
  }
}

TYPED_TEST(MetaSetTest, ContainsCompiletime) {
  EXPOSE_TYPE(PosT);
  EXPOSE_TYPE(E);
  EXPOSE_TYPE(Builder);

  {
    using BS = MakeMetaSet<Builder, PosT(1), PosT(2), PosT(3), PosT(4)>;
    static_assert(BS::contains(1));
    static_assert(BS::contains(2));
    static_assert(BS::contains(3));
    static_assert(BS::contains(4));

    static_assert(!BS::contains(0));
    static_assert(!BS::contains(-1));
    static_assert(!BS::contains(1000000));
  }

  {
    using BS = MakeMetaSet<Builder, E::A, E::C, E::D>;
    static_assert(BS::contains(E::A));
    static_assert(BS::contains(0));
    static_assert(BS::contains(E::C));
    static_assert(BS::contains(2));
    static_assert(BS::contains(E::D));
    static_assert(BS::contains(3));

    static_assert(!BS::contains(E::B));
    static_assert(!BS::contains(1));
    static_assert(!BS::contains(E::E));
    static_assert(!BS::contains(4));
  }

  if constexpr (std::is_signed_v<PosT>) {
    using BS = MakeMetaSet<Builder, 0, -126, 127>;

    if constexpr (Builder::IsBitset) {
      static_assert(
          std::is_same_v<BS, MetaBitset<PosT, -126, 1ULL, (1ULL << 62), 0ULL,
                                        (1ULL << 61)>>);
    }

    static_assert(BS::contains(0));
    static_assert(BS::contains(-126));
    static_assert(BS::contains(127));

    static_assert(!BS::contains(-127));
    static_assert(!BS::contains(128));
    static_assert(!BS::contains(-125));
    static_assert(!BS::contains(126));
    static_assert(!BS::contains(-1));
    static_assert(!BS::contains(1));
  }
}

TYPED_TEST(MetaSetTest, ContainsRuntime) {
  EXPOSE_TYPE(PosT);
  EXPOSE_TYPE(E);
  EXPOSE_TYPE(Builder);

  {
    using BS = MakeMetaSet<Builder, PosT(1), PosT(2), PosT(3), PosT(4)>;
    EXPECT_TRUE(BS::contains(1));
    EXPECT_TRUE(BS::contains(2));
    EXPECT_TRUE(BS::contains(3));
    EXPECT_TRUE(BS::contains(4));

    EXPECT_FALSE(BS::contains(0));
    EXPECT_FALSE(BS::contains(-1));
    EXPECT_FALSE(BS::contains(1000000));
  }

  {
    using BS = MakeMetaSet<Builder, E::A, E::C, E::D>;
    EXPECT_TRUE(BS::contains(E::A));
    EXPECT_TRUE(BS::contains(0));
    EXPECT_TRUE(BS::contains(E::C));
    EXPECT_TRUE(BS::contains(2));
    EXPECT_TRUE(BS::contains(E::D));
    EXPECT_TRUE(BS::contains(3));

    EXPECT_FALSE(BS::contains(E::B));
    EXPECT_FALSE(BS::contains(1));
    EXPECT_FALSE(BS::contains(E::E));
    EXPECT_FALSE(BS::contains(4));
  }

  if constexpr (std::is_signed_v<PosT>) {
    using BS = MakeMetaSet<Builder, 0, -126, 127>;
    EXPECT_TRUE(BS::contains(0));
    EXPECT_TRUE(BS::contains(-126));
    EXPECT_TRUE(BS::contains(127));

    EXPECT_FALSE(BS::contains(-127));
    EXPECT_FALSE(BS::contains(128));
    EXPECT_FALSE(BS::contains(-125));
    EXPECT_FALSE(BS::contains(126));
    EXPECT_FALSE(BS::contains(-1));
    EXPECT_FALSE(BS::contains(1));
  }
}

TYPED_TEST(MetaSetTest, Union) {
  EXPOSE_TYPE(PosT);
  EXPOSE_TYPE(Builder);
  EXPOSE_TYPE(ThisT);

  // Identity
  {
    using S = MakeMetaSet<Builder, 5, 10, 15, 65, 70, 75>;

    using Union = MetaSetUnion<S, S, 64>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 64, 5, 10, 15, 65, 70, 75>;

    static_assert(std::is_same_v<Union, ExpectedT>);
  }

  // Identity (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using S = MakeMetaSet<Builder, -75, -70, -65, 5, 10>;

    using Union = MetaSetUnion<S, S, 64>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 64, -75, -70, -65, 5, 10>;

    static_assert(std::is_same_v<Union, ExpectedT>);
  }

  // Disjoint
  {
    using L = MakeMetaSet<Builder, 1, 3, 5, 7, 9>;
    using R = MakeMetaSet<Builder, 2, 4, 6, 8, 10>;

    using Union = MetaSetUnion<L, R, 512>;

    using ExpectedT =
        MakeMetaSparseBitset<PosT, 512, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>;

    static_assert(std::is_same_v<Union, ExpectedT>);
  }

  // Disjoint (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using L = MakeMetaSet<Builder, -5, -3, -1, 1, 3, 5>;
    using R = MakeMetaSet<Builder, -4, -2, 0, 2, 4>;

    using Union = MetaSetUnion<L, R, 512>;

    using ExpectedT =
        MakeMetaSparseBitset<PosT, 512, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5>;

    static_assert(std::is_same_v<Union, ExpectedT>);
  }

  // Intersecting.
  {
    using L = MakeMetaSet<Builder, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>;
    using R = MakeMetaSet<Builder, 2, 4, 6, 8, 10>;

    using Union = MetaSetUnion<L, R, 512>;

    using ExpectedT =
        MakeMetaSparseBitset<PosT, 512, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>;

    static_assert(std::is_same_v<Union, ExpectedT>);
  }

  // Intersecting (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using L = MakeMetaSet<Builder, -6, -4, -2, 0, 2, 4, 6>;
    using R = MakeMetaSet<Builder, -4, -3, -2, -1, 0, 1, 2, 3, 4>;

    using Union = MetaSetUnion<L, R, 512>;

    using ExpectedT =
        MakeMetaSparseBitset<PosT, 512, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6>;

    static_assert(std::is_same_v<Union, ExpectedT>);
  }

  // Big gap.
  if constexpr (ThisT::max() >= 10000) {
    using Singletons = MakeMetaSet<Builder, 10000, 5000, 8000>;
    using SmallVals = MakeMetaSet<Builder, 10, 20, 30, 40, 50, 60>;

    using Union = MetaSetUnion<Singletons, SmallVals>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512, 10, 20, 30, 40, 50, 60,
                                           5000, 8000, 10000>;
    static_assert(std::is_same_v<Union, ExpectedT>);
  }
}

TYPED_TEST(MetaSetTest, Intersect) {
  EXPOSE_TYPE(PosT);
  EXPOSE_TYPE(Builder);
  EXPOSE_TYPE(ThisT);

  // Identity
  {
    using S = MakeMetaSet<Builder, 5, 10, 15, 65, 70, 75>;

    using Intersection = MetaSetIntersection<S, S, 64>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 64, 5, 10, 15, 65, 70, 75>;

    static_assert(std::is_same_v<Intersection, ExpectedT>);
  }

  // Identity (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using S = MakeMetaSet<Builder, -75, -70, -65, 5, 10>;

    using Intersection = MetaSetIntersection<S, S, 64>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 64, -75, -70, -65, 5, 10>;

    static_assert(std::is_same_v<Intersection, ExpectedT>);
  }

  // Disjoint
  {
    using L = MakeMetaSet<Builder, 1, 3, 5, 7, 9>;
    using R = MakeMetaSet<Builder, 2, 4, 6, 8, 10>;

    using Intersection = MetaSetIntersection<L, R, 512>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512>;

    static_assert(std::is_same_v<Intersection, ExpectedT>);
  }

  // Disjoint (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using L = MakeMetaSet<Builder, -5, -3, -1, 1, 3, 5>;
    using R = MakeMetaSet<Builder, -4, -2, 0, 2, 4>;

    using Intersection = MetaSetIntersection<L, R, 512>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512>;

    static_assert(std::is_same_v<Intersection, ExpectedT>);
  }

  // Intersecting.
  {
    using L = MakeMetaSet<Builder, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>;
    using R = MakeMetaSet<Builder, 2, 4, 6, 8, 10>;

    using Intersection = MetaSetIntersection<L, R, 512>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512, 2, 4, 6, 8, 10>;

    static_assert(std::is_same_v<Intersection, ExpectedT>);
  }

  // Intersecting (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using L = MakeMetaSet<Builder, -6, -4, -2, 0, 2, 4, 6>;
    using R = MakeMetaSet<Builder, -4, -3, -2, -1, 0, 1, 2, 3, 4>;

    using Intersection = MetaSetIntersection<L, R, 512>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512, -4, -2, 0, 2, 4>;

    static_assert(std::is_same_v<Intersection, ExpectedT>);
  }

  // Big gap.
  if constexpr (ThisT::max() >= 10000) {
    using Singletons = MakeMetaSet<Builder, 10000, 5000, 8000>;
    using SmallVals = MakeMetaSet<Builder, 10, 20, 30, 40, 50, 60>;

    using Intersection = MetaSetIntersection<Singletons, SmallVals>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512>;
    static_assert(std::is_same_v<Intersection, ExpectedT>);
  }
}

TYPED_TEST(MetaSetTest, Minus) {
  EXPOSE_TYPE(PosT);
  EXPOSE_TYPE(Builder);
  EXPOSE_TYPE(ThisT);

  // Identity
  {
    using S = MakeMetaSet<Builder, 5, 10, 15, 65, 70, 75>;

    using Minus = MetaSetMinus<S, S, 64>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 64>;

    static_assert(std::is_same_v<Minus, ExpectedT>);
  }

  // Identity (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using S = MakeMetaSet<Builder, -75, -70, -65, 5, 10>;

    using Minus = MetaSetMinus<S, S, 64>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 64>;

    static_assert(std::is_same_v<Minus, ExpectedT>);
  }

  // Disjoint
  {
    using L = MakeMetaSet<Builder, 1, 3, 5, 7, 9>;
    using R = MakeMetaSet<Builder, 2, 4, 6, 8, 10>;

    using Minus = MetaSetMinus<L, R, 512>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512, 1, 3, 5, 7, 9>;

    static_assert(std::is_same_v<Minus, ExpectedT>);
  }

  // Disjoint (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using L = MakeMetaSet<Builder, -5, -3, -1, 1, 3, 5>;
    using R = MakeMetaSet<Builder, -4, -2, 0, 2, 4>;

    using Minus = MetaSetMinus<L, R, 512>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512, -5, -3, -1, 1, 3, 5>;

    static_assert(std::is_same_v<Minus, ExpectedT>);
  }

  // Intersecting.
  {
    using L = MakeMetaSet<Builder, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>;
    using R = MakeMetaSet<Builder, 2, 4, 6, 8, 10>;

    using Minus = MetaSetMinus<L, R, 512>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512, 1, 3, 5, 7, 9>;

    static_assert(std::is_same_v<Minus, ExpectedT>);
  }

  // Intersecting (signed)
  if constexpr (std::is_signed_v<PosT>) {
    using L = MakeMetaSet<Builder, -6, -4, -2, 0, 2, 4, 6>;
    using R = MakeMetaSet<Builder, -4, -3, -2, -1, 0, 1, 2, 3, 4>;

    using Minus = MetaSetMinus<L, R, 512>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512, -6, 6>;

    static_assert(std::is_same_v<Minus, ExpectedT>);
  }

  // Big gap.
  if constexpr (ThisT::max() >= 10000) {
    using Singletons = MakeMetaSet<Builder, 10000, 5000, 8000>;
    using SmallVals = MakeMetaSet<Builder, 10, 20, 30, 40, 50, 60>;

    using Minus = MetaSetMinus<Singletons, SmallVals>;

    using ExpectedT = MakeMetaSparseBitset<PosT, 512, 10000, 5000, 8000>;
    static_assert(std::is_same_v<Minus, ExpectedT>);
  }
}

} // namespace
