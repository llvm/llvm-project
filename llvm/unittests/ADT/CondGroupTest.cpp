//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains testing for the condition group infrastructure.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/CondGroup.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <optional>
#include <type_traits>

using namespace llvm;
using namespace cgrp;

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

namespace cexpr {

template <int Val, bool = (Val == anyOf(1, 2, 3))>
class OneTwoThreeSimple : public std::false_type {};

template <int Val>
class OneTwoThreeSimple<Val, true> : public std::true_type {};

template <int Val, bool = (Val == anyOf(makeGroup(1, 2), makeGroup(3)))>
class OneTwoThreeNested : public std::false_type {};

template <int Val>
class OneTwoThreeNested<Val, true> : public std::true_type {};
} // namespace cexpr

enum class Number : int {
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
  Ten
};

enum class Letter : int { A, B, C, D, E, F, G, H, I, J };

enum LetterNonClass : int { A, B, C, D, E, F, G, H, I, J };

enum NumberNonClass : int {
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
  Ten
};

TEST(CondGroupTest, ConstExpr) {
  using namespace cexpr;

  // Ensure anyOf() can be called in the context of a `static_assert`
  // condition.
  static_assert(1 == anyOf(1, 2, 3), "1 is in group {1,2,3}");
  static_assert(4 != anyOf(1, 2, 3), "4 is not in group {1,2,3}");

  static_assert(1 == anyOf(makeGroup(1, 2, 3)), "1 is in group {1,2,3}");
  static_assert(9 != anyOf(makeGroup(1, 2, 3)), "9 is not in group {1,2,3}");

  static_assert(1 == anyOf(makeGroup(1, 2, 3)), "1 is in group {1,2,3}");
  static_assert(9 != anyOf(makeGroup(1, 2, 3)), "9 is not in group {1,2,3}");

  // Ensure that we can instantiate a template which calls anyOf() to derive
  // the value of an anonymous template parameter.
  static_assert(OneTwoThreeSimple<0>::value == false,
                "0 is not in group {1,2,3}");
  static_assert(OneTwoThreeNested<0>::value == false,
                "0 is not in group {1,2,3}");
  static_assert(OneTwoThreeSimple<2>::value == true, "2 is in group {1,2,3}");
  static_assert(OneTwoThreeNested<1>::value == true, "1 is in group {1,2,3}");
}

TEST(CondGroupTest, LiteralRepresentation) {
  // Single MetaBitset representation.
  {
    static constexpr auto Fives =
        cgrp::Literals<5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 10, 20, 30, 40,
                       50, 60, 70, 80, 90, 100>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Fives)>,
                       CondGroupMetaSet<MetaBitset<int, 5, 0x1084210842108421,
                                                   0x84210842>>>);
    static_assert(35 == anyOf(Fives));
  }

  // Multiple MetaBitsets for clusters (no singletons).
  {
    static constexpr auto Clusters =
        cgrp::Literals<10000, 10002, 10004, 10006, 10008, 1000, 1002, 1004,
                       1006, 1008>;

    static_assert(
        std::is_same_v<
            std::remove_cv_t<decltype(Clusters)>,
            CondGroupMetaSet<MetaSparseBitset<MetaBitset<int, 1000, 0x155>,
                                              MetaBitset<int, 10000, 0x155>>>>);
  }

  // Multiple MetaBitsets for clusters + MetaSequneceSet for singletons.
  {
    static constexpr auto ClustersAndSingletons =
        cgrp::Literals<10000, 10002, 10004, 10006, 10008, 1000, 1002, 1004,
                       1006, 1008, 5000, 8000>;

    static_assert(
        std::is_same_v<
            std::remove_cv_t<decltype(ClustersAndSingletons)>,
            CondGroupMetaSet<MetaSparseBitset<
                MetaBitset<int, 1000, 0x155>, MetaBitset<int, 10000, 0x155>,
                MakeMetaSequenceSet<int, 5000, 8000>>>>);
  }

  // No clusters, all singletons; MetaSequenceSet for all values.
  {
    static constexpr auto Singletons =
        cgrp::Literals<10000, 5000, 8000, 20000, 90000>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Singletons)>,
                       CondGroupMetaSet<MetaSparseBitset<MakeMetaSequenceSet<
                           int, 5000, 8000, 10000, 20000, 90000>>>>);
  }

  // Small set, no paritioning attempted; MetaSequenceSet with no sorting.
  {
    static constexpr auto SmallSet = cgrp::Literals<3, 1>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(SmallSet)>,
                       CondGroupMetaSet<
                           MetaSequenceSet<std::integer_sequence<int, 3, 1>>>>);
  }
}

TEST(CondGroupTest, Integers) {
  static constexpr auto Evens = makeGroup(2, 4, 6, 8, 10);
  static constexpr auto Odds = makeGroup(1, 3, 5, 7, 9);

  std::decay_t<decltype(Evens)> EvensC(Evens);

  static constexpr auto EvensL = Literals<2, 4, 6, 8, 10>;
  static_assert(10 == anyOf(EvensL));

  static constexpr auto LargeGap = Literals<128, 129, 131, 381, 382, 383>;

  static_assert(128 == anyOf(LargeGap));
  static_assert(129 == anyOf(LargeGap));
  static_assert(130 != anyOf(LargeGap));
  static_assert(383 == anyOf(LargeGap));
  static_assert(191 != anyOf(LargeGap));
  static_assert(192 != anyOf(LargeGap));
  static_assert(193 != anyOf(LargeGap));
  static_assert(255 != anyOf(LargeGap));
  static_assert(256 != anyOf(LargeGap));
  static_assert(257 != anyOf(LargeGap));

  // Empty group.
  EXPECT_TRUE(0 != anyOf());
  EXPECT_TRUE(anyOf() != 0);

  // Mixed group + value compare.
  EXPECT_TRUE(1 == anyOf(EvensC, 1));
  EXPECT_TRUE(anyOf(Evens, 1) == 1);

  // Multiple groups.
  EXPECT_TRUE(1 == anyOf(Evens, Odds));
  EXPECT_TRUE(anyOf(Evens, Odds) == 1);

  // Negative comparison.
  EXPECT_TRUE(0 != anyOf(1, 2, 3, 4, 5, 6, 7, 8, 9));
  EXPECT_TRUE(anyOf(1, 2, 3, 4, 5, 6, 7, 8, 9) != 0);

  // Negative comparison of multiple groups.
  EXPECT_TRUE(0 != anyOf(Evens, Odds));
  EXPECT_TRUE(anyOf(Evens, Odds) != 0);
}

TEST(CondGroupTest, OptionalIntegers) {
  using Opt = std::optional<int>;

  auto Evens = makeGroup(Opt(), Opt(2), Opt(4), Opt(6), Opt(8), Opt(10));

  auto Odds = makeGroup(Opt(), Opt(1), Opt(3), Opt(5), Opt(7), Opt(9));

  auto OddsNoNullopt = makeGroup(Opt(1), Opt(3), Opt(5), Opt(7), Opt(9));

  // Empty group.
  EXPECT_TRUE(Opt(0) != anyOf());
  EXPECT_TRUE(anyOf() != Opt(0));

  // Mixed group + value compare.
  EXPECT_TRUE(1 == anyOf(Evens, 1));
  EXPECT_TRUE(anyOf(Evens, 1) == 1);

  // Multiple groups.
  EXPECT_TRUE(1 == anyOf(Evens, Odds));
  EXPECT_TRUE(anyOf(Evens, Odds) == 1);

  // Negative comparison.
  EXPECT_TRUE(0 != anyOf(1, 2, 3, 4, 5, 6, 7, 8, 9));
  EXPECT_TRUE(anyOf(1, 2, 3, 4, 5, 6, 7, 8, 9) != 0);

  // Negative comparison of multiple groups.
  EXPECT_TRUE(0 != anyOf(Evens, Odds));
  EXPECT_TRUE(anyOf(Evens, Odds) != 0);

  // std::nullopt should match the default-constructed Opt() in the Evens group.
  EXPECT_TRUE(std::nullopt == anyOf(Evens));

  // Should be able to compare an Optional to a group containing a std::nullopt.
  EXPECT_TRUE(Opt(2) == anyOf(std::nullopt, 2));
  EXPECT_TRUE(anyOf(std::nullopt, 2) == Opt(2));

  EXPECT_FALSE(std::nullopt == anyOf(OddsNoNullopt));
  EXPECT_FALSE(anyOf(OddsNoNullopt) == std::nullopt);

  EXPECT_TRUE(std::nullopt != anyOf(OddsNoNullopt));
  EXPECT_TRUE(anyOf(OddsNoNullopt) != std::nullopt);
}

TEST(CondGroupTest, LiteralsTuple) {
  EXPECT_TRUE(1268 == (AnyOf<1, 5, 9, 1600, 905, 1268, 2402, 5732>));
  EXPECT_TRUE(1268 == (anyOf(Literals<1, 5, 9, 1600, 905, 1268, 2402, 5732>)));

  static constexpr auto Group = Literals<1, 5, 9, 1600, 905, 1268, 2402, 5732>;
  EXPECT_TRUE(1268 == anyOf(Group));
  EXPECT_FALSE(12 == anyOf(Group));

  {
    constexpr auto G =
        Literals<9971U, 9972U, 9973U, 9974U, 9975U, 9976U, 9977U, 9978U, 9985U,
                 9986U, 9987U, 9988U, 9989U, 9990U, 9991U, 9992U, 9993U, 9994U,
                 9995U, 10002U, 10003U, 10004U, 10005U, 10006U, 10007U, 10008U,
                 10009U, 10010U, 10011U, 10012U, 10013U, 9133U, 9134U>;

    static_assert(10009U == anyOf(G));
  }
}

namespace implicit_conversion {

struct Foo {
  // Support implicit conversions from int, unsigned or short.
  constexpr Foo(int V) : Val(V) {}
  constexpr Foo(unsigned V) : Val(static_cast<int>(V)) {}
  constexpr Foo(short V) : Val(static_cast<int>(V)) {}

  constexpr operator int() const { return Val; }

  int Val;
};

} // namespace implicit_conversion

TEST(CondGroupTest, ImplicitConversions) {
  using namespace implicit_conversion;

  static constexpr auto Evens =
      makeGroup(Foo(2), Foo(4U), Foo(short(6)), 8, 10);
  static constexpr auto Odds =
      makeGroup(Foo(1), Foo(3), Foo(5U), Foo(short(7)), 9);

  auto NegGroup =
      makeGroup(1, Foo(short(2)), 3, Foo(4U), short(5), Foo(6), 7, Foo(8), 9);

  //
  // Single value is `Foo` type.
  //

  // Empty group.
  EXPECT_TRUE(Foo(0) != anyOf());
  EXPECT_TRUE(anyOf() != Foo(0));

  // Mixed group + value compare.
  EXPECT_TRUE(Foo(1) == anyOf(Evens, 1));
  EXPECT_TRUE(anyOf(Evens, 1) == Foo(1));

  // Multiple groups.
  EXPECT_TRUE(Foo(1) == anyOf(Evens, Odds));
  EXPECT_TRUE(anyOf(Evens, Odds) == Foo(1));

  // Negative comparison.
  EXPECT_TRUE(Foo(0) != anyOf(NegGroup));
  EXPECT_TRUE(anyOf(NegGroup) != Foo(0));

  // Negative comparison of multiple groups.
  EXPECT_TRUE(Foo(0) != anyOf(Evens, Odds));
  EXPECT_TRUE(anyOf(Evens, Odds) != Foo(0));

  //
  // Single value is integral type.
  //

  // Empty group.
  EXPECT_TRUE(0 != anyOf());
  EXPECT_TRUE(anyOf() != 0);

  // Mixed group + value compare.
  EXPECT_TRUE(1 == anyOf(Evens, 1));
  EXPECT_TRUE(anyOf(Evens, 1) == 1);

  // Multiple groups.
  EXPECT_TRUE(1 == anyOf(Evens, Odds));
  EXPECT_TRUE(anyOf(Evens, Odds) == 1);

  // Negative comparison.
  EXPECT_TRUE(0 != anyOf(NegGroup));
  EXPECT_TRUE(anyOf(NegGroup) != 0);

  // Negative comparison of multiple groups.
  EXPECT_TRUE(0 != anyOf(Evens, Odds));
  EXPECT_TRUE(anyOf(Evens, Odds) != 0);
}

TEST(CondGroupTest, EnumLiterals) {
  using namespace implicit_conversion;

  //
  // Sequence set test cases.
  //

  // All enum group ; non-class (sequence).
  {
    static constexpr auto Letters = Literals<A, B, C>;

    static_assert(0 == anyOf(Letters));
    static_assert(A == anyOf(Letters));
    static_assert(1u == anyOf(Letters));
    static_assert(B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(Letter::D != anyOf(Letters));
    static_assert(E != anyOf(Letters));

    using ExpectedT = CondGroupMetaSet<MakeMetaSequenceSet<int, A, B, C>>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  // All enum group ; class (sequence).
  {
    static constexpr auto Letters = Literals<Letter::A, Letter::B, Letter::C>;

    static_assert(short(0) == anyOf(Letters));
    static_assert(Letter::A == anyOf(Letters));
    static_assert(1ul == anyOf(Letters));
    static_assert(Letter::B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(Letter::D != anyOf(Letters));
    static_assert(Letter::E != anyOf(Letters));

    using ExpectedT = CondGroupMetaSet<
        MakeMetaSequenceSet<int, Letter::A, Letter::B, Letter::C>, Letter>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  // Mixed enum + integral group ; non-class (sequence).
  {
    static constexpr auto Letters = Literals<0, B, Two>;

    static_assert(0ull == anyOf(Letters));
    static_assert(A == anyOf(Letters));
    static_assert(1 == anyOf(Letters));
    static_assert(B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(D != anyOf(Letters));
    static_assert(E != anyOf(Letters));

    using ExpectedT = CondGroupMetaSet<MakeMetaSequenceSet<int, 0, B, 2>>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  // Mixed group ; enum class (sequence).
  {
    static constexpr auto Letters = Literals<A, char(1), Letter::C>;

    static_assert(0 == anyOf(Letters));
    static_assert(Letter::A == anyOf(Letters));
    static_assert(1 == anyOf(Letters));
    static_assert(Letter::B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(C == anyOf(Letters));
    static_assert(Letter::D != anyOf(Letters));

    using ExpectedT =
        CondGroupMetaSet<MakeMetaSequenceSet<int, A, char(1), Letter::C>,
                         Letter>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  //
  // Bitset test cases.
  //

  // All enum group ; non-class (bitset).
  {
    static constexpr auto Letters = Literals<A, B, C, D>;

    static_assert(0 == anyOf(Letters));
    static_assert(A == anyOf(Letters));
    static_assert(1u == anyOf(Letters));
    static_assert(B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(D == anyOf(Letters));
    static_assert(E != anyOf(Letters));

    using ExpectedT = CondGroupMetaSet<MakeMetaBitset<int, A, B, C, D>>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  // All enum group ; class (bitset).
  {
    static constexpr auto Letters =
        Literals<Letter::A, Letter::B, Letter::C, Letter::D>;

    static_assert(short(0) == anyOf(Letters));
    static_assert(Letter::A == anyOf(Letters));
    static_assert(1ul == anyOf(Letters));
    static_assert(Letter::B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(Letter::D == anyOf(Letters));
    static_assert(Letter::E != anyOf(Letters));

    using ExpectedT = CondGroupMetaSet<
        MakeMetaBitset<int, Letter::A, Letter::B, Letter::C, Letter::D>,
        Letter>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  // Mixed enum + integral group ; non-class (bitset).
  {
    static constexpr auto Letters = Literals<0, B, 2, D>;

    static_assert(0ull == anyOf(Letters));
    static_assert(A == anyOf(Letters));
    static_assert(1 == anyOf(Letters));
    static_assert(B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(D == anyOf(Letters));
    static_assert(E != anyOf(Letters));

    using ExpectedT = CondGroupMetaSet<MakeMetaBitset<int, 0, B, 2, D>>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  // Mixed group ; enum class (bitset).
  {
    static constexpr auto Letters = Literals<0, char(1), Letter::C, Letter::D>;

    static_assert(0 == anyOf(Letters));
    static_assert(Letter::A == anyOf(Letters));
    static_assert(1 == anyOf(Letters));
    static_assert(Letter::B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(Letter::D == anyOf(Letters));
    static_assert(Letter::E != anyOf(Letters));

    using ExpectedT =
        CondGroupMetaSet<MakeMetaBitset<int, 0, char(1), Letter::C, Letter::D>,
                         Letter>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  //
  // Sparse bitset test cases.
  //

  // Mixed enum + integral group ; non-class (sparse bitset).
  {
    static constexpr auto Letters = Literals<0, B, 2, D, 1000>;

    static_assert(0ull == anyOf(Letters));
    static_assert(A == anyOf(Letters));
    static_assert(1 == anyOf(Letters));
    static_assert(B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(D == anyOf(Letters));
    static_assert(E != anyOf(Letters));
    static_assert(1000 == anyOf(Letters));

    using ExpectedT =
        CondGroupMetaSet<MetaSparseBitset<MakeMetaBitset<int, 0, B, 2, D>,
                                          MakeMetaSequenceSet<int, 1000>>>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

  // Mixed group ; enum class (sparse bitset).
  {
    static constexpr auto Letters =
        Literals<0, char(1), Letter::C, Letter::D, 5000>;

    static_assert(0 == anyOf(Letters));
    static_assert(Letter::A == anyOf(Letters));
    static_assert(1 == anyOf(Letters));
    static_assert(Letter::B == anyOf(Letters));
    static_assert(Foo(C) == anyOf(Letters));
    static_assert(Letter::D == anyOf(Letters));
    static_assert(Letter::E != anyOf(Letters));
    static_assert(4999 != anyOf(Letters));
    static_assert(5000 == anyOf(Letters));
    static_assert(5001 != anyOf(Letters));
    static_assert(Foo(1) == anyOf(Letters));

    using ExpectedT = CondGroupMetaSet<
        MetaSparseBitset<MakeMetaBitset<int, 0, char(1), Letter::C, Letter::D>,
                         MakeMetaSequenceSet<int, 5000>>,
        Letter>;

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Letters)>, ExpectedT>);
  }

#if 0 // compile errors
  {
    // Mixing signedness.
    [[maybe_unused]] static constexpr auto MixSignedness =
        Literals<1, 2u, 3, 4u>;
  }

  {
    // Multiple enum types: class + class
    [[maybe_unused]] static constexpr auto MalformedLetters =
        Literals<Number::Zero, Letter::B, Letter::C, Letter::D>;
  }

  {
    // enum class group.
    static constexpr auto Letters =
        Literals<Letter::A, Letter::B, Letter::C, Letter::D>;

    // Incompatible comparisons:
    EXPECT_TRUE(Number::One == anyOf(Letters));
  }
#endif
}

TEST(CondGroupTest, AllEmptyTuple) {
  auto Evens = Literals<2, 4, 6, 8, 10>;
  auto Odds = Literals<1, 3, 5, 7, 9>;

  static_assert(std::is_empty_v<decltype(Evens)> &&
                std::is_empty_v<decltype(Odds)>);

  // Create a tuple-representation group of two empty classes.
  auto G = makeGroup(Evens, Odds);
  EXPECT_TRUE(1 == anyOf(G));
}

TEST(CondGroupTest, LiteralsContainer) {

  // Empty set.
  {
    static constexpr auto Test = Literals<>;

    static constexpr std::array<int, 0> Expected{};

    static_assert(Test.empty() == true);
    static_assert(Test.size() == Expected.size());
    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }

  // Single value set.
  {
    static constexpr auto Test = Literals<5000>;

    static constexpr int Expected[]{5000};

    static_assert(Test.empty() == false);
    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[0] == Expected[0]);
    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }

  // Single bitset.
  {
    static constexpr auto Test = Literals<5, 10, 15, 20, 25>;

    static constexpr int Expected[]{5, 10, 15, 20, 25};

    static_assert(Test.empty() == false);
    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);
    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }

  // Single sequence.
  {
    static constexpr auto Test = Literals<0, 10000, 50000, 100000>;

    static constexpr int Expected[]{0, 10000, 50000, 100000};

    static_assert(Test.empty() == false);
    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);
    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }

  // Multiple clusters.
  {
    static constexpr auto Test =
        Literals<1005, 1004, 1003, 1002, 1001, 1000, 5, 4, 3, 2, 1, 0>;

    static constexpr int Expected[]{0,    1,    2,    3,    4,    5,
                                    1000, 1001, 1002, 1003, 1004, 1005};

    static_assert(Test.empty() == false);
    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);
    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }

  // Cluster + sequence.
  {
    static constexpr auto Test = Literals<1004, 1005, 5, 4, 3, 2, 1, 0>;

    static constexpr int Expected[]{0, 1, 2, 3, 4, 5, 1004, 1005};

    static_assert(Test.empty() == false);
    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);
    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }

  // Duplicate values (raw sequence).
  {
    static constexpr auto Test = Literals<3, 2, 3>;

    static constexpr int Expected[]{2, 3, 3};

    static_assert(Test.empty() == false);
    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);
    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }

  // Duplicate values (bitset representation).
  {
    static constexpr auto Test = Literals<3, 2, 3, 11, 6>;

    static constexpr int Expected[]{2, 3, 6, 11};

    static_assert(Test.empty() == false);
    static_assert(Test.size() == std::size(Expected));
    static_assert(Test.front() == Expected[0]);
    static_assert(Test.back() == Expected[std::size(Expected) - 1]);
    static_assert(Test[1] == Expected[1]);
    static_assert(ce_equal(Test, Expected));
    static_assert(ce_requal(Test, Expected));
  }
}

TEST(CondGroupTest, LiteralsUnion) {

  // Identity
  {
    constexpr auto Group = cgrp::Literals<5, 10, 15, 65, 70, 75>;

    static constexpr int Expected[]{5, 10, 15, 65, 70, 75};

    static_assert(ce_equal(Group | Group, Expected));
  }

  // Identity (signed)
  {
    constexpr auto Group = cgrp::Literals<75, -70, -65, 5, 10, 64>;

    static constexpr int Expected[]{-70, -65, 5, 10, 64, 75};

    static_assert(ce_equal(Group | Group, Expected));
  }

  // Disjoint
  {
    constexpr auto L = cgrp::Literals<1, 3, 5, 7, 9>;
    constexpr auto R = cgrp::Literals<2, 4, 6, 8, 10>;

    static constexpr int Expected[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    static_assert(ce_equal(L | R, Expected));
  }

  // Disjoint (signed)
  {
    constexpr auto L = cgrp::Literals<-5, -3, -1, 1, 3, 5>;
    constexpr auto R = cgrp::Literals<-4, -2, 0, 2, 4>;

    static constexpr int Expected[]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};

    static_assert(ce_equal(L | R, Expected));
  }

  // Intersecting.
  {
    constexpr auto L = cgrp::Literals<1, 2, 3, 4, 5, 6, 7, 8, 9, 10>;
    constexpr auto R = cgrp::Literals<2, 4, 6, 8, 10>;

    static constexpr int Expected[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    static_assert(ce_equal(L | R, Expected));
  }

  // Intersecting (signed)
  {
    constexpr auto L = cgrp::Literals<-6, -4, -2, 0, 2, 4, 6>;
    constexpr auto R = cgrp::Literals<-4, -3, -2, -1, 0, 1, 2, 3, 4>;

    static constexpr int Expected[]{-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6};

    static_assert(ce_equal(L | R, Expected));
  }

  // Big gap.
  {
    constexpr auto L = cgrp::Literals<10000, 5000, 8000>;
    constexpr auto R = cgrp::Literals<10, 20, 30, 40, 50, 60>;

    static constexpr int Expected[]{10, 20, 30, 40, 50, 60, 5000, 8000, 10000};

    static_assert(ce_equal(L | R, Expected));
  }
}

TEST(CondGroupTest, LiteralsIntersection) {

  // Identity
  {
    constexpr auto Group = cgrp::Literals<5, 10, 15, 65, 70, 75>;

    static constexpr int Expected[]{5, 10, 15, 65, 70, 75};

    static_assert(ce_equal(Group & Group, Expected));
  }

  // Identity (signed)
  {
    constexpr auto Group = cgrp::Literals<75, -70, -65, 5, 10, 64>;

    static constexpr int Expected[]{-70, -65, 5, 10, 64, 75};

    static_assert(ce_equal(Group & Group, Expected));
  }

  // Disjoint
  {
    constexpr auto L = cgrp::Literals<1, 3, 5, 7, 9>;
    constexpr auto R = cgrp::Literals<2, 4, 6, 8, 10>;

    static constexpr std::array<int, 0> Expected{};

    static_assert(ce_equal(L & R, Expected));
  }

  // Disjoint (signed)
  {
    constexpr auto L = cgrp::Literals<-5, -3, -1, 1, 3, 5>;
    constexpr auto R = cgrp::Literals<-4, -2, 0, 2, 4>;

    static constexpr std::array<int, 0> Expected{};

    static_assert(ce_equal(L & R, Expected));
  }

  // Intersecting.
  {
    constexpr auto L = cgrp::Literals<1, 2, 3, 4, 5, 6, 7, 8, 9, 10>;
    constexpr auto R = cgrp::Literals<2, 4, 6, 8, 10>;

    static constexpr int Expected[]{2, 4, 6, 8, 10};

    static_assert(ce_equal(L & R, Expected));
  }

  // Intersecting (signed)
  {
    constexpr auto L = cgrp::Literals<-6, -4, -2, 0, 2, 4, 6>;
    constexpr auto R = cgrp::Literals<-4, -3, -2, -1, 0, 1, 2, 3, 4>;

    static constexpr int Expected[]{-4, -2, 0, 2, 4};

    static_assert(ce_equal(L & R, Expected));
  }

  // Big gap.
  {
    constexpr auto L = cgrp::Literals<10000, 5000, 8000>;
    constexpr auto R = cgrp::Literals<10, 20, 30, 40, 50, 60>;

    static constexpr std::array<int, 0> Expected{};

    static_assert(ce_equal(L & R, Expected));
  }
}

TEST(CondGroupTest, LiteralsMinus) {

  // Same operands
  {
    constexpr auto Group = cgrp::Literals<5, 10, 15, 65, 70, 75>;

    static constexpr std::array<int, 0> Expected{};

    static_assert(ce_equal(Group - Group, Expected));
  }

  // Same operands (signed)
  {
    constexpr auto Group = cgrp::Literals<75, -70, -65, 5, 10, 64>;

    static constexpr std::array<int, 0> Expected{};

    static_assert(ce_equal(Group - Group, Expected));
  }

  // Disjoint
  {
    constexpr auto L = cgrp::Literals<1, 3, 5, 7, 9>;
    constexpr auto R = cgrp::Literals<2, 4, 6, 8, 10>;

    static constexpr int Expected[]{1, 3, 5, 7, 9};

    static_assert(ce_equal(L - R, Expected));
  }

  // Disjoint (signed)
  {
    constexpr auto L = cgrp::Literals<-5, -3, -1, 1, 3, 5>;
    constexpr auto R = cgrp::Literals<-4, -2, 0, 2, 4>;

    static constexpr int Expected[]{-5, -3, -1, 1, 3, 5};

    static_assert(ce_equal(L - R, Expected));
  }

  // Intersecting.
  {
    constexpr auto L = cgrp::Literals<1, 2, 3, 4, 5, 6, 7, 8, 9, 10>;
    constexpr auto R = cgrp::Literals<2, 4, 6, 8, 10>;

    static constexpr int Expected[]{1, 3, 5, 7, 9};

    static_assert(ce_equal(L - R, Expected));
  }

  // Intersecting (signed)
  {
    constexpr auto L = cgrp::Literals<-6, -4, -2, 0, 2, 4, 6>;
    constexpr auto R = cgrp::Literals<-4, -3, -2, -1, 0, 1, 2, 3, 4>;

    static constexpr int Expected[]{-6, 6};

    static_assert(ce_equal(L - R, Expected));
  }

  // Big gap.
  {
    constexpr auto L = cgrp::Literals<10000, 5000, 8000>;
    constexpr auto R = cgrp::Literals<10, 20, 30, 40, 50, 60>;

    static constexpr int Expected[]{5000, 8000, 10000};

    static_assert(ce_equal(L - R, Expected));
  }
}

} // namespace
