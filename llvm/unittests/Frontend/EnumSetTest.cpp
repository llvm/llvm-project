//===- llvm/unittests/Frontend/EnumSetTest.cpp - EnumSet unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <type_traits>

using namespace llvm;
using namespace llvm::omp;

namespace {
namespace detail {
template <typename Elem, Elem...> struct is_one_of {
  constexpr bool operator()(Elem) const { return false; }
};

template <typename Elem, Elem Value, Elem... Values>
struct is_one_of<Elem, Value, Values...> {
  constexpr bool operator()(Elem V) const {
    return V == Value || is_one_of<Elem, Values...>{}(V);
  }
};

template <typename Elem, Elem... Values, typename Range>
constexpr bool ElementsAre(Range &&R) {
  size_t Count = 0;
  // This also serves as an EnumSetIterator test.
  for (auto It = R.begin(), End = R.end(); It != End; ++It) {
    if (!is_one_of<Elem, Values...>{}(*It))
      return false;
    ++Count;
  }
  if (Count != sizeof...(Values))
    return false;
  return true;
}
} // namespace detail

using ClauseSet = EnumSet<Clause, Clause_enumSize>;

TEST(EnumSetTest, DefaultInitialization) {
  constexpr ClauseSet S;
  EXPECT_THAT(S, testing::IsEmpty());
  EXPECT_EQ(S.size(), static_cast<size_t>(0));

  static_assert(S.empty());
  static_assert(S.size() == 0);
}

TEST(EnumSetTest, ListInitialization) {
  constexpr ClauseSet S{Clause::OMPC_private, Clause::OMPC_shared};
  EXPECT_THAT(S, testing::ElementsAre(OMPC_private, OMPC_shared));

  static_assert(
      detail::ElementsAre<Clause, Clause::OMPC_private, Clause::OMPC_shared>(
          S));
}

TEST(EnumSetTest, CopyInitialization) {
  constexpr ClauseSet S(ClauseSet{Clause::OMPC_private, Clause::OMPC_shared});
  EXPECT_THAT(S, testing::ElementsAre(OMPC_private, OMPC_shared));

  static_assert(
      detail::ElementsAre<Clause, Clause::OMPC_private, Clause::OMPC_shared>(
          S));
}

TEST(EnumSetTest, Set) {
  ClauseSet S;
  S.set(Clause::OMPC_private);
  EXPECT_THAT(S, testing::ElementsAre(OMPC_private));

  static_assert(detail::ElementsAre<Clause, Clause::OMPC_private>(
      ClauseSet{}.set(Clause::OMPC_private)));
}

TEST(EnumSetTest, Reset) {
  ClauseSet S{Clause::OMPC_private, Clause::OMPC_shared};
  S.reset(Clause::OMPC_private);
  EXPECT_THAT(S, testing::ElementsAre(OMPC_shared));

  static_assert(detail::ElementsAre<Clause, Clause::OMPC_shared>(
      ClauseSet{Clause::OMPC_private, Clause::OMPC_shared}.reset(
          Clause::OMPC_private)));
}

TEST(EnumSetTest, Flip) {
  ClauseSet S{Clause::OMPC_private};
  S.flip(Clause::OMPC_private);
  S.flip(Clause::OMPC_shared);
  EXPECT_THAT(S, testing::ElementsAre(OMPC_shared));

  static_assert(detail::ElementsAre<Clause, Clause::OMPC_shared>(
      ClauseSet{Clause::OMPC_private}
          .flip(Clause::OMPC_private)
          .flip(Clause::OMPC_shared)));
}

TEST(EnumSetTest, Test) {
  constexpr ClauseSet S{Clause::OMPC_private};
  ASSERT_TRUE(S.test(Clause::OMPC_private));
  ASSERT_FALSE(S.test(Clause::OMPC_shared));

  static_assert(S.test(Clause::OMPC_private));
  static_assert(!S.test(Clause::OMPC_shared));
}

TEST(EnumSetTest, SquareBracket) {
  constexpr ClauseSet S{Clause::OMPC_private};
  ASSERT_TRUE(S[Clause::OMPC_private]);
  ASSERT_FALSE(S[Clause::OMPC_shared]);

  static_assert(S[Clause::OMPC_private]);
  static_assert(!S[Clause::OMPC_shared]);
}

TEST(EnumSetTest, UnionUpdate) {
  ClauseSet S{Clause::OMPC_private, OMPC_shared};
  ClauseSet A{Clause::OMPC_nowait};
  S |= A;
  EXPECT_THAT(S, testing::ElementsAre(Clause::OMPC_nowait, Clause::OMPC_private,
                                      OMPC_shared));
}

TEST(EnumSetTest, Union) {
  constexpr ClauseSet A{Clause::OMPC_private, OMPC_shared};
  constexpr ClauseSet B{Clause::OMPC_nowait};
  constexpr auto S = A | B;
  static_assert(std::is_same_v<llvm::remove_cvref_t<decltype(S)>, ClauseSet>);
  EXPECT_THAT(S, testing::ElementsAre(Clause::OMPC_nowait, Clause::OMPC_private,
                                      OMPC_shared));

  static_assert(detail::ElementsAre<Clause, Clause::OMPC_nowait,
                                    Clause::OMPC_private, OMPC_shared>(S));
}

TEST(EnumSetTest, IntersectionUpdate) {
  ClauseSet S{Clause::OMPC_private, OMPC_shared};
  ClauseSet A{Clause::OMPC_nowait, OMPC_shared};
  S &= A;
  EXPECT_THAT(S, testing::ElementsAre(OMPC_shared));
}

TEST(EnumSetTest, Intersection) {
  constexpr ClauseSet A{Clause::OMPC_private, OMPC_shared};
  constexpr ClauseSet B{Clause::OMPC_nowait, OMPC_shared};
  constexpr auto S = A & B;
  static_assert(std::is_same_v<llvm::remove_cvref_t<decltype(S)>, ClauseSet>);
  EXPECT_THAT(S, testing::ElementsAre(OMPC_shared));

  static_assert(detail::ElementsAre<Clause, OMPC_shared>(S));
}

TEST(EnumSetTest, WordCount) {
  // Check that for an enum that is exactly 64 in size, the iterator will not
  // try to access an out-of-bounds word of the underlying bitset.
  enum class E {
    V0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    V11,
    V12,
    V13,
    V14,
    V15,
    V16,
    V17,
    V18,
    V19,
    V20,
    V21,
    V22,
    V23,
    V24,
    V25,
    V26,
    V27,
    V28,
    V29,
    V30,
    V31,
    V32,
    V33,
    V34,
    V35,
    V36,
    V37,
    V38,
    V39,
    V40,
    V41,
    V42,
    V43,
    V44,
    V45,
    V46,
    V47,
    V48,
    V49,
    V50,
    V51,
    V52,
    V53,
    V54,
    V55,
    V56,
    V57,
    V58,
    V59,
    V60,
    V61,
    V62,
    V63
  };
  using ESet = EnumSet<E, 64>;

  constexpr ESet S{E::V0, E::V63};
  EXPECT_THAT(S, testing::ElementsAre(E::V0, E::V63));

  static_assert(detail::ElementsAre<E, E::V0, E::V63>(S));
}
} // namespace
