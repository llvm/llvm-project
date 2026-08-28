//===- RepeatedTest.cpp - Repeated unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Repeated.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;

namespace llvm {
namespace {

TEST(RepeatedTest, Construction) {
  {
    Repeated<int> Rep(5, 42);
    EXPECT_EQ(Rep.value(), 42);
    EXPECT_THAT(Rep, SizeIs(5));
    EXPECT_EQ(Rep[0], 42);
    EXPECT_EQ(Rep[4], 42);
    EXPECT_THAT(Rep, ElementsAre(42, 42, 42, 42, 42));
  }

  {
    Repeated<std::string> Rep(3, "hello");
    EXPECT_EQ(Rep.value(), "hello");
    EXPECT_THAT(Rep, SizeIs(3));
  }

  {
    // Move-only type.
    Repeated<std::unique_ptr<int>> Rep(1, std::make_unique<int>(42));
    EXPECT_EQ(*Rep.value(), 42);
    EXPECT_THAT(Rep, SizeIs(1));
  }

  {
    // Empty Rep.
    Repeated<int> EmptyRep(0, 42);
    EXPECT_THAT(EmptyRep, IsEmpty());
  }
}

TEST(RepeatedTest, CTAD) {
  static_assert(std::is_same_v<decltype(Repeated(3, 42)), Repeated<int>>);
  std::string S = "world";
  Repeated RepStr(2, S);
  static_assert(std::is_same_v<decltype(RepStr), Repeated<std::string>>);
  static_assert(
      std::is_same_v<decltype(Repeated(1, "literal")), Repeated<const char *>>);
  SUCCEED();
}

TEST(RepeatedTest, IteratorRandomAccess) {
  Repeated<int> Rep(10, 7);
  RepeatedIterator<int> It = Rep.begin();

  EXPECT_EQ(*It, 7);
  EXPECT_EQ(*(It + 5), 7);

  It += 10;
  EXPECT_EQ(It, Rep.end());
  --It;
  EXPECT_LT(It, Rep.end());
  EXPECT_EQ(Rep.end() - Rep.begin(), 10);
  ++It;
  EXPECT_EQ(It, Rep.end());
}

TEST(RepeatedTest, ReverseIterator) {
  Repeated<int> Rep(5, 42);
  std::vector<int> Reversed(Rep.rbegin(), Rep.rend());
  EXPECT_THAT(Reversed, SizeIs(5));
  EXPECT_THAT(Reversed, Each(42));
}

TEST(RepeatedTest, IteratorTraits) {
  using It = RepeatedIterator<int>;
  static_assert(std::is_default_constructible_v<It>);
  static_assert(std::is_same_v<std::iterator_traits<It>::iterator_category,
                               std::random_access_iterator_tag>);
  static_assert(std::is_same_v<std::iterator_traits<It>::value_type, int>);
  static_assert(
      std::is_same_v<std::iterator_traits<It>::difference_type, ptrdiff_t>);
  SUCCEED();
}

} // anonymous namespace
} // namespace llvm
