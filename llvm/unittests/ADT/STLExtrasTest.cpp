//===- STLExtrasTest.cpp - Unit tests for STL extras ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <vector>

using namespace llvm;

namespace {

int f(rank<0>) { return 0; }
int f(rank<1>) { return 1; }
int f(rank<2>) { return 2; }
int f(rank<4>) { return 4; }

TEST(STLExtrasTest, Rank) {
  // We shouldn't get ambiguities and should select the overload of the same
  // rank as the argument.
  EXPECT_EQ(0, f(rank<0>()));
  EXPECT_EQ(1, f(rank<1>()));
  EXPECT_EQ(2, f(rank<2>()));

  // This overload is missing so we end up back at 2.
  EXPECT_EQ(2, f(rank<3>()));

  // But going past 3 should work fine.
  EXPECT_EQ(4, f(rank<4>()));

  // And we can even go higher and just fall back to the last overload.
  EXPECT_EQ(4, f(rank<5>()));
  EXPECT_EQ(4, f(rank<6>()));
}

TEST(STLExtrasTest, Enumerate) {
  std::vector<char> foo = {'a', 'b', 'c'};

  std::vector<std::pair<std::size_t, char>> results;

  for (auto X : llvm::enumerate(foo)) {
    results.push_back(std::make_pair(X.Index, X.Value));
  }
  ASSERT_EQ(3u, results.size());
  EXPECT_EQ(0u, results[0].first);
  EXPECT_EQ('a', results[0].second);
  EXPECT_EQ(1u, results[1].first);
  EXPECT_EQ('b', results[1].second);
  EXPECT_EQ(2u, results[2].first);
  EXPECT_EQ('c', results[2].second);

  results.clear();
  const std::vector<int> bar = {'1', '2', '3'};
  for (auto X : llvm::enumerate(bar)) {
    results.push_back(std::make_pair(X.Index, X.Value));
  }
  EXPECT_EQ(0u, results[0].first);
  EXPECT_EQ('1', results[0].second);
  EXPECT_EQ(1u, results[1].first);
  EXPECT_EQ('2', results[1].second);
  EXPECT_EQ(2u, results[2].first);
  EXPECT_EQ('3', results[2].second);

  results.clear();
  const std::vector<int> baz;
  for (auto X : llvm::enumerate(baz)) {
    results.push_back(std::make_pair(X.Index, X.Value));
  }
  EXPECT_TRUE(baz.empty());
}

TEST(STLExtrasTest, EnumerateModify) {
  std::vector<char> foo = {'a', 'b', 'c'};

  for (auto X : llvm::enumerate(foo)) {
    ++X.Value;
  }

  EXPECT_EQ('b', foo[0]);
  EXPECT_EQ('c', foo[1]);
  EXPECT_EQ('d', foo[2]);
}
}
