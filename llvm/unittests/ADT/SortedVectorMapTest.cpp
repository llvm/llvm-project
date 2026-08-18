//===- llvm/unittest/ADT/SortedVectorMapTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SortedVectorMap.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {

TEST(SortedVectorMapTest, BasicOperations) {
  SortedVectorMap<int, std::string> Map;

  EXPECT_TRUE(Map.empty());
  EXPECT_EQ(Map.size(), 0u);

  Map[5] = "five";
  Map[2] = "two";
  Map[8] = "eight";

  EXPECT_FALSE(Map.empty());
  ASSERT_EQ(Map.size(), 3u);

  EXPECT_EQ(Map[2], "two");
  EXPECT_EQ(Map[5], "five");
  EXPECT_EQ(Map[8], "eight");

  // Verify elements are maintained in sorted key order
  EXPECT_THAT(Map, testing::ElementsAre(testing::Pair(2, "two"),
                                        testing::Pair(5, "five"),
                                        testing::Pair(8, "eight")));
}

TEST(SortedVectorMapTest, FindAndErase) {
  SortedVectorMap<int, int> Map;
  Map[10] = 100;
  Map[20] = 200;
  Map[30] = 300;

  auto It = Map.find(20);
  ASSERT_NE(It, Map.end());
  EXPECT_EQ(It->second, 200);

  EXPECT_EQ(Map.find(99), Map.end());

  It = Map.erase(It);
  EXPECT_EQ(Map.size(), 2u);
  EXPECT_EQ(Map.find(20), Map.end());
  ASSERT_NE(It, Map.end());
  EXPECT_EQ(It->first, 30);
}

TEST(SortedVectorMapTest, EqualityOperator) {
  SortedVectorMap<int, int> Map1;
  SortedVectorMap<int, int> Map2;

  Map1[1] = 10;
  Map1[2] = 20;

  Map2[2] = 20;
  Map2[1] = 10;

  EXPECT_EQ(Map1, Map2);
}

TEST(SortedVectorMapTest, InsertAndTryEmplace) {
  SortedVectorMap<int, std::string> Map;

  // Test insert with lvalue and rvalue pairs
  auto Pair1 = std::make_pair(3, "three");
  auto [It1, Inserted1] = Map.insert(Pair1);
  ASSERT_TRUE(Inserted1);
  EXPECT_EQ(It1->first, 3);
  EXPECT_EQ(It1->second, "three");

  auto [It2, Inserted2] = Map.insert(std::make_pair(1, "one"));
  ASSERT_TRUE(Inserted2);
  EXPECT_EQ(It2->first, 1);
  EXPECT_EQ(It2->second, "one");

  // Duplicate insert should fail and preserve existing value
  auto [ItDup, InsertedDup] = Map.insert(std::make_pair(3, "THREE"));
  ASSERT_FALSE(InsertedDup);
  EXPECT_EQ(ItDup->first, 3);
  EXPECT_EQ(ItDup->second, "three");

  // Test try_emplace in-place construction
  auto [It3, Inserted3] = Map.try_emplace(2, 4, 'x');
  ASSERT_TRUE(Inserted3);
  EXPECT_EQ(It3->first, 2);
  EXPECT_EQ(It3->second, "xxxx");

  // Duplicate try_emplace should not construct or overwrite
  auto [It4, Inserted4] = Map.try_emplace(2, "new_two");
  ASSERT_FALSE(Inserted4);
  EXPECT_EQ(It4->first, 2);
  EXPECT_EQ(It4->second, "xxxx");

  // Verify sorted order
  EXPECT_THAT(Map, testing::ElementsAre(testing::Pair(1, "one"),
                                        testing::Pair(2, "xxxx"),
                                        testing::Pair(3, "three")));
}

TEST(SortedVectorMapTest, ReserveAndCapacity) {
  SortedVectorMap<int, int> Map;
  EXPECT_EQ(Map.size(), 0u);
  Map.reserve(50);
  EXPECT_GE(Map.capacity(), 50u);
  Map[1] = 10;
  EXPECT_EQ(Map.size(), 1u);
  EXPECT_GE(Map.capacity(), 50u);
}

TEST(SortedVectorMapTest, Iterators) {
  SortedVectorMap<int, int> Map;
  Map[3] = 30;
  Map[1] = 10;
  Map[2] = 20;

  const auto &ConstMap = Map;
  ASSERT_EQ(std::distance(ConstMap.cbegin(), ConstMap.cend()), 3);
  EXPECT_EQ(ConstMap.cbegin()->first, 1);
  EXPECT_EQ(std::prev(ConstMap.cend())->first, 3);

  ASSERT_EQ(std::distance(ConstMap.crbegin(), ConstMap.crend()), 3);
  EXPECT_EQ(Map.rbegin()->first, 3);
  EXPECT_EQ(std::prev(Map.rend())->first, 1);
  EXPECT_EQ(ConstMap.crbegin()->first, 3);
  EXPECT_EQ(std::prev(ConstMap.crend())->first, 1);
}
} // namespace
