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

TEST(SortedVectorMapTest, ReserveAndCapacity) {
  SortedVectorMap<int, int> Map;
  EXPECT_EQ(Map.size(), 0u);
  Map.reserve(50);
  EXPECT_GE(Map.capacity(), 50u);
  Map[1] = 10;
  EXPECT_EQ(Map.size(), 1u);
  EXPECT_GE(Map.capacity(), 50u);
}
} // namespace
