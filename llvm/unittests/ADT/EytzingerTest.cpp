//===- EytzingerTest.cpp - EytzingerTableSpan unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Eytzinger.h"
#include "llvm/Support/Endian.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(EytzingerTest, EmptyTable) {
  // Default constructed table span should be empty and return nullopt.
  EytzingerTableSpan<int> Empty;
  EXPECT_TRUE(Empty.empty());
  EXPECT_EQ(Empty.size(), 0u);
  EXPECT_EQ(Empty.data(), nullptr);
  EXPECT_TRUE(Empty.isSorted());
  EXPECT_EQ(Empty.findIndex(42), std::nullopt);
  EXPECT_FALSE(Empty.contains(42));

  // Span initialized with nullptr and zero size should behave identically.
  EytzingerTableSpan<int> NullSpan(nullptr, 0);
  EXPECT_TRUE(NullSpan.empty());
  EXPECT_EQ(NullSpan.size(), 0u);
  EXPECT_TRUE(NullSpan.isSorted());
  EXPECT_EQ(NullSpan.findIndex(42), std::nullopt);
  EXPECT_FALSE(NullSpan.contains(42));
}

TEST(EytzingerTest, SingleElementTable) {
  // Table with a single element at index 0.
  const int Data[] = {100};
  EytzingerTableSpan<int> Span(Data, 1);

  EXPECT_FALSE(Span.empty());
  EXPECT_EQ(Span.size(), 1u);
  EXPECT_TRUE(Span.isSorted());
  EXPECT_EQ(Span[0], 100);

  // Successful lookup for the single element.
  EXPECT_EQ(Span.findIndex(100), 0u);
  EXPECT_TRUE(Span.contains(100));

  // Unsuccessful lookups for keys smaller and larger than the element.
  EXPECT_EQ(Span.findIndex(99), std::nullopt);
  EXPECT_FALSE(Span.contains(99));
  EXPECT_EQ(Span.findIndex(101), std::nullopt);
  EXPECT_FALSE(Span.contains(101));
}

TEST(EytzingerTest, BinaryTreeWithSevenElements) {
  // Binary tree of 7 elements (3 full levels).
  // In Eytzinger layout (breadth-first order of complete BST):
  //
  //         40
  //       /    \
  //     20      60
  //    /  \    /  \
  //  10   30  50   70
  const int Data[] = {40, 20, 60, 10, 30, 50, 70};
  EytzingerTableSpan<int> Span(Data, 7);

  EXPECT_EQ(Span.size(), 7u);
  EXPECT_TRUE(Span.isSorted());

  // Verify successful lookups for every node in the tree.
  EXPECT_EQ(Span.findIndex(40), 0u);
  EXPECT_TRUE(Span.contains(40));
  EXPECT_EQ(Span.findIndex(20), 1u);
  EXPECT_TRUE(Span.contains(20));
  EXPECT_EQ(Span.findIndex(60), 2u);
  EXPECT_TRUE(Span.contains(60));
  EXPECT_EQ(Span.findIndex(10), 3u);
  EXPECT_TRUE(Span.contains(10));
  EXPECT_EQ(Span.findIndex(30), 4u);
  EXPECT_TRUE(Span.contains(30));
  EXPECT_EQ(Span.findIndex(50), 5u);
  EXPECT_TRUE(Span.contains(50));
  EXPECT_EQ(Span.findIndex(70), 6u);
  EXPECT_TRUE(Span.contains(70));

  // Verify unsuccessful lookups for values not in the tree.
  EXPECT_EQ(Span.findIndex(0), std::nullopt);
  EXPECT_FALSE(Span.contains(0));
  EXPECT_EQ(Span.findIndex(15), std::nullopt);
  EXPECT_FALSE(Span.contains(15));
  EXPECT_EQ(Span.findIndex(25), std::nullopt);
  EXPECT_FALSE(Span.contains(25));
  EXPECT_EQ(Span.findIndex(35), std::nullopt);
  EXPECT_FALSE(Span.contains(35));
  EXPECT_EQ(Span.findIndex(45), std::nullopt);
  EXPECT_FALSE(Span.contains(45));
  EXPECT_EQ(Span.findIndex(55), std::nullopt);
  EXPECT_FALSE(Span.contains(55));
  EXPECT_EQ(Span.findIndex(65), std::nullopt);
  EXPECT_FALSE(Span.contains(65));
  EXPECT_EQ(Span.findIndex(80), std::nullopt);
  EXPECT_FALSE(Span.contains(80));
}

TEST(EytzingerTest, BinaryTreeWithFiveElements) {
  // Binary tree of 5 elements (non-power-of-two minus one).
  // In Eytzinger layout:
  //
  //       40
  //      /  \
  //    20    50
  //   /  \
  // 10    30
  const int Data[] = {40, 20, 50, 10, 30};
  EytzingerTableSpan<int> Span(Data, 5);

  EXPECT_EQ(Span.size(), 5u);
  EXPECT_TRUE(Span.isSorted());

  // Verify lookups on existing elements.
  EXPECT_EQ(Span.findIndex(40), 0u);
  EXPECT_TRUE(Span.contains(40));
  EXPECT_EQ(Span.findIndex(20), 1u);
  EXPECT_TRUE(Span.contains(20));
  EXPECT_EQ(Span.findIndex(50), 2u);
  EXPECT_TRUE(Span.contains(50));
  EXPECT_EQ(Span.findIndex(10), 3u);
  EXPECT_TRUE(Span.contains(10));
  EXPECT_EQ(Span.findIndex(30), 4u);
  EXPECT_TRUE(Span.contains(30));

  // Verify lookups on missing values across various boundary conditions.
  EXPECT_EQ(Span.findIndex(5), std::nullopt);
  EXPECT_FALSE(Span.contains(5));
  EXPECT_EQ(Span.findIndex(15), std::nullopt);
  EXPECT_FALSE(Span.contains(15));
  EXPECT_EQ(Span.findIndex(25), std::nullopt);
  EXPECT_FALSE(Span.contains(25));
  EXPECT_EQ(Span.findIndex(35), std::nullopt);
  EXPECT_FALSE(Span.contains(35));
  EXPECT_EQ(Span.findIndex(45), std::nullopt);
  EXPECT_FALSE(Span.contains(45));
  EXPECT_EQ(Span.findIndex(60), std::nullopt);
  EXPECT_FALSE(Span.contains(60));
}

TEST(EytzingerTest, EndianSpecificIntegerType) {
  // Verify compatibility with LLVM endian-specific wrapper types such as
  // support::ulittle64_t which are commonly used in binary profile formats.
  const support::ulittle64_t Data[] = {
      support::ulittle64_t(400), support::ulittle64_t(200),
      support::ulittle64_t(600), support::ulittle64_t(100),
      support::ulittle64_t(300), support::ulittle64_t(500),
      support::ulittle64_t(700)};
  EytzingerTableSpan<support::ulittle64_t> Span(Data, 7);
  EXPECT_TRUE(Span.isSorted());

  EXPECT_EQ(Span.findIndex(uint64_t(400)), 0u);
  EXPECT_TRUE(Span.contains(uint64_t(400)));
  EXPECT_EQ(Span.findIndex(uint64_t(200)), 1u);
  EXPECT_TRUE(Span.contains(uint64_t(200)));
  EXPECT_EQ(Span.findIndex(uint64_t(600)), 2u);
  EXPECT_TRUE(Span.contains(uint64_t(600)));
  EXPECT_EQ(Span.findIndex(uint64_t(100)), 3u);
  EXPECT_TRUE(Span.contains(uint64_t(100)));
  EXPECT_EQ(Span.findIndex(uint64_t(300)), 4u);
  EXPECT_TRUE(Span.contains(uint64_t(300)));
  EXPECT_EQ(Span.findIndex(uint64_t(500)), 5u);
  EXPECT_TRUE(Span.contains(uint64_t(500)));
  EXPECT_EQ(Span.findIndex(uint64_t(700)), 6u);
  EXPECT_TRUE(Span.contains(uint64_t(700)));

  EXPECT_EQ(Span.findIndex(uint64_t(999)), std::nullopt);
  EXPECT_FALSE(Span.contains(uint64_t(999)));
}

TEST(EytzingerTest, IsSortedVerification) {
  // Verify detection of local parent-child violations.
  // Root (40), left child (60 > 40), right child (20 < 40).
  const int InvalidChildOrder[] = {40, 60, 20};
  EXPECT_FALSE(EytzingerTableSpan<int>(InvalidChildOrder, 3).isSorted());

  // Verify detection of across-level ancestor bounds violations.
  // Root (40), left (20), right (60). Left of 20 is 10, right of 20 is 50.
  // Although 50 > 20 (local parent check passes), 50 > 40 violates the root
  // bound.
  const int AncestorViolation[] = {40, 20, 60, 10, 50, 55, 70};
  EXPECT_FALSE(EytzingerTableSpan<int>(AncestorViolation, 7).isSorted());

  // Verify that tables with duplicate values are flagged as unsorted because
  // EytzingerTableSpan requires strictly ascending in-order keys.
  const int Duplicates[] = {30, 30, 30};
  EXPECT_FALSE(EytzingerTableSpan<int>(Duplicates, 3).isSorted());
}

} // namespace
