//===- EytzingerTest.cpp - EytzingerTableSpan unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Eytzinger.h"
#include "llvm/Support/Endian.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(EytzingerTest, EmptyTable) {
  // Default constructed table span should be empty and return nullopt.
  EytzingerTableSpan<int> Empty;
  EXPECT_TRUE(Empty.empty());
  EXPECT_EQ(Empty.size(), 0u);
  EXPECT_EQ(Empty.data(), nullptr);
  EXPECT_EQ(Empty.begin(), Empty.end());
  EXPECT_EQ(Empty.begin(), nullptr);
  EXPECT_TRUE(Empty.isSorted());
  EXPECT_EQ(Empty.findIndex(42), std::nullopt);
  EXPECT_FALSE(Empty.contains(42));

  // Span initialized with nullptr and zero size should behave identically.
  EytzingerTableSpan<int> NullSpan(nullptr, 0);
  EXPECT_TRUE(NullSpan.empty());
  EXPECT_EQ(NullSpan.size(), 0u);
  EXPECT_EQ(NullSpan.begin(), NullSpan.end());
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

TEST(EytzingerTest, EytzingerTableCreateAndLookup) {
  // Construct table from an unsorted vector with duplicate keys for a non-full
  // bottom level (10 unique elements, size != 2^k - 1).
  std::vector<int> Unsorted = {70, 20, 40, 10,  60, 30, 50,
                               80, 90, 40, 100, 20, 10};
  auto Table = EytzingerTable<int>::create(std::move(Unsorted));

  // Should contain exactly 10 unique elements after deduplication.
  EXPECT_EQ(Table.size(), 10u);
  EXPECT_FALSE(Table.empty());
  EXPECT_TRUE(Table.isSorted());

  // Root of 10-element complete BST is 70.
  EXPECT_EQ(Table[0], 70);
  EXPECT_EQ(Table.findIndex(70), 0u);
  EXPECT_TRUE(Table.contains(70));

  // Verify successful lookups for all unique keys explicitly so line numbers
  // identify failures.
  EXPECT_TRUE(Table.contains(10));
  EXPECT_NE(Table.findIndex(10), std::nullopt);
  EXPECT_TRUE(Table.contains(20));
  EXPECT_NE(Table.findIndex(20), std::nullopt);
  EXPECT_TRUE(Table.contains(30));
  EXPECT_NE(Table.findIndex(30), std::nullopt);
  EXPECT_TRUE(Table.contains(40));
  EXPECT_NE(Table.findIndex(40), std::nullopt);
  EXPECT_TRUE(Table.contains(50));
  EXPECT_NE(Table.findIndex(50), std::nullopt);
  EXPECT_TRUE(Table.contains(60));
  EXPECT_NE(Table.findIndex(60), std::nullopt);
  EXPECT_TRUE(Table.contains(70));
  EXPECT_NE(Table.findIndex(70), std::nullopt);
  EXPECT_TRUE(Table.contains(80));
  EXPECT_NE(Table.findIndex(80), std::nullopt);
  EXPECT_TRUE(Table.contains(90));
  EXPECT_NE(Table.findIndex(90), std::nullopt);
  EXPECT_TRUE(Table.contains(100));
  EXPECT_NE(Table.findIndex(100), std::nullopt);

  // Verify missing keys across various bounds and gaps.
  EXPECT_FALSE(Table.contains(0));
  EXPECT_EQ(Table.findIndex(0), std::nullopt);
  EXPECT_FALSE(Table.contains(15));
  EXPECT_EQ(Table.findIndex(15), std::nullopt);
  EXPECT_FALSE(Table.contains(75));
  EXPECT_EQ(Table.findIndex(75), std::nullopt);
  EXPECT_FALSE(Table.contains(110));
  EXPECT_EQ(Table.findIndex(110), std::nullopt);
}

TEST(EytzingerTest, EytzingerTableEmptyAndSingle) {
  auto EmptyTable = EytzingerTable<int>::create(std::vector<int>{});
  EXPECT_TRUE(EmptyTable.empty());
  EXPECT_EQ(EmptyTable.size(), 0u);
  EXPECT_EQ(EmptyTable.begin(), EmptyTable.end());
  EXPECT_TRUE(EmptyTable.isSorted());
  EXPECT_FALSE(EmptyTable.contains(42));

  auto SingleTable = EytzingerTable<int>::create(std::vector<int>{42, 42});
  EXPECT_FALSE(SingleTable.empty());
  EXPECT_EQ(SingleTable.size(), 1u);
  EXPECT_NE(SingleTable.begin(), SingleTable.end());
  EXPECT_EQ(*SingleTable.begin(), 42);
  EXPECT_TRUE(SingleTable.isSorted());
  EXPECT_EQ(SingleTable[0], 42);
  EXPECT_TRUE(SingleTable.contains(42));
  EXPECT_EQ(SingleTable.findIndex(42), 0u);
}

TEST(EytzingerTest, EytzingerTableEndianSpecific) {
  std::vector<support::ulittle64_t> Input = {
      support::ulittle64_t(500), support::ulittle64_t(100),
      support::ulittle64_t(300), support::ulittle64_t(300)};
  auto Table = EytzingerTable<support::ulittle64_t>::create(Input);

  EXPECT_EQ(Table.size(), 3u);
  EXPECT_TRUE(Table.isSorted());
  EXPECT_TRUE(Table.contains(uint64_t(300)));
  EXPECT_EQ(Table.findIndex(uint64_t(300)), 0u);
  EXPECT_FALSE(Table.contains(uint64_t(999)));
}

TEST(EytzingerTest, EytzingerTableHeterogeneousCreate) {
  // Construct EytzingerTable<support::ulittle64_t> directly from a vector of
  // native uint64_t keys with an imperfect tree size (6 elements).
  std::vector<uint64_t> NativeKeys = {500ULL, 100ULL, 300ULL, 600ULL,
                                      200ULL, 400ULL, 300ULL};
  auto Table = EytzingerTable<support::ulittle64_t>::create(NativeKeys);

  EXPECT_EQ(Table.size(), 6u);
  EXPECT_TRUE(Table.isSorted());

  // Root of 6-element complete BST is 400.
  EXPECT_EQ(uint64_t(Table[0]), 400ULL);
  EXPECT_EQ(Table.findIndex(uint64_t(400ULL)), 0u);
  EXPECT_TRUE(Table.contains(uint64_t(100ULL)));
  EXPECT_TRUE(Table.contains(uint64_t(200ULL)));
  EXPECT_TRUE(Table.contains(uint64_t(300ULL)));
  EXPECT_TRUE(Table.contains(uint64_t(400ULL)));
  EXPECT_TRUE(Table.contains(uint64_t(500ULL)));
  EXPECT_TRUE(Table.contains(uint64_t(600ULL)));
  EXPECT_FALSE(Table.contains(uint64_t(999ULL)));
}

TEST(EytzingerTest, EytzingerTableSpanBeginEnd) {
  const int Data[] = {40, 20, 60, 10, 30, 50, 70};
  EytzingerTableSpan<int> Span(Data, 7);

  EXPECT_EQ(Span.begin(), Data);
  EXPECT_EQ(Span.end(), Data + 7);
  EXPECT_EQ(std::distance(Span.begin(), Span.end()), 7);

  std::vector<int> Traversed(Span.begin(), Span.end());
  const int Expected[] = {40, 20, 60, 10, 30, 50, 70};
  EXPECT_THAT(Traversed, testing::ElementsAreArray(Expected));

  // Range-based for loop verification.
  std::vector<int> RangeTraversed;
  for (const int &Val : Span)
    RangeTraversed.push_back(Val);
  EXPECT_THAT(RangeTraversed, testing::ElementsAreArray(Expected));
}

TEST(EytzingerTest, EytzingerTableBeginEnd) {
  std::vector<int> Unsorted = {70, 20, 40, 10, 60, 30, 50};
  auto Table = EytzingerTable<int>::create(std::move(Unsorted));

  EXPECT_EQ(std::distance(Table.begin(), Table.end()), 7);

  std::vector<int> Traversed(Table.begin(), Table.end());
  const int Expected[] = {40, 20, 60, 10, 30, 50, 70};
  EXPECT_THAT(Traversed, testing::ElementsAreArray(Expected));

  // Range-based for loop verification on both non-const and const tables.
  std::vector<int> RangeTraversed;
  for (const int &Val : Table)
    RangeTraversed.push_back(Val);
  EXPECT_THAT(RangeTraversed, testing::ElementsAreArray(Expected));

  const auto &ConstTable = Table;
  std::vector<int> ConstTraversed(ConstTable.begin(), ConstTable.end());
  EXPECT_THAT(ConstTraversed, testing::ElementsAreArray(Expected));
}

} // namespace
