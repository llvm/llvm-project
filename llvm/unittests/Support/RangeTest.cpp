//===- llvm/unittests/Support/RangeTest.cpp - Range tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Range.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <limits>

using namespace llvm;

TEST(RangeTest, BasicRange) {
  Range R(5, 10);
  EXPECT_EQ(R.getBegin(), 5);
  EXPECT_EQ(R.getEnd(), 10);
  EXPECT_TRUE(R.contains(5));
  EXPECT_TRUE(R.contains(7));
  EXPECT_TRUE(R.contains(10));
  EXPECT_FALSE(R.contains(4));
  EXPECT_FALSE(R.contains(11));
}

TEST(RangeTest, SingleValueRange) {
  Range R(42);
  EXPECT_EQ(R.getBegin(), 42);
  EXPECT_EQ(R.getEnd(), 42);
  EXPECT_TRUE(R.contains(42));
  EXPECT_FALSE(R.contains(41));
  EXPECT_FALSE(R.contains(43));
}

TEST(RangeTest, SizeBasic) {
  Range R1(5, 10);
  EXPECT_EQ(R1.size(), 6u);

  Range R2(0, 0);
  EXPECT_EQ(R2.size(), 1u);
}

TEST(RangeTest, SizeMixedSigns) {
  Range R1(-2, 2);
  EXPECT_EQ(R1.size(), 5u);

  Range R2(-1, 0);
  EXPECT_EQ(R2.size(), 2u);
}

TEST(RangeTest, SizeExtremesNonOverflow) {
  // [INT64_MIN, -1] has size 2^63
  Range R1(std::numeric_limits<int64_t>::min(), -1);
  EXPECT_EQ(R1.size(), (1ULL << 63));

  // [0, INT64_MAX] has size 2^63
  Range R2(0, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(R2.size(), (1ULL << 63));

  // [INT64_MIN, 0] has size 2^63 + 1
  Range R3(std::numeric_limits<int64_t>::min(), 0);
  EXPECT_EQ(R3.size(), (1ULL << 63) + 1);

  // Small extreme windows
  Range R4(std::numeric_limits<int64_t>::min(),
           std::numeric_limits<int64_t>::min() + 10);
  EXPECT_EQ(R4.size(), 11u);

  Range R5(std::numeric_limits<int64_t>::max() - 10,
           std::numeric_limits<int64_t>::max());
  EXPECT_EQ(R5.size(), 11u);
}

TEST(RangeTest, RangeOverlaps) {
  Range R1(1, 5);
  Range R2(3, 8);
  Range R3(6, 10);
  Range R4(11, 15);

  EXPECT_TRUE(R1.overlaps(R2));
  EXPECT_TRUE(R2.overlaps(R1));
  EXPECT_TRUE(R2.overlaps(R3));
  EXPECT_FALSE(R1.overlaps(R3));
  EXPECT_FALSE(R1.overlaps(R4));
  EXPECT_FALSE(R3.overlaps(R4));
}

TEST(RangeUtilsTest, ParseSingleNumber) {
  auto ER = RangeUtils::parseRanges("42");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_EQ(Ranges.size(), 1U);
  EXPECT_EQ(Ranges[0].getBegin(), 42);
  EXPECT_EQ(Ranges[0].getEnd(), 42);
}

TEST(RangeUtilsTest, ParseSingleRange) {
  auto ER = RangeUtils::parseRanges("10-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_EQ(Ranges.size(), 1U);
  EXPECT_EQ(Ranges[0].getBegin(), 10);
  EXPECT_EQ(Ranges[0].getEnd(), 20);
}

TEST(RangeUtilsTest, ParseMultipleRanges) {
  auto ER = RangeUtils::parseRanges("1-5,10,15-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_EQ(Ranges.size(), 3U);

  // Ranges are in input order (DebugCounter style).
  EXPECT_EQ(Ranges[0].getBegin(), 1);
  EXPECT_EQ(Ranges[0].getEnd(), 5);
  EXPECT_EQ(Ranges[1].getBegin(), 10);
  EXPECT_EQ(Ranges[1].getEnd(), 10);
  EXPECT_EQ(Ranges[2].getBegin(), 15);
  EXPECT_EQ(Ranges[2].getEnd(), 20);
}

TEST(RangeUtilsTest, ParseColonSeparated) {
  auto ER = RangeUtils::parseRanges("1-5:10:15-20", ':');
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_EQ(Ranges.size(), 3U);
  EXPECT_EQ(Ranges[0].getBegin(), 1);
  EXPECT_EQ(Ranges[0].getEnd(), 5);
  EXPECT_EQ(Ranges[1].getBegin(), 10);
  EXPECT_EQ(Ranges[1].getEnd(), 10);
  EXPECT_EQ(Ranges[2].getBegin(), 15);
  EXPECT_EQ(Ranges[2].getEnd(), 20);
}

TEST(RangeUtilsTest, ParseEmptyString) {
  auto ER = RangeUtils::parseRanges("");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_TRUE(Ranges.empty());
}

TEST(RangeUtilsTest, ParseInvalidRanges) {
  // Invalid number.
  auto ER1 = RangeUtils::parseRanges("abc");
  EXPECT_THAT_EXPECTED(ER1, Failed());
  consumeError(ER1.takeError());

  // Invalid range (begin > end).
  auto ER2 = RangeUtils::parseRanges("10-5");
  EXPECT_THAT_EXPECTED(ER2, Failed());
  consumeError(ER2.takeError());

  // Out of order ranges (DebugCounter constraint and overlap).
  auto ER3 = RangeUtils::parseRanges("10,5");
  EXPECT_THAT_EXPECTED(ER3, Failed());
  consumeError(ER3.takeError());

  auto ER4 = RangeUtils::parseRanges("1-5,3-7");
  EXPECT_THAT_EXPECTED(ER4, Failed());
  consumeError(ER4.takeError());
}

TEST(RangeUtilsTest, Contains) {
  auto ER = RangeUtils::parseRanges("1-5,10,15-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);

  EXPECT_TRUE(RangeUtils::contains(Ranges, 1));
  EXPECT_TRUE(RangeUtils::contains(Ranges, 3));
  EXPECT_TRUE(RangeUtils::contains(Ranges, 5));
  EXPECT_TRUE(RangeUtils::contains(Ranges, 10));
  EXPECT_TRUE(RangeUtils::contains(Ranges, 15));
  EXPECT_TRUE(RangeUtils::contains(Ranges, 18));
  EXPECT_TRUE(RangeUtils::contains(Ranges, 20));

  EXPECT_FALSE(RangeUtils::contains(Ranges, 6));
  EXPECT_FALSE(RangeUtils::contains(Ranges, 9));
  EXPECT_FALSE(RangeUtils::contains(Ranges, 11));
  EXPECT_FALSE(RangeUtils::contains(Ranges, 14));
  EXPECT_FALSE(RangeUtils::contains(Ranges, 21));
}

TEST(RangeUtilsTest, SeparatorParameter) {
  RangeUtils::RangeList ColonRanges, CommaRanges;

  // Test explicit separator parameters.
  auto ERC = RangeUtils::parseRanges("1-5:10:15-20", ':');
  ASSERT_THAT_EXPECTED(ERC, Succeeded());
  ColonRanges = std::move(*ERC);

  auto ERM = RangeUtils::parseRanges("1-5,10,15-20", ',');
  ASSERT_THAT_EXPECTED(ERM, Succeeded());
  CommaRanges = std::move(*ERM);

  EXPECT_EQ(ColonRanges.size(), CommaRanges.size());
  for (size_t I = 0; I < ColonRanges.size(); ++I) {
    EXPECT_EQ(ColonRanges[I].getBegin(), CommaRanges[I].getBegin());
    EXPECT_EQ(ColonRanges[I].getEnd(), CommaRanges[I].getEnd());
  }

  // Test that both work with contains().
  EXPECT_TRUE(RangeUtils::contains(ColonRanges, 3));
  EXPECT_TRUE(RangeUtils::contains(CommaRanges, 3));
  EXPECT_TRUE(RangeUtils::contains(ColonRanges, 10));
  EXPECT_TRUE(RangeUtils::contains(CommaRanges, 10));
  EXPECT_TRUE(RangeUtils::contains(ColonRanges, 18));
  EXPECT_TRUE(RangeUtils::contains(CommaRanges, 18));

  EXPECT_FALSE(RangeUtils::contains(ColonRanges, 8));
  EXPECT_FALSE(RangeUtils::contains(CommaRanges, 8));
}

TEST(RangeUtilsTest, DefaultCommaSeparator) {
  // Test that comma is the default separator.
  auto ER = RangeUtils::parseRanges("1-5,10,15-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_EQ(Ranges.size(), 3U);
  EXPECT_EQ(Ranges[0].getBegin(), 1);
  EXPECT_EQ(Ranges[0].getEnd(), 5);
  EXPECT_EQ(Ranges[1].getBegin(), 10);
  EXPECT_EQ(Ranges[1].getEnd(), 10);
  EXPECT_EQ(Ranges[2].getBegin(), 15);
  EXPECT_EQ(Ranges[2].getEnd(), 20);
}

TEST(RangeTest, MergeAdjacentRanges) {
  RangeUtils::RangeList Input, Expected, Result;

  // Empty input
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_TRUE(Result.empty());

  // Single range - no change.
  Input.push_back(Range(5, 10));
  Expected.push_back(Range(5, 10));
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);

  // Adjacent ranges should merge.
  Input.clear();
  Expected.clear();
  Input.push_back(Range(1, 3));
  Input.push_back(Range(4, 6));
  Input.push_back(Range(7, 9));
  Expected.push_back(Range(1, 9));
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);

  // Non-adjacent ranges should not merge.
  Input.clear();
  Expected.clear();
  Input.push_back(Range(1, 3));
  Input.push_back(Range(5, 7));   // Gap between 3 and 5.
  Input.push_back(Range(10, 12)); // Gap between 7 and 10.
  Expected.push_back(Range(1, 3));
  Expected.push_back(Range(5, 7));
  Expected.push_back(Range(10, 12));
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);

  // Mixed adjacent and non-adjacent.
  Input.clear();
  Expected.clear();
  Input.push_back(Range(1, 3));
  Input.push_back(Range(4, 6));     // Adjacent to first.
  Input.push_back(Range(8, 10));    // Gap.
  Input.push_back(Range(11, 13));   // Adjacent to third.
  Input.push_back(Range(14, 16));   // Adjacent to fourth.
  Expected.push_back(Range(1, 6));  // Merged 1-3 and 4-6.
  Expected.push_back(Range(8, 16)); // Merged 8-10, 11-13, 14-16.
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);

  // Single numbers that are adjacent.
  Input.clear();
  Expected.clear();
  Input.push_back(Range(5));
  Input.push_back(Range(6));
  Input.push_back(Range(7));
  Expected.push_back(Range(5, 7));
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);
}
