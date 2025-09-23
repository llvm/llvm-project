//===- llvm/unittests/Support/RangeTest.cpp - Range tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/IntegerInclusiveInterval.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(IntegerInclusiveIntervalTest, BasicInterval) {
  IntegerInclusiveInterval R(5, 10);
  EXPECT_EQ(R.getBegin(), 5);
  EXPECT_EQ(R.getEnd(), 10);
  EXPECT_TRUE(R.contains(5));
  EXPECT_TRUE(R.contains(7));
  EXPECT_TRUE(R.contains(10));
  EXPECT_FALSE(R.contains(4));
  EXPECT_FALSE(R.contains(11));
}

TEST(IntegerInclusiveIntervalTest, SingleValueInterval) {
  IntegerInclusiveInterval R(42);
  EXPECT_EQ(R.getBegin(), 42);
  EXPECT_EQ(R.getEnd(), 42);
  EXPECT_TRUE(R.contains(42));
  EXPECT_FALSE(R.contains(41));
  EXPECT_FALSE(R.contains(43));
}

TEST(IntegerInclusiveIntervalTest, IntervalOverlaps) {
  IntegerInclusiveInterval R1(1, 5);
  IntegerInclusiveInterval R2(3, 8);
  IntegerInclusiveInterval R3(6, 10);
  IntegerInclusiveInterval R4(11, 15);

  EXPECT_TRUE(R1.overlaps(R2));
  EXPECT_TRUE(R2.overlaps(R1));
  EXPECT_TRUE(R2.overlaps(R3));
  EXPECT_FALSE(R1.overlaps(R3));
  EXPECT_FALSE(R1.overlaps(R4));
  EXPECT_FALSE(R3.overlaps(R4));
}

TEST(IntegerIntervalUtilsTest, ParseSingleNumber) {
  auto ER = IntegerIntervalUtils::parseIntervals("42");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_EQ(Ranges.size(), 1U);
  EXPECT_EQ(Ranges[0].getBegin(), 42);
  EXPECT_EQ(Ranges[0].getEnd(), 42);
}

TEST(IntegerIntervalUtilsTest, ParseSingleInterval) {
  auto ER = IntegerIntervalUtils::parseIntervals("10-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_EQ(Ranges.size(), 1U);
  EXPECT_EQ(Ranges[0].getBegin(), 10);
  EXPECT_EQ(Ranges[0].getEnd(), 20);
}

TEST(IntegerIntervalUtilsTest, ParseMultipleIntervals) {
  auto ER = IntegerIntervalUtils::parseIntervals("1-5,10,15-20");
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

TEST(IntegerIntervalUtilsTest, ParseColonSeparatedIntervals) {
  auto ER = IntegerIntervalUtils::parseIntervals("1-5:10:15-20", ':');
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

TEST(IntegerIntervalUtilsTest, ParseEmptyString) {
  auto ER = IntegerIntervalUtils::parseIntervals("");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);
  EXPECT_TRUE(Ranges.empty());
}

TEST(IntegerIntervalUtilsTest, ParseInvalidIntervals) {
  // Invalid number.
  auto ER1 = IntegerIntervalUtils::parseIntervals("abc");
  EXPECT_THAT_EXPECTED(ER1, Failed());
  consumeError(ER1.takeError());

  // Invalid range (begin > end).
  auto ER2 = IntegerIntervalUtils::parseIntervals("10-5");
  EXPECT_THAT_EXPECTED(ER2, Failed());
  consumeError(ER2.takeError());

  // Out of order ranges (DebugCounter constraint and overlap).
  auto ER3 = IntegerIntervalUtils::parseIntervals("10,5");
  EXPECT_THAT_EXPECTED(ER3, Failed());
  consumeError(ER3.takeError());

  auto ER4 = IntegerIntervalUtils::parseIntervals("1-5,3-7");
  EXPECT_THAT_EXPECTED(ER4, Failed());
  consumeError(ER4.takeError());
}

TEST(IntegerIntervalUtilsTest, Contains) {
  auto ER = IntegerIntervalUtils::parseIntervals("1-5,10,15-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Ranges = std::move(*ER);

  EXPECT_TRUE(IntegerIntervalUtils::contains(Ranges, 1));
  EXPECT_TRUE(IntegerIntervalUtils::contains(Ranges, 3));
  EXPECT_TRUE(IntegerIntervalUtils::contains(Ranges, 5));
  EXPECT_TRUE(IntegerIntervalUtils::contains(Ranges, 10));
  EXPECT_TRUE(IntegerIntervalUtils::contains(Ranges, 15));
  EXPECT_TRUE(IntegerIntervalUtils::contains(Ranges, 18));
  EXPECT_TRUE(IntegerIntervalUtils::contains(Ranges, 20));

  EXPECT_FALSE(IntegerIntervalUtils::contains(Ranges, 6));
  EXPECT_FALSE(IntegerIntervalUtils::contains(Ranges, 9));
  EXPECT_FALSE(IntegerIntervalUtils::contains(Ranges, 11));
  EXPECT_FALSE(IntegerIntervalUtils::contains(Ranges, 14));
  EXPECT_FALSE(IntegerIntervalUtils::contains(Ranges, 21));
}

TEST(IntegerIntervalUtilsTest, SeparatorParameter) {
  IntegerIntervalUtils::IntervalList ColonRanges, CommaRanges;

  // Test explicit separator parameters.
  auto ERC = IntegerIntervalUtils::parseIntervals("1-5:10:15-20", ':');
  ASSERT_THAT_EXPECTED(ERC, Succeeded());
  ColonRanges = std::move(*ERC);

  auto ERM = IntegerIntervalUtils::parseIntervals("1-5,10,15-20", ',');
  ASSERT_THAT_EXPECTED(ERM, Succeeded());
  CommaRanges = std::move(*ERM);

  EXPECT_EQ(ColonRanges.size(), CommaRanges.size());
  for (size_t I = 0; I < ColonRanges.size(); ++I) {
    EXPECT_EQ(ColonRanges[I].getBegin(), CommaRanges[I].getBegin());
    EXPECT_EQ(ColonRanges[I].getEnd(), CommaRanges[I].getEnd());
  }

  // Test that both work with contains().
  EXPECT_TRUE(IntegerIntervalUtils::contains(ColonRanges, 3));
  EXPECT_TRUE(IntegerIntervalUtils::contains(CommaRanges, 3));
  EXPECT_TRUE(IntegerIntervalUtils::contains(ColonRanges, 10));
  EXPECT_TRUE(IntegerIntervalUtils::contains(CommaRanges, 10));
  EXPECT_TRUE(IntegerIntervalUtils::contains(ColonRanges, 18));
  EXPECT_TRUE(IntegerIntervalUtils::contains(CommaRanges, 18));

  EXPECT_FALSE(IntegerIntervalUtils::contains(ColonRanges, 8));
  EXPECT_FALSE(IntegerIntervalUtils::contains(CommaRanges, 8));
}

TEST(IntegerIntervalUtilsTest, DefaultCommaSeparator) {
  // Test that comma is the default separator.
  auto ER = IntegerIntervalUtils::parseIntervals("1-5,10,15-20");
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

TEST(IntegerInclusiveIntervalTest, MergeAdjacentIntervals) {
  IntegerIntervalUtils::IntervalList Input, Expected, Result;

  // Empty input
  Result = IntegerIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_TRUE(Result.empty());

  // Single range - no change.
  Input.push_back(IntegerInclusiveInterval(5, 10));
  Expected.push_back(IntegerInclusiveInterval(5, 10));
  Result = IntegerIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);

  // Adjacent ranges should merge.
  Input.clear();
  Expected.clear();
  Input.push_back(IntegerInclusiveInterval(1, 3));
  Input.push_back(IntegerInclusiveInterval(4, 6));
  Input.push_back(IntegerInclusiveInterval(7, 9));
  Expected.push_back(IntegerInclusiveInterval(1, 9));
  Result = IntegerIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);

  // Non-adjacent ranges should not merge.
  Input.clear();
  Expected.clear();
  Input.push_back(IntegerInclusiveInterval(1, 3));
  Input.push_back(IntegerInclusiveInterval(5, 7));   // Gap between 3 and 5.
  Input.push_back(IntegerInclusiveInterval(10, 12)); // Gap between 7 and 10.
  Expected.push_back(IntegerInclusiveInterval(1, 3));
  Expected.push_back(IntegerInclusiveInterval(5, 7));
  Expected.push_back(IntegerInclusiveInterval(10, 12));
  Result = IntegerIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);

  // Mixed adjacent and non-adjacent.
  Input.clear();
  Expected.clear();
  Input.push_back(IntegerInclusiveInterval(1, 3));
  Input.push_back(IntegerInclusiveInterval(4, 6));     // Adjacent to first.
  Input.push_back(IntegerInclusiveInterval(8, 10));    // Gap.
  Input.push_back(IntegerInclusiveInterval(11, 13));   // Adjacent to third.
  Input.push_back(IntegerInclusiveInterval(14, 16));   // Adjacent to fourth.
  Expected.push_back(IntegerInclusiveInterval(1, 6));  // Merged 1-3 and 4-6.
  Expected.push_back(IntegerInclusiveInterval(8, 16)); // Merged 8-10, 11-13, 14-16.
  Result = IntegerIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);

  // Single numbers that are adjacent.
  Input.clear();
  Expected.clear();
  Input.push_back(IntegerInclusiveInterval(5));
  Input.push_back(IntegerInclusiveInterval(6));
  Input.push_back(IntegerInclusiveInterval(7));
  Expected.push_back(IntegerInclusiveInterval(5, 7));
  Result = IntegerIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);
}
