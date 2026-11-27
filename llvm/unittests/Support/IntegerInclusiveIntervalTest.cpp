//===- llvm/unittest/Support/IntegerInclusiveIntervalTest.cpp -------------===//
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
  IntegerInclusiveInterval I(5, 10);
  EXPECT_EQ(I.getBegin(), 5);
  EXPECT_EQ(I.getEnd(), 10);
  EXPECT_TRUE(I.contains(5));
  EXPECT_TRUE(I.contains(7));
  EXPECT_TRUE(I.contains(10));
  EXPECT_FALSE(I.contains(4));
  EXPECT_FALSE(I.contains(11));
}

TEST(IntegerInclusiveIntervalTest, SingleValueInterval) {
  IntegerInclusiveInterval I(42);
  EXPECT_EQ(I.getBegin(), 42);
  EXPECT_EQ(I.getEnd(), 42);
  EXPECT_TRUE(I.contains(42));
  EXPECT_FALSE(I.contains(41));
  EXPECT_FALSE(I.contains(43));
}

TEST(IntegerInclusiveIntervalTest, IntervalOverlaps) {
  IntegerInclusiveInterval I1(1, 5);
  IntegerInclusiveInterval I2(3, 8);
  IntegerInclusiveInterval I3(6, 10);
  IntegerInclusiveInterval I4(11, 15);

  EXPECT_TRUE(I1.overlaps(I2));
  EXPECT_TRUE(I2.overlaps(I1));
  EXPECT_TRUE(I2.overlaps(I3));
  EXPECT_FALSE(I1.overlaps(I3));
  EXPECT_FALSE(I1.overlaps(I4));
  EXPECT_FALSE(I3.overlaps(I4));
}

TEST(IntegerInclusiveIntervalUtilsTest, ParseSingleNumber) {
  auto ER = IntegerInclusiveIntervalUtils::parseIntervals("42");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Intervals = std::move(*ER);
  EXPECT_EQ(Intervals.size(), 1U);
  EXPECT_EQ(Intervals[0].getBegin(), 42);
  EXPECT_EQ(Intervals[0].getEnd(), 42);
}

TEST(IntegerInclusiveIntervalUtilsTest, ParseSingleInterval) {
  auto ER = IntegerInclusiveIntervalUtils::parseIntervals("10-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Intervals = std::move(*ER);
  EXPECT_EQ(Intervals.size(), 1U);
  EXPECT_EQ(Intervals[0].getBegin(), 10);
  EXPECT_EQ(Intervals[0].getEnd(), 20);
}

TEST(IntegerInclusiveIntervalUtilsTest, ParseMultipleIntervals) {
  auto ER = IntegerInclusiveIntervalUtils::parseIntervals("1-5,10,15-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Intervals = std::move(*ER);
  EXPECT_EQ(Intervals.size(), 3U);

  // Intervals are in input order (DebugCounter style).
  EXPECT_EQ(Intervals[0].getBegin(), 1);
  EXPECT_EQ(Intervals[0].getEnd(), 5);
  EXPECT_EQ(Intervals[1].getBegin(), 10);
  EXPECT_EQ(Intervals[1].getEnd(), 10);
  EXPECT_EQ(Intervals[2].getBegin(), 15);
  EXPECT_EQ(Intervals[2].getEnd(), 20);
}

TEST(IntegerInclusiveIntervalUtilsTest, ParseColonSeparatedIntervals) {
  auto ER = IntegerInclusiveIntervalUtils::parseIntervals("1-5:10:15-20", ':');
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Intervals = std::move(*ER);
  EXPECT_EQ(Intervals.size(), 3U);
  EXPECT_EQ(Intervals[0].getBegin(), 1);
  EXPECT_EQ(Intervals[0].getEnd(), 5);
  EXPECT_EQ(Intervals[1].getBegin(), 10);
  EXPECT_EQ(Intervals[1].getEnd(), 10);
  EXPECT_EQ(Intervals[2].getBegin(), 15);
  EXPECT_EQ(Intervals[2].getEnd(), 20);
}

TEST(IntegerInclusiveIntervalUtilsTest, ParseEmptyString) {
  auto ER = IntegerInclusiveIntervalUtils::parseIntervals("");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Intervals = std::move(*ER);
  EXPECT_TRUE(Intervals.empty());
}

TEST(IntegerInclusiveIntervalUtilsTest, ParseInvalidIntervals) {
  // Invalid number.
  auto ER1 = IntegerInclusiveIntervalUtils::parseIntervals("abc");
  EXPECT_THAT_EXPECTED(ER1, Failed());
  consumeError(ER1.takeError());

  // Invalid interval (begin > end).
  auto ER2 = IntegerInclusiveIntervalUtils::parseIntervals("10-5");
  EXPECT_THAT_EXPECTED(ER2, Failed());
  consumeError(ER2.takeError());

  // Out of order intervals (DebugCounter constraint and overlap).
  auto ER3 = IntegerInclusiveIntervalUtils::parseIntervals("10,5");
  EXPECT_THAT_EXPECTED(ER3, Failed());
  consumeError(ER3.takeError());

  auto ER4 = IntegerInclusiveIntervalUtils::parseIntervals("1-5,3-7");
  EXPECT_THAT_EXPECTED(ER4, Failed());
  consumeError(ER4.takeError());
}

TEST(IntegerInclusiveIntervalUtilsTest, Contains) {
  auto ER = IntegerInclusiveIntervalUtils::parseIntervals("1-5,10,15-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Intervals = std::move(*ER);

  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(Intervals, 1));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(Intervals, 3));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(Intervals, 5));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(Intervals, 10));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(Intervals, 15));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(Intervals, 18));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(Intervals, 20));

  EXPECT_FALSE(IntegerInclusiveIntervalUtils::contains(Intervals, 6));
  EXPECT_FALSE(IntegerInclusiveIntervalUtils::contains(Intervals, 9));
  EXPECT_FALSE(IntegerInclusiveIntervalUtils::contains(Intervals, 11));
  EXPECT_FALSE(IntegerInclusiveIntervalUtils::contains(Intervals, 14));
  EXPECT_FALSE(IntegerInclusiveIntervalUtils::contains(Intervals, 21));
}

TEST(IntegerInclusiveIntervalUtilsTest, SeparatorParameter) {
  IntegerInclusiveIntervalUtils::IntervalList ColonIntervals, CommaIntervals;

  // Test explicit separator parameters.
  auto ERC = IntegerInclusiveIntervalUtils::parseIntervals("1-5:10:15-20", ':');
  ASSERT_THAT_EXPECTED(ERC, Succeeded());
  ColonIntervals = std::move(*ERC);

  auto ERM = IntegerInclusiveIntervalUtils::parseIntervals("1-5,10,15-20", ',');
  ASSERT_THAT_EXPECTED(ERM, Succeeded());
  CommaIntervals = std::move(*ERM);

  EXPECT_EQ(ColonIntervals.size(), CommaIntervals.size());
  for (size_t I = 0; I < ColonIntervals.size(); ++I) {
    EXPECT_EQ(ColonIntervals[I].getBegin(), CommaIntervals[I].getBegin());
    EXPECT_EQ(ColonIntervals[I].getEnd(), CommaIntervals[I].getEnd());
  }

  // Test that both work with contains().
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(ColonIntervals, 3));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(CommaIntervals, 3));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(ColonIntervals, 10));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(CommaIntervals, 10));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(ColonIntervals, 18));
  EXPECT_TRUE(IntegerInclusiveIntervalUtils::contains(CommaIntervals, 18));

  EXPECT_FALSE(IntegerInclusiveIntervalUtils::contains(ColonIntervals, 8));
  EXPECT_FALSE(IntegerInclusiveIntervalUtils::contains(CommaIntervals, 8));
}

TEST(IntegerInclusiveIntervalUtilsTest, DefaultCommaSeparator) {
  // Test that comma is the default separator.
  auto ER = IntegerInclusiveIntervalUtils::parseIntervals("1-5,10,15-20");
  ASSERT_THAT_EXPECTED(ER, Succeeded());
  auto Intervals = std::move(*ER);
  EXPECT_EQ(Intervals.size(), 3U);
  EXPECT_EQ(Intervals[0].getBegin(), 1);
  EXPECT_EQ(Intervals[0].getEnd(), 5);
  EXPECT_EQ(Intervals[1].getBegin(), 10);
  EXPECT_EQ(Intervals[1].getEnd(), 10);
  EXPECT_EQ(Intervals[2].getBegin(), 15);
  EXPECT_EQ(Intervals[2].getEnd(), 20);
}

TEST(IntegerInclusiveIntervalTest, MergeAdjacentIntervals) {
  IntegerInclusiveIntervalUtils::IntervalList Input, Expected, Result;

  // Empty input
  Result = IntegerInclusiveIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_TRUE(Result.empty());

  // Single interval - no change.
  Input.push_back(IntegerInclusiveInterval(5, 10));
  Expected.push_back(IntegerInclusiveInterval(5, 10));
  Result = IntegerInclusiveIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);

  // Adjacent intervals should merge.
  Input.clear();
  Expected.clear();
  Input.push_back(IntegerInclusiveInterval(1, 3));
  Input.push_back(IntegerInclusiveInterval(4, 6));
  Input.push_back(IntegerInclusiveInterval(7, 9));
  Expected.push_back(IntegerInclusiveInterval(1, 9));
  Result = IntegerInclusiveIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);

  // Non-adjacent intervals should not merge.
  Input.clear();
  Expected.clear();
  Input.push_back(IntegerInclusiveInterval(1, 3));
  Input.push_back(IntegerInclusiveInterval(5, 7));   // Gap between 3 and 5.
  Input.push_back(IntegerInclusiveInterval(10, 12)); // Gap between 7 and 10.
  Expected.push_back(IntegerInclusiveInterval(1, 3));
  Expected.push_back(IntegerInclusiveInterval(5, 7));
  Expected.push_back(IntegerInclusiveInterval(10, 12));
  Result = IntegerInclusiveIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);

  // Mixed adjacent and non-adjacent intervals.
  Input.clear();
  Expected.clear();
  Input.push_back(IntegerInclusiveInterval(1, 3));
  Input.push_back(IntegerInclusiveInterval(4, 6));    // Adjacent to first.
  Input.push_back(IntegerInclusiveInterval(8, 10));   // Gap.
  Input.push_back(IntegerInclusiveInterval(11, 13));  // Adjacent to third.
  Input.push_back(IntegerInclusiveInterval(14, 16));  // Adjacent to fourth.
  Expected.push_back(IntegerInclusiveInterval(1, 6)); // Merged 1-3 and 4-6.
  Expected.push_back(
      IntegerInclusiveInterval(8, 16)); // Merged 8-10, 11-13, 14-16.
  Result = IntegerInclusiveIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);

  // Single numbers that are adjacent.
  Input.clear();
  Expected.clear();
  Input.push_back(IntegerInclusiveInterval(5));
  Input.push_back(IntegerInclusiveInterval(6));
  Input.push_back(IntegerInclusiveInterval(7));
  Expected.push_back(IntegerInclusiveInterval(5, 7));
  Result = IntegerInclusiveIntervalUtils::mergeAdjacentIntervals(Input);
  EXPECT_EQ(Expected, Result);
}
