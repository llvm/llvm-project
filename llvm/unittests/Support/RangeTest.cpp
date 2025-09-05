//===- llvm/unittests/Support/RangeTest.cpp - Range tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Range.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(RangeTest, BasicRange) {
  Range R(5, 10);
  EXPECT_EQ(R.Begin, 5);
  EXPECT_EQ(R.End, 10);
  EXPECT_TRUE(R.contains(5));
  EXPECT_TRUE(R.contains(7));
  EXPECT_TRUE(R.contains(10));
  EXPECT_FALSE(R.contains(4));
  EXPECT_FALSE(R.contains(11));
}

TEST(RangeTest, SingleValueRange) {
  Range R(42);
  EXPECT_EQ(R.Begin, 42);
  EXPECT_EQ(R.End, 42);
  EXPECT_TRUE(R.contains(42));
  EXPECT_FALSE(R.contains(41));
  EXPECT_FALSE(R.contains(43));
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
  RangeUtils::RangeList Ranges;
  EXPECT_TRUE(RangeUtils::parseRanges("42", Ranges));
  EXPECT_EQ(Ranges.size(), 1U);
  EXPECT_EQ(Ranges[0].Begin, 42);
  EXPECT_EQ(Ranges[0].End, 42);
}

TEST(RangeUtilsTest, ParseSingleRange) {
  RangeUtils::RangeList Ranges;
  EXPECT_TRUE(RangeUtils::parseRanges("10-20", Ranges));
  EXPECT_EQ(Ranges.size(), 1U);
  EXPECT_EQ(Ranges[0].Begin, 10);
  EXPECT_EQ(Ranges[0].End, 20);
}

TEST(RangeUtilsTest, ParseMultipleRanges) {
  RangeUtils::RangeList Ranges;
  EXPECT_TRUE(RangeUtils::parseRanges("1-5,10,15-20", Ranges));
  EXPECT_EQ(Ranges.size(), 3U);

  // Ranges are in input order (DebugCounter style)
  EXPECT_EQ(Ranges[0].Begin, 1);
  EXPECT_EQ(Ranges[0].End, 5);
  EXPECT_EQ(Ranges[1].Begin, 10);
  EXPECT_EQ(Ranges[1].End, 10);
  EXPECT_EQ(Ranges[2].Begin, 15);
  EXPECT_EQ(Ranges[2].End, 20);
}

TEST(RangeUtilsTest, ParseColonSeparated) {
  RangeUtils::RangeList Ranges;
  EXPECT_TRUE(RangeUtils::parseRanges("1-5:10:15-20", Ranges, ':'));
  EXPECT_EQ(Ranges.size(), 3U);
  EXPECT_EQ(Ranges[0].Begin, 1);
  EXPECT_EQ(Ranges[0].End, 5);
  EXPECT_EQ(Ranges[1].Begin, 10);
  EXPECT_EQ(Ranges[1].End, 10);
  EXPECT_EQ(Ranges[2].Begin, 15);
  EXPECT_EQ(Ranges[2].End, 20);
}

TEST(RangeUtilsTest, ParseEmptyString) {
  RangeUtils::RangeList Ranges;
  EXPECT_TRUE(RangeUtils::parseRanges("", Ranges));
  EXPECT_TRUE(Ranges.empty());
}

TEST(RangeUtilsTest, ParseInvalidRanges) {
  RangeUtils::RangeList Ranges;

  // Invalid number
  EXPECT_FALSE(RangeUtils::parseRanges("abc", Ranges));

  // Invalid range (begin > end)
  EXPECT_FALSE(RangeUtils::parseRanges("10-5", Ranges));

  // Out of order ranges (DebugCounter constraint)
  EXPECT_FALSE(RangeUtils::parseRanges("10,5", Ranges));
  EXPECT_FALSE(
      RangeUtils::parseRanges("1-5,3-7", Ranges)); // Overlapping not allowed
}

TEST(RangeUtilsTest, Contains) {
  RangeUtils::RangeList Ranges;
  EXPECT_TRUE(RangeUtils::parseRanges("1-5,10,15-20", Ranges));

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

  // Test explicit separator parameters
  EXPECT_TRUE(RangeUtils::parseRanges("1-5:10:15-20", ColonRanges, ':'));
  EXPECT_TRUE(RangeUtils::parseRanges("1-5,10,15-20", CommaRanges, ','));

  EXPECT_EQ(ColonRanges.size(), CommaRanges.size());
  for (size_t I = 0; I < ColonRanges.size(); ++I) {
    EXPECT_EQ(ColonRanges[I].Begin, CommaRanges[I].Begin);
    EXPECT_EQ(ColonRanges[I].End, CommaRanges[I].End);
  }

  // Test that both work with contains()
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
  RangeUtils::RangeList Ranges;

  // Test that comma is the default separator
  EXPECT_TRUE(RangeUtils::parseRanges("1-5,10,15-20", Ranges));
  EXPECT_EQ(Ranges.size(), 3U);
  EXPECT_EQ(Ranges[0].Begin, 1);
  EXPECT_EQ(Ranges[0].End, 5);
  EXPECT_EQ(Ranges[1].Begin, 10);
  EXPECT_EQ(Ranges[1].End, 10);
  EXPECT_EQ(Ranges[2].Begin, 15);
  EXPECT_EQ(Ranges[2].End, 20);
}

TEST(RangeTest, MergeAdjacentRanges) {
  RangeUtils::RangeList Input, Expected, Result;

  // Empty input
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_TRUE(Result.empty());

  // Single range - no change
  Input.push_back(Range(5, 10));
  Expected.push_back(Range(5, 10));
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);

  // Adjacent ranges should merge
  Input.clear();
  Expected.clear();
  Input.push_back(Range(1, 3));
  Input.push_back(Range(4, 6));
  Input.push_back(Range(7, 9));
  Expected.push_back(Range(1, 9));
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);

  // Non-adjacent ranges should not merge
  Input.clear();
  Expected.clear();
  Input.push_back(Range(1, 3));
  Input.push_back(Range(5, 7));   // Gap between 3 and 5
  Input.push_back(Range(10, 12)); // Gap between 7 and 10
  Expected.push_back(Range(1, 3));
  Expected.push_back(Range(5, 7));
  Expected.push_back(Range(10, 12));
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);

  // Mixed adjacent and non-adjacent
  Input.clear();
  Expected.clear();
  Input.push_back(Range(1, 3));
  Input.push_back(Range(4, 6));     // Adjacent to first
  Input.push_back(Range(8, 10));    // Gap
  Input.push_back(Range(11, 13));   // Adjacent to third
  Input.push_back(Range(14, 16));   // Adjacent to fourth
  Expected.push_back(Range(1, 6));  // Merged 1-3 and 4-6
  Expected.push_back(Range(8, 16)); // Merged 8-10, 11-13, 14-16
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);

  // Single numbers that are adjacent
  Input.clear();
  Expected.clear();
  Input.push_back(Range(5));
  Input.push_back(Range(6));
  Input.push_back(Range(7));
  Expected.push_back(Range(5, 7));
  Result = RangeUtils::mergeAdjacentRanges(Input);
  EXPECT_EQ(Expected, Result);
}
