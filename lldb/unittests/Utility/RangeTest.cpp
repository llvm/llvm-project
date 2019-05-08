//===-- RangeTest.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RangeMap.h"
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(RangeTest, SizeTypes) {
  Range<lldb::addr_t, uint32_t> r;
  static_assert(std::is_same<lldb::addr_t, decltype(r.GetRangeBase())>::value,
                "RangeBase type is not equal to the given one.");
  static_assert(std::is_same<lldb::addr_t, decltype(r.GetRangeEnd())>::value,
                "RangeEnd type is not equal to the given one.");
  static_assert(std::is_same<uint32_t, decltype(r.GetByteSize())>::value,
                "Size type is not equal to the given one.");
}

typedef Range<lldb::addr_t, uint32_t> RangeT;

TEST(RangeTest, DefaultConstructor) {
  RangeT r;
  EXPECT_FALSE(r.IsValid());
  EXPECT_EQ(0U, r.GetByteSize());
  EXPECT_EQ(0U, r.GetRangeBase());
  EXPECT_EQ(0U, r.GetRangeEnd());
}

TEST(RangeTest, Constructor) {
  RangeT r(3, 5);
  EXPECT_TRUE(r.IsValid());
  EXPECT_EQ(5U, r.GetByteSize());
  EXPECT_EQ(3U, r.GetRangeBase());
  EXPECT_EQ(8U, r.GetRangeEnd());
}

TEST(RangeTest, Copy) {
  RangeT orig(3, 5);
  RangeT r = orig;
  EXPECT_TRUE(r.IsValid());
  EXPECT_EQ(5U, r.GetByteSize());
  EXPECT_EQ(3U, r.GetRangeBase());
  EXPECT_EQ(8U, r.GetRangeEnd());
}

TEST(RangeTest, Clear) {
  RangeT r(3, 5);
  r.Clear();
  EXPECT_TRUE(r == RangeT());
}

TEST(RangeTest, ClearWithStarAddress) {
  RangeT r(3, 5);
  r.Clear(4);
  EXPECT_TRUE(r == RangeT(4, 0));
}

TEST(RangeTest, SetRangeBase) {
  RangeT r(3, 5);
  r.SetRangeBase(6);
  EXPECT_EQ(6U, r.GetRangeBase());
  EXPECT_EQ(11U, r.GetRangeEnd());
  EXPECT_EQ(5U, r.GetByteSize());
}

TEST(RangeTest, Slide) {
  RangeT r(3, 5);
  r.Slide(1);
  EXPECT_EQ(4U, r.GetRangeBase());
  EXPECT_EQ(9U, r.GetRangeEnd());
  EXPECT_EQ(5U, r.GetByteSize());

  r.Slide(2);
  EXPECT_EQ(6U, r.GetRangeBase());
  EXPECT_EQ(11U, r.GetRangeEnd());
  EXPECT_EQ(5U, r.GetByteSize());
}

TEST(RangeTest, SlideZero) {
  RangeT r(3, 5);
  r.Slide(0);
  EXPECT_EQ(3U, r.GetRangeBase());
  EXPECT_EQ(8U, r.GetRangeEnd());
  EXPECT_EQ(5U, r.GetByteSize());
}

TEST(RangeTest, ContainsAddr) {
  RangeT r(3, 5);
  EXPECT_FALSE(r.Contains(0));
  EXPECT_FALSE(r.Contains(1));
  EXPECT_FALSE(r.Contains(2));
  EXPECT_TRUE(r.Contains(3));
  EXPECT_TRUE(r.Contains(4));
  EXPECT_TRUE(r.Contains(5));
  EXPECT_TRUE(r.Contains(6));
  EXPECT_TRUE(r.Contains(7));
  EXPECT_FALSE(r.Contains(8));
  EXPECT_FALSE(r.Contains(9));
  EXPECT_FALSE(r.Contains(10));
}

TEST(RangeTest, ContainsAddrInvalid) {
  RangeT r;
  EXPECT_FALSE(r.Contains(0));
  EXPECT_FALSE(r.Contains(1));
  EXPECT_FALSE(r.Contains(2));
  EXPECT_FALSE(r.Contains(3));
  EXPECT_FALSE(r.Contains(4));
}

TEST(RangeTest, ContainsEndInclusive) {
  RangeT r(3, 5);
  EXPECT_FALSE(r.ContainsEndInclusive(0));
  EXPECT_FALSE(r.ContainsEndInclusive(1));
  EXPECT_FALSE(r.ContainsEndInclusive(2));
  EXPECT_TRUE(r.ContainsEndInclusive(3));
  EXPECT_TRUE(r.ContainsEndInclusive(4));
  EXPECT_TRUE(r.ContainsEndInclusive(5));
  EXPECT_TRUE(r.ContainsEndInclusive(6));
  EXPECT_TRUE(r.ContainsEndInclusive(7));
  EXPECT_TRUE(r.ContainsEndInclusive(8));
  EXPECT_FALSE(r.ContainsEndInclusive(9));
  EXPECT_FALSE(r.ContainsEndInclusive(10));
}

TEST(RangeTest, ContainsEndInclusiveInvalid) {
  RangeT r;
  // FIXME: This is probably not intended.
  EXPECT_TRUE(r.ContainsEndInclusive(0));

  EXPECT_FALSE(r.ContainsEndInclusive(1));
  EXPECT_FALSE(r.ContainsEndInclusive(2));
}

TEST(RangeTest, ContainsRange) {
  RangeT r(3, 5);

  // Range always contains itself.
  EXPECT_TRUE(r.Contains(r));
  // Invalid range.
  EXPECT_FALSE(r.Contains(RangeT()));
  // Range starts and ends before.
  EXPECT_FALSE(r.Contains(RangeT(0, 3)));
  // Range starts before but contains beginning.
  EXPECT_FALSE(r.Contains(RangeT(0, 4)));
  // Range starts before but contains beginning and more.
  EXPECT_FALSE(r.Contains(RangeT(0, 5)));
  // Range starts before and contains the other.
  EXPECT_FALSE(r.Contains(RangeT(0, 9)));
  // Range is fully inside.
  EXPECT_TRUE(r.Contains(RangeT(4, 3)));
  // Range has same start, but not as large.
  EXPECT_TRUE(r.Contains(RangeT(3, 4)));
  // Range has same end, but starts earlier.
  EXPECT_TRUE(r.Contains(RangeT(4, 4)));
  // Range starts inside, but stops after the end of r.
  EXPECT_FALSE(r.Contains(RangeT(4, 5)));
  // Range starts directly after r.
  EXPECT_FALSE(r.Contains(RangeT(8, 2)));
  // Range starts directly after r.
  EXPECT_FALSE(r.Contains(RangeT(9, 2)));

  // Invalid range with different start.
  // FIXME: The first two probably not intended.
  EXPECT_TRUE(r.Contains(RangeT(3, 0)));
  EXPECT_TRUE(r.Contains(RangeT(4, 0)));
  EXPECT_FALSE(r.Contains(RangeT(8, 0)));
}

TEST(RangeTest, ContainsRangeStartingFromZero) {
  RangeT r(0, 3);
  EXPECT_TRUE(r.Contains(r));

  // FIXME: This is probably not intended.
  EXPECT_TRUE(r.Contains(RangeT()));
}

TEST(RangeTest, Union) {
  RangeT r(3, 5);

  // Ranges that we can't merge because it's not adjoin/intersecting.
  EXPECT_FALSE(r.Union(RangeT(9, 1)));
  // Check that we didn't modify our range.
  EXPECT_EQ(r, RangeT(3, 5));

  // Another range we can't merge, but before r.
  EXPECT_FALSE(r.Union(RangeT(1, 1)));
  EXPECT_EQ(r, RangeT(3, 5));

  // Merge an adjoin range after.
  EXPECT_TRUE(r.Union(RangeT(8, 2)));
  EXPECT_EQ(r, RangeT(3, 7));

  // Merge an adjoin range before.
  EXPECT_TRUE(r.Union(RangeT(1, 2)));
  EXPECT_EQ(r, RangeT(1, 9));

  // Merge an intersecting range after.
  EXPECT_TRUE(r.Union(RangeT(8, 3)));
  EXPECT_EQ(r, RangeT(1, 10));

  // Merge an intersecting range before.
  EXPECT_TRUE(r.Union(RangeT(0, 1)));
  EXPECT_EQ(r, RangeT(0, 11));

  // Merge a few ranges inside that shouldn't do anything.
  EXPECT_TRUE(r.Union(RangeT(0, 3)));
  EXPECT_EQ(r, RangeT(0, 11));
  EXPECT_TRUE(r.Union(RangeT(5, 1)));
  EXPECT_EQ(r, RangeT(0, 11));
  EXPECT_TRUE(r.Union(RangeT(9, 2)));
  EXPECT_EQ(r, RangeT(0, 11));
}

TEST(RangeTest, DoesAdjoinOrIntersect) {
  RangeT r(3, 4);

  EXPECT_FALSE(r.DoesAdjoinOrIntersect(RangeT(1, 1)));
  EXPECT_TRUE(r.DoesAdjoinOrIntersect(RangeT(1, 2)));
  EXPECT_TRUE(r.DoesAdjoinOrIntersect(RangeT(2, 2)));
  EXPECT_TRUE(r.DoesAdjoinOrIntersect(RangeT(4, 2)));
  EXPECT_TRUE(r.DoesAdjoinOrIntersect(RangeT(6, 2)));
  EXPECT_TRUE(r.DoesAdjoinOrIntersect(RangeT(7, 2)));
  EXPECT_FALSE(r.DoesAdjoinOrIntersect(RangeT(8, 2)));
}

TEST(RangeTest, DoesIntersect) {
  RangeT r(3, 4);

  EXPECT_FALSE(r.DoesIntersect(RangeT(1, 1)));
  EXPECT_FALSE(r.DoesIntersect(RangeT(1, 2)));
  EXPECT_TRUE(r.DoesIntersect(RangeT(2, 2)));
  EXPECT_TRUE(r.DoesIntersect(RangeT(4, 2)));
  EXPECT_TRUE(r.DoesIntersect(RangeT(6, 2)));
  EXPECT_FALSE(r.DoesIntersect(RangeT(7, 2)));
  EXPECT_FALSE(r.DoesIntersect(RangeT(8, 2)));
}

TEST(RangeTest, LessThan) {
  RangeT r(10, 20);

  // Equal range.
  EXPECT_FALSE(r < RangeT(10, 20));
  EXPECT_FALSE(RangeT(10, 20) < r);

  auto expect_ordered_less_than = [](RangeT r1, RangeT r2) {
    EXPECT_TRUE(r1 < r2);
    EXPECT_FALSE(r2 < r1);
  };

  // Same start, but bigger size.
  expect_ordered_less_than(r, RangeT(10, 21));

  // Start before and ends before.
  expect_ordered_less_than(RangeT(9, 20), r);

  // Start before and equal size.
  expect_ordered_less_than(RangeT(9, 21), r);

  // Start before and bigger size.
  expect_ordered_less_than(RangeT(9, 22), r);

  // Start after and ends before.
  expect_ordered_less_than(r, RangeT(11, 18));

  // Start after and equal size.
  expect_ordered_less_than(r, RangeT(11, 19));

  // Start after and bigger size.
  expect_ordered_less_than(r, RangeT(11, 20));
}

TEST(RangeTest, Equal) {
  RangeT r(10, 20);

  // Equal range.
  EXPECT_TRUE(r == RangeT(10, 20));

  // Same start, different size.
  EXPECT_FALSE(r == RangeT(10, 21));

  // Different start, same size.
  EXPECT_FALSE(r == RangeT(9, 20));

  // Different start, different size.
  EXPECT_FALSE(r == RangeT(9, 21));
  EXPECT_FALSE(r == RangeT(11, 19));
}

TEST(RangeTest, NotEqual) {
  RangeT r(10, 20);

  EXPECT_FALSE(r != RangeT(10, 20));

  EXPECT_TRUE(r != RangeT(10, 21));
  EXPECT_TRUE(r != RangeT(9, 20));
  EXPECT_TRUE(r != RangeT(9, 21));
}

// Comparison tests for invalid ranges (size == 0).

TEST(RangeTest, LessThanInvalid) {
  EXPECT_TRUE(RangeT() < RangeT(1, 0));
  EXPECT_TRUE(RangeT() < RangeT(2, 0));
  EXPECT_TRUE(RangeT(1, 0) < RangeT(2, 0));
}

TEST(RangeTest, EqualInvalid) {
  RangeT r;
  EXPECT_TRUE(r == RangeT());
  // Another invalid range, but with a different start.
  EXPECT_FALSE(r == RangeT(3, 0));
}

TEST(RangeTest, NotEqualInvalid) {
  RangeT r;
  EXPECT_FALSE(r != RangeT());
  EXPECT_FALSE(r == RangeT(3, 0));
}
