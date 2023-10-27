//===- llvm/unittest/Support/AddresRangeTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/AddressRanges.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

TEST(AddressRangeTest, TestRanges) {
  // test llvm::AddressRange.
  const uint64_t StartAddr = 0x1000;
  const uint64_t EndAddr = 0x2000;
  // Verify constructor and API to ensure it takes start and end address.
  const AddressRange Range(StartAddr, EndAddr);
  EXPECT_EQ(Range.size(), EndAddr - StartAddr);

  // Verify llvm::AddressRange::contains().
  EXPECT_FALSE(Range.contains(0));
  EXPECT_FALSE(Range.contains(StartAddr - 1));
  EXPECT_TRUE(Range.contains(StartAddr));
  EXPECT_TRUE(Range.contains(EndAddr - 1));
  EXPECT_FALSE(Range.contains(EndAddr));
  EXPECT_FALSE(Range.contains(UINT64_MAX));

  const AddressRange RangeSame(StartAddr, EndAddr);
  const AddressRange RangeDifferentStart(StartAddr + 1, EndAddr);
  const AddressRange RangeDifferentEnd(StartAddr, EndAddr + 1);
  const AddressRange RangeDifferentStartEnd(StartAddr + 1, EndAddr + 1);
  // Test == and != with values that are the same
  EXPECT_EQ(Range, RangeSame);
  EXPECT_FALSE(Range != RangeSame);
  // Test == and != with values that are the different
  EXPECT_NE(Range, RangeDifferentStart);
  EXPECT_NE(Range, RangeDifferentEnd);
  EXPECT_NE(Range, RangeDifferentStartEnd);
  EXPECT_FALSE(Range == RangeDifferentStart);
  EXPECT_FALSE(Range == RangeDifferentEnd);
  EXPECT_FALSE(Range == RangeDifferentStartEnd);

  // Test "bool operator<(const AddressRange &, const AddressRange &)".
  EXPECT_FALSE(Range < RangeSame);
  EXPECT_FALSE(RangeSame < Range);
  EXPECT_LT(Range, RangeDifferentStart);
  EXPECT_LT(Range, RangeDifferentEnd);
  EXPECT_LT(Range, RangeDifferentStartEnd);
  // Test "bool operator<(const AddressRange &, uint64_t)"
  EXPECT_LT(Range.start(), StartAddr + 1);
  // Test "bool operator<(uint64_t, const AddressRange &)"
  EXPECT_LT(StartAddr - 1, Range.start());

  // Verify llvm::AddressRange::isContiguousWith() and
  // llvm::AddressRange::intersects().
  const AddressRange EndsBeforeRangeStart(0, StartAddr - 1);
  const AddressRange EndsAtRangeStart(0, StartAddr);
  const AddressRange OverlapsRangeStart(StartAddr - 1, StartAddr + 1);
  const AddressRange InsideRange(StartAddr + 1, EndAddr - 1);
  const AddressRange OverlapsRangeEnd(EndAddr - 1, EndAddr + 1);
  const AddressRange StartsAtRangeEnd(EndAddr, EndAddr + 0x100);
  const AddressRange StartsAfterRangeEnd(EndAddr + 1, EndAddr + 0x100);

  EXPECT_FALSE(Range.intersects(EndsBeforeRangeStart));
  EXPECT_FALSE(Range.intersects(EndsAtRangeStart));
  EXPECT_TRUE(Range.intersects(OverlapsRangeStart));
  EXPECT_TRUE(Range.intersects(InsideRange));
  EXPECT_TRUE(Range.intersects(OverlapsRangeEnd));
  EXPECT_FALSE(Range.intersects(StartsAtRangeEnd));
  EXPECT_FALSE(Range.intersects(StartsAfterRangeEnd));

  // Test the functions that maintain address ranges:
  //  "bool AddressRange::contains(uint64_t Addr) const;"
  //  "void AddressRanges::insert(const AddressRange &R);"
  AddressRanges Ranges;
  Ranges.insert(AddressRange(0x1000, 0x2000));
  Ranges.insert(AddressRange(0x2000, 0x3000));
  Ranges.insert(AddressRange(0x4000, 0x5000));

  EXPECT_FALSE(Ranges.contains(0));
  EXPECT_FALSE(Ranges.contains(0x1000 - 1));
  EXPECT_TRUE(Ranges.contains(0x1000));
  EXPECT_TRUE(Ranges.contains(0x2000));
  EXPECT_TRUE(Ranges.contains(0x4000));
  EXPECT_TRUE(Ranges.contains(0x2000 - 1));
  EXPECT_TRUE(Ranges.contains(0x3000 - 1));
  EXPECT_FALSE(Ranges.contains(0x3000 + 1));
  EXPECT_TRUE(Ranges.contains(0x5000 - 1));
  EXPECT_FALSE(Ranges.contains(0x5000 + 1));
  EXPECT_FALSE(Ranges.contains(UINT64_MAX));

  EXPECT_FALSE(Ranges.contains(AddressRange()));
  EXPECT_FALSE(Ranges.contains(AddressRange(0x1000 - 1, 0x1000)));
  EXPECT_FALSE(Ranges.contains(AddressRange(0x1000, 0x1000)));
  EXPECT_TRUE(Ranges.contains(AddressRange(0x1000, 0x1000 + 1)));
  EXPECT_TRUE(Ranges.contains(AddressRange(0x1000, 0x2000)));
  EXPECT_TRUE(Ranges.contains(AddressRange(0x1000, 0x2001)));
  EXPECT_TRUE(Ranges.contains(AddressRange(0x2000, 0x3000)));
  EXPECT_FALSE(Ranges.contains(AddressRange(0x2000, 0x3001)));
  EXPECT_FALSE(Ranges.contains(AddressRange(0x3000, 0x3001)));
  EXPECT_FALSE(Ranges.contains(AddressRange(0x1500, 0x4500)));
  EXPECT_FALSE(Ranges.contains(AddressRange(0x5000, 0x5001)));

  // Verify that intersecting ranges get combined
  Ranges.clear();
  Ranges.insert(AddressRange(0x1100, 0x1F00));
  // Verify a wholy contained range that is added doesn't do anything.
  Ranges.insert(AddressRange(0x1500, 0x1F00));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1100, 0x1F00));

  // Verify a range that starts before and intersects gets combined.
  Ranges.insert(AddressRange(0x1000, Ranges[0].start() + 1));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x1F00));

  // Verify a range that starts inside and extends ranges gets combined.
  Ranges.insert(AddressRange(Ranges[0].end() - 1, 0x2000));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x2000));

  // Verify that adjacent ranges get combined
  Ranges.insert(AddressRange(0x2000, 0x2fff));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x2fff));

  // Verify that ranges having 1 byte gap do not get combined
  Ranges.insert(AddressRange(0x3000, 0x4000));
  EXPECT_EQ(Ranges.size(), 2u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x2fff));
  EXPECT_EQ(Ranges[1], AddressRange(0x3000, 0x4000));

  // Verify if we add an address range that intersects two ranges
  // that they get combined
  Ranges.insert(AddressRange(Ranges[0].end() - 1, Ranges[1].start() + 1));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x4000));

  Ranges.insert(AddressRange(0x3000, 0x4000));
  Ranges.insert(AddressRange(0x4000, 0x5000));
  Ranges.insert(AddressRange(0x2000, 0x4500));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0], AddressRange(0x1000, 0x5000));
}

TEST(AddressRangeTest, TestRangesRandom) {
  AddressRanges Ranges;
  size_t NumElements = 100;

  std::srand(std::time(nullptr));

  // Fill ranges.
  for (size_t Idx = 0; Idx < NumElements; Idx++) {
    uint64_t Start = static_cast<uint64_t>(std::rand() % 1000);
    uint64_t End = Start + static_cast<uint64_t>(std::rand() % 1000);
    Ranges.insert({Start, End});
  }

  // Check ranges.
  for (size_t Idx = 0; Idx + 1 < Ranges.size(); Idx++) {
    // Check that ranges are not intersected.
    EXPECT_FALSE(Ranges[Idx].intersects(Ranges[Idx + 1]));

    // Check that ranges are sorted and not adjusted.
    EXPECT_TRUE(Ranges[Idx].end() < Ranges[Idx + 1].start());
  }
}

TEST(AddressRangeTest, TestRangesMap) {
  AddressRangesMap Ranges;

  EXPECT_EQ(Ranges.size(), 0u);
  EXPECT_TRUE(Ranges.empty());

  // Add single range.
  Ranges.insert(AddressRange(0x1000, 0x2000), 0xfe);
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_FALSE(Ranges.empty());
  EXPECT_TRUE(Ranges.contains(0x1500));
  EXPECT_TRUE(Ranges.contains(AddressRange(0x1000, 0x2000)));

  ///////////////////////////////////////
  /// Check ranges with the same mapped value.

  // Clear ranges.
  Ranges.clear();
  EXPECT_EQ(Ranges.size(), 0u);
  EXPECT_TRUE(Ranges.empty());

  // Add range and check mapped value.
  Ranges.insert(AddressRange(0x1000, 0x2000), 0x11);
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0x11);

  // Add adjacent range and check mapped value.
  Ranges.insert(AddressRange(0x2000, 0x3000), 0x11);
  EXPECT_EQ(Ranges.size(), 2u);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0x11);
  EXPECT_EQ(Ranges.getRangeThatContains(0x2000)->Value, 0x11);
  EXPECT_EQ(Ranges.getRangeThatContains(0x2900)->Value, 0x11);
  EXPECT_FALSE(Ranges.getRangeThatContains(0x3000));

  // Add intersecting range and check mapped value.
  Ranges.insert(AddressRange(0x1000, 0x3000), 0x11);
  EXPECT_EQ(Ranges.size(), 2u);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0x11);

  // Add second range and check mapped values.
  Ranges.insert(AddressRange(0x4000, 0x5000), 0x11);
  EXPECT_EQ(Ranges.size(), 3u);
  EXPECT_EQ(Ranges[0].Range, AddressRange(0x1000, 0x2000));
  EXPECT_EQ(Ranges[0].Value, 0x11);
  EXPECT_EQ(Ranges[1].Range, AddressRange(0x2000, 0x3000));
  EXPECT_EQ(Ranges[1].Value, 0x11);
  EXPECT_EQ(Ranges[2].Range, AddressRange(0x4000, 0x5000));
  EXPECT_EQ(Ranges[2].Value, 0x11);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0x11);
  EXPECT_EQ(Ranges.getRangeThatContains(0x4000)->Value, 0x11);

  // Add intersecting range and check mapped value.
  Ranges.insert(AddressRange(0x0, 0x6000), 0x11);
  EXPECT_EQ(Ranges.size(), 6u);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0x11);

  // Check that mapped values are correctly preserved for combined ranges.
  Ranges.clear();
  Ranges.insert(AddressRange(0x0, 0xff), 0x11);
  Ranges.insert(AddressRange(0x100, 0x1ff), 0x11);
  Ranges.insert(AddressRange(0x200, 0x2ff), 0x11);
  Ranges.insert(AddressRange(0x500, 0x5ff), 0x11);
  Ranges.insert(AddressRange(0x300, 0x3ff), 0x11);
  Ranges.insert(AddressRange(0x400, 0x4ff), 0x11);
  Ranges.insert(AddressRange(0x600, 0x6ff), 0x11);
  EXPECT_EQ(Ranges.size(), 7u);

  Ranges.insert(AddressRange(0x150, 0x350), 0x11);
  EXPECT_EQ(Ranges.size(), 9u);
  EXPECT_EQ(Ranges[0].Range, AddressRange(0x0, 0xff));
  EXPECT_EQ(Ranges[0].Value, 0x11);
  EXPECT_EQ(Ranges[1].Range, AddressRange(0x100, 0x1ff));
  EXPECT_EQ(Ranges[1].Value, 0x11);
  EXPECT_EQ(Ranges[2].Range, AddressRange(0x1ff, 0x200));
  EXPECT_EQ(Ranges[2].Value, 0x11);
  EXPECT_EQ(Ranges[3].Range, AddressRange(0x200, 0x2ff));
  EXPECT_EQ(Ranges[3].Value, 0x11);
  EXPECT_EQ(Ranges[4].Range, AddressRange(0x2ff, 0x300));
  EXPECT_EQ(Ranges[4].Value, 0x11);
  EXPECT_EQ(Ranges[5].Range, AddressRange(0x300, 0x3ff));
  EXPECT_EQ(Ranges[5].Value, 0x11);
  EXPECT_EQ(Ranges[6].Range, AddressRange(0x400, 0x4ff));
  EXPECT_EQ(Ranges[6].Value, 0x11);
  EXPECT_EQ(Ranges[7].Range, AddressRange(0x500, 0x5ff));
  EXPECT_EQ(Ranges[7].Value, 0x11);
  EXPECT_EQ(Ranges[8].Range, AddressRange(0x600, 0x6ff));
  EXPECT_EQ(Ranges[8].Value, 0x11);

  Ranges.insert(AddressRange(0x3ff, 0x400), 0x11);
  EXPECT_EQ(Ranges.size(), 10u);
  EXPECT_EQ(Ranges[0].Range, AddressRange(0x0, 0xff));
  EXPECT_EQ(Ranges[0].Value, 0x11);
  EXPECT_EQ(Ranges[1].Range, AddressRange(0x100, 0x1ff));
  EXPECT_EQ(Ranges[1].Value, 0x11);
  EXPECT_EQ(Ranges[2].Range, AddressRange(0x1ff, 0x200));
  EXPECT_EQ(Ranges[2].Value, 0x11);
  EXPECT_EQ(Ranges[3].Range, AddressRange(0x200, 0x2ff));
  EXPECT_EQ(Ranges[3].Value, 0x11);
  EXPECT_EQ(Ranges[4].Range, AddressRange(0x2ff, 0x300));
  EXPECT_EQ(Ranges[4].Value, 0x11);
  EXPECT_EQ(Ranges[5].Range, AddressRange(0x300, 0x3ff));
  EXPECT_EQ(Ranges[5].Value, 0x11);
  EXPECT_EQ(Ranges[6].Range, AddressRange(0x3ff, 0x400));
  EXPECT_EQ(Ranges[6].Value, 0x11);
  EXPECT_EQ(Ranges[7].Range, AddressRange(0x400, 0x4ff));
  EXPECT_EQ(Ranges[7].Value, 0x11);
  EXPECT_EQ(Ranges[8].Range, AddressRange(0x500, 0x5ff));
  EXPECT_EQ(Ranges[8].Value, 0x11);
  EXPECT_EQ(Ranges[9].Range, AddressRange(0x600, 0x6ff));
  EXPECT_EQ(Ranges[9].Value, 0x11);

  /////////////////////////////////////////////
  /// Check ranges with various mapped values.

  // Clear ranges.
  Ranges.clear();
  EXPECT_EQ(Ranges.size(), 0u);
  EXPECT_TRUE(Ranges.empty());

  // Add range and check mapped value.
  Ranges.insert(AddressRange(0x1000, 0x2000), 0xfe);
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0xfe);

  // Add adjacent range and check mapped value.
  Ranges.insert(AddressRange(0x2000, 0x3000), 0xfc);
  EXPECT_EQ(Ranges.size(), 2u);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0xfe);
  EXPECT_EQ(Ranges.getRangeThatContains(0x2000)->Value, 0xfc);
  EXPECT_EQ(Ranges.getRangeThatContains(0x2900)->Value, 0xfc);
  EXPECT_FALSE(Ranges.getRangeThatContains(0x3000));

  // Add intersecting range and check mapped value.
  Ranges.insert(AddressRange(0x1000, 0x3000), 0xff);
  EXPECT_EQ(Ranges.size(), 2u);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0xfe);

  // Add one more range and check mapped values.
  Ranges.insert(AddressRange(0x4000, 0x5000), 0x0);
  EXPECT_EQ(Ranges.size(), 3u);
  EXPECT_EQ(Ranges[0].Value, 0xfe);
  EXPECT_EQ(Ranges[1].Value, 0xfc);
  EXPECT_EQ(Ranges[2].Value, 0x0);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0xfe);
  EXPECT_EQ(Ranges.getRangeThatContains(0x4000)->Value, 0x0);

  // Add intersecting range and check mapped value.
  Ranges.insert(AddressRange(0x0, 0x6000), 0x1);
  EXPECT_EQ(Ranges.size(), 6u);
  EXPECT_EQ(Ranges[0].Value, 0x1);
  EXPECT_EQ(Ranges[1].Value, 0xfe);
  EXPECT_EQ(Ranges[2].Value, 0xfc);
  EXPECT_EQ(Ranges[3].Value, 0x1);
  EXPECT_EQ(Ranges[4].Value, 0x0);
  EXPECT_EQ(Ranges[5].Value, 0x1);
  EXPECT_EQ(Ranges.getRangeThatContains(0x1000)->Value, 0xfe);

  // Check that mapped values are correctly preserved for combined ranges.
  Ranges.clear();
  Ranges.insert(AddressRange(0x0, 0xff), 0x1);
  Ranges.insert(AddressRange(0x100, 0x1ff), 0x2);
  Ranges.insert(AddressRange(0x200, 0x2ff), 0x3);
  Ranges.insert(AddressRange(0x300, 0x3ff), 0x4);
  Ranges.insert(AddressRange(0x500, 0x5ff), 0x6);
  Ranges.insert(AddressRange(0x400, 0x4ff), 0x5);
  Ranges.insert(AddressRange(0x600, 0x6ff), 0x7);
  EXPECT_EQ(Ranges.size(), 7u);

  Ranges.insert(AddressRange(0x150, 0x350), 0xff);
  EXPECT_EQ(Ranges.size(), 9u);
  EXPECT_EQ(Ranges[0].Range, AddressRange(0x0, 0xff));
  EXPECT_EQ(Ranges[0].Value, 0x1);
  EXPECT_EQ(Ranges[1].Range, AddressRange(0x100, 0x1ff));
  EXPECT_EQ(Ranges[1].Value, 0x2);
  EXPECT_EQ(Ranges[2].Range, AddressRange(0x1ff, 0x200));
  EXPECT_EQ(Ranges[2].Value, 0xff);
  EXPECT_EQ(Ranges[3].Range, AddressRange(0x200, 0x2ff));
  EXPECT_EQ(Ranges[3].Value, 0x3);
  EXPECT_EQ(Ranges[4].Range, AddressRange(0x2ff, 0x300));
  EXPECT_EQ(Ranges[4].Value, 0xff);
  EXPECT_EQ(Ranges[5].Range, AddressRange(0x300, 0x3ff));
  EXPECT_EQ(Ranges[5].Value, 0x4);
  EXPECT_EQ(Ranges[6].Range, AddressRange(0x400, 0x4ff));
  EXPECT_EQ(Ranges[6].Value, 0x5);
  EXPECT_EQ(Ranges[7].Range, AddressRange(0x500, 0x5ff));
  EXPECT_EQ(Ranges[7].Value, 0x6);
  EXPECT_EQ(Ranges[8].Range, AddressRange(0x600, 0x6ff));
  EXPECT_EQ(Ranges[8].Value, 0x7);

  Ranges.insert(AddressRange(0x650, 0x700), 0x8);
  Ranges.insert(AddressRange(0x3ff, 0x400), 0x5);
  Ranges.insert(AddressRange(0x0, 0x40), 0xee);
  EXPECT_EQ(Ranges.size(), 11u);
  EXPECT_EQ(Ranges[0].Range, AddressRange(0x0, 0xff));
  EXPECT_EQ(Ranges[0].Value, 0x1);
  EXPECT_EQ(Ranges[1].Range, AddressRange(0x100, 0x1ff));
  EXPECT_EQ(Ranges[1].Value, 0x2);
  EXPECT_EQ(Ranges[2].Range, AddressRange(0x1ff, 0x200));
  EXPECT_EQ(Ranges[2].Value, 0xff);
  EXPECT_EQ(Ranges[3].Range, AddressRange(0x200, 0x2ff));
  EXPECT_EQ(Ranges[3].Value, 0x3);
  EXPECT_EQ(Ranges[4].Range, AddressRange(0x2ff, 0x300));
  EXPECT_EQ(Ranges[4].Value, 0xff);
  EXPECT_EQ(Ranges[5].Range, AddressRange(0x300, 0x3ff));
  EXPECT_EQ(Ranges[5].Value, 0x4);
  EXPECT_EQ(Ranges[6].Range, AddressRange(0x3ff, 0x400));
  EXPECT_EQ(Ranges[6].Value, 0x5);
  EXPECT_EQ(Ranges[7].Range, AddressRange(0x400, 0x4ff));
  EXPECT_EQ(Ranges[7].Value, 0x5);
  EXPECT_EQ(Ranges[8].Range, AddressRange(0x500, 0x5ff));
  EXPECT_EQ(Ranges[8].Value, 0x6);
  EXPECT_EQ(Ranges[9].Range, AddressRange(0x600, 0x6ff));
  EXPECT_EQ(Ranges[9].Value, 0x7);
  EXPECT_EQ(Ranges[10].Range, AddressRange(0x6ff, 0x700));
  EXPECT_EQ(Ranges[10].Value, 0x8);
}

TEST(AddressRangeTest, TestRangesMapRandom) {
  AddressRangesMap Ranges;
  size_t NumElements = 100;

  std::srand(std::time(nullptr));

  // Fill ranges. Use the same mapped value.
  for (size_t Idx = 0; Idx < NumElements; Idx++) {
    uint64_t Start = static_cast<uint64_t>(std::rand() % 1000);
    uint64_t End = Start + static_cast<uint64_t>(std::rand() % 1000);
    Ranges.insert({Start, End}, 0xffLL);
  }

  // Check ranges.
  for (size_t Idx = 0; Idx + 1 < Ranges.size(); Idx++) {
    // Check that ranges are not intersected.
    EXPECT_FALSE(Ranges[Idx].Range.intersects(Ranges[Idx + 1].Range));

    // Check that ranges are sorted and not adjusted.
    EXPECT_TRUE(Ranges[Idx].Range.end() <= Ranges[Idx + 1].Range.start());
  }

  Ranges.clear();
  // Fill ranges. Use the various mapped value.
  for (size_t Idx = 0; Idx < NumElements; Idx++) {
    uint64_t Start = static_cast<uint64_t>(std::rand() % 1000);
    uint64_t End = Start + static_cast<uint64_t>(std::rand() % 1000);
    int64_t Value = static_cast<int64_t>(std::rand() % 10);
    Ranges.insert({Start, End}, Value);
  }

  // Check ranges.
  for (size_t Idx = 0; Idx + 1 < Ranges.size(); Idx++) {
    // Check that ranges are not intersected.
    EXPECT_FALSE(Ranges[Idx].Range.intersects(Ranges[Idx + 1].Range));

    // Check that ranges are sorted and not adjusted.
    EXPECT_TRUE(Ranges[Idx].Range.end() <= Ranges[Idx + 1].Range.start());
  }
}
