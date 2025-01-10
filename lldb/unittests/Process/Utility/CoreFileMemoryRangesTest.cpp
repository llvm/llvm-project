//===-- CoreFileMemoryRangesTests.cpp
//---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "lldb/Target/CoreFileMemoryRanges.h"
#include "lldb/lldb-types.h"

using namespace lldb_private;

TEST(CoreFileMemoryRangesTest, MapOverlappingRanges) {
  lldb_private::CoreFileMemoryRanges ranges;
  const lldb::addr_t start_addr = 0x1000;
  const lldb::addr_t increment_addr = 0x1000;
  const size_t iterations = 10;
  for (size_t i = 0; i < iterations; i++) {
    const lldb::addr_t start = start_addr + (i * increment_addr);
    const lldb::addr_t end = start + increment_addr;
    // Arbitrary value
    const uint32_t permissions = 0x3;
    llvm::AddressRange range(start, end);
    const CoreFileMemoryRange core_range = {range, permissions};
    // The range data is Start, Size, While the range is start-end.
    CoreFileMemoryRanges::Entry entry = {start, end - start, core_range};
    ranges.Append(entry);
  }

  Status error = ranges.FinalizeCoreFileSaveRanges();
  EXPECT_TRUE(error.Success());
  ASSERT_THAT(1, ranges.GetSize());
  const auto range = ranges.GetEntryAtIndex(0);
  ASSERT_TRUE(range);
  ASSERT_THAT(start_addr, range->GetRangeBase());
  ASSERT_THAT(start_addr + (iterations * increment_addr), range->GetRangeEnd());
}

TEST(CoreFileMemoryRangesTest, RangesSplitByPermissions) {
  lldb_private::CoreFileMemoryRanges ranges;
  const lldb::addr_t start_addr = 0x1000;
  const lldb::addr_t increment_addr = 0x1000;
  const size_t iterations = 10;
  for (size_t i = 0; i < iterations; i++) {
    const lldb::addr_t start = start_addr + (i * increment_addr);
    const lldb::addr_t end = start + increment_addr;
    const uint32_t permissions = i;
    llvm::AddressRange range(start, end);
    const CoreFileMemoryRange core_range = {range, permissions};
    // The range data is Start, Size, While the range is start-end.
    CoreFileMemoryRanges::Entry entry = {start, end - start, core_range};
    ranges.Append(entry);
  }

  Status error = ranges.FinalizeCoreFileSaveRanges();
  EXPECT_TRUE(error.Success());
  ASSERT_THAT(10, ranges.GetSize());
  const auto range = ranges.GetEntryAtIndex(0);
  ASSERT_TRUE(range);
  ASSERT_THAT(start_addr, range->GetRangeBase());
  ASSERT_THAT(start_addr + increment_addr, range->GetRangeEnd());
}

TEST(CoreFileMemoryRangesTest, MapPartialOverlappingRanges) {
  lldb_private::CoreFileMemoryRanges ranges;
  const lldb::addr_t start_addr = 0x1000;
  const lldb::addr_t increment_addr = 0x1000;
  const size_t iterations = 10;
  for (size_t i = 0; i < iterations; i++) {
    const lldb::addr_t start = start_addr + (i * increment_addr);
    const lldb::addr_t end = start + increment_addr;
    // Arbitrary value
    const uint32_t permissions = 0x3;
    llvm::AddressRange range(start, end);
    const CoreFileMemoryRange core_range = {range, permissions};
    // The range data is Start, Size, While the range is start-end.
    CoreFileMemoryRanges::Entry entry = {start, end - start, core_range};
    ranges.Append(entry);
  }

  const lldb::addr_t unique_start = 0x7fff0000;
  const lldb::addr_t unique_end = unique_start + increment_addr;
  llvm::AddressRange range(unique_start, unique_end);
  const uint32_t permissions = 0x3;
  const CoreFileMemoryRange core_range = {range, permissions};
  // The range data is Start, Size, While the range is start-end.
  CoreFileMemoryRanges::Entry entry = {unique_start, unique_end - unique_start,
                                       core_range};
  ranges.Append(entry);

  Status error = ranges.FinalizeCoreFileSaveRanges();
  EXPECT_TRUE(error.Success());
  ASSERT_THAT(2, ranges.GetSize());
  const auto merged_range = ranges.GetEntryAtIndex(0);
  ASSERT_TRUE(merged_range);
  ASSERT_THAT(start_addr, merged_range->GetRangeBase());
  ASSERT_THAT(start_addr + (iterations * increment_addr),
              merged_range->GetRangeEnd());
  const auto unique_range = ranges.GetEntryAtIndex(1);
  ASSERT_TRUE(unique_range);
  ASSERT_THAT(unique_start, unique_range->GetRangeBase());
  ASSERT_THAT(unique_end, unique_range->GetRangeEnd());
}

TEST(CoreFileMemoryRangesTest, SuperiorAndInferiorRanges_SamePermissions) {
  lldb_private::CoreFileMemoryRanges ranges;
  const lldb::addr_t start_addr = 0x1000;
  const lldb::addr_t increment_addr = 0x1000;
  const lldb::addr_t superior_region_end = start_addr + increment_addr * 10;
  llvm::AddressRange range(start_addr, superior_region_end);
  const CoreFileMemoryRange core_range = {range, 0x3};
  CoreFileMemoryRanges::Entry entry = {
      start_addr, superior_region_end - start_addr, core_range};
  ranges.Append(entry);
  const lldb::addr_t inferior_region_end = start_addr + increment_addr;
  llvm::AddressRange inferior_range(start_addr, inferior_region_end);
  const CoreFileMemoryRange inferior_core_range = {inferior_range, 0x3};
  CoreFileMemoryRanges::Entry inferior_entry = {
      start_addr, inferior_region_end - start_addr, inferior_core_range};
  ranges.Append(inferior_entry);

  Status error = ranges.FinalizeCoreFileSaveRanges();
  EXPECT_TRUE(error.Success());
  ASSERT_THAT(1, ranges.GetSize());
  const auto searched_range = ranges.GetEntryAtIndex(0);
  ASSERT_TRUE(searched_range);
  ASSERT_THAT(start_addr, searched_range->GetRangeBase());
  ASSERT_THAT(superior_region_end, searched_range->GetRangeEnd());
}

TEST(CoreFileMemoryRangesTest, SuperiorAndInferiorRanges_DifferentPermissions) {
  lldb_private::CoreFileMemoryRanges ranges;
  const lldb::addr_t start_addr = 0x1000;
  const lldb::addr_t increment_addr = 0x1000;
  const lldb::addr_t superior_region_end = start_addr + increment_addr * 10;
  llvm::AddressRange range(start_addr, superior_region_end);
  const CoreFileMemoryRange core_range = {range, 0x3};
  CoreFileMemoryRanges::Entry entry = {
      start_addr, superior_region_end - start_addr, core_range};
  ranges.Append(entry);
  const lldb::addr_t inferior_region_end = start_addr + increment_addr;
  llvm::AddressRange inferior_range(start_addr, inferior_region_end);
  const CoreFileMemoryRange inferior_core_range = {inferior_range, 0x4};
  CoreFileMemoryRanges::Entry inferior_entry = {
      start_addr, inferior_region_end - start_addr, inferior_core_range};
  ranges.Append(inferior_entry);

  Status error = ranges.FinalizeCoreFileSaveRanges();
  EXPECT_TRUE(error.Fail());
}

TEST(CoreFileMemoryRangesTest, NonIntersectingRangesSamePermissions) {
  const int permissions = 0x7;
  lldb_private::CoreFileMemoryRanges ranges;
  const lldb::addr_t region_one_start = 0x1000;
  const lldb::addr_t region_one_end = 0x2000;
  llvm::AddressRange range_one(region_one_start, region_one_end);
  const CoreFileMemoryRange core_range_one = {range_one, permissions};
  CoreFileMemoryRanges::Entry entry_one = {
      region_one_start, region_one_end - region_one_start, core_range_one};
  ranges.Append(entry_one);
  const lldb::addr_t region_two_start = 0xb000;
  const lldb::addr_t region_two_end = 0xc000;
  llvm::AddressRange range_two(region_two_start, region_two_end);
  const CoreFileMemoryRange core_range_two = {range_two, permissions};
  CoreFileMemoryRanges::Entry entry_two = {
      region_two_start, region_two_end - region_two_start, core_range_two};
  ranges.Append(entry_two);

  Status error = ranges.FinalizeCoreFileSaveRanges();
  EXPECT_TRUE(error.Success());
  ASSERT_THAT(2UL, ranges.GetSize());
  ASSERT_THAT(region_one_start, ranges.GetEntryAtIndex(0)->GetRangeBase());
  ASSERT_THAT(region_two_start, ranges.GetEntryAtIndex(1)->GetRangeBase());
}

TEST(CoreFileMemoryRangesTest, PartialOverlapping) {
  const int permissions = 0x3;
  lldb_private::CoreFileMemoryRanges ranges;
  const lldb::addr_t start_addr = 0x1000;
  const lldb::addr_t end_addr = 0x2000;
  llvm::AddressRange range_one(start_addr, end_addr);
  const CoreFileMemoryRange core_range_one = {range_one, permissions};
  CoreFileMemoryRanges::Entry entry_one = {start_addr, end_addr - start_addr,
                                           core_range_one};
  llvm::AddressRange range_two(start_addr / 2, end_addr / 2);
  const CoreFileMemoryRange core_range_two = {range_two, permissions};
  CoreFileMemoryRanges::Entry entry_two = {
      start_addr / 2, end_addr / 2 - start_addr / 2, core_range_two};
  ranges.Append(entry_one);
  ranges.Append(entry_two);

  Status error = ranges.FinalizeCoreFileSaveRanges();
  EXPECT_TRUE(error.Success());
  ASSERT_THAT(1, ranges.GetSize());
  const auto searched_range = ranges.GetEntryAtIndex(0);
  ASSERT_TRUE(searched_range);
  ASSERT_THAT(start_addr / 2, searched_range->GetRangeBase());
  ASSERT_THAT(end_addr, searched_range->GetRangeEnd());
}
