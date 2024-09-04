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
