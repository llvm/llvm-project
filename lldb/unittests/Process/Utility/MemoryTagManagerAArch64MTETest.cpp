//===-- MemoryTagManagerAArch64MTETest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/Utility/MemoryTagManagerAArch64MTE.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(MemoryTagManagerAArch64MTETest, UnpackTagsData) {
  MemoryTagManagerAArch64MTE manager;

  // Error for insufficient tag data
  std::vector<uint8_t> input;
  ASSERT_THAT_EXPECTED(
      manager.UnpackTagsData(input, 2),
      llvm::FailedWithMessage(
          "Packed tag data size does not match expected number of tags. "
          "Expected 2 tag(s) for 2 granules, got 0 tag(s)."));

  // This is out of the valid tag range
  input.push_back(0x1f);
  ASSERT_THAT_EXPECTED(
      manager.UnpackTagsData(input, 1),
      llvm::FailedWithMessage(
          "Found tag 0x1f which is > max MTE tag value of 0xf."));

  // MTE tags are 1 per byte
  input.pop_back();
  input.push_back(0xe);
  input.push_back(0xf);

  std::vector<lldb::addr_t> expected{0xe, 0xf};

  llvm::Expected<std::vector<lldb::addr_t>> got =
      manager.UnpackTagsData(input, 2);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_THAT(expected, testing::ContainerEq(*got));
}

TEST(MemoryTagManagerAArch64MTETest, GetLogicalTag) {
  MemoryTagManagerAArch64MTE manager;

  // Set surrounding bits to check shift is correct
  ASSERT_EQ((lldb::addr_t)0, manager.GetLogicalTag(0xe0e00000ffffffff));
  // Max tag value
  ASSERT_EQ((lldb::addr_t)0xf, manager.GetLogicalTag(0x0f000000ffffffff));
  ASSERT_EQ((lldb::addr_t)2, manager.GetLogicalTag(0x02000000ffffffff));
}

TEST(MemoryTagManagerAArch64MTETest, ExpandToGranule) {
  MemoryTagManagerAArch64MTE manager;
  // Reading nothing, no alignment needed
  ASSERT_EQ(
      MemoryTagManagerAArch64MTE::TagRange(0, 0),
      manager.ExpandToGranule(MemoryTagManagerAArch64MTE::TagRange(0, 0)));

  // Ranges with 0 size are unchanged even if address is non 0
  // (normally 0x1234 would be aligned to 0x1230)
  ASSERT_EQ(
      MemoryTagManagerAArch64MTE::TagRange(0x1234, 0),
      manager.ExpandToGranule(MemoryTagManagerAArch64MTE::TagRange(0x1234, 0)));

  // Ranges already aligned don't change
  ASSERT_EQ(
      MemoryTagManagerAArch64MTE::TagRange(0x100, 64),
      manager.ExpandToGranule(MemoryTagManagerAArch64MTE::TagRange(0x100, 64)));

  // Any read of less than 1 granule is rounded up to reading 1 granule
  ASSERT_EQ(
      MemoryTagManagerAArch64MTE::TagRange(0, 16),
      manager.ExpandToGranule(MemoryTagManagerAArch64MTE::TagRange(0, 1)));

  // Start address is aligned down, and length modified accordingly
  // Here bytes 8 through 24 straddle 2 granules. So the resulting range starts
  // at 0 and covers 32 bytes.
  ASSERT_EQ(
      MemoryTagManagerAArch64MTE::TagRange(0, 32),
      manager.ExpandToGranule(MemoryTagManagerAArch64MTE::TagRange(8, 16)));

  // Here only the size of the range needs aligning
  ASSERT_EQ(
      MemoryTagManagerAArch64MTE::TagRange(16, 32),
      manager.ExpandToGranule(MemoryTagManagerAArch64MTE::TagRange(16, 24)));

  // Start and size need aligning here but we only need 1 granule to cover it
  ASSERT_EQ(
      MemoryTagManagerAArch64MTE::TagRange(16, 16),
      manager.ExpandToGranule(MemoryTagManagerAArch64MTE::TagRange(18, 4)));
}

TEST(MemoryTagManagerAArch64MTETest, RemoveNonAddressBits) {
  MemoryTagManagerAArch64MTE manager;

  ASSERT_EQ(0, 0);
  ASSERT_EQ((lldb::addr_t)0x00ffeedd11223344,
            manager.RemoveNonAddressBits(0x00ffeedd11223344));
  ASSERT_EQ((lldb::addr_t)0x0000000000000000,
            manager.RemoveNonAddressBits(0xFF00000000000000));
  ASSERT_EQ((lldb::addr_t)0x0055555566666666,
            manager.RemoveNonAddressBits(0xee55555566666666));
}

TEST(MemoryTagManagerAArch64MTETest, AddressDiff) {
  MemoryTagManagerAArch64MTE manager;

  ASSERT_EQ(0, manager.AddressDiff(0, 0));
  // Result is signed
  ASSERT_EQ(10, manager.AddressDiff(10, 0));
  ASSERT_EQ(-10, manager.AddressDiff(0, 10));
  // Anything in the top byte is ignored
  ASSERT_EQ(0, manager.AddressDiff(0x2211222233334444, 0x3311222233334444));
  ASSERT_EQ(-32, manager.AddressDiff(0x5511222233334400, 0x4411222233334420));
  ASSERT_EQ(65, manager.AddressDiff(0x9911222233334441, 0x6611222233334400));
}
