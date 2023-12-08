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
          "Expected 2 tag(s) for 2 granule(s), got 0 tag(s)."));

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

  // Error for too much tag data
  ASSERT_THAT_EXPECTED(
      manager.UnpackTagsData(input, 1),
      llvm::FailedWithMessage(
          "Packed tag data size does not match expected number of tags. "
          "Expected 1 tag(s) for 1 granule(s), got 2 tag(s)."));

  // By default, we don't check number of tags
  llvm::Expected<std::vector<lldb::addr_t>> got_zero =
      manager.UnpackTagsData(input);
  ASSERT_THAT_EXPECTED(got_zero, llvm::Succeeded());
  ASSERT_THAT(expected, testing::ContainerEq(*got));

  // Which is the same as granules=0
  got_zero = manager.UnpackTagsData(input, 0);
  ASSERT_THAT_EXPECTED(got_zero, llvm::Succeeded());
  ASSERT_THAT(expected, testing::ContainerEq(*got));
}

TEST(MemoryTagManagerAArch64MTETest, PackTags) {
  MemoryTagManagerAArch64MTE manager;

  // Error for tag out of range
  llvm::Expected<std::vector<uint8_t>> invalid_tag_err =
      manager.PackTags({0x10});
  ASSERT_THAT_EXPECTED(
      invalid_tag_err,
      llvm::FailedWithMessage(
          "Found tag 0x10 which is > max MTE tag value of 0xf."));

  // 0xf here is the max tag value that we can pack
  std::vector<lldb::addr_t> tags{0, 1, 0xf};
  std::vector<uint8_t> expected{0, 1, 0xf};
  llvm::Expected<std::vector<uint8_t>> packed = manager.PackTags(tags);
  ASSERT_THAT_EXPECTED(packed, llvm::Succeeded());
  ASSERT_THAT(expected, testing::ContainerEq(*packed));
}

TEST(MemoryTagManagerAArch64MTETest, UnpackTagsFromCoreFileSegment) {
  MemoryTagManagerAArch64MTE manager;
  // This is our fake segment data where tags are compressed as 2 4 bit tags
  // per byte.
  std::vector<uint8_t> tags_data;
  MemoryTagManager::CoreReaderFn reader =
      [&tags_data](lldb::offset_t offset, size_t length, void *dst) {
        std::memcpy(dst, tags_data.data() + offset, length);
        return length;
      };

  // Zero length is ok.
  std::vector<lldb::addr_t> tags =
      manager.UnpackTagsFromCoreFileSegment(reader, 0, 0, 0, 0);
  ASSERT_EQ(tags.size(), (size_t)0);

  // In the simplest case we read 2 tags which are in the same byte.
  tags_data.push_back(0x21);
  // The least significant bits are the first tag in memory.
  std::vector<lldb::addr_t> expected{1, 2};
  tags = manager.UnpackTagsFromCoreFileSegment(reader, 0, 0, 0, 32);
  ASSERT_THAT(expected, testing::ContainerEq(tags));

  // If we read just one then it will have to trim off the second one.
  expected = std::vector<lldb::addr_t>{1};
  tags = manager.UnpackTagsFromCoreFileSegment(reader, 0, 0, 0, 16);
  ASSERT_THAT(expected, testing::ContainerEq(tags));

  // If we read the second tag only then the first one must be trimmed.
  expected = std::vector<lldb::addr_t>{2};
  tags = manager.UnpackTagsFromCoreFileSegment(reader, 0, 0, 16, 16);
  ASSERT_THAT(expected, testing::ContainerEq(tags));

  // This trimming logic applies if you read a larger set of tags.
  tags_data = std::vector<uint8_t>{0x21, 0x43, 0x65, 0x87};

  // Trailing tag should be trimmed.
  expected = std::vector<lldb::addr_t>{1, 2, 3};
  tags = manager.UnpackTagsFromCoreFileSegment(reader, 0, 0, 0, 48);
  ASSERT_THAT(expected, testing::ContainerEq(tags));

  // Leading tag should be trimmed.
  expected = std::vector<lldb::addr_t>{2, 3, 4};
  tags = manager.UnpackTagsFromCoreFileSegment(reader, 0, 0, 16, 48);
  ASSERT_THAT(expected, testing::ContainerEq(tags));

  // Leading and trailing trimmmed.
  expected = std::vector<lldb::addr_t>{2, 3, 4, 5};
  tags = manager.UnpackTagsFromCoreFileSegment(reader, 0, 0, 16, 64);
  ASSERT_THAT(expected, testing::ContainerEq(tags));

  // The address given is an offset into the whole file so the address requested
  // from the reader should be beyond that.
  tags_data = std::vector<uint8_t>{0xFF, 0xFF, 0x21, 0x43, 0x65, 0x87};
  expected = std::vector<lldb::addr_t>{1, 2};
  tags = manager.UnpackTagsFromCoreFileSegment(reader, 0, 2, 0, 32);
  ASSERT_THAT(expected, testing::ContainerEq(tags));

  // addr is a virtual address that we expect to be >= the tag segment's
  // starting virtual address. So again an offset must be made from the
  // difference.
  expected = std::vector<lldb::addr_t>{3, 4};
  tags = manager.UnpackTagsFromCoreFileSegment(reader, 32, 2, 64, 32);
  ASSERT_THAT(expected, testing::ContainerEq(tags));
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

static MemoryRegionInfo MakeRegionInfo(lldb::addr_t base, lldb::addr_t size,
                                       bool tagged) {
  return MemoryRegionInfo(
      MemoryRegionInfo::RangeType(base, size),
      MemoryRegionInfo::eYes, MemoryRegionInfo::eYes, MemoryRegionInfo::eYes,
      MemoryRegionInfo::eNo,
      MemoryRegionInfo::eYes,
      ConstString(), MemoryRegionInfo::eNo, 0,
      /*memory_tagged=*/
      tagged ? MemoryRegionInfo::eYes : MemoryRegionInfo::eNo,
      MemoryRegionInfo::eDontKnow);
}

TEST(MemoryTagManagerAArch64MTETest, MakeTaggedRange) {
  MemoryTagManagerAArch64MTE manager;
  MemoryRegionInfos memory_regions;

  // No regions means no tagged regions, error
  ASSERT_THAT_EXPECTED(
      manager.MakeTaggedRange(0, 0x10, memory_regions),
      llvm::FailedWithMessage(
          "Address range 0x0:0x10 is not in a memory tagged region"));

  // Alignment is done before checking regions.
  // Here 1 is rounded up to the granule size of 0x10.
  ASSERT_THAT_EXPECTED(
      manager.MakeTaggedRange(0, 1, memory_regions),
      llvm::FailedWithMessage(
          "Address range 0x0:0x10 is not in a memory tagged region"));

  // Range must not be inverted
  ASSERT_THAT_EXPECTED(
      manager.MakeTaggedRange(1, 0, memory_regions),
      llvm::FailedWithMessage(
          "End address (0x0) must be greater than the start address (0x1)"));

  // The inversion check ignores tags in the addresses (MTE tags start at bit
  // 56).
  ASSERT_THAT_EXPECTED(
      manager.MakeTaggedRange((lldb::addr_t)1 << 56,
                              ((lldb::addr_t)2 << 56) + 0x10, memory_regions),
      llvm::FailedWithMessage(
          "Address range 0x0:0x10 is not in a memory tagged region"));

  // Adding a single region to cover the whole range
  memory_regions.push_back(MakeRegionInfo(0, 0x1000, true));

  // Range can have different tags for begin and end
  // (which would make it look inverted if we didn't remove them)
  // Note that range comes back with an untagged base and alginment
  // applied.
  MemoryTagManagerAArch64MTE::TagRange expected_range(0x0, 0x10);
  llvm::Expected<MemoryTagManagerAArch64MTE::TagRange> got =
      manager.MakeTaggedRange(0x0f00000000000000, 0x0e00000000000001,
                              memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, expected_range);

  // Error if the range isn't within any region
  ASSERT_THAT_EXPECTED(
      manager.MakeTaggedRange(0x1000, 0x1010, memory_regions),
      llvm::FailedWithMessage(
          "Address range 0x1000:0x1010 is not in a memory tagged region"));

  // Error if the first part of a range isn't tagged
  memory_regions.clear();
  const char *err_msg =
      "Address range 0x0:0x1000 is not in a memory tagged region";

  // First because it has no region entry
  memory_regions.push_back(MakeRegionInfo(0x10, 0x1000, true));
  ASSERT_THAT_EXPECTED(manager.MakeTaggedRange(0, 0x1000, memory_regions),
                       llvm::FailedWithMessage(err_msg));

  // Then because the first region is untagged
  memory_regions.push_back(MakeRegionInfo(0, 0x10, false));
  ASSERT_THAT_EXPECTED(manager.MakeTaggedRange(0, 0x1000, memory_regions),
                       llvm::FailedWithMessage(err_msg));

  // If we tag that first part it succeeds
  memory_regions.back().SetMemoryTagged(MemoryRegionInfo::eYes);
  expected_range = MemoryTagManagerAArch64MTE::TagRange(0x0, 0x1000);
  got = manager.MakeTaggedRange(0, 0x1000, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, expected_range);

  // Error if the end of a range is untagged
  memory_regions.clear();

  // First because it has no region entry
  memory_regions.push_back(MakeRegionInfo(0, 0xF00, true));
  ASSERT_THAT_EXPECTED(manager.MakeTaggedRange(0, 0x1000, memory_regions),
                       llvm::FailedWithMessage(err_msg));

  // Then because the last region is untagged
  memory_regions.push_back(MakeRegionInfo(0xF00, 0x100, false));
  ASSERT_THAT_EXPECTED(manager.MakeTaggedRange(0, 0x1000, memory_regions),
                       llvm::FailedWithMessage(err_msg));

  // If we tag the last part it succeeds
  memory_regions.back().SetMemoryTagged(MemoryRegionInfo::eYes);
  got = manager.MakeTaggedRange(0, 0x1000, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, expected_range);

  // Error if the middle of a range is untagged
  memory_regions.clear();

  // First because it has no entry
  memory_regions.push_back(MakeRegionInfo(0, 0x500, true));
  memory_regions.push_back(MakeRegionInfo(0x900, 0x700, true));
  ASSERT_THAT_EXPECTED(manager.MakeTaggedRange(0, 0x1000, memory_regions),
                       llvm::FailedWithMessage(err_msg));

  // Then because it's untagged
  memory_regions.push_back(MakeRegionInfo(0x500, 0x400, false));
  ASSERT_THAT_EXPECTED(manager.MakeTaggedRange(0, 0x1000, memory_regions),
                       llvm::FailedWithMessage(err_msg));

  // If we tag the middle part it succeeds
  memory_regions.back().SetMemoryTagged(MemoryRegionInfo::eYes);
  got = manager.MakeTaggedRange(0, 0x1000, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, expected_range);
}

TEST(MemoryTagManagerAArch64MTETest, MakeTaggedRanges) {
  MemoryTagManagerAArch64MTE manager;
  MemoryRegionInfos memory_regions;

  // Note that MakeTaggedRanges takes start/end address.
  // Whereas TagRanges and regions take start address and size.

  // Range must not be inverted
  ASSERT_THAT_EXPECTED(
      manager.MakeTaggedRanges(1, 0, memory_regions),
      llvm::FailedWithMessage(
          "End address (0x0) must be greater than the start address (0x1)"));

  // We remove tags before doing the inversion check, so this is not an error.
  // Also no regions means no tagged regions returned.
  // (bit 56 is where MTE tags begin)
  llvm::Expected<std::vector<MemoryTagManager::TagRange>> got =
      manager.MakeTaggedRanges((lldb::addr_t)2 << 56,
                               ((lldb::addr_t)1 << 56) + 0x10, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, std::vector<MemoryTagManager::TagRange>{});

  // Cover whole range, untagged. No ranges returned.
  memory_regions.push_back(MakeRegionInfo(0, 0x20, false));
  got = manager.MakeTaggedRanges(0, 0x20, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, std::vector<MemoryTagManager::TagRange>{});

  // Make the region tagged and it'll be the one range returned.
  memory_regions.back().SetMemoryTagged(MemoryRegionInfo::eYes);
  got = manager.MakeTaggedRanges(0, 0x20, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, std::vector<MemoryTagManager::TagRange>{
                      MemoryTagManager::TagRange(0, 0x20)});

  // This region will be trimmed if it's larger than the whole range.
  memory_regions.clear();
  memory_regions.push_back(MakeRegionInfo(0, 0x40, true));
  got = manager.MakeTaggedRanges(0x10, 0x30, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, std::vector<MemoryTagManager::TagRange>{
                      MemoryTagManager::TagRange(0x10, 0x20)});

  memory_regions.clear();

  // For the following tests we keep the input regions
  // in ascending order as MakeTaggedRanges expects.

  // Only start of range is tagged, only that is returned.
  // Start the region just before the requested range to check
  // we limit the result to the requested range.
  memory_regions.push_back(MakeRegionInfo(0, 0x20, true));
  got = manager.MakeTaggedRanges(0x10, 0x100, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, std::vector<MemoryTagManager::TagRange>{
                      MemoryTagManager::TagRange(0x10, 0x10)});

  // Add a tagged region at the end, now we get both
  // and the middle is untagged.
  // <tagged: [0x0, 0x20)>
  // <...>
  // <tagged: [0xE0, 0x120)>
  // The range added here is deliberately over the end of the
  // requested range to show that we trim the end.
  memory_regions.push_back(MakeRegionInfo(0xE0, 0x40, true));
  got = manager.MakeTaggedRanges(0x10, 0x110, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());

  std::vector<MemoryTagManager::TagRange> expected{
      MemoryTagManager::TagRange(0x10, 0x10),
      MemoryTagManager::TagRange(0xE0, 0x30)};
  ASSERT_EQ(*got, expected);

  // Now add a middle tagged region.
  // <tagged: [0x0, 0x20)>
  // <...>
  // <tagged: [0x90, 0xB0)>
  // <...>
  // <tagged: [0xE0, 0x120)>
  memory_regions.insert(std::next(memory_regions.begin()),
                        MakeRegionInfo(0x90, 0x20, true));

  // As the given regions are in ascending order, the resulting
  // tagged ranges are also. So this new range goes in the middle.
  expected.insert(std::next(expected.begin()),
                  MemoryTagManager::TagRange(0x90, 0x20));
  got = manager.MakeTaggedRanges(0x10, 0x110, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, expected);

  // Then if we add untagged regions in between the tagged,
  // the output should stay the same.
  // <tagged:   [0x0, 0x20)>
  // <untagged: [0x20, 0x90)>
  // <tagged:   [0x90, 0xB0)>
  // <untagged: [0xB0, 0xE0)>
  // <tagged:   [0xE0, 0x120)>
  memory_regions.insert(std::next(memory_regions.begin()),
                        MakeRegionInfo(0x20, 0x70, false));
  memory_regions.insert(std::prev(memory_regions.end()),
                        MakeRegionInfo(0xB0, 0x30, false));
  got = manager.MakeTaggedRanges(0x10, 0x110, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, expected);

  // Finally check that we handle only having the end of the range.
  memory_regions.clear();
  expected.clear();

  memory_regions.push_back(MakeRegionInfo(0x100, 0x10, true));
  expected.push_back(MemoryTagManager::TagRange(0x100, 0x10));
  got = manager.MakeTaggedRanges(0x10, 0x110, memory_regions);
  ASSERT_THAT_EXPECTED(got, llvm::Succeeded());
  ASSERT_EQ(*got, expected);
}

TEST(MemoryTagManagerAArch64MTETest, RemoveTagBits) {
  MemoryTagManagerAArch64MTE manager;

  ASSERT_EQ(0, 0);
  // Removes the whole top byte
  ASSERT_EQ((lldb::addr_t)0x00ffeedd11223344,
            manager.RemoveTagBits(0x00ffeedd11223344));
  ASSERT_EQ((lldb::addr_t)0x0000000000000000,
            manager.RemoveTagBits(0xff00000000000000));
  ASSERT_EQ((lldb::addr_t)0x0055555566666666,
            manager.RemoveTagBits(0xee55555566666666));
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

// Helper to check that repeating "tags" over "range" gives you
// "expected_tags".
static void
test_repeating_tags(const std::vector<lldb::addr_t> &tags,
                    MemoryTagManagerAArch64MTE::TagRange range,
                    const std::vector<lldb::addr_t> &expected_tags) {
  MemoryTagManagerAArch64MTE manager;
  llvm::Expected<std::vector<lldb::addr_t>> tags_or_err =
      manager.RepeatTagsForRange(tags, range);
  ASSERT_THAT_EXPECTED(tags_or_err, llvm::Succeeded());
  ASSERT_THAT(expected_tags, testing::ContainerEq(*tags_or_err));
}

TEST(MemoryTagManagerAArch64MTETest, RepeatTagsForRange) {
  MemoryTagManagerAArch64MTE manager;

  // Must have some tags if your range is not empty
  llvm::Expected<std::vector<lldb::addr_t>> no_tags_err =
      manager.RepeatTagsForRange({},
                                 MemoryTagManagerAArch64MTE::TagRange{0, 16});
  ASSERT_THAT_EXPECTED(
      no_tags_err, llvm::FailedWithMessage(
                       "Expected some tags to cover given range, got zero."));

  // If the range is empty, you get no tags back
  test_repeating_tags({1, 2, 3}, MemoryTagManagerAArch64MTE::TagRange{0, 0},
                      {});
  // And you don't need tags for an empty range
  test_repeating_tags({}, MemoryTagManagerAArch64MTE::TagRange{0, 0}, {});

  // A single tag will just be multiplied as many times as needed
  test_repeating_tags({5}, MemoryTagManagerAArch64MTE::TagRange{0, 16}, {5});
  test_repeating_tags({6}, MemoryTagManagerAArch64MTE::TagRange{0, 32}, {6, 6});

  // If you've got as many tags as granules, it's a roundtrip
  test_repeating_tags({7, 8}, MemoryTagManagerAArch64MTE::TagRange{0, 32},
                      {7, 8});

  // If you've got fewer tags than granules, they repeat. Exactly or partially
  // as needed.
  test_repeating_tags({7, 8}, MemoryTagManagerAArch64MTE::TagRange{0, 64},
                      {7, 8, 7, 8});
  test_repeating_tags({7, 8}, MemoryTagManagerAArch64MTE::TagRange{0, 48},
                      {7, 8, 7});

  // If you've got more tags than granules you get back only those needed
  test_repeating_tags({1, 2, 3, 4}, MemoryTagManagerAArch64MTE::TagRange{0, 32},
                      {1, 2});
}
