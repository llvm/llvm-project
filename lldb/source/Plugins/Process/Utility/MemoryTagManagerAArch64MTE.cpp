//===-- MemoryTagManagerAArch64MTE.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemoryTagManagerAArch64MTE.h"

using namespace lldb_private;

static const unsigned MTE_START_BIT = 56;
static const unsigned MTE_TAG_MAX = 0xf;
static const unsigned MTE_GRANULE_SIZE = 16;

lldb::addr_t
MemoryTagManagerAArch64MTE::GetLogicalTag(lldb::addr_t addr) const {
  return (addr >> MTE_START_BIT) & MTE_TAG_MAX;
}

lldb::addr_t
MemoryTagManagerAArch64MTE::RemoveNonAddressBits(lldb::addr_t addr) const {
  // Here we're ignoring the whole top byte. If you've got MTE
  // you must also have TBI (top byte ignore).
  // The other 4 bits could contain other extension bits or
  // user metadata.
  return addr & ~((lldb::addr_t)0xFF << MTE_START_BIT);
}

ptrdiff_t MemoryTagManagerAArch64MTE::AddressDiff(lldb::addr_t addr1,
                                                  lldb::addr_t addr2) const {
  return RemoveNonAddressBits(addr1) - RemoveNonAddressBits(addr2);
}

lldb::addr_t MemoryTagManagerAArch64MTE::GetGranuleSize() const {
  return MTE_GRANULE_SIZE;
}

int32_t MemoryTagManagerAArch64MTE::GetAllocationTagType() const {
  return eMTE_allocation;
}

size_t MemoryTagManagerAArch64MTE::GetTagSizeInBytes() const { return 1; }

MemoryTagManagerAArch64MTE::TagRange
MemoryTagManagerAArch64MTE::ExpandToGranule(TagRange range) const {
  // Ignore reading a length of 0
  if (!range.IsValid())
    return range;

  const size_t granule = GetGranuleSize();

  // Align start down to granule start
  lldb::addr_t new_start = range.GetRangeBase();
  lldb::addr_t align_down_amount = new_start % granule;
  new_start -= align_down_amount;

  // Account for the distance we moved the start above
  size_t new_len = range.GetByteSize() + align_down_amount;
  // Then align up to the end of the granule
  size_t align_up_amount = granule - (new_len % granule);
  if (align_up_amount != granule)
    new_len += align_up_amount;

  return TagRange(new_start, new_len);
}

llvm::Expected<std::vector<lldb::addr_t>>
MemoryTagManagerAArch64MTE::UnpackTagsData(const std::vector<uint8_t> &tags,
                                           size_t granules) const {
  size_t num_tags = tags.size() / GetTagSizeInBytes();
  if (num_tags != granules) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Packed tag data size does not match expected number of tags. "
        "Expected %zu tag(s) for %zu granules, got %zu tag(s).",
        granules, granules, num_tags);
  }

  // (if bytes per tag was not 1, we would reconstruct them here)

  std::vector<lldb::addr_t> unpacked;
  unpacked.reserve(tags.size());
  for (auto it = tags.begin(); it != tags.end(); ++it) {
    // Check all tags are in range
    if (*it > MTE_TAG_MAX) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Found tag 0x%x which is > max MTE tag value of 0x%x.", *it,
          MTE_TAG_MAX);
    }
    unpacked.push_back(*it);
  }

  return unpacked;
}
