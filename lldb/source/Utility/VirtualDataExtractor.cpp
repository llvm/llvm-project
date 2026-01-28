//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/VirtualDataExtractor.h"
#include <cassert>

using namespace lldb;
using namespace lldb_private;

VirtualDataExtractor::VirtualDataExtractor(const void *data,
                                           offset_t data_length,
                                           ByteOrder byte_order,
                                           uint32_t addr_size,
                                           LookupTable lookup_table)
    : DataExtractor(data, data_length, byte_order, addr_size),
      m_lookup_table(std::move(lookup_table)) {
  m_lookup_table.Sort();
}

VirtualDataExtractor::VirtualDataExtractor(const DataBufferSP &data_sp,
                                           ByteOrder byte_order,
                                           uint32_t addr_size,
                                           LookupTable lookup_table)
    : DataExtractor(data_sp, byte_order, addr_size),
      m_lookup_table(std::move(lookup_table)) {
  m_lookup_table.Sort();
}

VirtualDataExtractor::VirtualDataExtractor(const DataBufferSP &data_sp,
                                           LookupTable lookup_table)
    : DataExtractor(data_sp), m_lookup_table(std::move(lookup_table)) {
  m_lookup_table.Sort();
}

const VirtualDataExtractor::LookupTable::Entry *
VirtualDataExtractor::FindEntry(offset_t virtual_addr) const {
  // Use RangeDataVector's binary search instead of linear search.
  return m_lookup_table.FindEntryThatContains(virtual_addr);
}

bool VirtualDataExtractor::ValidateVirtualRead(offset_t virtual_addr,
                                               offset_t length) const {
  const LookupTable::Entry *entry = FindEntry(virtual_addr);
  if (!entry)
    return false;

  // Assert that the read does not cross entry boundaries.
  // RangeData.Contains() checks if a range is fully contained.
  assert(entry->Contains(LookupTable::Range(virtual_addr, length)) &&
         "Read crosses lookup table entry boundary");

  // Also validate that the physical offset is within the data buffer.
  // RangeData.data contains the physical offset.
  offset_t physical_offset = entry->data + (virtual_addr - entry->base);
  return ValidOffsetForDataOfSize(physical_offset, length);
}

const void *VirtualDataExtractor::GetData(offset_t *offset_ptr,
                                          offset_t length) const {
  // Override to treat offset as virtual address.
  if (!offset_ptr)
    return nullptr;

  offset_t virtual_addr = *offset_ptr;

  if (!ValidateVirtualRead(virtual_addr, length))
    return nullptr;

  const LookupTable::Entry *entry = FindEntry(virtual_addr);
  assert(entry && "ValidateVirtualRead should have found an entry");

  offset_t physical_offset = entry->data + (virtual_addr - entry->base);
  // Use base class PeekData directly to avoid recursion.
  const void *result = DataExtractor::PeekData(physical_offset, length);

  if (result) {
    // Advance the virtual offset pointer.
    *offset_ptr += length;
  }

  return result;
}

const uint8_t *VirtualDataExtractor::PeekData(offset_t offset,
                                              offset_t length) const {
  // Override to treat offset as virtual address.
  if (!ValidateVirtualRead(offset, length))
    return nullptr;

  const LookupTable::Entry *entry = FindEntry(offset);
  assert(entry && "ValidateVirtualRead should have found an entry");

  offset_t physical_offset = entry->data + (offset - entry->base);
  // Use the base class PeekData with the physical offset.
  return DataExtractor::PeekData(physical_offset, length);
}

uint8_t VirtualDataExtractor::GetU8_unchecked(offset_t *offset_ptr) const {
  offset_t virtual_addr = *offset_ptr;
  const LookupTable::Entry *entry = FindEntry(virtual_addr);
  assert(entry && "Unchecked methods require valid virtual address");

  offset_t physical_offset = entry->data + (virtual_addr - entry->base);
  uint8_t result = DataExtractor::GetU8_unchecked(&physical_offset);
  *offset_ptr += 1;
  return result;
}

uint16_t VirtualDataExtractor::GetU16_unchecked(offset_t *offset_ptr) const {
  offset_t virtual_addr = *offset_ptr;
  const LookupTable::Entry *entry = FindEntry(virtual_addr);
  assert(entry && "Unchecked methods require valid virtual address");

  offset_t physical_offset = entry->data + (virtual_addr - entry->base);
  uint16_t result = DataExtractor::GetU16_unchecked(&physical_offset);
  *offset_ptr += 2;
  return result;
}

uint32_t VirtualDataExtractor::GetU32_unchecked(offset_t *offset_ptr) const {
  offset_t virtual_addr = *offset_ptr;
  const LookupTable::Entry *entry = FindEntry(virtual_addr);
  assert(entry && "Unchecked methods require valid virtual address");

  offset_t physical_offset = entry->data + (virtual_addr - entry->base);
  uint32_t result = DataExtractor::GetU32_unchecked(&physical_offset);
  *offset_ptr += 4;
  return result;
}

uint64_t VirtualDataExtractor::GetU64_unchecked(offset_t *offset_ptr) const {
  offset_t virtual_addr = *offset_ptr;
  const LookupTable::Entry *entry = FindEntry(virtual_addr);
  assert(entry && "Unchecked methods require valid virtual address");

  offset_t physical_offset = entry->data + (virtual_addr - entry->base);
  uint64_t result = DataExtractor::GetU64_unchecked(&physical_offset);
  *offset_ptr += 8;
  return result;
}

DataExtractorSP
VirtualDataExtractor::GetSubsetExtractorSP(offset_t virtual_offset,
                                           offset_t virtual_length) {
  const LookupTable::Entry *entry = FindEntry(virtual_offset);
  assert(
      entry &&
      "VirtualDataExtractor subset extractor requires valid virtual address");
  if (!entry)
    return {};

  // Entry::data is the offset into the DataBuffer's actual start/end range
  // Entry::base is the virtual address at the start of this region of data
  offset_t offset_into_entry_range = virtual_offset - entry->base;
  assert(
      offset_into_entry_range + virtual_length <= entry->size &&
      "VirtualDataExtractor subset may not span multiple LookupTable entries");
  if (offset_into_entry_range + virtual_length > entry->size)
    return {};

  // We could support a Subset VirtualDataExtractor which covered
  // multiple LookupTable virtual entries, but we'd need to mutate
  // all of the LookupTable entries that were properly included in
  // the Subset, a bit tricky.  So we won't implement that until it's
  // needed.

  offset_t physical_start = entry->data + offset_into_entry_range;
  std::shared_ptr<DataExtractor> new_sp = std::make_shared<DataExtractor>(
      GetSharedDataBuffer(), GetByteOrder(), GetAddressByteSize());
  new_sp->SetData(GetSharedDataBuffer(), physical_start, virtual_length);
  return new_sp;
}

// Return a DataExtractorSP that contains a single LookupTable's entry; all
// bytes are guaranteed to be readable.
DataExtractorSP
VirtualDataExtractor::GetSubsetExtractorSP(offset_t virtual_offset) {
  const LookupTable::Entry *entry = FindEntry(virtual_offset);
  assert(
      entry &&
      "VirtualDataExtractor subset extractor requires valid virtual address");
  if (!entry)
    return {};

  // Entry::data is the offset into the DataBuffer's actual start/end range
  // Entry::base is the virtual address at the start of this region of data
  offset_t offset_into_entry_range = virtual_offset - entry->base;

  offset_t physical_start = entry->data + offset_into_entry_range;
  std::shared_ptr<DataExtractor> new_sp = std::make_shared<DataExtractor>(
      GetSharedDataBuffer(), GetByteOrder(), GetAddressByteSize());
  new_sp->SetData(GetSharedDataBuffer(), physical_start,
                  entry->size - offset_into_entry_range);
  return new_sp;
}

// Return an ArrayRef to the first contiguous region of the LookupTable
// only.  The LookupTable entries may have gaps of unmapped data, and we
// can't include those in the ArrayRef or something may touch those pages.
llvm::ArrayRef<uint8_t> VirtualDataExtractor::GetData() const {
  const LookupTable::Entry *entry = FindEntry(0);
  assert(entry &&
         "VirtualDataExtractor GetData requires valid virtual address");
  if (!entry)
    return {};
  return {m_start + static_cast<size_t>(entry->data), static_cast<size_t>(entry->size)};
}
