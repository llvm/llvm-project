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
