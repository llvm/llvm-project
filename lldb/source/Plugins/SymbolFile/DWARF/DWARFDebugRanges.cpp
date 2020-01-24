//===-- DWARFDebugRanges.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugRanges.h"
#include "DWARFUnit.h"
#include "lldb/Utility/Stream.h"

using namespace lldb_private;

static dw_addr_t GetBaseAddressMarker(uint32_t addr_size) {
  switch(addr_size) {
    case 2:
      return 0xffff;
    case 4:
      return 0xffffffff;
    case 8:
      return 0xffffffffffffffff;
  }
  llvm_unreachable("GetBaseAddressMarker unsupported address size.");
}

DWARFDebugRanges::DWARFDebugRanges() : m_range_map() {}

void DWARFDebugRanges::Extract(DWARFContext &context) {
  DWARFRangeList range_list;
  lldb::offset_t offset = 0;
  dw_offset_t debug_ranges_offset = offset;
  while (Extract(context, &offset, range_list)) {
    range_list.Sort();
    m_range_map[debug_ranges_offset] = range_list;
    debug_ranges_offset = offset;
  }
}

bool DWARFDebugRanges::Extract(DWARFContext &context,
                               lldb::offset_t *offset_ptr,
                               DWARFRangeList &range_list) {
  range_list.Clear();

  lldb::offset_t range_offset = *offset_ptr;
  const DWARFDataExtractor &debug_ranges_data = context.getOrLoadRangesData();
  uint32_t addr_size = debug_ranges_data.GetAddressByteSize();
  dw_addr_t base_addr = 0;
  dw_addr_t base_addr_marker = GetBaseAddressMarker(addr_size);

  while (
      debug_ranges_data.ValidOffsetForDataOfSize(*offset_ptr, 2 * addr_size)) {
    dw_addr_t begin = debug_ranges_data.GetMaxU64(offset_ptr, addr_size);
    dw_addr_t end = debug_ranges_data.GetMaxU64(offset_ptr, addr_size);

    if (!begin && !end) {
      // End of range list
      break;
    }

    if (begin == base_addr_marker) {
      base_addr = end;
      continue;
    }

    // Filter out empty ranges
    if (begin < end)
      range_list.Append(DWARFRangeList::Entry(begin + base_addr, end - begin));
  }

  // Make sure we consumed at least something
  return range_offset != *offset_ptr;
}

void DWARFDebugRanges::Dump(Stream &s,
                            const DWARFDataExtractor &debug_ranges_data,
                            lldb::offset_t *offset_ptr,
                            dw_addr_t cu_base_addr) {
  uint32_t addr_size = s.GetAddressByteSize();

  dw_addr_t base_addr = cu_base_addr;
  while (
      debug_ranges_data.ValidOffsetForDataOfSize(*offset_ptr, 2 * addr_size)) {
    dw_addr_t begin = debug_ranges_data.GetMaxU64(offset_ptr, addr_size);
    dw_addr_t end = debug_ranges_data.GetMaxU64(offset_ptr, addr_size);
    // Extend 4 byte addresses that consists of 32 bits of 1's to be 64 bits of
    // ones
    if (begin == 0xFFFFFFFFull && addr_size == 4)
      begin = LLDB_INVALID_ADDRESS;

    s.Indent();
    if (begin == 0 && end == 0) {
      s.PutCString(" End");
      break;
    } else if (begin == LLDB_INVALID_ADDRESS) {
      // A base address selection entry
      base_addr = end;
      DumpAddress(s.AsRawOstream(), base_addr, sizeof(dw_addr_t),
                  " Base address = ");
    } else {
      // Convert from offset to an address
      dw_addr_t begin_addr = begin + base_addr;
      dw_addr_t end_addr = end + base_addr;

      DumpAddressRange(s.AsRawOstream(), begin_addr, end_addr,
                       sizeof(dw_addr_t), nullptr);
    }
  }
}

bool DWARFDebugRanges::FindRanges(const DWARFUnit *cu,
                                  dw_offset_t debug_ranges_offset,
                                  DWARFRangeList &range_list) const {
  dw_addr_t debug_ranges_address = cu->GetRangesBase() + debug_ranges_offset;
  range_map_const_iterator pos = m_range_map.find(debug_ranges_address);
  if (pos != m_range_map.end()) {
    range_list = pos->second;

    // All DW_AT_ranges are relative to the base address of the compile
    // unit. We add the compile unit base address to make sure all the
    // addresses are properly fixed up.
    range_list.Slide(cu->GetBaseAddress());
    return true;
  }
  return false;
}
