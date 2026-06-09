//===-- MemoryRegionInfoCache.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/MemoryRegionInfoCache.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/Status.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

void MemoryRegionInfoCache::Clear() { m_region_infos.Clear(); }

void MemoryRegionInfoCache::Flush(addr_t load_addr, addr_t size) {
  m_region_infos.Erase(load_addr, size);
  m_region_infos.Sort();
}

Status
MemoryRegionInfoCache::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                           MemoryRegionInfo &region_info) {
  uint32_t index = m_region_infos.FindEntryIndexThatContains(load_addr);
  if (index != UINT32_MAX) {
    region_info = m_region_infos.GetEntryAtIndex(index)->data;
    return Status();
  }

  load_addr = m_process.FixAnyAddress(load_addr);
  Status error = m_process.DoGetMemoryRegionInfo(load_addr, region_info);
  // Reject a region that does not contain the requested address.
  if (error.Success() && !region_info.GetRange().Contains(load_addr))
    error = Status::FromErrorString("Invalid memory region");

  if (error.Success())
    AddRegion(region_info);

  return error;
}

void MemoryRegionInfoCache::AddRegion(const MemoryRegionInfo &ri) {
  InfoMap::Entry new_entry(ri.GetRange().GetRangeBase(),
                           ri.GetRange().GetByteSize(), ri);
  m_region_infos.Append(new_entry);
  m_region_infos.Sort();
}
