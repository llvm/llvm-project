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

using namespace lldb;
using namespace lldb_private;

void MemoryRegionInfoCache::Clear() { m_region_infos.Clear(); }

void MemoryRegionInfoCache::Erase(addr_t load_addr, addr_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  uint32_t start_idx = m_region_infos.FindEntryIndexThatContains(load_addr);
  uint32_t end_idx =
      m_region_infos.FindEntryIndexThatContains(load_addr + size);
  if (start_idx == UINT32_MAX && end_idx == UINT32_MAX)
    return;

  if (start_idx == UINT32_MAX)
    m_region_infos.Erase(end_idx, end_idx);
  else if (end_idx == UINT32_MAX)
    m_region_infos.Erase(start_idx, start_idx);
  else
    m_region_infos.Erase(start_idx, end_idx);
  m_region_infos.Sort();
}

std::optional<MemoryRegionInfo>
MemoryRegionInfoCache::GetMemoryRegion(addr_t load_addr) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  uint32_t index = m_region_infos.FindEntryIndexThatContains(load_addr);
  if (index != UINT32_MAX)
    return m_region_infos.GetEntryAtIndex(index)->data;

  return std::nullopt;
}

void MemoryRegionInfoCache::AddRegion(const MemoryRegionInfo &ri) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  InfoMap::Entry new_entry(ri.GetRange().GetRangeBase(),
                           ri.GetRange().GetByteSize(), ri);
  m_region_infos.Append(new_entry);
  m_region_infos.Sort();
}
