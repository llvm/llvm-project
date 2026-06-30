//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/MemoryRegionInfoCache.h"
#include "lldb/Target/MemoryRegionInfo.h"

using namespace lldb;
using namespace lldb_private;

void MemoryRegionInfoCache::Clear() {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  m_region_infos.Clear();
  m_is_sorted = true;
}

void MemoryRegionInfoCache::EraseRange(addr_t load_addr, size_t size) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  // If load_addr+size would overflow, do nothing.
  // Likely this is an LLDB_INVALID_ADDRESS plus something.
  uint64_t max_minus_addr = std::numeric_limits<addr_t>::max() - load_addr;
  if (size > max_minus_addr)
    return;

  if (!m_is_sorted)
    m_region_infos.Sort();
  uint32_t start_idx = m_region_infos.FindEntryIndexThatContains(load_addr);
  uint32_t end_idx =
      m_region_infos.FindEntryIndexThatContains(load_addr + size);
  if (start_idx == UINT32_MAX && end_idx == UINT32_MAX)
    return;

  if (start_idx == UINT32_MAX)
    m_region_infos.Erase(end_idx, end_idx + 1);
  else if (end_idx == UINT32_MAX)
    m_region_infos.Erase(start_idx, start_idx + 1);
  else
    m_region_infos.Erase(start_idx, end_idx + 1);
  m_is_sorted = false;
}

void MemoryRegionInfoCache::EraseContaining(addr_t load_addr) {
  EraseRange(load_addr, 1);
}

std::optional<MemoryRegionInfo>
MemoryRegionInfoCache::GetMemoryRegion(addr_t load_addr) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (!m_is_sorted) {
    m_region_infos.Sort();
    m_is_sorted = true;
  }
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
  m_is_sorted = false;
}
