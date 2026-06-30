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
  std::lock_guard<std::mutex> guard(m_mutex);
  m_region_infos.Clear();
  m_is_sorted = true;
}

size_t MemoryRegionInfoCache::GetSize() {
  std::lock_guard<std::mutex> guard(m_mutex);
  return m_region_infos.GetSize();
}

std::optional<MemoryRegionInfo>
MemoryRegionInfoCache::GetMemoryRegion(addr_t load_addr) {
  std::lock_guard<std::mutex> guard(m_mutex);
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
  std::lock_guard<std::mutex> guard(m_mutex);
  InfoMap::Entry new_entry(ri.GetRange().GetRangeBase(),
                           ri.GetRange().GetByteSize(), ri);
  m_region_infos.Append(new_entry);
  m_is_sorted = false;
}
