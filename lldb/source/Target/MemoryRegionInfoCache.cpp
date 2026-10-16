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
  m_region_infos.clear();
}

size_t MemoryRegionInfoCache::GetSize() {
  std::lock_guard<std::mutex> guard(m_mutex);
  return m_region_infos.size();
}

std::optional<MemoryRegionInfo>
MemoryRegionInfoCache::GetMemoryRegion(addr_t load_addr) {
  std::lock_guard<std::mutex> guard(m_mutex);
  auto it = m_region_infos.upper_bound(load_addr);
  if (it == m_region_infos.begin())
    return std::nullopt;
  --it;
  if (load_addr < it->second.GetRange().GetRangeEnd())
    return it->second;

  return std::nullopt;
}

void MemoryRegionInfoCache::AddRegion(const MemoryRegionInfo &ri) {
  std::lock_guard<std::mutex> guard(m_mutex);
  m_region_infos.insert_or_assign(ri.GetRange().GetRangeBase(), ri);
}
