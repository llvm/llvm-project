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

void MemoryRegionInfoCache::Clear() { m_region_infos.Lock()->clear(); }

size_t MemoryRegionInfoCache::GetSize() {
  return m_region_infos.Lock()->size();
}

std::optional<MemoryRegionInfo>
MemoryRegionInfoCache::GetMemoryRegion(addr_t load_addr) {
  auto region_infos = m_region_infos.Lock();
  auto it = region_infos->upper_bound(load_addr);
  if (it == region_infos->begin())
    return std::nullopt;
  --it;
  if (load_addr < it->second.GetRange().GetRangeEnd())
    return it->second;

  return std::nullopt;
}

void MemoryRegionInfoCache::AddRegion(const MemoryRegionInfo &ri) {
  m_region_infos.Lock()->insert_or_assign(ri.GetRange().GetRangeBase(), ri);
}
