//===-- NameToDIE.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NameToDIE.h"
#include "DWARFUnit.h"
#include "Plugins/SymbolFile/DWARF/DIERef.h"
#include "lldb/Core/DataFileCache.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"
#include <algorithm>
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

bool NameToDIE::Find(llvm::StringRef name,
                     llvm::function_ref<bool(DIERef ref)> callback) const {
  uint32_t hash = djbHash(name);
  auto &map = m_map[hash >> 30];
  auto range = std::equal_range(map.begin(), map.end(),
                                NameToDIEX::Tuple(hash, 0, nullptr, 0),
                                llvm::less_first());
  for (const auto &[_, len, n, ref] : llvm::make_range(range))
    if (name == llvm::StringRef(n, len) && !callback(DIERef(ref)))
      return false;
  return true;
}

bool NameToDIE::Find(const RegularExpression &regex,
                     llvm::function_ref<bool(DIERef ref)> callback) const {
  for (const auto &map : m_map) {
    for (const auto &[_, len, name, ref] : map) {
      if (regex.Execute(llvm::StringRef(name, len))) {
        if (!callback(DIERef(ref)))
          return false;
      }
    }
  }
  return true;
}

void NameToDIE::FindAllEntriesForUnit(
    DWARFUnit &s_unit, llvm::function_ref<bool(DIERef ref)> callback) const {
  const DWARFUnit &ns_unit = s_unit.GetNonSkeletonUnit();
  for (const auto &map : m_map) {
    for (const auto &[_, len, name, ref] : map) {
      DIERef die_ref(ref);
      if (ns_unit.GetDebugSection() == die_ref.section() &&
          ns_unit.GetSymbolFileDWARF().GetFileIndex() == die_ref.file_index() &&
          ns_unit.GetOffset() <= die_ref.die_offset() &&
          die_ref.die_offset() < ns_unit.GetNextUnitOffset()) {
        if (!callback(die_ref))
          return;
      }
    }
  }
}

void NameToDIE::Dump(Stream *s) {
}

void NameToDIE::ForEach(
    llvm::function_ref<bool(llvm::StringRef name, const DIERef &die_ref)>
        callback) const {
  for (const auto &map : m_map) {
    for (const auto &[_, len, name, ref] : map) {
      if (!callback(llvm::StringRef(name, len), DIERef(ref)))
        return;
    }
  }
}

void NameToDIE::Append(const NameToDIEX &other, unsigned slice) {
  auto begin = other.m_starts[slice];
  auto end = slice + 1 < other.m_starts.size() ? other.m_starts[slice + 1]
                                               : other.m_map.end();
  std::copy(begin, end, std::back_inserter(m_map[slice]));
}

bool NameToDIE::Decode(const DataExtractor &data, lldb::offset_t *offset_ptr,
                       const StringTableReader &strtab) {
  for (auto &map : m_map)
    map.clear();
  return true;
}

void NameToDIE::Encode(DataEncoder &encoder, ConstStringTable &strtab) const {
}

bool NameToDIE::operator==(const NameToDIE &rhs) const {
  return m_map == rhs.m_map;
}
