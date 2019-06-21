//===-- NameToDIE.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NameToDIE.h"
#include "DWARFUnit.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

void NameToDIE::Finalize() {
  m_map.Sort();
  m_map.SizeToFit();
}

void NameToDIE::Insert(ConstString name, const DIERef &die_ref) {
  m_map.Append(name, die_ref);
}

size_t NameToDIE::Find(ConstString name, DIEArray &info_array) const {
  return m_map.GetValues(name, info_array);
}

size_t NameToDIE::Find(const RegularExpression &regex,
                       DIEArray &info_array) const {
  return m_map.GetValues(regex, info_array);
}

size_t NameToDIE::FindAllEntriesForUnit(const DWARFUnit &unit,
                                        DIEArray &info_array) const {
  const size_t initial_size = info_array.size();
  const uint32_t size = m_map.GetSize();
  for (uint32_t i = 0; i < size; ++i) {
    const DIERef &die_ref = m_map.GetValueAtIndexUnchecked(i);
    if (unit.GetSymbolFileDWARF().GetDwoNum() == die_ref.dwo_num() &&
        unit.GetDebugSection() == die_ref.section() &&
        unit.GetOffset() <= die_ref.die_offset() &&
        die_ref.die_offset() < unit.GetNextUnitOffset())
      info_array.push_back(die_ref);
  }
  return info_array.size() - initial_size;
}

void NameToDIE::Dump(Stream *s) {
  const uint32_t size = m_map.GetSize();
  for (uint32_t i = 0; i < size; ++i) {
    s->Format("{0} \"{1}\"\n", m_map.GetValueAtIndexUnchecked(i),
              m_map.GetCStringAtIndexUnchecked(i));
  }
}

void NameToDIE::ForEach(
    std::function<bool(ConstString name, const DIERef &die_ref)> const
        &callback) const {
  const uint32_t size = m_map.GetSize();
  for (uint32_t i = 0; i < size; ++i) {
    if (!callback(m_map.GetCStringAtIndexUnchecked(i),
                  m_map.GetValueAtIndexUnchecked(i)))
      break;
  }
}

void NameToDIE::Append(const NameToDIE &other) {
  const uint32_t size = other.m_map.GetSize();
  for (uint32_t i = 0; i < size; ++i) {
    m_map.Append(other.m_map.GetCStringAtIndexUnchecked(i),
                 other.m_map.GetValueAtIndexUnchecked(i));
  }
}
