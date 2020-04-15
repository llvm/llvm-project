//===-- NameToDIE.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H

#include <functional>

#include "DIERef.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Core/dwarf.h"
#include "lldb/lldb-defines.h"

class DWARFUnit;

class NameToDIE {
public:
  NameToDIE() : m_map() {}

  ~NameToDIE() {}

  void Dump(lldb_private::Stream *s);

  void Insert(lldb_private::ConstString name, const DIERef &die_ref);

  void Append(const NameToDIE &other);

  void Finalize();

  size_t Find(lldb_private::ConstString name,
              DIEArray &info_array) const;

  size_t Find(const lldb_private::RegularExpression &regex,
              DIEArray &info_array) const;

  size_t FindAllEntriesForUnit(const DWARFUnit &unit,
                               DIEArray &info_array) const;

  void
  ForEach(std::function<bool(lldb_private::ConstString name,
                             const DIERef &die_ref)> const
              &callback) const;

protected:
  lldb_private::UniqueCStringMap<DIERef> m_map;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H
