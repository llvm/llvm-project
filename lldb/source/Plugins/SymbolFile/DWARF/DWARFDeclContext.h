//===-- DWARFDeclContext.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDECLCONTEXT_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDECLCONTEXT_H

#include "lldb/Utility/ConstString.h"
#include "DWARFDefines.h"

#include <cassert>
#include <string>
#include <vector>

namespace lldb_private::plugin {
namespace dwarf {
// DWARFDeclContext
//
// A class that represents a declaration context all the way down to a
// DIE. This is useful when trying to find a DIE in one DWARF to a DIE
// in another DWARF file.

class DWARFDeclContext {
public:
  struct Entry {
    Entry() = default;
    Entry(dw_tag_t t, const char *n) : tag(t), name(n) {}

    bool NameMatches(const Entry &rhs) const {
      if (name == rhs.name)
        return true;
      else if (name && rhs.name)
        return strcmp(name, rhs.name) == 0;
      return false;
    }

    // Test operator
    explicit operator bool() const { return tag != 0; }

    dw_tag_t tag = llvm::dwarf::DW_TAG_null;
    const char *name = nullptr;
  };

  DWARFDeclContext() : m_entries() {}

  DWARFDeclContext(llvm::ArrayRef<Entry> entries) {
    llvm::append_range(m_entries, entries);
  }

  void AppendDeclContext(dw_tag_t tag, const char *name) {
    m_entries.push_back(Entry(tag, name));
  }

  bool operator==(const DWARFDeclContext &rhs) const;
  bool operator!=(const DWARFDeclContext &rhs) const { return !(*this == rhs); }

  uint32_t GetSize() const { return m_entries.size(); }

  Entry &operator[](uint32_t idx) {
    assert(idx < m_entries.size() && "invalid index");
    return m_entries[idx];
  }

  const Entry &operator[](uint32_t idx) const {
    assert(idx < m_entries.size() && "invalid index");
    return m_entries[idx];
  }

  const char *GetQualifiedName() const;

  // Same as GetQualifiedName, but the life time of the returned string will
  // be that of the LLDB session.
  ConstString GetQualifiedNameAsConstString() const {
    return ConstString(GetQualifiedName());
  }

  void Clear() {
    m_entries.clear();
    m_qualified_name.clear();
  }

protected:
  typedef std::vector<Entry> collection;
  collection m_entries;
  mutable std::string m_qualified_name;
};
} // namespace dwarf
} // namespace lldb_private::plugin

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDECLCONTEXT_H
