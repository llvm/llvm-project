//===-- UniqueDWARFASTType.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_UNIQUEDWARFASTTYPE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_UNIQUEDWARFASTTYPE_H

#include <vector>

#include "llvm/ADT/DenseMap.h"

#include "DWARFDIE.h"
#include "lldb/Core/Declaration.h"

class UniqueDWARFASTType {
public:
  // Constructors and Destructors
  UniqueDWARFASTType() : m_type_sp(), m_die(), m_declaration() {}

  UniqueDWARFASTType(lldb::TypeSP &type_sp, const DWARFDIE &die,
                     const lldb_private::Declaration &decl, int32_t byte_size)
      : m_type_sp(type_sp), m_die(die), m_declaration(decl),
        m_byte_size(byte_size) {}

  UniqueDWARFASTType(const UniqueDWARFASTType &rhs)
      : m_type_sp(rhs.m_type_sp), m_die(rhs.m_die),
        m_declaration(rhs.m_declaration), m_byte_size(rhs.m_byte_size) {}

  ~UniqueDWARFASTType() = default;

  UniqueDWARFASTType &operator=(const UniqueDWARFASTType &rhs) {
    if (this != &rhs) {
      m_type_sp = rhs.m_type_sp;
      m_die = rhs.m_die;
      m_declaration = rhs.m_declaration;
      m_byte_size = rhs.m_byte_size;
    }
    return *this;
  }

  lldb::TypeSP m_type_sp;
  DWARFDIE m_die;
  lldb_private::Declaration m_declaration;
  int32_t m_byte_size = -1;
};

class UniqueDWARFASTTypeList {
public:
  UniqueDWARFASTTypeList() : m_collection() {}

  ~UniqueDWARFASTTypeList() = default;

  uint32_t GetSize() { return (uint32_t)m_collection.size(); }

  void Append(const UniqueDWARFASTType &entry) {
    m_collection.push_back(entry);
  }

  bool Find(const DWARFDIE &die, const lldb_private::Declaration &decl,
            const int32_t byte_size, UniqueDWARFASTType &entry) const;

protected:
  typedef std::vector<UniqueDWARFASTType> collection;
  collection m_collection;
};

class UniqueDWARFASTTypeMap {
public:
  UniqueDWARFASTTypeMap() : m_collection() {}

  ~UniqueDWARFASTTypeMap() = default;

  void Insert(lldb_private::ConstString name,
              const UniqueDWARFASTType &entry) {
    m_collection[name.GetCString()].Append(entry);
  }

  bool Find(lldb_private::ConstString name, const DWARFDIE &die,
            const lldb_private::Declaration &decl, const int32_t byte_size,
            UniqueDWARFASTType &entry) const {
    const char *unique_name_cstr = name.GetCString();
    collection::const_iterator pos = m_collection.find(unique_name_cstr);
    if (pos != m_collection.end()) {
      return pos->second.Find(die, decl, byte_size, entry);
    }
    return false;
  }

protected:
  // A unique name string should be used
  typedef llvm::DenseMap<const char *, UniqueDWARFASTTypeList> collection;
  collection m_collection;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_UNIQUEDWARFASTTYPE_H
