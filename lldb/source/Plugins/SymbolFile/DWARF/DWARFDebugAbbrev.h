//===-- DWARFDebugAbbrev.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGABBREV_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGABBREV_H

#include <list>
#include <map>

#include "lldb/lldb-private.h"

#include "DWARFAbbreviationDeclaration.h"
#include "DWARFDefines.h"

typedef std::vector<DWARFAbbreviationDeclaration>
    DWARFAbbreviationDeclarationColl;
typedef DWARFAbbreviationDeclarationColl::iterator
    DWARFAbbreviationDeclarationCollIter;
typedef DWARFAbbreviationDeclarationColl::const_iterator
    DWARFAbbreviationDeclarationCollConstIter;

class DWARFAbbreviationDeclarationSet {
public:
  DWARFAbbreviationDeclarationSet() : m_offset(DW_INVALID_OFFSET), m_decls() {}

  DWARFAbbreviationDeclarationSet(dw_offset_t offset, uint32_t idx_offset)
      : m_offset(offset), m_idx_offset(idx_offset), m_decls() {}

  void Clear();
  dw_offset_t GetOffset() const { return m_offset; }

  /// Extract all abbrev decls in a set.  Returns llvm::ErrorSuccess() on
  /// success, and an appropriate llvm::Error object otherwise.
  llvm::Error extract(const lldb_private::DWARFDataExtractor &data,
                      lldb::offset_t *offset_ptr);
  // void Encode(BinaryStreamBuf& debug_abbrev_buf) const;
  void GetUnsupportedForms(std::set<dw_form_t> &invalid_forms) const;

  const DWARFAbbreviationDeclaration *
  GetAbbreviationDeclaration(uint32_t abbrCode) const;

  /// Unit test accessor functions.
  /// @{
  uint32_t GetIndexOffset() const { return m_idx_offset; }
  /// @}
private:
  dw_offset_t m_offset;
  uint32_t m_idx_offset = 0;
  std::vector<DWARFAbbreviationDeclaration> m_decls;
};

typedef std::map<dw_offset_t, DWARFAbbreviationDeclarationSet>
    DWARFAbbreviationDeclarationCollMap;
typedef DWARFAbbreviationDeclarationCollMap::iterator
    DWARFAbbreviationDeclarationCollMapIter;
typedef DWARFAbbreviationDeclarationCollMap::const_iterator
    DWARFAbbreviationDeclarationCollMapConstIter;

class DWARFDebugAbbrev {
public:
  DWARFDebugAbbrev();
  const DWARFAbbreviationDeclarationSet *
  GetAbbreviationDeclarationSet(dw_offset_t cu_abbr_offset) const;
  /// Extract all abbreviations for a particular compile unit.  Returns
  /// llvm::ErrorSuccess() on success, and an appropriate llvm::Error object
  /// otherwise.
  llvm::Error parse(const lldb_private::DWARFDataExtractor &data);
  void GetUnsupportedForms(std::set<dw_form_t> &invalid_forms) const;

protected:
  DWARFAbbreviationDeclarationCollMap m_abbrevCollMap;
  mutable DWARFAbbreviationDeclarationCollMapConstIter m_prev_abbr_offset_pos;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGABBREV_H
