//===-- DWARFDebugAbbrev.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGABBREV_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGABBREV_H

#include "DWARFDefines.h"
#include "lldb/lldb-private.h"

#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"

#include <map>

using DWARFAbbreviationDeclaration = llvm::DWARFAbbreviationDeclaration;
using DWARFAbbreviationDeclarationSet = llvm::DWARFAbbreviationDeclarationSet;

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
