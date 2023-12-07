//===-- DWARFDebugAbbrev.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugAbbrev.h"
#include "DWARFDataExtractor.h"
#include "DWARFFormValue.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

// DWARFDebugAbbrev constructor
DWARFDebugAbbrev::DWARFDebugAbbrev()
    : m_abbrevCollMap(), m_prev_abbr_offset_pos(m_abbrevCollMap.end()) {}

// DWARFDebugAbbrev::Parse()
llvm::Error DWARFDebugAbbrev::parse(const DWARFDataExtractor &data) {
  llvm::DataExtractor llvm_data = data.GetAsLLVM();
  lldb::offset_t offset = 0;

  while (llvm_data.isValidOffset(offset)) {
    uint32_t initial_cu_offset = offset;
    DWARFAbbreviationDeclarationSet abbrevDeclSet;

    llvm::Error error = abbrevDeclSet.extract(llvm_data, &offset);
    if (error)
      return error;

    m_abbrevCollMap[initial_cu_offset] = abbrevDeclSet;
  }
  m_prev_abbr_offset_pos = m_abbrevCollMap.end();
  return llvm::ErrorSuccess();
}

// DWARFDebugAbbrev::GetAbbreviationDeclarationSet()
const DWARFAbbreviationDeclarationSet *
DWARFDebugAbbrev::GetAbbreviationDeclarationSet(
    dw_offset_t cu_abbr_offset) const {
  DWARFAbbreviationDeclarationCollMapConstIter end = m_abbrevCollMap.end();
  DWARFAbbreviationDeclarationCollMapConstIter pos;
  if (m_prev_abbr_offset_pos != end &&
      m_prev_abbr_offset_pos->first == cu_abbr_offset)
    return &(m_prev_abbr_offset_pos->second);
  else {
    pos = m_abbrevCollMap.find(cu_abbr_offset);
    m_prev_abbr_offset_pos = pos;
  }

  if (pos != m_abbrevCollMap.end())
    return &(pos->second);
  return nullptr;
}

// DWARFDebugAbbrev::GetUnsupportedForms()
void DWARFDebugAbbrev::GetUnsupportedForms(
    std::set<dw_form_t> &invalid_forms) const {
  for (const auto &pair : m_abbrevCollMap)
    for (const auto &decl : pair.second)
      for (const auto &attr : decl.attributes())
        if (!DWARFFormValue::FormIsSupported(attr.Form))
          invalid_forms.insert(attr.Form);
}
