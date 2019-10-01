//===-- DWARFDebugAbbrev.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugAbbrev.h"
#include "DWARFDataExtractor.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;
using namespace std;

// DWARFAbbreviationDeclarationSet::Clear()
void DWARFAbbreviationDeclarationSet::Clear() {
  m_idx_offset = 0;
  m_decls.clear();
}

// DWARFAbbreviationDeclarationSet::Extract()
llvm::Error
DWARFAbbreviationDeclarationSet::extract(const DWARFDataExtractor &data,
                                         lldb::offset_t *offset_ptr) {
  const lldb::offset_t begin_offset = *offset_ptr;
  m_offset = begin_offset;
  Clear();
  DWARFAbbreviationDeclaration abbrevDeclaration;
  dw_uleb128_t prev_abbr_code = 0;
  while (true) {
    llvm::Expected<DWARFEnumState> es =
        abbrevDeclaration.extract(data, offset_ptr);
    if (!es)
      return es.takeError();
    if (*es == DWARFEnumState::Complete)
      break;
    m_decls.push_back(abbrevDeclaration);
    if (m_idx_offset == 0)
      m_idx_offset = abbrevDeclaration.Code();
    else if (prev_abbr_code + 1 != abbrevDeclaration.Code()) {
      // Out of order indexes, we can't do O(1) lookups...
      m_idx_offset = UINT32_MAX;
    }
    prev_abbr_code = abbrevDeclaration.Code();
  }
  return llvm::ErrorSuccess();
}

// DWARFAbbreviationDeclarationSet::GetAbbreviationDeclaration()
const DWARFAbbreviationDeclaration *
DWARFAbbreviationDeclarationSet::GetAbbreviationDeclaration(
    dw_uleb128_t abbrCode) const {
  if (m_idx_offset == UINT32_MAX) {
    DWARFAbbreviationDeclarationCollConstIter pos;
    DWARFAbbreviationDeclarationCollConstIter end = m_decls.end();
    for (pos = m_decls.begin(); pos != end; ++pos) {
      if (pos->Code() == abbrCode)
        return &(*pos);
    }
  } else {
    uint32_t idx = abbrCode - m_idx_offset;
    if (idx < m_decls.size())
      return &m_decls[idx];
  }
  return nullptr;
}


// DWARFAbbreviationDeclarationSet::GetUnsupportedForms()
void DWARFAbbreviationDeclarationSet::GetUnsupportedForms(
    std::set<dw_form_t> &invalid_forms) const {
  for (const auto &abbr_decl : m_decls) {
    const size_t num_attrs = abbr_decl.NumAttributes();
    for (size_t i=0; i<num_attrs; ++i) {
      dw_form_t form = abbr_decl.GetFormByIndex(i);
      if (!DWARFFormValue::FormIsSupported(form))
        invalid_forms.insert(form);
    }
  }
}

// Encode
//
// Encode the abbreviation table onto the end of the buffer provided into a
// byte representation as would be found in a ".debug_abbrev" debug information
// section.
// void
// DWARFAbbreviationDeclarationSet::Encode(BinaryStreamBuf& debug_abbrev_buf)
// const
//{
//  DWARFAbbreviationDeclarationCollConstIter pos;
//  DWARFAbbreviationDeclarationCollConstIter end = m_decls.end();
//  for (pos = m_decls.begin(); pos != end; ++pos)
//      pos->Append(debug_abbrev_buf);
//  debug_abbrev_buf.Append8(0);
//}

// DWARFDebugAbbrev constructor
DWARFDebugAbbrev::DWARFDebugAbbrev()
    : m_abbrevCollMap(), m_prev_abbr_offset_pos(m_abbrevCollMap.end()) {}

// DWARFDebugAbbrev::Parse()
llvm::Error DWARFDebugAbbrev::parse(const DWARFDataExtractor &data) {
  lldb::offset_t offset = 0;

  while (data.ValidOffset(offset)) {
    uint32_t initial_cu_offset = offset;
    DWARFAbbreviationDeclarationSet abbrevDeclSet;

    llvm::Error error = abbrevDeclSet.extract(data, &offset);
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
    pair.second.GetUnsupportedForms(invalid_forms);
}
