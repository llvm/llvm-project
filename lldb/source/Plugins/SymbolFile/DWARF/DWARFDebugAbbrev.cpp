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
DWARFDebugAbbrev::DWARFDebugAbbrev(const DWARFDataExtractor &data)
    : m_abbrevCollMap(), m_prev_abbr_offset_pos(m_abbrevCollMap.end()),
      m_data(data.GetAsLLVM()) {}

// DWARFDebugAbbrev::Parse()
llvm::Error DWARFDebugAbbrev::parse() {
  if (!m_data)
    return llvm::Error::success();

  lldb::offset_t offset = 0;

  while (m_data->isValidOffset(offset)) {
    uint32_t initial_cu_offset = offset;
    DWARFAbbreviationDeclarationSet abbrevDeclSet;

    llvm::Error error = abbrevDeclSet.extract(*m_data, &offset);
    if (error) {
      m_data = std::nullopt;
      return error;
    }

    m_abbrevCollMap[initial_cu_offset] = abbrevDeclSet;
  }
  m_data = std::nullopt;
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
