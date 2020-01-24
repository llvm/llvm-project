//===-- SymbolFileDWARFDwoDwp.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARFDwoDwp.h"

#include "lldb/Core/Section.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/LLDBAssert.h"

#include "DWARFUnit.h"
#include "DWARFDebugInfo.h"

using namespace lldb;
using namespace lldb_private;

char SymbolFileDWARFDwoDwp::ID;

SymbolFileDWARFDwoDwp::SymbolFileDWARFDwoDwp(SymbolFileDWARFDwp *dwp_symfile,
                                             ObjectFileSP objfile,
                                             DWARFCompileUnit &dwarf_cu,
                                             uint64_t dwo_id)
    : SymbolFileDWARFDwo(objfile, dwarf_cu), m_dwp_symfile(dwp_symfile),
      m_dwo_id(dwo_id) {}

void SymbolFileDWARFDwoDwp::LoadSectionData(lldb::SectionType sect_type,
                                            DWARFDataExtractor &data) {
  if (m_dwp_symfile->LoadSectionData(m_dwo_id, sect_type, data))
    return;

  SymbolFileDWARF::LoadSectionData(sect_type, data);
}
