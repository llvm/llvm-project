//===-- SymbolFileDWARFDwoDwp.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARFDwoDwp_SymbolFileDWARFDwoDwp_h_
#define SymbolFileDWARFDwoDwp_SymbolFileDWARFDwoDwp_h_

#include "SymbolFileDWARFDwo.h"
#include "SymbolFileDWARFDwp.h"

class SymbolFileDWARFDwoDwp : public SymbolFileDWARFDwo {
public:
  SymbolFileDWARFDwoDwp(SymbolFileDWARFDwp *dwp_symfile,
                        lldb::ObjectFileSP objfile, DWARFCompileUnit &dwarf_cu,
                        uint64_t dwo_id);

protected:
  void LoadSectionData(lldb::SectionType sect_type,
                       lldb_private::DWARFDataExtractor &data) override;

  SymbolFileDWARFDwp *m_dwp_symfile;
  uint64_t m_dwo_id;
};

#endif // SymbolFileDWARFDwoDwp_SymbolFileDWARFDwoDwp_h_
