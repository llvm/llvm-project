//===-- DWARFCompileUnit.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFCompileUnit_h_
#define SymbolFileDWARF_DWARFCompileUnit_h_

#include "DWARFUnit.h"

class DWARFCompileUnit : public DWARFUnit {
  friend class DWARFUnit;

public:
  static DWARFUnitSP Extract(SymbolFileDWARF *dwarf2Data,
      lldb::offset_t *offset_ptr);
  void Dump(lldb_private::Stream *s) const override;

private:
  DWARFCompileUnit(SymbolFileDWARF *dwarf2Data);
  DISALLOW_COPY_AND_ASSIGN(DWARFCompileUnit);
};

#endif // SymbolFileDWARF_DWARFCompileUnit_h_
