//===-- DIERef.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DIERef.h"
#include "DWARFUnit.h"
#include "DWARFDebugInfo.h"
#include "DWARFFormValue.h"
#include "SymbolFileDWARF.h"
#include "SymbolFileDWARFDebugMap.h"

DIERef::DIERef(const DWARFFormValue &form_value)
    : cu_offset(DW_INVALID_OFFSET), die_offset(DW_INVALID_OFFSET) {
  if (form_value.IsValid()) {
    const DWARFUnit *dwarf_cu = form_value.GetCompileUnit();
    if (dwarf_cu) {
      if (dwarf_cu->GetBaseObjOffset() != DW_INVALID_OFFSET)
        cu_offset = dwarf_cu->GetBaseObjOffset();
      else
        cu_offset = dwarf_cu->GetOffset();
    }
    die_offset = form_value.Reference();
  }
}
