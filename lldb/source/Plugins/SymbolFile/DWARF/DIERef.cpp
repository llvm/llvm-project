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
    if (const DWARFUnit *unit = form_value.GetUnit()) {
      if (unit->GetBaseObjOffset() != DW_INVALID_OFFSET)
        cu_offset = unit->GetBaseObjOffset();
      else
        cu_offset = unit->GetOffset();
    }
    die_offset = form_value.Reference();
  }
}
