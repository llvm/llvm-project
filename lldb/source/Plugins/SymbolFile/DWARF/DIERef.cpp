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

DIERef::DIERef(const DWARFFormValue &form_value) {
  if (form_value.IsValid()) {
    DWARFDIE die = form_value.Reference();
    die_offset = die.GetOffset();
    if (die) {
      section = die.GetCU()->GetDebugSection();
      if (die.GetCU()->GetBaseObjOffset() != DW_INVALID_OFFSET)
        cu_offset = die.GetCU()->GetBaseObjOffset();
      else
        cu_offset = die.GetCU()->GetOffset();
    }
  }
}
