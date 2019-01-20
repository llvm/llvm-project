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

DIERef::DIERef(lldb::user_id_t uid, SymbolFileDWARF *dwarf)
    : cu_offset(DW_INVALID_OFFSET), die_offset(uid & 0xffffffff) {
  SymbolFileDWARFDebugMap *debug_map = dwarf->GetDebugMapSymfile();
  if (debug_map) {
    const uint32_t oso_idx = debug_map->GetOSOIndexFromUserID(uid);
    SymbolFileDWARF *actual_dwarf = debug_map->GetSymbolFileByOSOIndex(oso_idx);
    if (actual_dwarf) {
      DWARFDebugInfo *debug_info = actual_dwarf->DebugInfo();
      if (debug_info) {
        DWARFUnit *dwarf_cu =
            debug_info->GetCompileUnitContainingDIEOffset(die_offset);
        if (dwarf_cu) {
          cu_offset = dwarf_cu->GetOffset();
          return;
        }
      }
    }
    die_offset = DW_INVALID_OFFSET;
  } else {
    cu_offset = uid >> 32;
  }
}

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

lldb::user_id_t DIERef::GetUID(SymbolFileDWARF *dwarf) const {
  //----------------------------------------------------------------------
  // Each SymbolFileDWARF will set its ID to what is expected.
  //
  // SymbolFileDWARF, when used for DWARF with .o files on MacOSX, has the
  // ID set to the compile unit index.
  //
  // SymbolFileDWARFDwo sets the ID to the compile unit offset.
  //----------------------------------------------------------------------
  if (dwarf && die_offset != DW_INVALID_OFFSET)
    return dwarf->GetID() | die_offset;
  else
    return LLDB_INVALID_UID;
}
