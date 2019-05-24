//===-- DIERef.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DIERef_h_
#define SymbolFileDWARF_DIERef_h_

#include "lldb/Core/dwarf.h"
#include "lldb/lldb-defines.h"

class DWARFFormValue;
class SymbolFileDWARF;

struct DIERef {
  enum Section : uint8_t { DebugInfo, DebugTypes };

  DIERef() = default;

  DIERef(Section s, dw_offset_t c, dw_offset_t d)
      : section(s), cu_offset(c), die_offset(d) {}

  explicit DIERef(const DWARFFormValue &form_value);

  bool operator<(const DIERef &ref) const {
    return die_offset < ref.die_offset;
  }

  explicit operator bool() const {
    return cu_offset != DW_INVALID_OFFSET || die_offset != DW_INVALID_OFFSET;
  }

  Section section = Section::DebugInfo;
  dw_offset_t cu_offset = DW_INVALID_OFFSET;
  dw_offset_t die_offset = DW_INVALID_OFFSET;
};

typedef std::vector<DIERef> DIEArray;

#endif // SymbolFileDWARF_DIERef_h_
