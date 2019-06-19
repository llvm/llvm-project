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
#include <vector>

struct DIERef {
  enum Section : uint8_t { DebugInfo, DebugTypes };

  DIERef(Section s, dw_offset_t c, dw_offset_t d)
      : section(s), cu_offset(c), die_offset(d) {}

  Section section;
  dw_offset_t cu_offset;
  dw_offset_t die_offset;
};

typedef std::vector<DIERef> DIEArray;

#endif // SymbolFileDWARF_DIERef_h_
