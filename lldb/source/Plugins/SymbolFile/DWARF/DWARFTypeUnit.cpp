//===-- DWARFTypeUnit.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFTypeUnit.h"

#include "SymbolFileDWARF.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

void DWARFTypeUnit::Dump(Stream *s) const {
  s->Printf("0x%8.8x: Type Unit: length = 0x%8.8x, version = 0x%4.4x, "
            "abbr_offset = 0x%8.8x, addr_size = 0x%2.2x (next CU at "
            "{0x%8.8x})\n",
            GetOffset(), GetLength(), GetVersion(), GetAbbrevOffset(),
            GetAddressByteSize(), GetNextUnitOffset());
}
