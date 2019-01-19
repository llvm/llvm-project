//===-- DWARFDebugMacinfo.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugMacinfo.h"

#include "DWARFDebugMacinfoEntry.h"
#include "SymbolFileDWARF.h"

#include "lldb/Utility/Stream.h"

using namespace lldb_private;
using namespace std;

DWARFDebugMacinfo::DWARFDebugMacinfo() {}

DWARFDebugMacinfo::~DWARFDebugMacinfo() {}

void DWARFDebugMacinfo::Dump(Stream *s, const DWARFDataExtractor &macinfo_data,
                             lldb::offset_t offset) {
  DWARFDebugMacinfoEntry maninfo_entry;
  if (macinfo_data.GetByteSize() == 0) {
    s->PutCString("< EMPTY >\n");
    return;
  }
  if (offset == LLDB_INVALID_OFFSET) {
    offset = 0;
    while (maninfo_entry.Extract(macinfo_data, &offset))
      maninfo_entry.Dump(s);
  } else {
    if (maninfo_entry.Extract(macinfo_data, &offset))
      maninfo_entry.Dump(s);
  }
}
