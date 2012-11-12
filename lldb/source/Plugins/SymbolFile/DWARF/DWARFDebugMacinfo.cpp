//===-- DWARFDebugMacinfo.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugMacinfo.h"

#include "DWARFDebugMacinfoEntry.h"
#include "SymbolFileDWARF.h"

#include "lldb/Core/Stream.h"

using namespace lldb_private;
using namespace std;

DWARFDebugMacinfo::DWARFDebugMacinfo()
{
}

DWARFDebugMacinfo::~DWARFDebugMacinfo()
{
}

void
DWARFDebugMacinfo::Dump(Stream *s, const DataExtractor& macinfo_data, dw_offset_t offset)
{
    DWARFDebugMacinfoEntry maninfo_entry;
    if (macinfo_data.GetByteSize() == 0)
    {
        s->PutCString("< EMPTY >\n");
        return;
    }
    if (offset == DW_INVALID_OFFSET)
    {
        offset = 0;
        while (maninfo_entry.Extract(macinfo_data, &offset))
            maninfo_entry.Dump(s);
    }
    else
    {
        if (maninfo_entry.Extract(macinfo_data, &offset))
            maninfo_entry.Dump(s);
    }
}
