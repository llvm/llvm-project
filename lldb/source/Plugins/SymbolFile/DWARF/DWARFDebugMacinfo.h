//===-- DWARFDebugMacinfo.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugLine_h_
#define SymbolFileDWARF_DWARFDebugLine_h_

#include "SymbolFileDWARF.h"

class DWARFDebugMacinfo
{
public:
    DWARFDebugMacinfo();

    ~DWARFDebugMacinfo();

    static void
    Dump (lldb_private::Stream *s,
          const lldb_private::DataExtractor& macinfo_data,
          dw_offset_t offset = DW_INVALID_OFFSET);
};


#endif  // SymbolFileDWARF_DWARFDebugLine_h_
