//===-- DWARFLocationList.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFLocationList_h_
#define SymbolFileDWARF_DWARFLocationList_h_

#include "SymbolFileDWARF.h"

class DWARFLocationList
{
public:
    static dw_offset_t
    Dump (lldb_private::Stream &s,
          const DWARFCompileUnit* cu,
          const lldb_private::DataExtractor& debug_loc_data,
          dw_offset_t offset);

    static bool
    Extract (const lldb_private::DataExtractor& debug_loc_data,
             dw_offset_t* offset_ptr,
             lldb_private::DataExtractor& location_list_data);

    static size_t
    Size (const lldb_private::DataExtractor& debug_loc_data,
          dw_offset_t offset);

};
#endif  // SymbolFileDWARF_DWARFLocationList_h_
