//===-- DWARFDebugRanges.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugRanges_h_
#define SymbolFileDWARF_DWARFDebugRanges_h_

#include "SymbolFileDWARF.h"

#include <map>
#include <vector>

#include "lldb/Core/RangeMap.h"

class DWARFDebugRanges
{
public:
    typedef lldb_private::RangeArray<dw_addr_t, dw_addr_t, 2> RangeList;
    typedef RangeList::Entry Range;

    DWARFDebugRanges();
    ~DWARFDebugRanges();
    void Extract(SymbolFileDWARF* dwarf2Data);
    static void Dump(lldb_private::Stream &s, const lldb_private::DataExtractor& debug_ranges_data, lldb::offset_t *offset_ptr, dw_addr_t cu_base_addr);
    bool FindRanges(dw_offset_t debug_ranges_offset, DWARFDebugRanges::RangeList& range_list) const;

protected:

    bool
    Extract (SymbolFileDWARF* dwarf2Data, 
             lldb::offset_t *offset_ptr, 
             RangeList &range_list);

    typedef std::map<dw_offset_t, RangeList>    range_map;
    typedef range_map::iterator                 range_map_iterator;
    typedef range_map::const_iterator           range_map_const_iterator;
    range_map m_range_map;
};


#endif  // SymbolFileDWARF_DWARFDebugRanges_h_
