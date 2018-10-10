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

#include "DWARFDIE.h"
#include "SymbolFileDWARF.h"

#include <map>

class DWARFDebugRanges {
public:
  DWARFDebugRanges();
  virtual ~DWARFDebugRanges();
  virtual void Extract(SymbolFileDWARF *dwarf2Data);

  static void Dump(lldb_private::Stream &s,
                   const lldb_private::DWARFDataExtractor &debug_ranges_data,
                   lldb::offset_t *offset_ptr, dw_addr_t cu_base_addr);
  bool FindRanges(dw_addr_t debug_ranges_base,
                  dw_offset_t debug_ranges_offset,
                  DWARFRangeList &range_list) const;

protected:
  bool Extract(SymbolFileDWARF *dwarf2Data, lldb::offset_t *offset_ptr,
               DWARFRangeList &range_list);

  typedef std::map<dw_offset_t, DWARFRangeList> range_map;
  typedef range_map::iterator range_map_iterator;
  typedef range_map::const_iterator range_map_const_iterator;
  range_map m_range_map;
};

// DWARF v5 .debug_rnglists section.
class DWARFDebugRngLists : public DWARFDebugRanges {
public:
  void Extract(SymbolFileDWARF *dwarf2Data) override;

protected:
  bool ExtractRangeList(const lldb_private::DWARFDataExtractor &data,
                        uint8_t addrSize, lldb::offset_t *offset_ptr,
                        DWARFRangeList &list);

  std::vector<uint64_t> Offsets;
};

#endif // SymbolFileDWARF_DWARFDebugRanges_h_
