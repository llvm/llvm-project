//===-- DWARFDebugRanges.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugRanges_h_
#define SymbolFileDWARF_DWARFDebugRanges_h_

#include "lldb/Core/dwarf.h"
#include <map>

class DWARFUnit;
namespace lldb_private {
class DWARFContext;
}

class DWARFDebugRangesBase {
public:
  virtual ~DWARFDebugRangesBase(){};

  virtual void Extract(lldb_private::DWARFContext &context) = 0;
  virtual bool FindRanges(const DWARFUnit *cu, dw_offset_t debug_ranges_offset,
                          DWARFRangeList &range_list) const = 0;
};

class DWARFDebugRanges final : public DWARFDebugRangesBase {
public:
  DWARFDebugRanges();

  void Extract(lldb_private::DWARFContext &context) override;
  bool FindRanges(const DWARFUnit *cu, dw_offset_t debug_ranges_offset,
                  DWARFRangeList &range_list) const override;

  static void Dump(lldb_private::Stream &s,
                   const lldb_private::DWARFDataExtractor &debug_ranges_data,
                   lldb::offset_t *offset_ptr, dw_addr_t cu_base_addr);

protected:
  bool Extract(lldb_private::DWARFContext &context, lldb::offset_t *offset_ptr,
               DWARFRangeList &range_list);

  typedef std::map<dw_offset_t, DWARFRangeList> range_map;
  typedef range_map::iterator range_map_iterator;
  typedef range_map::const_iterator range_map_const_iterator;
  range_map m_range_map;
};

#endif // SymbolFileDWARF_DWARFDebugRanges_h_
