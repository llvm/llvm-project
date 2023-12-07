//===-- DWARFDebugRanges.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGRANGES_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGRANGES_H

#include "lldb/Core/dwarf.h"
#include <map>

class DWARFUnit;
namespace lldb_private {
class DWARFContext;
}

class DWARFDebugRanges {
public:
  DWARFDebugRanges();

  void Extract(lldb_private::DWARFContext &context);
  DWARFRangeList FindRanges(const DWARFUnit *cu,
                            dw_offset_t debug_ranges_offset) const;

protected:
  std::map<dw_offset_t, DWARFRangeList> m_range_map;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGRANGES_H
