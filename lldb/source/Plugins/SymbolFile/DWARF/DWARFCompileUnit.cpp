//===-- DWARFCompileUnit.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFCompileUnit.h"
#include "DWARFDebugAranges.h"
#include "SymbolFileDWARFDebugMap.h"

#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

void DWARFCompileUnit::Dump(Stream *s) const {
  s->Printf("0x%8.8x: Compile Unit: length = 0x%8.8x, version = 0x%4.4x, "
            "abbr_offset = 0x%8.8x, addr_size = 0x%2.2x (next CU at "
            "{0x%8.8x})\n",
            GetOffset(), GetLength(), GetVersion(), GetAbbrevOffset(),
            GetAddressByteSize(), GetNextUnitOffset());
}

void DWARFCompileUnit::BuildAddressRangeTable(
    DWARFDebugAranges *debug_aranges) {
  // This function is usually called if there in no .debug_aranges section in
  // order to produce a compile unit level set of address ranges that is
  // accurate.

  size_t num_debug_aranges = debug_aranges->GetNumRanges();

  // First get the compile unit DIE only and check if it has a DW_AT_ranges
  const DWARFDebugInfoEntry *die = GetUnitDIEPtrOnly();

  const dw_offset_t cu_offset = GetOffset();
  if (die) {
    DWARFRangeList ranges;
    const size_t num_ranges =
        die->GetAttributeAddressRanges(this, ranges, false);
    if (num_ranges > 0) {
      // This compile unit has DW_AT_ranges, assume this is correct if it is
      // present since clang no longer makes .debug_aranges by default and it
      // emits DW_AT_ranges for DW_TAG_compile_units. GCC also does this with
      // recent GCC builds.
      for (size_t i = 0; i < num_ranges; ++i) {
        const DWARFRangeList::Entry &range = ranges.GetEntryRef(i);
        debug_aranges->AppendRange(cu_offset, range.GetRangeBase(),
                                   range.GetRangeEnd());
      }

      return; // We got all of our ranges from the DW_AT_ranges attribute
    }
  }
  // We don't have a DW_AT_ranges attribute, so we need to parse the DWARF

  // If the DIEs weren't parsed, then we don't want all dies for all compile
  // units to stay loaded when they weren't needed. So we can end up parsing
  // the DWARF and then throwing them all away to keep memory usage down.
  ScopedExtractDIEs clear_dies(ExtractDIEsScoped());

  die = DIEPtr();
  if (die)
    die->BuildAddressRangeTable(this, debug_aranges);

  if (debug_aranges->GetNumRanges() == num_debug_aranges) {
    // We got nothing from the functions, maybe we have a line tables only
    // situation. Check the line tables and build the arange table from this.
    SymbolContext sc;
    sc.comp_unit = m_dwarf.GetCompUnitForDWARFCompUnit(*this);
    if (sc.comp_unit) {
      SymbolFileDWARFDebugMap *debug_map_sym_file =
          m_dwarf.GetDebugMapSymfile();
      if (debug_map_sym_file == nullptr) {
        if (LineTable *line_table = sc.comp_unit->GetLineTable()) {
          LineTable::FileAddressRanges file_ranges;
          const bool append = true;
          const size_t num_ranges =
              line_table->GetContiguousFileAddressRanges(file_ranges, append);
          for (uint32_t idx = 0; idx < num_ranges; ++idx) {
            const LineTable::FileAddressRanges::Entry &range =
                file_ranges.GetEntryRef(idx);
            debug_aranges->AppendRange(cu_offset, range.GetRangeBase(),
                                       range.GetRangeEnd());
          }
        }
      } else
        debug_map_sym_file->AddOSOARanges(&m_dwarf, debug_aranges);
    }
  }

  if (debug_aranges->GetNumRanges() == num_debug_aranges) {
    // We got nothing from the functions, maybe we have a line tables only
    // situation. Check the line tables and build the arange table from this.
    SymbolContext sc;
    sc.comp_unit = m_dwarf.GetCompUnitForDWARFCompUnit(*this);
    if (sc.comp_unit) {
      if (LineTable *line_table = sc.comp_unit->GetLineTable()) {
        LineTable::FileAddressRanges file_ranges;
        const bool append = true;
        const size_t num_ranges =
            line_table->GetContiguousFileAddressRanges(file_ranges, append);
        for (uint32_t idx = 0; idx < num_ranges; ++idx) {
          const LineTable::FileAddressRanges::Entry &range =
              file_ranges.GetEntryRef(idx);
          debug_aranges->AppendRange(GetOffset(), range.GetRangeBase(),
                                     range.GetRangeEnd());
        }
      }
    }
  }
}
