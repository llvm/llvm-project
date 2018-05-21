//===-- DWARFIndex.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFIndex.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"

using namespace lldb_private;
using namespace lldb;

DWARFIndex::~DWARFIndex() = default;

void DWARFIndex::ParseFunctions(
    const DIEArray &offsets, DWARFDebugInfo &info,
    llvm::function_ref<bool(const DWARFDIE &die, bool include_inlines,
                            lldb_private::SymbolContextList &sc_list)>
        resolve_function,
    bool include_inlines, SymbolContextList &sc_list) {
  const size_t num_matches = offsets.size();
  for (size_t i = 0; i < num_matches; ++i) {
    DWARFDIE die = info.GetDIE(offsets[i]);
    if (die)
      resolve_function(die, include_inlines, sc_list);
  }
}
