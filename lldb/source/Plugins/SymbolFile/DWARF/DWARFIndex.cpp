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
