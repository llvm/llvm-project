//===-- DWARFWasm.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFWASM_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFWASM_H

#include "Plugins/SymbolFile/DWARF/DWARFWasm.h"

namespace lldb_private::plugin {
namespace dwarf {

enum DWARFWasmLocation { eLocal = 0, eGlobal, eOperandStack, eGlobalU32 };

} // namespace dwarf
} // namespace lldb_private::plugin

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFWASM_H
