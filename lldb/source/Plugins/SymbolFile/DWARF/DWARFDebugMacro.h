//===-- DWARFDebugMacro.h ----------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGMACRO_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGMACRO_H

#include <map>

#include "lldb/Core/dwarf.h"
#include "lldb/Symbol/DebugMacros.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

class DWARFDataExtractor;

} // namespace lldb_private

class SymbolFileDWARF;

class DWARFDebugMacroHeader {
public:
  enum HeaderFlagMask {
    OFFSET_SIZE_MASK = 0x1,
    DEBUG_LINE_OFFSET_MASK = 0x2,
    OPCODE_OPERANDS_TABLE_MASK = 0x4
  };

  static DWARFDebugMacroHeader
  ParseHeader(const lldb_private::DWARFDataExtractor &debug_macro_data,
              lldb::offset_t *offset);

  bool OffsetIs64Bit() const { return m_offset_is_64_bit; }

private:
  static void
  SkipOperandTable(const lldb_private::DWARFDataExtractor &debug_macro_data,
                   lldb::offset_t *offset);

  uint16_t m_version;
  bool m_offset_is_64_bit;
  uint64_t m_debug_line_offset;
};

class DWARFDebugMacroEntry {
public:
  static void
  ReadMacroEntries(const lldb_private::DWARFDataExtractor &debug_macro_data,
                   const lldb_private::DWARFDataExtractor &debug_str_data,
                   const bool offset_is_64_bit, lldb::offset_t *sect_offset,
                   SymbolFileDWARF *sym_file_dwarf,
                   lldb_private::DebugMacrosSP &debug_macros_sp);
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEBUGMACRO_H
