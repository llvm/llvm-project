//===-- DWARFDefines.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEFINES_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEFINES_H

#include "lldb/Core/dwarf.h"
#include <stdint.h>

namespace lldb_private {

enum class DWARFEnumState { MoreItems, Complete };

typedef uint32_t DRC_class; // Holds DRC_* class bitfields

const char *DW_TAG_value_to_name(uint32_t val);

const char *DW_AT_value_to_name(uint32_t val);

const char *DW_FORM_value_to_name(uint32_t val);

const char *DW_OP_value_to_name(uint32_t val);

DRC_class DW_OP_value_to_class(uint32_t val);

const char *DW_ATE_value_to_name(uint32_t val);

const char *DW_LANG_value_to_name(uint32_t val);

const char *DW_LNS_value_to_name(uint32_t val);

/* These DRC are entirely our own construction,
    although they are derived from various comments in the DWARF standard.
    Most of these are not useful to the parser, but the DW_AT and DW_FORM
    classes should prove to be usable in some fashion.  */

#define DRC_0x65 0x1
#define DRC_ADDRESS 0x2
#define DRC_BLOCK 0x4
#define DRC_CONSTANT 0x8
#define DRC_DWARFv3 0x10
#define DRC_FLAG 0x20
#define DRC_INDIRECT_SPECIAL 0x40
#define DRC_LINEPTR 0x80
#define DRC_LOCEXPR 0x100
#define DRC_LOCLISTPTR 0x200
#define DRC_MACPTR 0x400
#define DRC_ONEOPERAND 0x800
#define DRC_OPERANDONE_1BYTE_DELTA 0x1000
#define DRC_OPERANDONE_2BYTE_DELTA 0x2000
#define DRC_OPERANDONE_4BYTE_DELTA 0x4000
#define DRC_OPERANDONE_ADDRESS 0x8000
#define DRC_OPERANDONE_BLOCK 0x10000
#define DRC_OPERANDONE_SLEB128_OFFSET 0x20000
#define DRC_OPERANDONE_ULEB128_OFFSET 0x40000
#define DRC_OPERANDONE_ULEB128_REGISTER 0x80000
#define DRC_OPERANDTWO_BLOCK 0x100000
#define DRC_OPERANDTWO_SLEB128_OFFSET 0x200000
#define DRC_OPERANDTWO_ULEB128_OFFSET 0x400000
#define DRC_OPERANDTWO_ULEB128_REGISTER 0x800000
#define DRC_OPERNADONE_ULEB128_REGISTER 0x1000000
#define DRC_RANGELISTPTR 0x2000000
#define DRC_REFERENCE 0x4000000
#define DRC_STRING 0x8000000
#define DRC_TWOOPERANDS 0x10000000
#define DRC_VENDOR_GNU 0x20000000
#define DRC_VENDOR_MIPS 0x40000000
#define DRC_ZEROOPERANDS 0x80000000

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEFINES_H
