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

const char *DW_ATE_value_to_name(uint32_t val);

const char *DW_LANG_value_to_name(uint32_t val);

const char *DW_LNS_value_to_name(uint32_t val);

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDEFINES_H
