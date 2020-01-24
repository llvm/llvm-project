//===-- LogChannelDWARF.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LogChannelDWARF.h"

using namespace lldb_private;

static constexpr Log::Category g_categories[] = {
    {{"comp"},
     {"log insertions of object files into DWARF debug maps"},
     DWARF_LOG_TYPE_COMPLETION},
    {{"info"}, {"log the parsing of .debug_info"}, DWARF_LOG_DEBUG_INFO},
    {{"line"}, {"log the parsing of .debug_line"}, DWARF_LOG_DEBUG_LINE},
    {{"lookups"},
     {"log any lookups that happen by name, regex, or address"},
     DWARF_LOG_LOOKUPS},
    {{"map"},
     {"log struct/unions/class type completions"},
     DWARF_LOG_DEBUG_MAP},
};

Log::Channel LogChannelDWARF::g_channel(g_categories, DWARF_LOG_DEFAULT);

void LogChannelDWARF::Initialize() {
  Log::Register("dwarf", g_channel);
}

void LogChannelDWARF::Terminate() { Log::Unregister("dwarf"); }
