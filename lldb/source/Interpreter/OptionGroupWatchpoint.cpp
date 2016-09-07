//===-- OptionGroupWatchpoint.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupWatchpoint.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Utility/Utils.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;

static OptionEnumValueElement g_watch_type[] = {
    {OptionGroupWatchpoint::eWatchRead, "read", "Watch for read"},
    {OptionGroupWatchpoint::eWatchWrite, "write", "Watch for write"},
    {OptionGroupWatchpoint::eWatchReadWrite, "read_write",
     "Watch for read/write"},
    {0, nullptr, nullptr}};

static OptionEnumValueElement g_watch_size[] = {
    {1, "1", "Watch for byte size of 1"},
    {2, "2", "Watch for byte size of 2"},
    {4, "4", "Watch for byte size of 4"},
    {8, "8", "Watch for byte size of 8"},
    {0, nullptr, nullptr}};

static OptionDefinition g_option_table[] = {
    {LLDB_OPT_SET_1, false, "watch", 'w', OptionParser::eRequiredArgument,
     nullptr, g_watch_type, 0, eArgTypeWatchType,
     "Specify the type of watching to perform."},
    {LLDB_OPT_SET_1, false, "size", 's', OptionParser::eRequiredArgument,
     nullptr, g_watch_size, 0, eArgTypeByteSize,
     "Number of bytes to use to watch a region."}};

bool OptionGroupWatchpoint::IsWatchSizeSupported(uint32_t watch_size) {
  for (uint32_t i = 0; i < llvm::array_lengthof(g_watch_size); ++i) {
    if (g_watch_size[i].value == 0)
      break;
    if (watch_size == g_watch_size[i].value)
      return true;
  }
  return false;
}

OptionGroupWatchpoint::OptionGroupWatchpoint() : OptionGroup() {}

OptionGroupWatchpoint::~OptionGroupWatchpoint() {}

Error OptionGroupWatchpoint::SetOptionValue(
    uint32_t option_idx, const char *option_arg,
    ExecutionContext *execution_context) {
  Error error;
  const int short_option = g_option_table[option_idx].short_option;
  switch (short_option) {
  case 'w': {
    WatchType tmp_watch_type;
    tmp_watch_type = (WatchType)Args::StringToOptionEnum(
        option_arg, g_option_table[option_idx].enum_values, 0, error);
    if (error.Success()) {
      watch_type = tmp_watch_type;
      watch_type_specified = true;
    }
    break;
  }
  case 's':
    watch_size = (uint32_t)Args::StringToOptionEnum(
        option_arg, g_option_table[option_idx].enum_values, 0, error);
    break;

  default:
    error.SetErrorStringWithFormat("unrecognized short option '%c'",
                                   short_option);
    break;
  }

  return error;
}

void OptionGroupWatchpoint::OptionParsingStarting(
    ExecutionContext *execution_context) {
  watch_type_specified = false;
  watch_type = eWatchInvalid;
  watch_size = 0;
}

const OptionDefinition *OptionGroupWatchpoint::GetDefinitions() {
  return g_option_table;
}

uint32_t OptionGroupWatchpoint::GetNumDefinitions() {
  return llvm::array_lengthof(g_option_table);
}
