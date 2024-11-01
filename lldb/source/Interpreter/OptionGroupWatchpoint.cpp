//===-- OptionGroupWatchpoint.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupWatchpoint.h"

#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Target/Language.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;

static constexpr OptionEnumValueElement g_watch_type[] = {
    {
        OptionGroupWatchpoint::eWatchRead,
        "read",
        "Watch for read",
    },
    {
        OptionGroupWatchpoint::eWatchWrite,
        "write",
        "Watch for write",
    },
    {
        OptionGroupWatchpoint::eWatchReadWrite,
        "read_write",
        "Watch for read/write",
    },
};

static constexpr OptionEnumValueElement g_watch_size[] = {
    {
        1,
        "1",
        "Watch for byte size of 1",
    },
    {
        2,
        "2",
        "Watch for byte size of 2",
    },
    {
        4,
        "4",
        "Watch for byte size of 4",
    },
    {
        8,
        "8",
        "Watch for byte size of 8",
    },
};

static constexpr OptionDefinition g_option_table[] = {
    {LLDB_OPT_SET_1, false, "watch", 'w', OptionParser::eRequiredArgument,
     nullptr, OptionEnumValues(g_watch_type), 0, eArgTypeWatchType,
     "Specify the type of watching to perform."},
    {LLDB_OPT_SET_1, false, "size", 's', OptionParser::eRequiredArgument,
     nullptr, OptionEnumValues(g_watch_size), 0, eArgTypeByteSize,
     "Number of bytes to use to watch a region."},
    {LLDB_OPT_SET_2,
     false,
     "language",
     'l',
     OptionParser::eRequiredArgument,
     nullptr,
     {},
     0,
     eArgTypeLanguage,
     "Language of expression to run"}};

bool OptionGroupWatchpoint::IsWatchSizeSupported(uint32_t watch_size) {
  for (const auto& size : g_watch_size) {
    if (0  == size.value)
      break;
    if (watch_size == size.value)
      return true;
  }
  return false;
}

Status
OptionGroupWatchpoint::SetOptionValue(uint32_t option_idx,
                                      llvm::StringRef option_arg,
                                      ExecutionContext *execution_context) {
  Status error;
  const int short_option = g_option_table[option_idx].short_option;
  switch (short_option) {
  case 'l': {
    language_type = Language::GetLanguageTypeFromString(option_arg);
    if (language_type == eLanguageTypeUnknown) {
      StreamString sstr;
      sstr.Printf("Unknown language type: '%s' for expression. List of "
                  "supported languages:\n",
                  option_arg.str().c_str());
      Language::PrintSupportedLanguagesForExpressions(sstr, " ", "\n");
      error.SetErrorString(sstr.GetString());
    }
    break;
  }
  case 'w': {
    WatchType tmp_watch_type;
    tmp_watch_type = (WatchType)OptionArgParser::ToOptionEnum(
        option_arg, g_option_table[option_idx].enum_values, 0, error);
    if (error.Success()) {
      watch_type = tmp_watch_type;
      watch_type_specified = true;
    }
    break;
  }
  case 's':
    watch_size = (uint32_t)OptionArgParser::ToOptionEnum(
        option_arg, g_option_table[option_idx].enum_values, 0, error);
    break;

  default:
    llvm_unreachable("Unimplemented option");
  }

  return error;
}

void OptionGroupWatchpoint::OptionParsingStarting(
    ExecutionContext *execution_context) {
  watch_type_specified = false;
  watch_type = eWatchInvalid;
  watch_size = 0;
  language_type = eLanguageTypeUnknown;
}

llvm::ArrayRef<OptionDefinition> OptionGroupWatchpoint::GetDefinitions() {
  return llvm::ArrayRef(g_option_table);
}
