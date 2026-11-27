//===-- OptionGroupWatchpoint.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupWatchpoint.h"

#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Target/Language.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_watchpoint
#include "InterpreterOptions.inc"

Status
OptionGroupWatchpoint::SetOptionValue(uint32_t option_idx,
                                      llvm::StringRef option_arg,
                                      ExecutionContext *execution_context) {
  Status error;
  const int short_option = g_watchpoint_options[option_idx].short_option;
  switch (short_option) {
  case 'l': {
    language_type = Language::GetLanguageTypeFromString(option_arg);
    if (language_type == eLanguageTypeUnknown) {
      StreamString sstr;
      sstr.Printf("Unknown language type: '%s' for expression. List of "
                  "supported languages:\n",
                  option_arg.str().c_str());
      Language::PrintSupportedLanguagesForExpressions(sstr, " ", "\n");
      error = Status(sstr.GetString().str());
    }
    break;
  }
  case 'w': {
    WatchType tmp_watch_type;
    tmp_watch_type = (WatchType)OptionArgParser::ToOptionEnum(
        option_arg, g_watchpoint_options[option_idx].enum_values, 0, error);
    if (error.Success()) {
      watch_type = tmp_watch_type;
      watch_type_specified = true;
    }
    break;
  }
  case 's':
    error = watch_size.SetValueFromString(option_arg);
    if (watch_size.GetCurrentValue() == 0)
      error = Status::FromErrorStringWithFormat(
          "invalid --size option value '%s'", option_arg.str().c_str());
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
  watch_size.Clear();
  language_type = eLanguageTypeUnknown;
}

llvm::ArrayRef<OptionDefinition> OptionGroupWatchpoint::GetDefinitions() {
  return llvm::ArrayRef(g_watchpoint_options);
}
