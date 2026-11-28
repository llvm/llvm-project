//===-- OptionGroupVariable.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupVariable.h"

#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_variable
#include "InterpreterOptions.inc"

static Status ValidateNamedSummary(const char *str, void *) {
  if (!str || !str[0])
    return Status::FromErrorStringWithFormat(
        "must specify a valid named summary");
  TypeSummaryImplSP summary_sp;
  if (!DataVisualization::NamedSummaryFormats::GetSummaryFormat(
          ConstString(str), summary_sp))
    return Status::FromErrorStringWithFormat(
        "must specify a valid named summary");
  return Status();
}

static Status ValidateSummaryString(const char *str, void *) {
  if (!str || !str[0])
    return Status::FromErrorStringWithFormat(
        "must specify a non-empty summary string");
  return Status();
}

OptionGroupVariable::OptionGroupVariable(bool show_frame_options)
    : include_frame_options(show_frame_options), show_args(false),
      show_recognized_args(false), show_locals(false), show_globals(false),
      show_synthetic(true), use_regex(false), show_scope(false),
      show_decl(false), summary(ValidateNamedSummary),
      summary_string(ValidateSummaryString) {}

Status
OptionGroupVariable::SetOptionValue(uint32_t option_idx,
                                    llvm::StringRef option_arg,
                                    ExecutionContext *execution_context) {
  Status error;
  const int short_option = GetDefinitions()[option_idx].short_option;
  switch (short_option) {
  case 'r':
    use_regex = true;
    break;
  case 'a':
    show_args = false;
    break;
  case 'l':
    show_locals = false;
    break;
  case 'g':
    show_globals = true;
    break;
  case 'e':
    show_synthetic = false;
    break;
  case 'c':
    show_decl = true;
    break;
  case 's':
    show_scope = true;
    break;
  case 't':
    show_recognized_args = false;
    break;
  case 'y':
    error = summary.SetCurrentValue(option_arg);
    break;
  case 'z':
    error = summary_string.SetCurrentValue(option_arg);
    break;
  default:
    llvm_unreachable("Unimplemented option");
  }

  return error;
}

void OptionGroupVariable::OptionParsingStarting(
    ExecutionContext *execution_context) {
  show_args = true;            // Frame option only
  show_recognized_args = true; // Frame option only
  show_locals = true;          // Frame option only
  show_globals = false;        // Frame option only
  show_synthetic = true;       // Frame option only
  show_decl = false;
  use_regex = false;
  show_scope = false;
  summary.Clear();
  summary_string.Clear();
}

llvm::ArrayRef<OptionDefinition> OptionGroupVariable::GetDefinitions() {
  // Show the "--no-args", "--no-recognized-args", "--no-locals" and
  // "--show-globals" options if we are showing frame specific options
  static constexpr auto num_frame_options = 4;
  static_assert(g_variable_options[0].short_option == 'a' &&
                    g_variable_options[1].short_option == 'l' &&
                    g_variable_options[2].short_option == 't' &&
                    g_variable_options[3].short_option == 'g',
                "frame-only options must come first");
  llvm::ArrayRef<OptionDefinition> result(g_variable_options);
  return include_frame_options ? result : result.drop_front(num_frame_options);
}
