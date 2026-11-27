//===-- OptionGroupValueObjectDisplay.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupValueObjectDisplay.h"

#include "lldb/DataFormatters/ValueObjectPrinter.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/ArrayRef.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_value_object_display
#include "InterpreterOptions.inc"

llvm::ArrayRef<OptionDefinition>
OptionGroupValueObjectDisplay::GetDefinitions() {
  return llvm::ArrayRef(g_value_object_display_options);
}

Status OptionGroupValueObjectDisplay::SetOptionValue(
    uint32_t option_idx, llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  const int short_option =
      g_value_object_display_options[option_idx].short_option;
  bool success = false;

  switch (short_option) {
  case 'd': {
    int32_t result;
    result = OptionArgParser::ToOptionEnum(
        option_arg, g_value_object_display_options[option_idx].enum_values, 2,
        error);
    if (error.Success())
      use_dynamic = (lldb::DynamicValueType)result;
  } break;
  case 'T':
    show_types = true;
    break;
  case 'L':
    show_location = true;
    break;
  case 'F':
    flat_output = true;
    break;
  case 'O':
    use_object_desc = true;
    break;
  case 'R':
    be_raw = true;
    break;
  case 'A':
    ignore_cap = true;
    break;

  case 'D':
    if (option_arg.getAsInteger(0, max_depth)) {
      max_depth = UINT32_MAX;
      error = Status::FromErrorStringWithFormat("invalid max depth '%s'",
                                                option_arg.str().c_str());
    } else {
      max_depth_is_default = false;
    }
    break;

  case 'Z':
    if (option_arg.getAsInteger(0, elem_count)) {
      elem_count = UINT32_MAX;
      error = Status::FromErrorStringWithFormat("invalid element count '%s'",
                                                option_arg.str().c_str());
    }
    break;

  case 'P':
    if (option_arg.getAsInteger(0, ptr_depth)) {
      ptr_depth = 0;
      error = Status::FromErrorStringWithFormat("invalid pointer depth '%s'",
                                                option_arg.str().c_str());
    }
    break;

  case 'Y':
    if (option_arg.empty())
      no_summary_depth = 1;
    else if (option_arg.getAsInteger(0, no_summary_depth)) {
      no_summary_depth = 0;
      error = Status::FromErrorStringWithFormat("invalid pointer depth '%s'",
                                                option_arg.str().c_str());
    }
    break;

  case 'S':
    use_synth = OptionArgParser::ToBoolean(option_arg, true, &success);
    if (!success)
      error = Status::FromErrorStringWithFormat("invalid synthetic-type '%s'",
                                                option_arg.str().c_str());
    break;

  case 'V':
    run_validator = OptionArgParser::ToBoolean(option_arg, true, &success);
    if (!success)
      error = Status::FromErrorStringWithFormat("invalid validate '%s'",
                                                option_arg.str().c_str());
    break;

  default:
    llvm_unreachable("Unimplemented option");
  }

  return error;
}

void OptionGroupValueObjectDisplay::OptionParsingStarting(
    ExecutionContext *execution_context) {
  // If these defaults change, be sure to modify AnyOptionWasSet().
  show_types = false;
  no_summary_depth = 0;
  show_location = false;
  flat_output = false;
  use_object_desc = false;
  max_depth = UINT32_MAX;
  max_depth_is_default = true;
  ptr_depth = 0;
  elem_count = 0;
  use_synth = true;
  be_raw = false;
  ignore_cap = false;
  run_validator = false;

  TargetSP target_sp =
      execution_context ? execution_context->GetTargetSP() : TargetSP();
  if (target_sp) {
    use_dynamic = target_sp->GetPreferDynamicValue();
    auto max_depth_config = target_sp->GetMaximumDepthOfChildrenToDisplay();
    max_depth = std::get<uint32_t>(max_depth_config);
    max_depth_is_default = std::get<bool>(max_depth_config);
  } else {
    // If we don't have any targets, then dynamic values won't do us much good.
    use_dynamic = lldb::eNoDynamicValues;
  }
}

DumpValueObjectOptions OptionGroupValueObjectDisplay::GetAsDumpOptions(
    LanguageRuntimeDescriptionDisplayVerbosity lang_descr_verbosity,
    lldb::Format format, lldb::TypeSummaryImplSP summary_sp) {
  DumpValueObjectOptions options;
  options.SetMaximumPointerDepth(ptr_depth);
  if (use_object_desc)
    options.SetShowSummary(false);
  else
    options.SetOmitSummaryDepth(no_summary_depth);
  options.SetMaximumDepth(max_depth, max_depth_is_default)
      .SetShowTypes(show_types)
      .SetShowLocation(show_location)
      .SetUseObjectDescription(use_object_desc)
      .SetUseDynamicType(use_dynamic)
      .SetUseSyntheticValue(use_synth)
      .SetFlatOutput(flat_output)
      .SetIgnoreCap(ignore_cap)
      .SetFormat(format)
      .SetSummary(summary_sp);

  if (lang_descr_verbosity ==
      eLanguageRuntimeDescriptionDisplayVerbosityCompact)
    options.SetHideRootType(use_object_desc)
        .SetHideName(use_object_desc)
        .SetHideValue(use_object_desc);

  if (be_raw)
    options.SetRawDisplay();

  options.SetRunValidator(run_validator);

  options.SetElementCount(elem_count);

  return options;
}
