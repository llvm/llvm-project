//===-- EvaluateRequestHandler.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPError.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/DAPTypes.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBExpressionOptions.h"
#include "lldb/API/SBLanguageRuntime.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

static lldb::LanguageType GetLanguage(lldb::SBDebugger &debugger) {
  const lldb::SBStructuredData string_result =
      debugger.GetSetting("target.language");

  return lldb::SBLanguageRuntime::GetLanguageTypeFromString(
      GetStringValue(string_result).c_str());
}

static lldb::DynamicValueType
GetPreferDynamicValue(lldb::SBDebugger &debugger) {
  const lldb::SBStructuredData string_result =
      debugger.GetSetting("target.prefer-dynamic-value");

  return llvm::StringSwitch<lldb::DynamicValueType>(
             GetStringValue(string_result))
      .Case("no-dynamic-values", lldb::eNoDynamicValues)
      .Case("run-target", lldb::eDynamicCanRunTarget)
      .Case("no-run-target", lldb::eDynamicDontRunTarget)
      .Default(lldb::eDynamicDontRunTarget);
}

static bool RunExpressionAsLLDBCommand(DAP &dap, lldb::SBFrame &frame,
                                       std::string &expression,
                                       EvaluateContext context) {
  if (context != eEvaluateContextRepl && context != eEvaluateContextUnknown)
    return false;

  // Since we don't know this context do not try to repeat the last command;
  if (context == eEvaluateContextUnknown && expression.empty())
    return false;

  const bool repeat_last_command =
      expression.empty() && dap.last_valid_variable_expression.empty();
  if (repeat_last_command)
    return true;

  const ReplMode repl_mode = dap.DetectReplMode(frame, expression, false);
  return repl_mode == ReplMode::Command;
}

static lldb::SBValue EvaluateVariableExpression(lldb::SBTarget &target,
                                                lldb::SBFrame &frame,
                                                const std::string &expression,
                                                bool run_as_expression) {
  const char *expression_cstr = expression.c_str();

  // First, try resolving as a variable path (e.g. 'foo->bar' finds 'bar'). This
  // is more reliable than the expression parser in many cases and is faster.
  if (frame) {
    const lldb::SBValue value = frame.GetValueForVariablePath(
        expression_cstr, lldb::eDynamicDontRunTarget, lldb::eDILModeLegacy);
    if (value)
      return value;
  }

  if (!run_as_expression)
    return lldb::SBValue{};

  // Fall back to full expression evaluation.
  lldb::SBDebugger debugger = target.GetDebugger();
  lldb::SBExpressionOptions options;
  options.SetUnwindOnError(true);
  options.SetIgnoreBreakpoints(true);
  options.SetFetchDynamicValue(GetPreferDynamicValue(debugger));
  constexpr auto evaluate_timeout_us = lldb_dap::k_evaluate_timeout_ms * 1000;
  options.SetTimeoutInMicroSeconds(evaluate_timeout_us);

  if (frame) {
    lldb::LanguageType eval_language = GetLanguage(debugger);
    if (eval_language == lldb::eLanguageTypeUnknown)
      eval_language = frame.GuessLanguage();
    options.SetLanguage(eval_language);

    return frame.EvaluateExpression(expression_cstr, options);
  }
  return target.EvaluateExpression(expression_cstr, options);
}

namespace lldb_dap {

/// Evaluates the given expression in the context of a stack frame.
///
/// The expression has access to any variables and arguments that are in scope.
Expected<EvaluateResponseBody>
EvaluateRequestHandler::Run(const EvaluateArguments &arguments) const {

  EvaluateResponseBody body;
  lldb::SBFrame frame = dap.GetLLDBFrame(arguments.frameId);
  std::string expression = llvm::StringRef(arguments.expression).trim().str();
  const EvaluateContext evaluate_context = arguments.context;
  const bool is_repl_context = evaluate_context == eEvaluateContextRepl;

  if (RunExpressionAsLLDBCommand(dap, frame, expression, evaluate_context)) {
    // Since the current expression is not for a variable, clear the
    // last_valid_variable_expression field.
    dap.last_valid_variable_expression.clear();
    // If we're evaluating a command relative to the current frame, set the
    // focus_tid to the current frame for any thread related events.
    if (frame.IsValid()) {
      dap.focus_tid = frame.GetThread().GetThreadID();
    }

    bool required_command_failed = false;
    body.result = RunLLDBCommands(
        dap.debugger, dap.GetAPIMutex(), llvm::StringRef(), {expression},
        required_command_failed,
        /*parse_command_directives=*/false, /*echo_commands=*/false);
    return body;
  }

  if (dap.ProcessIsNotStopped())
    return llvm::make_error<DAPError>(
        "Cannot evaluate expressions while the process is running. Pause "
        "the process and try again.",
        /**error_code=*/llvm::inconvertibleErrorCode(),
        /**show_user=*/false);

  // If the user expression is empty, evaluate the last valid variable
  // expression.
  if (expression.empty() && is_repl_context)
    expression = dap.last_valid_variable_expression;

  const bool run_as_expression = evaluate_context != eEvaluateContextHover;
  lldb::SBValue value = EvaluateVariableExpression(
      dap.target, frame, expression, run_as_expression);

  if (value.GetError().Fail())
    return ToError(value.GetError(), /*show_user=*/false);

  if (is_repl_context) {
    // save the new variable expression
    dap.last_valid_variable_expression = std::move(expression);

    // Freeze dry the value in case users expand it later in the debug console
    value = value.Persist();
  }

  const bool hex = arguments.format ? arguments.format->hex : false;
  VariableDescription desc(value, dap.configuration.enableAutoVariableSummaries,
                           hex);

  body.result = desc.GetResult(evaluate_context);
  body.type = desc.display_type_name;

  if (value.MightHaveChildren() || ValuePointsToCode(value))
    body.variablesReference = dap.reference_storage.Insert(
        value, /*is_permanent=*/is_repl_context, /*is_internal=*/false);

  if (lldb::addr_t addr = value.GetLoadAddress(); addr != LLDB_INVALID_ADDRESS)
    body.memoryReference = EncodeMemoryReference(addr);

  if (ValuePointsToCode(value) &&
      body.variablesReference.Kind() != eReferenceKindInvalid)
    body.valueLocationReference =
        PackLocation(body.variablesReference.AsUInt32(), true);

  return body;
}

} // namespace lldb_dap
