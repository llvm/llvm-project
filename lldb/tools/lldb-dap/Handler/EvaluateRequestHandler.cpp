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
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBValue.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstddef>

using namespace llvm;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

namespace lldb_dap {

static lldb::SBValue EvaluateVariableExpression(lldb::SBTarget &target,
                                                lldb::SBFrame &frame,
                                                const std::string &expression,
                                                bool run_as_expression) {
  const char *expression_cstr = expression.c_str();

  lldb::SBValue value;
  if (frame) {
    // Check if it is a variable or an expression path for a variable. i.e.
    // 'foo->bar' finds the 'bar' variable. It is more reliable than the
    // expression parser in many cases and it is faster.
    value = frame.GetValueForVariablePath(expression_cstr,
                                          lldb::eDynamicDontRunTarget);
    if (value || !run_as_expression)
      return value;

    return frame.EvaluateExpression(expression_cstr);
  }

  if (run_as_expression)
    value = target.EvaluateExpression(expression_cstr);

  return value;
}

/// Check if we need to run `expr` as an lldb command or as a variable
/// expression.
static bool IsReplCommand(DAP &dap, lldb::SBFrame &frame, std::string &expr) {
  return dap.DetectReplMode(frame, expr, false) == ReplMode::Command;
}

/// Evaluates the given expression in the context of a stack frame.
///
/// The expression has access to any variables and arguments that are in scope.
Expected<EvaluateResponseBody>
EvaluateRequestHandler::Run(const EvaluateArguments &arguments) const {
  EvaluateResponseBody body;
  lldb::SBFrame frame = dap.GetLLDBFrame(arguments.frameId);
  std::string expression = arguments.expression;
  const bool is_repl_context = arguments.context == eEvaluateContextRepl;
  const bool run_as_expression = arguments.context != eEvaluateContextHover;
  lldb::SBValue value;

  // If we're evaluating a command relative to the current frame, set the
  // focus_tid to the current frame for any thread related events.
  if (frame.IsValid())
    dap.focus_tid = frame.GetThread().GetThreadID();

  if (is_repl_context && IsReplCommand(dap, frame, expression)) {
    Expected<std::pair<std::string, lldb::SBValueList>> result =
        EvaluateContext::Run(dap, expression + "\n");
    if (!result)
      return result.takeError();

    lldb::SBValueList values;
    std::tie(body.result, values) = *result;

    if (values.GetSize() == 1) {
      value = values.GetValueAtIndex(0);
      body.type = value.GetDisplayTypeName();
    } else if (values.GetSize()) {
      body.variablesReference = dap.variables.Insert(result->second);
    }
  } else {
    if (dap.ProcessIsNotStopped())
      return llvm::make_error<NotStoppedError>();

    value = EvaluateVariableExpression(dap.target, frame, expression,
                                       run_as_expression);

    if (value.GetError().Fail())
      return ToError(value.GetError(), /*show_user=*/false);

    // Freeze dry the value in case users expand it later in the debug console
    if (is_repl_context)
      value = value.Persist();

    const bool hex = arguments.format ? arguments.format->hex : false;
    VariableDescription desc(
        value, dap.configuration.enableAutoVariableSummaries, hex);

    body.result = desc.GetResult(arguments.context);
    body.type = desc.display_type_name;
  }

  if (value.MightHaveChildren() || ValuePointsToCode(value))
    body.variablesReference =
        dap.variables.InsertVariable(value, /*is_permanent=*/is_repl_context);

  if (lldb::addr_t addr = value.GetLoadAddress(); addr != LLDB_INVALID_ADDRESS)
    body.memoryReference = EncodeMemoryReference(addr);

  if (ValuePointsToCode(value) &&
      body.variablesReference != LLDB_DAP_INVALID_VAR_REF)
    body.valueLocationReference = PackLocation(body.variablesReference, true);

  return body;
}

} // namespace lldb_dap
