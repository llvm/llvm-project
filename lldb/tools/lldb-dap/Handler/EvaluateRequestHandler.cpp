//===-- EvaluateRequestHandler.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolEvents.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "Variables.h"
#include "lldb/API/SBValue.h"
#include "lldb/Host/File.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Support/Error.h"
#include <chrono>
#include <cstddef>

using namespace llvm;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// Evaluates the given expression in the context of a stack frame.
///
/// The expression has access to any variables and arguments that are in scope.
Expected<EvaluateResponseBody>
EvaluateRequestHandler::Run(const EvaluateArguments &arguments) const {
  EvaluateResponseBody body;
  lldb::SBFrame frame = dap.GetLLDBFrame(arguments.frameId);
  std::string expression = arguments.expression;

  if (arguments.context == eEvaluateContextRepl &&
      dap.DetectReplMode(frame, expression, false) == ReplMode::Command) {
    // If we're evaluating a command relative to the current frame, set the
    // focus_tid to the current frame for any thread related events.
    if (frame.IsValid()) {
      dap.focus_tid = frame.GetThread().GetThreadID();
    }

    for (const auto &line_ref : llvm::split(expression, "\n")) {
      ReplContext context{dap, line_ref};
      if (llvm::Error err = context.Run())
        return err;

      if (!context.succeeded)
        return llvm::make_error<DAPError>(std::string(context.output),
                                          /*show_user=*/false);

      body.result += std::string(context.output);

      if (context.values && context.values.GetSize()) {
        if (context.values.GetSize() == 1) {
          lldb::SBValue v = context.values.GetValueAtIndex(0);
          if (!IsPersistent(v))
            v = v.Persist();
          VariableDescription desc(
              v, dap.configuration.enableAutoVariableSummaries);
          body.type = desc.display_type_name;
          if (v.MightHaveChildren() || ValuePointsToCode(v))
            body.variablesReference = dap.variables.InsertVariable(v);
        } else {
          body.variablesReference =
              dap.variables.InsertVariables(context.values);
        }
      }
    }

    return body;
  }

  // Always try to get the answer from the local variables if possible. If
  // this fails, then if the context is not "hover", actually evaluate an
  // expression using the expression parser.
  //
  // "frame variable" is more reliable than the expression parser in
  // many cases and it is faster.
  lldb::SBValue value = frame.GetValueForVariablePath(
      expression.data(), lldb::eDynamicDontRunTarget);

  // Freeze dry the value in case users expand it later in the debug console
  if (value.GetError().Success() && arguments.context == eEvaluateContextRepl)
    value = value.Persist();

  if (value.GetError().Fail() && arguments.context != eEvaluateContextHover)
    value = frame.EvaluateExpression(expression.data());

  if (value.GetError().Fail())
    return ToError(value.GetError(), /*show_user=*/false);

  const bool hex = arguments.format ? arguments.format->hex : false;

  VariableDescription desc(value, dap.configuration.enableAutoVariableSummaries,
                           hex);

  body.result = desc.GetResult(arguments.context);
  body.type = desc.display_type_name;
  if (value.MightHaveChildren() || ValuePointsToCode(value))
    body.variablesReference = dap.variables.InsertVariable(value);
  if (lldb::addr_t addr = value.GetLoadAddress(); addr != LLDB_INVALID_ADDRESS)
    body.memoryReference = EncodeMemoryReference(addr);

  if (ValuePointsToCode(value) &&
      body.variablesReference != LLDB_DAP_INVALID_VAR_REF)
    body.valueLocationReference = PackLocation(body.variablesReference, true);

  return body;
}

} // namespace lldb_dap
