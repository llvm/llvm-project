//===-- StepInRequestHandler.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "RequestHandler.h"

namespace lldb_dap {

// "StepInRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "StepIn request; value of command field is 'stepIn'. The
//     request starts the debuggee to step into a function/method if possible.
//     If it cannot step into a target, 'stepIn' behaves like 'next'. The debug
//     adapter first sends the StepInResponse and then a StoppedEvent (event
//     type 'step') after the step has completed. If there are multiple
//     function/method calls (or other targets) on the source line, the optional
//     argument 'targetId' can be used to control into which target the 'stepIn'
//     should occur. The list of possible targets for a given source line can be
//     retrieved via the 'stepInTargets' request.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "stepIn" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/StepInArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "StepInArguments": {
//   "type": "object",
//   "description": "Arguments for 'stepIn' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Execute 'stepIn' for this thread."
//     },
//     "targetId": {
//       "type": "integer",
//       "description": "Optional id of the target to step into."
//     },
//     "granularity": {
//       "$ref": "#/definitions/SteppingGranularity",
//       "description": "Stepping granularity. If no granularity is specified, a
//                       granularity of `statement` is assumed."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "StepInResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'stepIn' request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// }
void StepInRequestHandler::operator()(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");

  std::string step_in_target;
  uint64_t target_id = GetUnsigned(arguments, "targetId", 0);
  auto it = dap.step_in_targets.find(target_id);
  if (it != dap.step_in_targets.end())
    step_in_target = it->second;

  const bool single_thread = GetBoolean(arguments, "singleThread", false);
  lldb::RunMode run_mode =
      single_thread ? lldb::eOnlyThisThread : lldb::eOnlyDuringStepping;
  lldb::SBThread thread = dap.GetLLDBThread(*arguments);
  if (thread.IsValid()) {
    // Remember the thread ID that caused the resume so we can set the
    // "threadCausedFocus" boolean value in the "stopped" events.
    dap.focus_tid = thread.GetThreadID();
    if (HasInstructionGranularity(*arguments)) {
      thread.StepInstruction(/*step_over=*/false);
    } else {
      thread.StepInto(step_in_target.c_str(), run_mode);
    }
  } else {
    response["success"] = llvm::json::Value(false);
  }
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
