//===-- NextRequestHandler.cpp --------------------------------------------===//
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

// "NextRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Next request; value of command field is 'next'. The
//                     request starts the debuggee to run again for one step.
//                     The debug adapter first sends the NextResponse and then
//                     a StoppedEvent (event type 'step') after the step has
//                     completed.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "next" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/NextArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "NextArguments": {
//   "type": "object",
//   "description": "Arguments for 'next' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Execute 'next' for this thread."
//     },
//     "granularity": {
//       "$ref": "#/definitions/SteppingGranularity",
//       "description": "Stepping granularity. If no granularity is specified, a
//                       granularity of `statement` is assumed."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "NextResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'next' request. This is just an
//                     acknowledgement, so no body field is required."
//   }]
// }
void NextRequestHandler::operator()(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  lldb::SBThread thread = dap.GetLLDBThread(*arguments);
  if (thread.IsValid()) {
    // Remember the thread ID that caused the resume so we can set the
    // "threadCausedFocus" boolean value in the "stopped" events.
    dap.focus_tid = thread.GetThreadID();
    if (HasInstructionGranularity(*arguments)) {
      thread.StepInstruction(/*step_over=*/true);
    } else {
      thread.StepOver();
    }
  } else {
    response["success"] = llvm::json::Value(false);
  }
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
