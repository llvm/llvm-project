//===-- StepOutRequestHandler.cpp -----------------------------------------===//
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

// "StepOutRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "StepOut request; value of command field is 'stepOut'. The
//     request starts the debuggee to run again for one step. The debug adapter
//     first sends the StepOutResponse and then a StoppedEvent (event type
//     'step') after the step has completed.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "stepOut" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/StepOutArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "StepOutArguments": {
//   "type": "object",
//   "description": "Arguments for 'stepOut' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Execute 'stepOut' for this thread."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "StepOutResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'stepOut' request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// }
void StepOutRequestHandler::operator()(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  lldb::SBThread thread = dap.GetLLDBThread(*arguments);
  if (thread.IsValid()) {
    // Remember the thread ID that caused the resume so we can set the
    // "threadCausedFocus" boolean value in the "stopped" events.
    dap.focus_tid = thread.GetThreadID();
    thread.StepOut();
  } else {
    response["success"] = llvm::json::Value(false);
  }
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
