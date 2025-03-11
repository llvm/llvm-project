//===-- GoToRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"

namespace lldb_dap {

// "GotoRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "The request sets the location where the debuggee will
//     continue to run.\nThis makes it possible to skip the execution of code or
//     to execute code again.\nThe code between the current location and the
//     goto target is not executed but skipped.\nThe debug adapter first sends
//     the response and then a `stopped` event with reason `goto`.\nClients
//     should only call this request if the corresponding capability
//     `supportsGotoTargetsRequest` is true (because only then goto targets
//     exist that can be passed as arguments).",
//.    "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "goto" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/GotoArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// }
// "GotoArguments": {
//   "type": "object",
//   "description": "Arguments for `goto` request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Set the goto target for this thread."
//     },
//     "targetId": {
//       "type": "integer",
//       "description": "The location where the debuggee will continue to run."
//     }
//   },
//   "required": [ "threadId", "targetId" ]
// }
// "GotoResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `goto` request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// }
void GoToRequestHandler::operator()(const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);

  auto SendError = [&](auto &&message) {
    response["success"] = false;
    response["message"] = message;
    dap.SendJSON(llvm::json::Value(std::move(response)));
  };

  const auto *goto_arguments = request.getObject("arguments");
  if (goto_arguments == nullptr) {
    SendError("Arguments is empty");
    return;
  }

  lldb::SBThread current_thread = dap.GetLLDBThread(*goto_arguments);
  if (!current_thread.IsValid()) {
    SendError(llvm::formatv("Thread id `{0}` is not valid",
                            current_thread.GetThreadID()));
    return;
  }

  const auto target_id = GetInteger<uint64_t>(goto_arguments, "targetId");
  const auto line_entry = dap.goto_id_map.GetLineEntry(target_id.value());
  if (!target_id || !line_entry) {
    SendError(llvm::formatv("Target id `{0}` is not valid",
                            current_thread.GetThreadID()));
    return;
  }

  auto file_spec = line_entry->GetFileSpec();
  const auto error =
      current_thread.JumpToLine(file_spec, line_entry->GetLine());
  if (error.Fail()) {
    SendError(error.GetCString());
    return;
  }

  dap.SendJSON(llvm::json::Value(std::move(response)));

  SendThreadStoppedEvent(dap);
}

} // namespace lldb_dap