//===-- DisconnectRequestHandler.cpp --------------------------------------===//
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

// "DisconnectRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Disconnect request; value of command field is
//                     'disconnect'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "disconnect" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/DisconnectArguments"
//       }
//     },
//     "required": [ "command" ]
//   }]
// },
// "DisconnectArguments": {
//   "type": "object",
//   "description": "Arguments for 'disconnect' request.",
//   "properties": {
//     "terminateDebuggee": {
//       "type": "boolean",
//       "description": "Indicates whether the debuggee should be terminated
//                       when the debugger is disconnected. If unspecified,
//                       the debug adapter is free to do whatever it thinks
//                       is best. A client can only rely on this attribute
//                       being properly honored if a debug adapter returns
//                       true for the 'supportTerminateDebuggee' capability."
//     },
//     "restart": {
//       "type": "boolean",
//       "description": "Indicates whether the debuggee should be restart
//                       the process."
//     }
//   }
// },
// "DisconnectResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'disconnect' request. This is just an
//                     acknowledgement, so no body field is required."
//   }]
// }
void DisconnectRequestHandler::operator()(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");

  bool defaultTerminateDebuggee = dap.is_attach ? false : true;
  bool terminateDebuggee =
      GetBoolean(arguments, "terminateDebuggee", defaultTerminateDebuggee);
  lldb::SBProcess process = dap.target.GetProcess();
  auto state = process.GetState();
  switch (state) {
  case lldb::eStateInvalid:
  case lldb::eStateUnloaded:
  case lldb::eStateDetached:
  case lldb::eStateExited:
    break;
  case lldb::eStateConnected:
  case lldb::eStateAttaching:
  case lldb::eStateLaunching:
  case lldb::eStateStepping:
  case lldb::eStateCrashed:
  case lldb::eStateSuspended:
  case lldb::eStateStopped:
  case lldb::eStateRunning:
    dap.debugger.SetAsync(false);
    lldb::SBError error = terminateDebuggee ? process.Kill() : process.Detach();
    if (!error.Success())
      EmplaceSafeString(response, "error", error.GetCString());
    dap.debugger.SetAsync(true);
    break;
  }
  SendTerminatedEvent(dap);
  dap.SendJSON(llvm::json::Value(std::move(response)));
  if (dap.event_thread.joinable()) {
    dap.broadcaster.BroadcastEventByType(eBroadcastBitStopEventThread);
    dap.event_thread.join();
  }
  if (dap.progress_event_thread.joinable()) {
    dap.broadcaster.BroadcastEventByType(eBroadcastBitStopProgressThread);
    dap.progress_event_thread.join();
  }
  dap.StopIO();
  dap.disconnecting = true;
}
} // namespace lldb_dap
