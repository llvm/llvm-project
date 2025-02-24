//===-- RestartRequestHandler.cpp -----------------------------------------===//
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
#include "lldb/API/SBListener.h"
#include "llvm/Support/FileSystem.h"

namespace lldb_dap {

// "RestartRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Restarts a debug session. Clients should only call this
//     request if the corresponding capability `supportsRestartRequest` is
//     true.\nIf the capability is missing or has the value false, a typical
//     client emulates `restart` by terminating the debug adapter first and then
//     launching it anew.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "restart" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/RestartArguments"
//       }
//     },
//     "required": [ "command" ]
//   }]
// },
// "RestartArguments": {
//   "type": "object",
//   "description": "Arguments for `restart` request.",
//   "properties": {
//     "arguments": {
//       "oneOf": [
//         { "$ref": "#/definitions/LaunchRequestArguments" },
//         { "$ref": "#/definitions/AttachRequestArguments" }
//       ],
//       "description": "The latest version of the `launch` or `attach`
//       configuration."
//     }
//   }
// },
// "RestartResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `restart` request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// },
void RestartRequestHandler::operator()(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  if (!dap.last_launch_or_attach_request) {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message",
                      "Restart request received but no process was launched.");
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }
  // Check if we were in a "launch" session or an "attach" session.
  //
  // Restarting is not well defined when we started the session by attaching to
  // an existing process, because we don't know how the process was started, so
  // we don't support it.
  //
  // Note that when using runInTerminal we're technically attached, but it's an
  // implementation detail. The adapter *did* launch the process in response to
  // a "launch" command, so we can still stop it and re-run it. This is why we
  // don't just check `dap.is_attach`.
  if (GetString(*dap.last_launch_or_attach_request, "command") == "attach") {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message",
                      "Restarting an \"attach\" session is not supported.");
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  // The optional `arguments` field in RestartRequest can contain an updated
  // version of the launch arguments. If there's one, use it.
  const auto *restart_arguments = request.getObject("arguments");
  if (restart_arguments) {
    const auto *launch_request_arguments =
        restart_arguments->getObject("arguments");
    if (launch_request_arguments) {
      (*dap.last_launch_or_attach_request)["arguments"] =
          llvm::json::Value(llvm::json::Object(*launch_request_arguments));
    }
  }

  // Keep track of the old PID so when we get a "process exited" event from the
  // killed process we can detect it and not shut down the whole session.
  lldb::SBProcess process = dap.target.GetProcess();
  dap.restarting_process_id = process.GetProcessID();

  // Stop the current process if necessary. The logic here is similar to
  // CommandObjectProcessLaunchOrAttach::StopProcessIfNecessary, except that
  // we don't ask the user for confirmation.
  dap.debugger.SetAsync(false);
  if (process.IsValid()) {
    lldb::StateType state = process.GetState();
    if (state != lldb::eStateConnected) {
      process.Kill();
    }
    // Clear the list of thread ids to avoid sending "thread exited" events
    // for threads of the process we are terminating.
    dap.thread_ids.clear();
  }
  dap.debugger.SetAsync(true);
  LaunchProcess(*dap.last_launch_or_attach_request);

  // This is normally done after receiving a "configuration done" request.
  // Because we're restarting, configuration has already happened so we can
  // continue the process right away.
  if (dap.stop_at_entry) {
    SendThreadStoppedEvent(dap);
  } else {
    dap.target.GetProcess().Continue();
  }

  dap.SendJSON(llvm::json::Value(std::move(response)));
}
} // namespace lldb_dap
