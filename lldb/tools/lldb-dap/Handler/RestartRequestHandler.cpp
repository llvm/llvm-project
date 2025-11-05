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
#include "LLDBUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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
void RestartRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  if (!dap.target.GetProcess().IsValid()) {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message",
                      "Restart request received but no process was launched.");
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  const llvm::json::Object *arguments = request.getObject("arguments");
  if (arguments) {
    // The optional `arguments` field in RestartRequest can contain an updated
    // version of the launch arguments. If there's one, use it.
    if (const llvm::json::Value *restart_arguments =
            arguments->get("arguments")) {
      protocol::LaunchRequestArguments updated_arguments;
      llvm::json::Path::Root root;
      if (!fromJSON(*restart_arguments, updated_arguments, root)) {
        response["success"] = llvm::json::Value(false);
        EmplaceSafeString(
            response, "message",
            llvm::formatv("Failed to parse updated launch arguments: {0}",
                          llvm::toString(root.getError()))
                .str());
        dap.SendJSON(llvm::json::Value(std::move(response)));
        return;
      }
      dap.last_launch_request = updated_arguments;
      // Update DAP configuration based on the latest copy of the launch
      // arguments.
      dap.SetConfiguration(updated_arguments.configuration, false);
      dap.ConfigureSourceMaps();
    }
  }

  // Keep track of the old PID so when we get a "process exited" event from the
  // killed process we can detect it and not shut down the whole session.
  lldb::SBProcess process = dap.target.GetProcess();
  dap.restarting_process_id = process.GetProcessID();

  // Stop the current process if necessary. The logic here is similar to
  // CommandObjectProcessLaunchOrAttach::StopProcessIfNecessary, except that
  // we don't ask the user for confirmation.
  if (process.IsValid()) {
    ScopeSyncMode scope_sync_mode(dap.debugger);
    lldb::StateType state = process.GetState();
    if (state != lldb::eStateConnected) {
      process.Kill();
    }
    // Clear the list of thread ids to avoid sending "thread exited" events
    // for threads of the process we are terminating.
    dap.thread_ids.clear();
  }

  // FIXME: Should we run 'preRunCommands'?
  // FIXME: Should we add a 'preRestartCommands'?
  if (llvm::Error err = LaunchProcess(*dap.last_launch_request)) {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  SendProcessEvent(dap, Launch);

  // This is normally done after receiving a "configuration done" request.
  // Because we're restarting, configuration has already happened so we can
  // continue the process right away.
  if (dap.stop_at_entry) {
    if (llvm::Error err = SendThreadStoppedEvent(dap, /*on_entry=*/true)) {
      EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }
  } else {
    dap.target.GetProcess().Continue();
  }

  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
