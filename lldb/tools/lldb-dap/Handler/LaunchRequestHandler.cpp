//===-- LaunchRequestHandler.cpp ------------------------------------------===//
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
#include "llvm/Support/FileSystem.h"

namespace lldb_dap {

// "LaunchRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Launch request; value of command field is 'launch'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "launch" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/LaunchRequestArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "LaunchRequestArguments": {
//   "type": "object",
//   "description": "Arguments for 'launch' request.",
//   "properties": {
//     "noDebug": {
//       "type": "boolean",
//       "description": "If noDebug is true the launch request should launch
//                       the program without enabling debugging."
//     }
//   }
// },
// "LaunchResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'launch' request. This is just an
//                     acknowledgement, so no body field is required."
//   }]
// }
void LaunchRequestHandler::operator()(const llvm::json::Object &request) const {
  dap.is_attach = false;
  dap.last_launch_or_attach_request = request;
  llvm::json::Object response;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  dap.init_commands = GetStrings(arguments, "initCommands");
  dap.pre_run_commands = GetStrings(arguments, "preRunCommands");
  dap.stop_commands = GetStrings(arguments, "stopCommands");
  dap.exit_commands = GetStrings(arguments, "exitCommands");
  dap.terminate_commands = GetStrings(arguments, "terminateCommands");
  dap.post_run_commands = GetStrings(arguments, "postRunCommands");
  dap.stop_at_entry = GetBoolean(arguments, "stopOnEntry", false);
  const llvm::StringRef debuggerRoot = GetString(arguments, "debuggerRoot");
  dap.enable_auto_variable_summaries =
      GetBoolean(arguments, "enableAutoVariableSummaries", false);
  dap.enable_synthetic_child_debugging =
      GetBoolean(arguments, "enableSyntheticChildDebugging", false);
  dap.display_extended_backtrace =
      GetBoolean(arguments, "displayExtendedBacktrace", false);
  dap.command_escape_prefix = GetString(arguments, "commandEscapePrefix", "`");
  dap.SetFrameFormat(GetString(arguments, "customFrameFormat"));
  dap.SetThreadFormat(GetString(arguments, "customThreadFormat"));

  PrintWelcomeMessage();

  // This is a hack for loading DWARF in .o files on Mac where the .o files
  // in the debug map of the main executable have relative paths which
  // require the lldb-dap binary to have its working directory set to that
  // relative root for the .o files in order to be able to load debug info.
  if (!debuggerRoot.empty())
    llvm::sys::fs::set_current_path(debuggerRoot);

  // Run any initialize LLDB commands the user specified in the launch.json.
  // This is run before target is created, so commands can't do anything with
  // the targets - preRunCommands are run with the target.
  if (llvm::Error err = dap.RunInitCommands()) {
    response["success"] = false;
    EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  SetSourceMapFromArguments(*arguments);

  lldb::SBError status;
  dap.SetTarget(dap.CreateTargetFromArguments(*arguments, status));
  if (status.Fail()) {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message", status.GetCString());
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  // Run any pre run LLDB commands the user specified in the launch.json
  if (llvm::Error err = dap.RunPreRunCommands()) {
    response["success"] = false;
    EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  status = LaunchProcess(request);

  if (status.Fail()) {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message", std::string(status.GetCString()));
  } else {
    dap.RunPostRunCommands();
  }

  dap.SendJSON(llvm::json::Value(std::move(response)));

  if (!status.Fail()) {
    if (dap.is_attach)
      SendProcessEvent(dap, Attach); // this happens when doing runInTerminal
    else
      SendProcessEvent(dap, Launch);
  }
  dap.SendJSON(CreateEventObject("initialized"));
}
} // namespace lldb_dap
