//===-- AttachRequestHandler.cpp ------------------------------------------===//
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

// "AttachRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Attach request; value of command field is 'attach'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "attach" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/AttachRequestArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "AttachRequestArguments": {
//   "type": "object",
//   "description": "Arguments for 'attach' request.\nThe attach request has no
//   standardized attributes."
// },
// "AttachResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'attach' request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// }

void AttachRequestHandler::operator()(const llvm::json::Object &request) const {
  dap.is_attach = true;
  dap.last_launch_or_attach_request = request;
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  lldb::SBAttachInfo attach_info;
  const int invalid_port = 0;
  const auto *arguments = request.getObject("arguments");
  const lldb::pid_t pid =
      GetInteger<uint64_t>(arguments, "pid").value_or(LLDB_INVALID_PROCESS_ID);
  const auto gdb_remote_port =
      GetInteger<uint64_t>(arguments, "gdb-remote-port").value_or(invalid_port);
  const auto gdb_remote_hostname =
      GetString(arguments, "gdb-remote-hostname", "localhost");
  if (pid != LLDB_INVALID_PROCESS_ID)
    attach_info.SetProcessID(pid);
  const auto wait_for = GetBoolean(arguments, "waitFor").value_or(false);
  attach_info.SetWaitForLaunch(wait_for, false /*async*/);
  dap.init_commands = GetStrings(arguments, "initCommands");
  dap.pre_run_commands = GetStrings(arguments, "preRunCommands");
  dap.stop_commands = GetStrings(arguments, "stopCommands");
  dap.exit_commands = GetStrings(arguments, "exitCommands");
  dap.terminate_commands = GetStrings(arguments, "terminateCommands");
  auto attachCommands = GetStrings(arguments, "attachCommands");
  llvm::StringRef core_file = GetString(arguments, "coreFile");
  const auto timeout_seconds =
      GetInteger<uint64_t>(arguments, "timeout").value_or(30);
  dap.stop_at_entry = core_file.empty()
                          ? GetBoolean(arguments, "stopOnEntry").value_or(false)
                          : true;
  dap.post_run_commands = GetStrings(arguments, "postRunCommands");
  const llvm::StringRef debuggerRoot = GetString(arguments, "debuggerRoot");
  dap.enable_auto_variable_summaries =
      GetBoolean(arguments, "enableAutoVariableSummaries").value_or(false);
  dap.enable_synthetic_child_debugging =
      GetBoolean(arguments, "enableSyntheticChildDebugging").value_or(false);
  dap.display_extended_backtrace =
      GetBoolean(arguments, "displayExtendedBacktrace").value_or(false);
  dap.command_escape_prefix = GetString(arguments, "commandEscapePrefix", "`");
  dap.SetFrameFormat(GetString(arguments, "customFrameFormat"));
  dap.SetThreadFormat(GetString(arguments, "customThreadFormat"));

  PrintWelcomeMessage();

  // This is a hack for loading DWARF in .o files on Mac where the .o files
  // in the debug map of the main executable have relative paths which require
  // the lldb-dap binary to have its working directory set to that relative
  // root for the .o files in order to be able to load debug info.
  if (!debuggerRoot.empty())
    llvm::sys::fs::set_current_path(debuggerRoot);

  // Run any initialize LLDB commands the user specified in the launch.json
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

  if ((pid == LLDB_INVALID_PROCESS_ID || gdb_remote_port == invalid_port) &&
      wait_for) {
    char attach_msg[256];
    auto attach_msg_len = snprintf(attach_msg, sizeof(attach_msg),
                                   "Waiting to attach to \"%s\"...",
                                   dap.target.GetExecutable().GetFilename());
    dap.SendOutput(OutputType::Console,
                   llvm::StringRef(attach_msg, attach_msg_len));
  }
  if (attachCommands.empty()) {
    // No "attachCommands", just attach normally.
    // Disable async events so the attach will be successful when we return from
    // the launch call and the launch will happen synchronously
    dap.debugger.SetAsync(false);
    if (core_file.empty()) {
      if ((pid != LLDB_INVALID_PROCESS_ID) &&
          (gdb_remote_port != invalid_port)) {
        // If both pid and port numbers are specified.
        error.SetErrorString("The user can't specify both pid and port");
      } else if (gdb_remote_port != invalid_port) {
        // If port is specified and pid is not.
        lldb::SBListener listener = dap.debugger.GetListener();

        // If the user hasn't provided the hostname property, default localhost
        // being used.
        std::string connect_url =
            llvm::formatv("connect://{0}:", gdb_remote_hostname);
        connect_url += std::to_string(gdb_remote_port);
        dap.target.ConnectRemote(listener, connect_url.c_str(), "gdb-remote",
                                 error);
      } else {
        // Attach by process name or id.
        dap.target.Attach(attach_info, error);
      }
    } else
      dap.target.LoadCore(core_file.data(), error);
    // Reenable async events
    dap.debugger.SetAsync(true);
  } else {
    // We have "attachCommands" that are a set of commands that are expected
    // to execute the commands after which a process should be created. If there
    // is no valid process after running these commands, we have failed.
    if (llvm::Error err = dap.RunAttachCommands(attachCommands)) {
      response["success"] = false;
      EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }
    // The custom commands might have created a new target so we should use the
    // selected target after these commands are run.
    dap.target = dap.debugger.GetSelectedTarget();

    // Make sure the process is attached and stopped before proceeding as the
    // the launch commands are not run using the synchronous mode.
    error = dap.WaitForProcessToStop(timeout_seconds);
  }

  if (error.Success() && core_file.empty()) {
    auto attached_pid = dap.target.GetProcess().GetProcessID();
    if (attached_pid == LLDB_INVALID_PROCESS_ID) {
      if (attachCommands.empty())
        error.SetErrorString("failed to attach to a process");
      else
        error.SetErrorString("attachCommands failed to attach to a process");
    }
  }

  if (error.Fail()) {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message", std::string(error.GetCString()));
  } else {
    dap.RunPostRunCommands();
  }

  dap.SendJSON(llvm::json::Value(std::move(response)));
  if (error.Success()) {
    SendProcessEvent(dap, Attach);
    dap.SendJSON(CreateEventObject("initialized"));
  }
}

} // namespace lldb_dap
