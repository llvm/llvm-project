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
#include "LLDBUtils.h"
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
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  const int invalid_port = 0;
  const auto *arguments = request.getObject("arguments");
  const lldb::pid_t pid =
      GetInteger<uint64_t>(arguments, "pid").value_or(LLDB_INVALID_PROCESS_ID);
  const auto gdb_remote_port =
      GetInteger<uint64_t>(arguments, "gdb-remote-port").value_or(invalid_port);
  const auto gdb_remote_hostname =
      GetString(arguments, "gdb-remote-hostname").value_or("localhost");
  const auto wait_for = GetBoolean(arguments, "waitFor").value_or(false);
  dap.configuration.initCommands = GetStrings(arguments, "initCommands");
  dap.configuration.preRunCommands = GetStrings(arguments, "preRunCommands");
  dap.configuration.postRunCommands = GetStrings(arguments, "postRunCommands");
  dap.configuration.stopCommands = GetStrings(arguments, "stopCommands");
  dap.configuration.exitCommands = GetStrings(arguments, "exitCommands");
  dap.configuration.terminateCommands =
      GetStrings(arguments, "terminateCommands");
  auto attachCommands = GetStrings(arguments, "attachCommands");
  llvm::StringRef core_file = GetString(arguments, "coreFile").value_or("");
  const uint64_t timeout_seconds =
      GetInteger<uint64_t>(arguments, "timeout").value_or(30);
  dap.stop_at_entry = core_file.empty()
                          ? GetBoolean(arguments, "stopOnEntry").value_or(false)
                          : true;
  const llvm::StringRef debuggerRoot =
      GetString(arguments, "debuggerRoot").value_or("");
  dap.configuration.enableAutoVariableSummaries =
      GetBoolean(arguments, "enableAutoVariableSummaries").value_or(false);
  dap.configuration.enableSyntheticChildDebugging =
      GetBoolean(arguments, "enableSyntheticChildDebugging").value_or(false);
  dap.configuration.displayExtendedBacktrace =
      GetBoolean(arguments, "displayExtendedBacktrace").value_or(false);
  dap.configuration.commandEscapePrefix =
      GetString(arguments, "commandEscapePrefix").value_or("`");
  dap.configuration.program = GetString(arguments, "program");
  dap.configuration.targetTriple = GetString(arguments, "targetTriple");
  dap.configuration.platformName = GetString(arguments, "platformName");
  dap.SetFrameFormat(GetString(arguments, "customFrameFormat").value_or(""));
  dap.SetThreadFormat(GetString(arguments, "customThreadFormat").value_or(""));

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
  dap.SetTarget(dap.CreateTarget(status));
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

  {
    // Perform the launch in synchronous mode so that we don't have to worry
    // about process state changes during the launch.
    ScopeSyncMode scope_sync_mode(dap.debugger);
    if (attachCommands.empty()) {
      // No "attachCommands", just attach normally.
      if (core_file.empty()) {
        if ((pid != LLDB_INVALID_PROCESS_ID) &&
            (gdb_remote_port != invalid_port)) {
          // If both pid and port numbers are specified.
          error.SetErrorString("The user can't specify both pid and port");
        } else if (gdb_remote_port != invalid_port) {
          // If port is specified and pid is not.
          lldb::SBListener listener = dap.debugger.GetListener();

          // If the user hasn't provided the hostname property, default
          // localhost being used.
          std::string connect_url =
              llvm::formatv("connect://{0}:", gdb_remote_hostname);
          connect_url += std::to_string(gdb_remote_port);
          dap.target.ConnectRemote(listener, connect_url.c_str(), "gdb-remote",
                                   error);
        } else {
          // Attach by pid or process name.
          lldb::SBAttachInfo attach_info;
          if (pid != LLDB_INVALID_PROCESS_ID)
            attach_info.SetProcessID(pid);
          else if (dap.configuration.program.has_value())
            attach_info.SetExecutable(dap.configuration.program->data());
          attach_info.SetWaitForLaunch(wait_for, false /*async*/);
          dap.target.Attach(attach_info, error);
        }
      } else {
        dap.target.LoadCore(core_file.data(), error);
      }
    } else {
      // We have "attachCommands" that are a set of commands that are expected
      // to execute the commands after which a process should be created. If
      // there is no valid process after running these commands, we have failed.
      if (llvm::Error err = dap.RunAttachCommands(attachCommands)) {
        response["success"] = false;
        EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
        dap.SendJSON(llvm::json::Value(std::move(response)));
        return;
      }
      // The custom commands might have created a new target so we should use
      // the selected target after these commands are run.
      dap.target = dap.debugger.GetSelectedTarget();
    }
  }

  // Make sure the process is attached and stopped.
  error = dap.WaitForProcessToStop(std::chrono::seconds(timeout_seconds));

  // Clients can request a baseline of currently existing threads after
  // we acknowledge the configurationDone request.
  // Client requests the baseline of currently existing threads after
  // a successful or attach by sending a 'threads' request
  // right after receiving the configurationDone response.
  // Obtain the list of threads before we resume the process
  dap.initial_thread_list =
      GetThreads(dap.target.GetProcess(), dap.thread_format);

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

  // FIXME: Move this into PostRun.
  if (error.Success()) {
    if (dap.target.GetProcess().IsValid()) {
      SendProcessEvent(dap, Attach);

      if (dap.stop_at_entry)
        SendThreadStoppedEvent(dap);
      else
        dap.target.GetProcess().Continue();
    }
  }
}

} // namespace lldb_dap
