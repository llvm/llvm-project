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
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "lldb/API/SBListener.h"
#include "lldb/lldb-defines.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// The `attach` request is sent from the client to the debug adapter to attach
/// to a debuggee that is already running.
///
/// Since attaching is debugger/runtime specific, the arguments for this request
/// are not part of this specification.
Error AttachRequestHandler::Run(const AttachRequestArguments &args) const {
  dap.SetConfiguration(args.configuration, true);
  if (!args.coreFile.empty())
    dap.stop_at_entry = true;

  // If both pid and port numbers are specified.
  if ((args.pid != LLDB_INVALID_PROCESS_ID) &&
      (args.gdbRemotePort != LLDB_DAP_INVALID_PORT))
    return make_error<DAPError>(
        "pid and gdb-remote-port are mutually exclusive");

  PrintWelcomeMessage();

  // This is a hack for loading DWARF in .o files on Mac where the .o files
  // in the debug map of the main executable have relative paths which
  // require the lldb-dap binary to have its working directory set to that
  // relative root for the .o files in order to be able to load debug info.
  if (!dap.configuration.debuggerRoot.empty())
    sys::fs::set_current_path(dap.configuration.debuggerRoot);

  // Run any initialize LLDB commands the user specified in the launch.json
  if (llvm::Error err = dap.RunInitCommands())
    return err;

  dap.ConfigureSourceMaps();

  lldb::SBError error;
  lldb::SBTarget target = dap.CreateTarget(error);
  if (error.Fail())
    return ToError(error);

  dap.SetTarget(target);

  // Run any pre run LLDB commands the user specified in the launch.json
  if (Error err = dap.RunPreRunCommands())
    return err;

  if ((args.pid == LLDB_INVALID_PROCESS_ID ||
       args.gdbRemotePort == LLDB_DAP_INVALID_PORT) &&
      args.waitFor) {
    char attach_msg[256];
    auto attach_msg_len = snprintf(attach_msg, sizeof(attach_msg),
                                   "Waiting to attach to \"%s\"...",
                                   dap.target.GetExecutable().GetFilename());
    dap.SendOutput(OutputType::Console, StringRef(attach_msg, attach_msg_len));
  }

  if (args.attachCommands.empty()) {
    // No "attachCommands", just attach normally.

    // Disable async events so the attach will be successful when we return from
    // the launch call and the launch will happen synchronously
    ScopeSyncMode scope_sync_mode(dap.debugger);

    if (args.coreFile.empty()) {
      if (args.gdbRemotePort != LLDB_DAP_INVALID_PORT) {
        lldb::SBListener listener = dap.debugger.GetListener();
        std::string connect_url =
            llvm::formatv("connect://{0}:", args.gdbRemoteHostname);
        connect_url += std::to_string(args.gdbRemotePort);
        dap.target.ConnectRemote(listener, connect_url.data(), "gdb-remote",
                                 error);
      } else {
        // Attach by pid or process name.
        lldb::SBAttachInfo attach_info;
        if (args.pid != LLDB_INVALID_PROCESS_ID)
          attach_info.SetProcessID(args.pid);
        else if (!args.configuration.program.empty())
          attach_info.SetExecutable(args.configuration.program.data());
        attach_info.SetWaitForLaunch(args.waitFor, false /*async*/);
        dap.target.Attach(attach_info, error);
      }
    } else
      dap.target.LoadCore(args.coreFile.data(), error);
  } else {
    // We have "attachCommands" that are a set of commands that are expected
    // to execute the commands after which a process should be created. If there
    // is no valid process after running these commands, we have failed.
    if (llvm::Error err = dap.RunAttachCommands(args.attachCommands))
      return err;

    // The custom commands might have created a new target so we should use the
    // selected target after these commands are run.
    dap.target = dap.debugger.GetSelectedTarget();
    if (!dap.target.IsValid())
      return make_error<DAPError>(
          "attachCommands failed to create a valid target");

    // Make sure the process is attached and stopped before proceeding as the
    // the launch commands are not run using the synchronous mode.
    error = dap.WaitForProcessToStop(dap.configuration.timeout);
  }

  if (error.Fail())
    return ToError(error);

  if (args.coreFile.empty() && !dap.target.GetProcess().IsValid())
    return make_error<DAPError>("failed to attach to process");

  dap.RunPostRunCommands();

  return Error::success();
}

void AttachRequestHandler::PostRun() const {
  if (dap.target.GetProcess().IsValid()) {
    SendProcessEvent(dap, Attach);
  }

  dap.SendJSON(CreateEventObject("initialized"));
}

} // namespace lldb_dap
