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
#include "lldb/API/SBAttachInfo.h"
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
  // Validate that we have a well formed attach request.
  if (args.attachCommands.empty() && args.coreFile.empty() &&
      args.configuration.program.empty() &&
      args.pid == LLDB_INVALID_PROCESS_ID &&
      args.gdbRemotePort == LLDB_DAP_INVALID_PORT)
    return make_error<DAPError>(
        "expected one of 'pid', 'program', 'attachCommands', "
        "'coreFile' or 'gdb-remote-port' to be specified");

  // Check if we have mutually exclusive arguments.
  if ((args.pid != LLDB_INVALID_PROCESS_ID) &&
      (args.gdbRemotePort != LLDB_DAP_INVALID_PORT))
    return make_error<DAPError>(
        "'pid' and 'gdb-remote-port' are mutually exclusive");

  dap.SetConfiguration(args.configuration, /*is_attach=*/true);
  if (!args.coreFile.empty())
    dap.stop_at_entry = true;

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
    dap.SendOutput(OutputType::Console,
                   llvm::formatv("Waiting to attach to \"{0}\"...",
                                 dap.target.GetExecutable().GetFilename())
                       .str());
  }

  {
    // Perform the launch in synchronous mode so that we don't have to worry
    // about process state changes during the launch.
    ScopeSyncMode scope_sync_mode(dap.debugger);

    if (!args.attachCommands.empty()) {
      // Run the attach commands, after which we expect the debugger's selected
      // target to contain a valid and stopped process. Otherwise inform the
      // user that their command failed or the debugger is in an unexpected
      // state.
      if (llvm::Error err = dap.RunAttachCommands(args.attachCommands))
        return err;

      dap.target = dap.debugger.GetSelectedTarget();

      // Validate the attachCommand results.
      if (!dap.target.GetProcess().IsValid())
        return make_error<DAPError>(
            "attachCommands failed to attach to a process");
    } else if (!args.coreFile.empty()) {
      dap.target.LoadCore(args.coreFile.data(), error);
    } else if (args.gdbRemotePort != LLDB_DAP_INVALID_PORT) {
      lldb::SBListener listener = dap.debugger.GetListener();

      // If the user hasn't provided the hostname property, default
      // localhost being used.
      std::string connect_url =
          llvm::formatv("connect://{0}:", args.gdbRemoteHostname);
      connect_url += std::to_string(args.gdbRemotePort);
      dap.target.ConnectRemote(listener, connect_url.c_str(), "gdb-remote",
                               error);
    } else {
      // Attach by pid or process name.
      lldb::SBAttachInfo attach_info;
      if (args.pid != LLDB_INVALID_PROCESS_ID)
        attach_info.SetProcessID(args.pid);
      else if (!dap.configuration.program.empty())
        attach_info.SetExecutable(dap.configuration.program.data());
      attach_info.SetWaitForLaunch(args.waitFor, /*async=*/false);
      dap.target.Attach(attach_info, error);
    }
    if (error.Fail())
      return ToError(error);
  }

  // Make sure the process is attached and stopped.
  error = dap.WaitForProcessToStop(args.configuration.timeout);
  if (error.Fail())
    return ToError(error);

  if (args.coreFile.empty() && !dap.target.GetProcess().IsValid())
    return make_error<DAPError>("failed to attach to process");

  dap.RunPostRunCommands();

  return Error::success();
}

void AttachRequestHandler::PostRun() const {
  dap.SendJSON(CreateEventObject("initialized"));
}

} // namespace lldb_dap
