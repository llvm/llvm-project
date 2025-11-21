//===-- InitializeRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandPlugins.h"
#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "lldb/API/SBTarget.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

/// Initialize request; value of command field is 'initialize'.
llvm::Expected<InitializeResponse> InitializeRequestHandler::Run(
    const InitializeRequestArguments &arguments) const {
  dap.clientFeatures = arguments.supportedFeatures;

  // Do not source init files until in/out/err are configured.
  dap.debugger = lldb::SBDebugger::Create(false);
  dap.debugger.SetInputFile(dap.in);
  dap.target = dap.debugger.GetDummyTarget();

  llvm::Expected<int> out_fd = dap.out.GetWriteFileDescriptor();
  if (!out_fd)
    return out_fd.takeError();
  dap.debugger.SetOutputFile(lldb::SBFile(*out_fd, "w", false));

  llvm::Expected<int> err_fd = dap.err.GetWriteFileDescriptor();
  if (!err_fd)
    return err_fd.takeError();
  dap.debugger.SetErrorFile(lldb::SBFile(*err_fd, "w", false));

  auto interp = dap.debugger.GetCommandInterpreter();

  // The sourceInitFile option is not part of the DAP specification. It is an
  // extension used by the test suite to prevent sourcing `.lldbinit` and
  // changing its behavior. The CLI flag --no-lldbinit takes precedence over
  // the DAP parameter.
  bool should_source_init_files =
      !dap.no_lldbinit && arguments.lldbExtSourceInitFile.value_or(true);
  if (should_source_init_files) {
    dap.debugger.SkipLLDBInitFiles(false);
    dap.debugger.SkipAppInitFiles(false);
    lldb::SBCommandReturnObject init;
    interp.SourceInitFileInGlobalDirectory(init);
    interp.SourceInitFileInHomeDirectory(init);
  }

  if (llvm::Error err = dap.RunPreInitCommands())
    return err;

  auto cmd = dap.debugger.GetCommandInterpreter().AddMultiwordCommand(
      "lldb-dap", "Commands for managing lldb-dap.");
  if (arguments.supportedFeatures.contains(
          eClientFeatureStartDebuggingRequest)) {
    cmd.AddCommand(
        "start-debugging", new StartDebuggingCommand(dap),
        "Sends a startDebugging request from the debug adapter to the client "
        "to start a child debug session of the same type as the caller.");
  }
  cmd.AddCommand(
      "repl-mode", new ReplModeCommand(dap),
      "Get or set the repl behavior of lldb-dap evaluation requests.");
  cmd.AddCommand("send-event", new SendEventCommand(dap),
                 "Sends an DAP event to the client.");

  if (arguments.supportedFeatures.contains(eClientFeatureProgressReporting))
    dap.StartProgressEventThread();

  // Start our event thread so we can receive events from the debugger, target,
  // process and more.
  dap.StartEventThread();

  return dap.GetCapabilities();
}
