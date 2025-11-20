//===-- InitializeRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

/// Initialize request; value of command field is 'initialize'.
llvm::Expected<InitializeResponse> InitializeRequestHandler::Run(
    const InitializeRequestArguments &arguments) const {
  if (auto err = dap.CreateDebugger(arguments.supportedFeatures))
    return err;

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

  return dap.GetCapabilities();
}
