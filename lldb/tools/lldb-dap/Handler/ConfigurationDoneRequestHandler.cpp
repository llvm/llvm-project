//===-- ConfigurationDoneRequestHandler..cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "ProtocolUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBDebugger.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// This request indicates that the client has finished initialization of the
/// debug adapter.
///
/// So it is the last request in the sequence of configuration requests (which
/// was started by the `initialized` event).
///
/// Clients should only call this request if the corresponding capability
/// `supportsConfigurationDoneRequest` is true.
llvm::Error
ConfigurationDoneRequestHandler::Run(const ConfigurationDoneArguments &) const {
  dap.configuration_done = true;

  // Ensure any command scripts did not leave us in an unexpected state.
  lldb::SBProcess process = dap.target.GetProcess();
  if (!process.IsValid() ||
      !lldb::SBDebugger::StateIsStoppedState(process.GetState()))
    return make_error<DAPError>(
        "Expected process to be stopped.\r\n\r\nProcess is in an unexpected "
        "state and may have missed an initial configuration. Please check that "
        "any debugger command scripts are not resuming the process during the "
        "launch sequence.");

  // Waiting until 'configurationDone' to send target based capabilities in case
  // the launch or attach scripts adjust the target. The initial dummy target
  // may have different capabilities than the final target.
  SendTargetBasedCapabilities(dap);

  // Clients can request a baseline of currently existing threads after
  // we acknowledge the configurationDone request.
  // Client requests the baseline of currently existing threads after
  // a successful or attach by sending a 'threads' request
  // right after receiving the configurationDone response.
  // Obtain the list of threads before we resume the process
  dap.initial_thread_list = GetThreads(process, dap.thread_format);

  SendProcessEvent(dap, dap.is_attach ? Attach : Launch);

  if (dap.stop_at_entry)
    return SendThreadStoppedEvent(dap, /*on_entry=*/true);

  return ToError(process.Continue());
}

} // namespace lldb_dap
