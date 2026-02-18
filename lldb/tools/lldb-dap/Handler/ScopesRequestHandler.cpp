//===-- ScopesRequestHandler.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "RequestHandler.h"
#include "Variables.h"

using namespace lldb_dap::protocol;
namespace lldb_dap {

llvm::Expected<ScopesResponseBody>
ScopesRequestHandler::Run(const ScopesArguments &args) const {
  lldb::SBFrame frame = dap.GetLLDBFrame(args.frameId);

  // As the user selects different stack frames in the GUI, a "scopes" request
  // will be sent to the DAP. This is the only way we know that the user has
  // selected a frame in a thread. There are no other notifications that are
  // sent and VS code doesn't allow multiple frames to show variables
  // concurrently. If we select the thread and frame as the "scopes" requests
  // are sent, this allows users to type commands in the debugger console
  // with a backtick character to run lldb commands and these lldb commands
  // will now have the right context selected as they are run. If the user
  // types "`bt" into the debugger console, and we had another thread selected
  // in the LLDB library, we would show the wrong thing to the user. If the
  // users switch threads with a lldb command like "`thread select 14", the
  // GUI will not update as there are no "event" notification packets that
  // allow us to change the currently selected thread or frame in the GUI that
  // I am aware of.
  if (frame.IsValid()) {
    frame.GetThread().GetProcess().SetSelectedThread(frame.GetThread());
    frame.GetThread().SetSelectedFrame(frame.GetFrameID());
  }

  std::vector<protocol::Scope> scopes =
      dap.reference_storage.CreateScopes(frame);

  return ScopesResponseBody{std::move(scopes)};
}

} // namespace lldb_dap
