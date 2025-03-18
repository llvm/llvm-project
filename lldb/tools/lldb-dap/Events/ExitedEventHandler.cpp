//===-- ExitedEventHandler.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Events/EventHandler.h"
#include "lldb/API/SBProcess.h"

namespace lldb_dap {

void ExitedEventHandler::operator()(lldb::SBProcess &process) const {
  if (!process.IsValid())
    return;

  protocol::ExitedEventBody body;
  body.exitCode = process.GetExitStatus();
  dap.Send(protocol::Event{/*event=*/ExitedEventHandler::event.str(),
                           /*body=*/std::move(body)});
}

} // namespace lldb_dap
