//===-- ExitedEventHandler.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Events/EventHandler.h"
#include "lldb/API/SBProcess.h"

namespace lldb_dap {

protocol::ExitedEventBody ExitedEventHandler::Handler() const {
  protocol::ExitedEventBody body;
  body.exitCode = dap.target.GetProcess().GetExitStatus();
  return body;
}

} // namespace lldb_dap
