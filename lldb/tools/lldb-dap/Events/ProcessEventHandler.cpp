//===-- ProcessEventHandler.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Events/EventHandler.h"
#include "Protocol/ProtocolEvents.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap::protocol;

namespace lldb_dap {

void ProcessEventHandler::operator()(SBTarget &target,
                                     ProcessStartMethod startMethod) const {
  SBProcess process = target.GetProcess();
  if (!target.IsValid() || !process.IsValid())
    return;

  char path[PATH_MAX] = {0};
  target.GetExecutable().GetPath(path, sizeof(path));

  ProcessEventBody body;
  body.name = path;
  body.systemProcessId = process.GetProcessID();
  body.isLocalProcess = target.GetPlatform().GetName() ==
                        lldb::SBPlatform::GetHostPlatform().GetName();
  body.startMethod = startMethod;
  body.pointerSize = target.GetAddressByteSize();

  dap.Send(protocol::Event{/*event=*/ProcessEventHandler::event.str(),
                           /*body*/ body});
}

} // namespace lldb_dap
