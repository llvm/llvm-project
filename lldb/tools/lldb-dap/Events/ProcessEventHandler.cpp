//===-- ProcessEventHandler.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Events/EventHandler.h"
#include "Protocol/ProtocolEvents.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

ProcessEventBody ProcessEventHandler::Handler() const {
  ProcessEventBody body;

  char path[PATH_MAX] = {0};
  dap.target.GetExecutable().GetPath(path, sizeof(path));
  body.name = path;
  body.systemProcessId = dap.target.GetProcess().GetProcessID();
  body.isLocalProcess = dap.target.GetPlatform().GetName() ==
                        lldb::SBPlatform::GetHostPlatform().GetName();
  body.startMethod = dap.is_attach ? ProcessEventBody::StartMethod::attach
                                   : ProcessEventBody::StartMethod::launch;
  body.pointerSize = dap.target.GetAddressByteSize();
  return body;
}

} // namespace lldb_dap
