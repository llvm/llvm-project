//===-- EventHelper.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_EVENTHELPER_H
#define LLDB_TOOLS_LLDB_DAP_EVENTHELPER_H

#include "DAPForward.h"

namespace lldb_dap {
struct DAP;

enum LaunchMethod { Launch, Attach, AttachForSuspendedLaunch };

void SendProcessEvent(DAP &dap, LaunchMethod launch_method);

void SendThreadStoppedEvent(DAP &dap);

void SendTerminatedEvent(DAP &dap);

void SendStdOutStdErr(DAP &dap, lldb::SBProcess &process);

void SendContinuedEvent(DAP &dap);

void SendProcessExitedEvent(DAP &dap, lldb::SBProcess &process);

} // namespace lldb_dap

#endif
