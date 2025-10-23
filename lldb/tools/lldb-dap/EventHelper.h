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
#include "Protocol/ProtocolEvents.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace lldb_dap {
struct DAP;

enum LaunchMethod { Launch, Attach, AttachForSuspendedLaunch };

/// Sends target based capabilities and lldb-dap custom capabilities.
void SendExtraCapabilities(DAP &dap);

void SendProcessEvent(DAP &dap, LaunchMethod launch_method);

llvm::Error SendThreadStoppedEvent(DAP &dap, bool on_entry = false);

void SendTerminatedEvent(DAP &dap);

void SendStdOutStdErr(DAP &dap, lldb::SBProcess &process);

void SendContinuedEvent(DAP &dap);

void SendProcessExitedEvent(DAP &dap, lldb::SBProcess &process);

void SendInvalidatedEvent(
    DAP &dap, llvm::ArrayRef<protocol::InvalidatedEventBody::Area> areas);

void SendMemoryEvent(DAP &dap, lldb::SBValue variable);

/// Event thread function that handles debugger events for multiple DAP sessions
/// sharing the same debugger instance. This runs in its own thread and
/// dispatches events to the appropriate DAP instance.
///
/// \param debugger The debugger instance to listen for events from.
/// \param broadcaster The broadcaster for stop event thread notifications.
/// \param client_name The client name for thread naming/logging purposes.
/// \param log The log instance for logging.
void EventThread(lldb::SBDebugger debugger, lldb::SBBroadcaster broadcaster,
                 llvm::StringRef client_name, Log *log);

} // namespace lldb_dap

#endif
