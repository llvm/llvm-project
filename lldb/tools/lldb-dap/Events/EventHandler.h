//===-- EventHandler.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_EVENTS_EVENT_HANDLER
#define LLDB_TOOLS_LLDB_DAP_EVENTS_EVENT_HANDLER

#include "DAPForward.h"
#include "Protocol/ProtocolEvents.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_dap {

/// An event handler for triggering DAP events.
class EventHandler {
public:
  EventHandler(DAP &dap) : dap(dap) {}
  virtual ~EventHandler() = default;

protected:
  DAP &dap;
};

/// Handler for the event indicates that the debuggee has exited and returns its
/// exit code.
class ExitedEventHandler : public EventHandler {
public:
  using EventHandler::EventHandler;
  static constexpr llvm::StringLiteral event = "exited";
  void operator()(lldb::SBProcess &process) const;
};

using ProcessStartMethod = protocol::ProcessEventBody::StartMethod;

/// Handler for the event indicates that the debugger has begun debugging a new
/// process. Either one that it has launched, or one that it has attached to.
class ProcessEventHandler : public EventHandler {
public:
  using EventHandler::EventHandler;
  static constexpr llvm::StringLiteral event = "process";
  void operator()(lldb::SBTarget &target, ProcessStartMethod startMethod) const;
};

using OutputCategory = protocol::OutputEventBody::Category;

/// Handle for the event indicates that the target has produced some output.
class OutputEventHandler : public EventHandler {
public:
  using EventHandler::EventHandler;
  static constexpr llvm::StringLiteral event = "output";
  void operator()(llvm::StringRef output,
                  OutputCategory category = OutputCategory::Console) const;
};

} // end namespace lldb_dap

#endif
