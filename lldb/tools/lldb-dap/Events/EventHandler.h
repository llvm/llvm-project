//===-- EventHandler.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_EVENTS_EVENT_HANDLER
#define LLDB_TOOLS_LLDB_DAP_EVENTS_EVENT_HANDLER

#include "DAP.h"
#include "Protocol/ProtocolBase.h"
#include "Protocol/ProtocolEvents.h"
#include "lldb/API/SBProcess.h"

namespace lldb_dap {

template <typename Body, typename... Args> class BaseEventHandler {
public:
  BaseEventHandler(DAP &dap) : dap(dap) {}

  virtual ~BaseEventHandler() = default;

  virtual llvm::StringLiteral getEvent() const = 0;
  virtual Body Handler(Args...) const = 0;

  void operator()(Args... args) const {
    Body body = Handler(args...);
    protocol::Event event{/*event=*/getEvent().str(), /*body=*/std::move(body)};
    dap.Send(event);
  }

protected:
  DAP &dap;
};

/// Handler for the event indicates that the debuggee has exited and returns its
/// exit code.
class ExitedEventHandler : public BaseEventHandler<protocol::ExitedEventBody> {
public:
  using BaseEventHandler::BaseEventHandler;
  llvm::StringLiteral getEvent() const override { return "exited"; }
  protocol::ExitedEventBody Handler() const override;
};

class ProcessEventHandler
    : public BaseEventHandler<protocol::ProcessEventBody> {
public:
  using BaseEventHandler::BaseEventHandler;
  llvm::StringLiteral getEvent() const override { return "process"; }
  protocol::ProcessEventBody Handler() const override;
};

} // end namespace lldb_dap

#endif
