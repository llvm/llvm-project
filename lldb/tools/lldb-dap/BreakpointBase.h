//===-- BreakpointBase.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_BREAKPOINTBASE_H
#define LLDB_TOOLS_LLDB_DAP_BREAKPOINTBASE_H

#include "DAPForward.h"
#include "Protocol/ProtocolTypes.h"
#include <optional>
#include <string>

namespace lldb_dap {

class BreakpointBase {
public:
  explicit BreakpointBase(DAP &d) : m_dap(d) {}
  BreakpointBase(DAP &d, const std::optional<std::string> &condition,
                 const std::optional<std::string> &hit_condition);
  virtual ~BreakpointBase() = default;

  virtual void SetCondition() = 0;
  virtual void SetHitCondition() = 0;
  virtual protocol::Breakpoint ToProtocolBreakpoint() = 0;

  void UpdateBreakpoint(const BreakpointBase &request_bp);

  /// Breakpoints in LLDB can have names added to them which are kind of like
  /// labels or categories. All breakpoints that are set through DAP get sent
  /// through the various DAP set*Breakpoint packets, and these breakpoints will
  /// be labeled with this name so if breakpoint update events come in for
  /// breakpoints that the client doesn't know about, like if a breakpoint is
  /// set manually using the debugger console, we won't report any updates on
  /// them and confused the client. This label gets added by all of the
  /// breakpoint classes after they set breakpoints to mark a breakpoint as a
  /// DAP breakpoint. We can later check a lldb::SBBreakpoint object that comes
  /// in via LLDB breakpoint changed events and check the breakpoint by calling
  /// "bool lldb::SBBreakpoint::MatchesName(const char *)" to check if a
  /// breakpoint in one of the DAP breakpoints that we should report changes
  /// for.
  static constexpr const char *kDAPBreakpointLabel = "dap";

protected:
  /// Associated DAP session.
  DAP &m_dap;

  /// An optional expression for conditional breakpoints.
  std::string m_condition;

  /// An optional expression that controls how many hits of the breakpoint are
  /// ignored. The backend is expected to interpret the expression as needed
  std::string m_hit_condition;
};

} // namespace lldb_dap

#endif
