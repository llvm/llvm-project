//===-- Watchpoint.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_WATCHPOINT_H
#define LLDB_TOOLS_LLDB_DAP_WATCHPOINT_H

#include "BreakpointBase.h"
#include "DAPForward.h"
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBWatchpoint.h"
#include "lldb/API/SBWatchpointOptions.h"
#include "lldb/lldb-types.h"
#include <cstddef>

namespace lldb_dap {

class Watchpoint : public BreakpointBase {
public:
  Watchpoint(DAP &d, const protocol::DataBreakpointInfo &breakpoint);
  Watchpoint(DAP &d, lldb::SBWatchpoint wp) : BreakpointBase(d), m_wp(wp) {}

  void SetCondition() override;
  void SetHitCondition() override;

  protocol::Breakpoint ToProtocolBreakpoint() override;

  void SetWatchpoint();

  lldb::addr_t GetAddress() const { return m_addr; }

protected:
  lldb::addr_t m_addr;
  size_t m_size;
  lldb::SBWatchpointOptions m_options;
  /// The LLDB breakpoint associated wit this watchpoint.
  lldb::SBWatchpoint m_wp;
  lldb::SBError m_error;
};
} // namespace lldb_dap

#endif
