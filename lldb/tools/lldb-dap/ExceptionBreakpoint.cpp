//===-- ExceptionBreakpoint.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionBreakpoint.h"
#include "BreakpointBase.h"
#include "DAP.h"
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBMutex.h"
#include "lldb/API/SBTarget.h"
#include <mutex>

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

protocol::Breakpoint ExceptionBreakpoint::SetBreakpoint(StringRef condition) {
  lldb::SBMutex lock = m_dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  if (!m_bp.IsValid()) {
    m_bp = m_dap.target.BreakpointCreateForException(
        m_language, m_kind == eExceptionKindCatch,
        m_kind == eExceptionKindThrow);
    m_bp.AddName(BreakpointBase::kDAPBreakpointLabel);
  }

  m_bp.SetCondition(condition.data());

  protocol::Breakpoint breakpoint;
  breakpoint.id = m_bp.GetID();
  breakpoint.verified = m_bp.IsValid();
  return breakpoint;
}

void ExceptionBreakpoint::ClearBreakpoint() {
  if (!m_bp.IsValid())
    return;
  m_dap.target.BreakpointDelete(m_bp.GetID());
  m_bp = lldb::SBBreakpoint();
}

} // namespace lldb_dap
