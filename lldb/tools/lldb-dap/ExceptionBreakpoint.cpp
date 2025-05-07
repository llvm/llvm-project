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
#include "lldb/API/SBMutex.h"
#include "lldb/API/SBTarget.h"
#include <mutex>

namespace lldb_dap {

void ExceptionBreakpoint::SetBreakpoint() {
  lldb::SBMutex lock = m_dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  if (m_bp.IsValid())
    return;
  bool catch_value = m_filter.find("_catch") != std::string::npos;
  bool throw_value = m_filter.find("_throw") != std::string::npos;
  m_bp = m_dap.target.BreakpointCreateForException(m_language, catch_value,
                                                   throw_value);
  m_bp.AddName(BreakpointBase::kDAPBreakpointLabel);
}

void ExceptionBreakpoint::ClearBreakpoint() {
  if (!m_bp.IsValid())
    return;
  m_dap.target.BreakpointDelete(m_bp.GetID());
  m_bp = lldb::SBBreakpoint();
}

} // namespace lldb_dap
