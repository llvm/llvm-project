//===-- FunctionBreakpoint.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionBreakpoint.h"
#include "DAP.h"
#include "JSONUtils.h"
#include "lldb/API/SBMutex.h"
#include <mutex>

namespace lldb_dap {

FunctionBreakpoint::FunctionBreakpoint(DAP &d, const llvm::json::Object &obj)
    : Breakpoint(d, obj),
      m_function_name(std::string(GetString(obj, "name").value_or(""))) {}

void FunctionBreakpoint::SetBreakpoint() {
  lldb::SBMutex lock = m_dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  if (m_function_name.empty())
    return;
  m_bp = m_dap.target.BreakpointCreateByName(m_function_name.c_str());
  Breakpoint::SetBreakpoint();
}

} // namespace lldb_dap
