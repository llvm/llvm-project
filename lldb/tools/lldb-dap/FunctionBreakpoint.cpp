//===-- FunctionBreakpoint.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionBreakpoint.h"
#include "DAP.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFileSpecList.h"
#include "lldb/API/SBMutex.h"
#include <mutex>
#include <string>

namespace lldb_dap {

FunctionBreakpoint::FunctionBreakpoint(
    DAP &d, const protocol::FunctionBreakpoint &breakpoint)
    : Breakpoint(d, breakpoint.condition, breakpoint.hitCondition),
      m_function_name(breakpoint.name) {
  for (const protocol::String &module : breakpoint.modules)
    if (!module.empty())
      m_modules.insert(module.str());
}

void FunctionBreakpoint::SetBreakpoint() {
  lldb::SBMutex lock = m_dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  if (m_function_name.empty())
    return;

  lldb::SBFileSpecList module_list;
  for (const std::string &module : m_modules)
    module_list.AppendIfUnique(
        lldb::SBFileSpec(module.c_str(), /*resolve=*/false));

  m_bp = m_dap.target.BreakpointCreateByName(
      m_function_name.c_str(), module_list, lldb::SBFileSpecList());
  Breakpoint::SetBreakpoint();
}

} // namespace lldb_dap
