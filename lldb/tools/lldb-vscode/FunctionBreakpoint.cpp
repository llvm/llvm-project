//===-- FunctionBreakpoint.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionBreakpoint.h"
#include "VSCode.h"

namespace lldb_vscode {

FunctionBreakpoint::FunctionBreakpoint(const llvm::json::Object &obj)
    : BreakpointBase(obj), functionName(std::string(GetString(obj, "name"))) {}

void FunctionBreakpoint::SetBreakpoint() {
  if (functionName.empty())
    return;
  bp = g_vsc.target.BreakpointCreateByName(functionName.c_str());
  // See comments in BreakpointBase::GetBreakpointLabel() for details of why
  // we add a label to our breakpoints.
  bp.AddName(GetBreakpointLabel());
  if (!condition.empty())
    SetCondition();
  if (!hitCondition.empty())
    SetHitCondition();
  if (!logMessage.empty())
    SetLogMessage();
}

} // namespace lldb_vscode
