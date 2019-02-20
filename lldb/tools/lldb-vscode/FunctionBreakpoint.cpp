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
    : BreakpointBase(obj), functionName(GetString(obj, "name")) {}

void FunctionBreakpoint::SetBreakpoint() {
  if (functionName.empty())
    return;
  bp = g_vsc.target.BreakpointCreateByName(functionName.c_str());
  if (!condition.empty())
    SetCondition();
  if (!hitCondition.empty())
    SetHitCondition();
}

}
