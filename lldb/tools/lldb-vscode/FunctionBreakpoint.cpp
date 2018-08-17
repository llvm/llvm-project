//===-- FunctionBreakpoint.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
