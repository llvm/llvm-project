//===-- SourceBreakpoint.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SourceBreakpoint.h"
#include "VSCode.h"

namespace lldb_vscode {

SourceBreakpoint::SourceBreakpoint(const llvm::json::Object &obj)
    : BreakpointBase(obj), line(GetUnsigned(obj, "line", 0)),
      column(GetUnsigned(obj, "column", 0)) {}

void SourceBreakpoint::SetBreakpoint(const llvm::StringRef source_path) {
  bp = g_vsc.target.BreakpointCreateByLocation(source_path.str().c_str(), line);
  if (!condition.empty())
    SetCondition();
  if (!hitCondition.empty())
    SetHitCondition();
}

} // namespace lldb_vscode
