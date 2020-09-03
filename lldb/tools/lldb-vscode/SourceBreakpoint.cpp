//===-- SourceBreakpoint.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  // See comments in BreakpointBase::GetBreakpointLabel() for details of why
  // we add a label to our breakpoints.
  bp.AddName(GetBreakpointLabel());
  if (!condition.empty())
    SetCondition();
  if (!hitCondition.empty())
    SetHitCondition();
}

} // namespace lldb_vscode
