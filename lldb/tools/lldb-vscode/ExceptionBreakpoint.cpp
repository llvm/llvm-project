//===-- ExceptionBreakpoint.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExceptionBreakpoint.h"
#include "VSCode.h"

namespace lldb_vscode {

void ExceptionBreakpoint::SetBreakpoint() {
  if (bp.IsValid())
    return;
  bool catch_value = filter.find("_catch") != std::string::npos;
  bool throw_value = filter.find("_throw") != std::string::npos;
  bp = g_vsc.target.BreakpointCreateForException(language, catch_value,
                                                 throw_value);
}

void ExceptionBreakpoint::ClearBreakpoint() {
  if (!bp.IsValid())
    return;
  g_vsc.target.BreakpointDelete(bp.GetID());
  bp = lldb::SBBreakpoint();
}

} // namespace lldb_vscode

