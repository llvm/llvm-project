//===-- FunctionBreakpoint.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_VSCODE_FUNCTIONBREAKPOINT_H
#define LLDB_TOOLS_LLDB_VSCODE_FUNCTIONBREAKPOINT_H

#include "BreakpointBase.h"

namespace lldb_vscode {

struct FunctionBreakpoint : public BreakpointBase {
  std::string functionName;

  FunctionBreakpoint() = default;
  FunctionBreakpoint(const llvm::json::Object &obj);

  // Set this breakpoint in LLDB as a new breakpoint
  void SetBreakpoint();
};

} // namespace lldb_vscode

#endif
