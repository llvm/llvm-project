//===-- FunctionBreakpoint.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDBVSCODE_FUNCTIONBREAKPOINT_H_
#define LLDBVSCODE_FUNCTIONBREAKPOINT_H_

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
