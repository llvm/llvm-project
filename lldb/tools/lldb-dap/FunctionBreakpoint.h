//===-- FunctionBreakpoint.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_FUNCTIONBREAKPOINT_H
#define LLDB_TOOLS_LLDB_DAP_FUNCTIONBREAKPOINT_H

#include "Breakpoint.h"
#include "DAPForward.h"

namespace lldb_dap {

class FunctionBreakpoint : public Breakpoint {
public:
  FunctionBreakpoint(DAP &dap, const llvm::json::Object &obj);

  /// Set this breakpoint in LLDB as a new breakpoint.
  void SetBreakpoint();

  llvm::StringRef GetFunctionName() const { return m_function_name; }

protected:
  std::string m_function_name;
};

} // namespace lldb_dap

#endif
