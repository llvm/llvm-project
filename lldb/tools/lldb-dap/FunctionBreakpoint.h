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
#include "Protocol/ProtocolTypes.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <utility>

namespace lldb_dap {

class FunctionBreakpoint : public Breakpoint {
public:
  /// Identifies a function breakpoint: (module name, function name).
  /// The module name is empty if the breakpoint isn't restricted to a module.
  using Key = std::pair<std::string, std::string>;

  FunctionBreakpoint(DAP &dap, const protocol::FunctionBreakpoint &breakpoint);

  /// Set this breakpoint in LLDB as a new breakpoint.
  void SetBreakpoint();

  llvm::StringRef GetFunctionName() const { return m_function_name; }
  /// The module the lookup is restricted to, or empty if it isn't restricted.
  llvm::StringRef GetModuleName() const { return m_module_name; }

  Key GetKey() const { return {m_module_name, m_function_name}; }

protected:
  std::string m_function_name;
  /// Module to restrict the lookup to, empty if there is no restriction.
  /// Together with the function name it identifies the breakpoint, so it is
  /// never updated in place.
  std::string m_module_name;
};

} // namespace lldb_dap

#endif
