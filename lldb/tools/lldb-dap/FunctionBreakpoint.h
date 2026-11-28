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
#include <set>
#include <string>
#include <utility>

namespace lldb_dap {

class FunctionBreakpoint : public Breakpoint {
public:
  /// Identifies a function breakpoint: (function name, set of module names).
  /// The set of modules is empty if the breakpoint isn't restricted.
  using Key = std::pair<std::string, std::set<std::string>>;

  FunctionBreakpoint(DAP &dap, const protocol::FunctionBreakpoint &breakpoint);

  /// Set this breakpoint in LLDB as a new breakpoint.
  void SetBreakpoint();

  llvm::StringRef GetFunctionName() const { return m_function_name; }

  Key GetKey() const { return {m_function_name, m_modules}; }

protected:
  std::string m_function_name;
  /// Names of the modules to restrict the lookup to, without duplicates. Empty
  /// if there is no restriction. Together with the function name they identify
  /// the breakpoint, so they are never updated in place.
  std::set<std::string> m_modules;
};

} // namespace lldb_dap

#endif
