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
  /// How a module is identified.
  enum class ModuleRefKind { Name, Id };
  /// A module the lookup is restricted to: its name (file name or path), or its
  /// canonical id (see `Module.id`).
  using ModuleRef = std::pair<ModuleRefKind, std::string>;

  /// Identifies a function breakpoint: (function name, set of modules).
  /// The set of modules is empty if the breakpoint isn't restricted.
  using Key = std::pair<std::string, std::set<ModuleRef>>;

  FunctionBreakpoint(DAP &dap, const protocol::FunctionBreakpoint &breakpoint);

  /// Set this breakpoint in LLDB as a new breakpoint.
  void SetBreakpoint();

  protocol::Breakpoint ToProtocolBreakpoint() override;

  llvm::StringRef GetFunctionName() const { return m_function_name; }
  /// The modules the lookup is restricted to, without duplicates.
  /// Empty if the lookup isn't restricted.
  const std::set<ModuleRef> &GetModules() const { return m_modules; }

  Key GetKey() const { return {m_function_name, m_modules}; }

protected:
  std::string m_function_name;
  /// Modules to restrict the lookup to, without duplicates. Empty if there is
  /// no restriction. Together with the function name they identify the
  /// breakpoint, so they are never updated in place.
  std::set<ModuleRef> m_modules;
  /// Why the breakpoint couldn't be created, if it couldn't.
  std::string m_error;
};

} // namespace lldb_dap

#endif
