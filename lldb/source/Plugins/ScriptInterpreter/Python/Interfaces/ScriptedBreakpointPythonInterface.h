//===-- ScriptedBreakpointPythonInterface.h -----------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDBREAKPOINTPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDBREAKPOINTPYTHONINTERFACE_H

#include "lldb/Host/Config.h"
#include "lldb/Interpreter/Interfaces/ScriptedBreakpointInterface.h"

#if LLDB_ENABLE_PYTHON

#include "ScriptedPythonInterface.h"

namespace lldb_private {
class ScriptedBreakpointPythonInterface : public ScriptedBreakpointInterface,
                                          public ScriptedPythonInterface,
                                          public PluginInterface {
public:
  ScriptedBreakpointPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, lldb::BreakpointSP break_sp,
                     const StructuredDataImpl &args_sp) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>({{"__callback__", 2}});
  }

  bool ResolverCallback(SymbolContext sym_ctx) override;
  lldb::SearchDepth GetDepth() override;
  std::optional<std::string> GetShortHelp() override;
  lldb::BreakpointLocationSP
  WasHit(lldb::StackFrameSP frame_sp,
         lldb::BreakpointLocationSP bp_loc_sp) override;
  virtual std::optional<std::string>
  GetLocationDescription(lldb::BreakpointLocationSP bp_loc_sp,
                         lldb::DescriptionLevel level) override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedBreakpointPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDBREAKPOINTPYTHONINTERFACE_H
