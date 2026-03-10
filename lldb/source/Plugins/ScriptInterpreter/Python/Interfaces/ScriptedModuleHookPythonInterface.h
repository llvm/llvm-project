//===-- ScriptedModuleHookPythonInterface.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDMODULEHOOKPYTHONINTERFACE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDMODULEHOOKPYTHONINTERFACE_H

#include "lldb/Interpreter/Interfaces/ScriptedModuleHookInterface.h"

#include "ScriptedPythonInterface.h"

namespace lldb_private {
class ScriptedModuleHookPythonInterface : public ScriptedModuleHookInterface,
                                          public ScriptedPythonInterface,
                                          public PluginInterface {
public:
  ScriptedModuleHookPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, lldb::TargetSP target_sp,
                     const StructuredDataImpl &args_sp) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>(
        {{"handle_module_loaded", 1}});
  }

  void HandleModuleLoaded(lldb::StreamSP &output_sp) override;

  /// Optional: only called if the Python class implements
  /// handle_module_unloaded. Silently does nothing otherwise.
  void HandleModuleUnloaded(lldb::StreamSP &output_sp) override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedModuleHookPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDMODULEHOOKPYTHONINTERFACE_H
