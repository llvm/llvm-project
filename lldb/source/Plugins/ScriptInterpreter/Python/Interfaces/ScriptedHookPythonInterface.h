//===-- ScriptedHookPythonInterface.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDHOOKPYTHONINTERFACE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDHOOKPYTHONINTERFACE_H

#include "lldb/Interpreter/Interfaces/ScriptedHookInterface.h"

#include "ScriptedPythonInterface.h"

namespace lldb_private {
class ScriptedHookPythonInterface : public ScriptedHookInterface,
                                    public ScriptedPythonInterface,
                                    public PluginInterface {
public:
  ScriptedHookPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, lldb::TargetSP target_sp,
                     const StructuredDataImpl &args_sp) override;

  /// A hook class must implement at least one callback. All three are
  /// individually optional; hooks that implement none will be rejected
  /// at creation time.
  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return {};
  }

  /// Check which of the three hook methods the Python class implements.
  SupportedHookMethods GetSupportedMethods() override;

  void HandleModuleLoaded(lldb::StreamSP &output_sp) override;
  void HandleModuleUnloaded(lldb::StreamSP &output_sp) override;
  llvm::Expected<bool> HandleStop(ExecutionContext &exe_ctx,
                                  lldb::StreamSP &output_sp) override;

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedHookPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDHOOKPYTHONINTERFACE_H
