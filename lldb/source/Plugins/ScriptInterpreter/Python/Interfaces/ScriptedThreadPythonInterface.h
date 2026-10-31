//===-- ScriptedThreadPythonInterface.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDTHREADPYTHONINTERFACE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDTHREADPYTHONINTERFACE_H

#include "ScriptedPythonInterface.h"
#include "lldb/Interpreter/Interfaces/ScriptedThreadInterface.h"
#include <optional>

namespace lldb_private {
class ScriptedThreadPythonInterface : public ScriptedThreadInterface,
                                      public ScriptedPythonInterface,
                                      virtual public PluginInterface {
public:
  ScriptedThreadPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(const ScriptedMetadata &scripted_metadata,
                     ExecutionContext &exe_ctx,
                     StructuredData::Generic *script_obj = nullptr) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>(
        {{"get_stop_reason"}, {"get_register_context"}});
  }

  lldb::tid_t GetThreadID() override;

  std::optional<std::string> GetName() override;

  lldb::StateType GetState() override;

  std::optional<std::string> GetQueue() override;

  StructuredData::DictionarySP GetStopReason() override;

  StructuredData::ArraySP GetStackFrames() override;

  StructuredData::DictionarySP GetRegisterInfo() override;

  std::optional<std::string> GetRegisterContext() override;

  StructuredData::ArraySP GetExtendedInfo() override;

  std::optional<std::string> GetScriptedFramePluginName() override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedThreadPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  lldb::ScriptedFrameInterfaceSP CreateScriptedFrameInterface() override;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDTHREADPYTHONINTERFACE_H
