//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDCOMMANDPYTHONINTERFACE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDCOMMANDPYTHONINTERFACE_H

#include "lldb/Interpreter/Interfaces/ScriptedCommandInterface.h"

#include "ScriptedPythonInterface.h"
namespace lldb_private {

class ScriptedCommandPythonInterface : public ScriptedCommandInterface,
                                       public ScriptedPythonInterface,
                                       public PluginInterface {
public:
  ScriptedCommandPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name,
                     lldb::DebuggerSP debugger_sp) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>();
  }

  bool RunRawCommand(llvm::StringRef args,
                     ScriptedCommandSynchronicity synchronicity,
                     CommandReturnObject &cmd_retobj, Status &error,
                     const ExecutionContext &exe_ctx) override;

  bool RunParsedCommand(Args &args, ScriptedCommandSynchronicity synchronicity,
                        CommandReturnObject &cmd_retobj, Status &error,
                        const ExecutionContext &exe_ctx) override;

  std::optional<std::string> GetRepeatCommand(Args &args) override;

  StructuredData::DictionarySP
  HandleArgumentCompletion(std::vector<std::string> &args, size_t args_pos,
                           size_t char_in_arg) override;

  StructuredData::DictionarySP
  HandleOptionArgumentCompletion(llvm::StringRef &long_option,
                                 size_t char_in_arg) override;

  bool GetShortHelp(std::string &dest) override;

  bool GetLongHelp(std::string &dest) override;

  uint32_t GetFlags() override;

  StructuredData::ObjectSP GetOptionsDefinition() override;

  StructuredData::ObjectSP GetArgumentsDefinition() override;

  void OptionParsingStarted() override;

  bool SetOptionValue(ExecutionContext *exe_ctx, llvm::StringRef long_option,
                      llvm::StringRef value) override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedCommandPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  lldb::DebuggerSP m_debugger_sp;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDCOMMANDPYTHONINTERFACE_H
