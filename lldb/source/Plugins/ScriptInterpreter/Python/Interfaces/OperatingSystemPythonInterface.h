//===-- OperatingSystemPythonInterface.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_OPERATINGSYSTEMPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_OPERATINGSYSTEMPYTHONINTERFACE_H

#include "lldb/Host/Config.h"
#include "lldb/Interpreter/Interfaces/OperatingSystemInterface.h"

#if LLDB_ENABLE_PYTHON

#include "ScriptedThreadPythonInterface.h"

#include <optional>

namespace lldb_private {
class OperatingSystemPythonInterface
    : virtual public OperatingSystemInterface,
      virtual public ScriptedThreadPythonInterface,
      public PluginInterface {
public:
  OperatingSystemPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp,
                     StructuredData::Generic *script_obj = nullptr) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>({{"get_thread_info"}});
  }

  StructuredData::DictionarySP CreateThread(lldb::tid_t tid,
                                            lldb::addr_t context) override;

  StructuredData::ArraySP GetThreadInfo() override;

  StructuredData::DictionarySP GetRegisterInfo() override;

  std::optional<std::string> GetRegisterContextForTID(lldb::tid_t tid) override;

  std::optional<bool> DoesPluginReportAllThreads() override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "OperatingSystemPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_OPERATINGSYSTEMPYTHONINTERFACE_H
