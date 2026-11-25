//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSTRINGSUMMARYPYTHONINTERFACE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSTRINGSUMMARYPYTHONINTERFACE_H

#include "lldb/Interpreter/Interfaces/ScriptedStringSummaryInterface.h"

#include "ScriptedPythonInterface.h"
namespace lldb_private {

class ScriptedStringSummaryPythonInterface
    : public ScriptedStringSummaryInterface,
      public ScriptedPythonInterface,
      public PluginInterface {
public:
  ScriptedStringSummaryPythonInterface(
      ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>({{"get_summary", 3}});
  }

  llvm::Expected<std::string>
  GetSummary(ValueObject &valobj, const TypeSummaryOptions &options) override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedStringSummaryPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSTRINGSUMMARYPYTHONINTERFACE_H
