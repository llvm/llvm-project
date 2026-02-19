//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSYMBOLLOCATORPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSYMBOLLOCATORPYTHONINTERFACE_H

#include "lldb/Host/Config.h"
#include "lldb/Interpreter/Interfaces/ScriptedSymbolLocatorInterface.h"

#if LLDB_ENABLE_PYTHON

#include "ScriptedPythonInterface.h"

#include <optional>

namespace lldb_private {
class ScriptedSymbolLocatorPythonInterface
    : public ScriptedSymbolLocatorInterface,
      public ScriptedPythonInterface,
      public PluginInterface {
public:
  ScriptedSymbolLocatorPythonInterface(
      ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(const llvm::StringRef class_name,
                     ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp,
                     StructuredData::Generic *script_obj = nullptr) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>(
        {{"locate_source_file", 2}});
  }

  std::optional<FileSpec> LocateSourceFile(const lldb::ModuleSP &module_sp,
                                           const FileSpec &original_source_file,
                                           Status &error) override;

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedSymbolLocatorPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSYMBOLLOCATORPYTHONINTERFACE_H
