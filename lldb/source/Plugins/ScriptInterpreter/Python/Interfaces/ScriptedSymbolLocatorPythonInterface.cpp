//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// clang-format off
// LLDB Python header must be included first
#include "../lldb-python.h"
// clang-format on

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedSymbolLocatorPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;

ScriptedSymbolLocatorPythonInterface::ScriptedSymbolLocatorPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedSymbolLocatorInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedSymbolLocatorPythonInterface::CreatePluginObject(
    const llvm::StringRef class_name, ExecutionContext &exe_ctx,
    StructuredData::DictionarySP args_sp, StructuredData::Generic *script_obj) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  StructuredDataImpl sd_impl(args_sp);
  return ScriptedPythonInterface::CreatePluginObject(class_name, script_obj,
                                                     exe_ctx_ref_sp, sd_impl);
}

std::optional<FileSpec> ScriptedSymbolLocatorPythonInterface::LocateSourceFile(
    const lldb::ModuleSP &module_sp, const FileSpec &original_source_file,
    Status &error) {
  std::string source_path = original_source_file.GetPath();
  lldb::ModuleSP module_copy(module_sp);

  FileSpec file_spec =
      Dispatch<FileSpec>("locate_source_file", error, module_copy, source_path);

  if (error.Fail() || !file_spec)
    return {};

  return file_spec;
}

void ScriptedSymbolLocatorPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "target symbols scripted register -C "
      "<script-class> [-k <key> -v <value> ...]"};
  const std::vector<llvm::StringRef> api_usages = {
      "SBTarget.RegisterScriptedSymbolLocator(class_name, args_dict)"};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      llvm::StringRef("Scripted symbol locator Python interface"),
      CreateInstance, eScriptLanguagePython, {ci_usages, api_usages});
}

void ScriptedSymbolLocatorPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

#endif // LLDB_ENABLE_PYTHON
