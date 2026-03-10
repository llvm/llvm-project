//===-- ScriptedModuleHookPythonInterface.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "../lldb-python.h"
#include "ScriptedModuleHookPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;

ScriptedModuleHookPythonInterface::ScriptedModuleHookPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedModuleHookInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedModuleHookPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, lldb::TargetSP target_sp,
    const StructuredDataImpl &args_sp) {
  return ScriptedPythonInterface::CreatePluginObject(class_name, nullptr,
                                                     target_sp, args_sp);
}

void ScriptedModuleHookPythonInterface::HandleModuleLoaded(
    lldb::StreamSP &output_sp) {
  Status error;
  // We pass only the output stream to Python. The Python class can access
  // self.target (set during __init__) to query loaded modules if needed.
  Dispatch("handle_module_loaded", error, output_sp);
}

void ScriptedModuleHookPythonInterface::HandleModuleUnloaded(
    lldb::StreamSP &output_sp) {
  Status error;
  // Optional method. If the Python class does not implement
  // handle_module_unloaded, the dispatch will fail silently.
  Dispatch("handle_module_unloaded", error, output_sp);
}

void ScriptedModuleHookPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "target hook add -P <script-name> [-k key -v value ...]"};
  const std::vector<llvm::StringRef> api_usages = {};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      llvm::StringRef("Perform actions whenever modules are loaded into the "
                      "target."),
      CreateInstance, eScriptLanguagePython, {ci_usages, api_usages});
}

void ScriptedModuleHookPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
