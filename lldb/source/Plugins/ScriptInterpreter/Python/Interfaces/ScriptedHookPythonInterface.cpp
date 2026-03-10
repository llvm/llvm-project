//===-- ScriptedHookPythonInterface.cpp
//------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "../lldb-python.h"
#include "ScriptedHookPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;

ScriptedHookPythonInterface::ScriptedHookPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedHookInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedHookPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, lldb::TargetSP target_sp,
    const StructuredDataImpl &args_sp) {
  return ScriptedPythonInterface::CreatePluginObject(class_name, nullptr,
                                                     target_sp, args_sp);
}

void ScriptedHookPythonInterface::HandleModuleLoaded(
    lldb::StreamSP &output_sp) {
  Status error;
  Dispatch("handle_module_loaded", error, output_sp);
}

void ScriptedHookPythonInterface::HandleModuleUnloaded(
    lldb::StreamSP &output_sp) {
  Status error;
  Dispatch("handle_module_unloaded", error, output_sp);
}

llvm::Expected<bool>
ScriptedHookPythonInterface::HandleStop(ExecutionContext &exe_ctx,
                                        lldb::StreamSP &output_sp) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  Status error;
  StructuredData::ObjectSP obj =
      Dispatch("handle_stop", error, exe_ctx_ref_sp, output_sp);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error)) {
    if (!obj)
      return true;
    return error.ToError();
  }

  return obj->GetBooleanValue();
}

void ScriptedHookPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "target hook add -P <script-name> [-k key -v value ...]"};
  const std::vector<llvm::StringRef> api_usages = {};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      llvm::StringRef("Perform actions on target lifecycle events (module "
                      "load/unload, process stop)."),
      CreateInstance, eScriptLanguagePython, {ci_usages, api_usages});
}

void ScriptedHookPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
