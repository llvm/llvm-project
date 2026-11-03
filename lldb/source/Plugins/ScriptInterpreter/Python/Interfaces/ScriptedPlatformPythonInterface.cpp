//===-- ScriptedPlatformPythonInterface.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedPlatformPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedPlatformPythonInterface::ScriptedPlatformPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedPlatformInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedPlatformPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, ExecutionContext &exe_ctx,
    StructuredData::DictionarySP args_sp, StructuredData::Generic *script_obj) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  ScriptedMetadata scripted_metadata(class_name, args_sp);
  return ScriptedPythonInterface::CreatePluginObject(
      scripted_metadata, script_obj, exe_ctx_ref_sp, args_sp);
}

StructuredData::DictionarySP ScriptedPlatformPythonInterface::ListProcesses() {
  StructuredData::DictionarySP dict_sp =
      LogAndDefault(Dispatch<StructuredData::DictionarySP>("list_processes"),
                    LLVM_PRETTY_FUNCTION);
  if (!dict_sp || !dict_sp->IsValid())
    return {};

  return dict_sp;
}

StructuredData::DictionarySP
ScriptedPlatformPythonInterface::GetProcessInfo(lldb::pid_t pid) {
  StructuredData::DictionarySP dict_sp = LogAndDefault(
      Dispatch<StructuredData::DictionarySP>("get_process_info", pid),
      LLVM_PRETTY_FUNCTION);
  if (!dict_sp || !dict_sp->IsValid())
    return {};

  return dict_sp;
}

Status ScriptedPlatformPythonInterface::AttachToProcess(
    ProcessAttachInfoSP attach_info) {
  // FIXME: Pass `attach_info` to method call
  return GetStatusFromMethod("attach_to_process");
}

Status ScriptedPlatformPythonInterface::LaunchProcess(
    ProcessLaunchInfoSP launch_info) {
  // FIXME: Pass `launch_info` to method call
  return GetStatusFromMethod("launch_process");
}

Status ScriptedPlatformPythonInterface::KillProcess(lldb::pid_t pid) {
  return GetStatusFromMethod("kill_process", pid);
}

void ScriptedPlatformPythonInterface::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "Mock platform and interact with its processes.",
      CreateInstance, eScriptedExtensionScriptedPlatform, eScriptLanguagePython,
      {});
}

void ScriptedPlatformPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
