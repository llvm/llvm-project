//===-- ScriptedPlatformPythonInterface.cpp -------------------------------===//
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
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// clang-format off
// LLDB Python header must be included first
#include "../../lldb-python.h"
//clang-format on

#include "../../SWIGPythonBridge.h"
#include "../../ScriptInterpreterPythonImpl.h"
#include "ScriptedPlatformPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

LLDB_PLUGIN_DEFINE_ADV(ScriptedPlatformPythonInterface, ScriptInterpreterPythonScriptedPlatformPythonInterface)

ScriptedPlatformPythonInterface::ScriptedPlatformPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedPlatformInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedPlatformPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, ExecutionContext &exe_ctx,
    StructuredData::DictionarySP args_sp, StructuredData::Generic *script_obj) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  StructuredDataImpl sd_impl(args_sp);
  return ScriptedPythonInterface::CreatePluginObject(class_name, script_obj,
                                                     exe_ctx_ref_sp, sd_impl);
}

StructuredData::ArraySP
ScriptedPlatformPythonInterface::GetSupportedArchitectures() {
  Status error;
  StructuredData::ArraySP arr =
      Dispatch<StructuredData::ArraySP>("get_supported_architectures", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, arr,
                                                    error))
    return {};

  return arr;
}

StructuredData::DictionarySP ScriptedPlatformPythonInterface::ListProcesses() {
  Status error;
  StructuredData::DictionarySP dict =
      Dispatch<StructuredData::DictionarySP>("list_processes", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, dict,
                                                    error))
    return {};

  return dict;
}

StructuredData::DictionarySP
ScriptedPlatformPythonInterface::GetProcessInfo(lldb::pid_t pid) {
  Status error;
  StructuredData::DictionarySP dict_sp =
      Dispatch<StructuredData::DictionarySP>("get_process_info", error, pid);

  if (!dict_sp || !dict_sp->IsValid() || error.Fail()) {
    return ScriptedInterface::ErrorWithMessage<StructuredData::DictionarySP>(
        LLVM_PRETTY_FUNCTION,
        llvm::Twine("Null or invalid object (" +
                    llvm::Twine(error.AsCString()) + llvm::Twine(")."))
            .str(),
        error);
  }

  return dict_sp;
}

lldb::ProcessSP ScriptedPlatformPythonInterface::AttachToProcess(
    lldb::ProcessAttachInfoSP attach_info_sp, lldb::TargetSP target_sp,
    lldb::DebuggerSP debugger_sp, Status &error) {
  Status py_error;
  ProcessSP process_sp =
      Dispatch<ProcessSP>("attach_to_process", py_error, attach_info_sp,
                          target_sp, debugger_sp, error);

  if (!process_sp || error.Fail()) {
    return ScriptedInterface::ErrorWithMessage<ProcessSP>(
        LLVM_PRETTY_FUNCTION,
        llvm::Twine("Null or invalid object (" +
                    llvm::Twine(error.AsCString()) + llvm::Twine(")."))
            .str(),
        error);
  }

  return process_sp;
}

Status ScriptedPlatformPythonInterface::LaunchProcess(
    ProcessLaunchInfoSP launch_info_sp) {
  return GetStatusFromMethod("launch_process", launch_info_sp);
}

Status ScriptedPlatformPythonInterface::KillProcess(lldb::pid_t pid) {
  return GetStatusFromMethod("kill_process", pid);
}

void ScriptedPlatformPythonInterface::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "Mock platform and interact with its processes.",
      CreateInstance, eScriptLanguagePython, {});
}

void ScriptedPlatformPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

#endif // LLDB_ENABLE_PYTHON
