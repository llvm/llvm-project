//===-- ScriptedThreadPythonInterface.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "OperatingSystemPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

OperatingSystemPythonInterface::OperatingSystemPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : OperatingSystemInterface(), ScriptedThreadPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
OperatingSystemPythonInterface::CreatePluginObject(
    const ScriptedMetadata &scripted_metadata, ExecutionContext &exe_ctx,
    StructuredData::Generic *script_obj) {
  return ScriptedPythonInterface::CreatePluginObject(scripted_metadata, nullptr,
                                                     exe_ctx.GetProcessSP());
}

StructuredData::DictionarySP
OperatingSystemPythonInterface::CreateThread(lldb::tid_t tid,
                                             lldb::addr_t context) {
  StructuredData::DictionarySP dict = LogAndDefault(
      Dispatch<StructuredData::DictionarySP>("create_thread", tid, context),
      LLVM_PRETTY_FUNCTION);
  if (!dict)
    return {};

  return dict;
}

StructuredData::ArraySP OperatingSystemPythonInterface::GetThreadInfo() {
  StructuredData::ArraySP arr =
      LogAndDefault(Dispatch<StructuredData::ArraySP>("get_thread_info"),
                    LLVM_PRETTY_FUNCTION);
  if (!arr)
    return {};

  return arr;
}

StructuredData::DictionarySP OperatingSystemPythonInterface::GetRegisterInfo() {
  return ScriptedThreadPythonInterface::GetRegisterInfo();
}

std::optional<std::string>
OperatingSystemPythonInterface::GetRegisterContextForTID(lldb::tid_t tid) {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_register_data", tid), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

std::optional<bool> OperatingSystemPythonInterface::DoesPluginReportAllThreads() {
  StructuredData::ObjectSP obj = LogAndDefault(
      Dispatch("does_plugin_report_all_threads"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetBooleanValue();
}

void OperatingSystemPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "settings set target.process.python-os-plugin-path <script-path>",
      "settings set process.experimental.os-plugin-reports-all-threads [0/1]"};
  const std::vector<llvm::StringRef> api_usages = {};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), llvm::StringRef("Mock thread state"),
      CreateInstance, eScriptedExtensionOperatingSystem, eScriptLanguagePython,
      {ci_usages, api_usages});
}

void OperatingSystemPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
