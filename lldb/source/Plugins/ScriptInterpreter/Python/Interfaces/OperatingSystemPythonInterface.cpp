//===-- ScriptedThreadPythonInterface.cpp ---------------------------------===//
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
//clang-format on

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
    llvm::StringRef class_name, ExecutionContext &exe_ctx,
    StructuredData::DictionarySP args_sp, StructuredData::Generic *script_obj) {
  return ScriptedPythonInterface::CreatePluginObject(class_name, nullptr,
                                                     exe_ctx.GetProcessSP());
}

StructuredData::DictionarySP
OperatingSystemPythonInterface::CreateThread(lldb::tid_t tid,
                                             lldb::addr_t context) {
  Status error;
  StructuredData::DictionarySP dict = Dispatch<StructuredData::DictionarySP>(
      "create_thread", error, tid, context);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, dict,
                                                    error))
    return {};

  return dict;
}

StructuredData::ArraySP OperatingSystemPythonInterface::GetThreadInfo() {
  Status error;
  StructuredData::ArraySP arr =
      Dispatch<StructuredData::ArraySP>("get_thread_info", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, arr,
                                                    error))
    return {};

  return arr;
}

StructuredData::DictionarySP OperatingSystemPythonInterface::GetRegisterInfo() {
  return ScriptedThreadPythonInterface::GetRegisterInfo();
}

std::optional<std::string>
OperatingSystemPythonInterface::GetRegisterContextForTID(lldb::tid_t tid) {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_register_data", error, tid);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return {};

  return obj->GetAsString()->GetValue().str();
}

std::optional<bool> OperatingSystemPythonInterface::DoesPluginReportAllThreads() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("does_plugin_report_all_threads", error);
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return {};

  return obj->GetAsBoolean()->GetValue();
}

void OperatingSystemPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "settings set target.process.python-os-plugin-path <script-path>",
      "settings set process.experimental.os-plugin-reports-all-threads [0/1]"};
  const std::vector<llvm::StringRef> api_usages = {};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), llvm::StringRef("Mock thread state"),
      CreateInstance, eScriptLanguagePython, {ci_usages, api_usages});
}

void OperatingSystemPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

#endif
