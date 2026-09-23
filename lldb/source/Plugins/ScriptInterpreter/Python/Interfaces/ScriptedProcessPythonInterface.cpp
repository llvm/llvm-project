//===-- ScriptedProcessPythonInterface.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedThreadPythonInterface.h"
#include "ScriptedProcessPythonInterface.h"

#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedProcessPythonInterface::ScriptedProcessPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedProcessInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedProcessPythonInterface::CreatePluginObject(
    const ScriptedMetadata &scripted_metadata, ExecutionContext &exe_ctx,
    StructuredData::Generic *script_obj) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  return ScriptedPythonInterface::CreatePluginObject(
      scripted_metadata, script_obj, exe_ctx_ref_sp,
      scripted_metadata.GetArgsSP());
}

StructuredData::DictionarySP ScriptedProcessPythonInterface::GetCapabilities() {
  StructuredData::DictionarySP dict =
      LogAndDefault(Dispatch<StructuredData::DictionarySP>("get_capabilities"),
                    LLVM_PRETTY_FUNCTION);
  if (!dict)
    return {};

  return dict;
}

StructuredData::DictionarySP
ScriptedProcessPythonInterface::GetAddressableBits() {
  StructuredData::DictionarySP dict = LogAndDefault(
      Dispatch<StructuredData::DictionarySP>("get_addressable_bits"),
      LLVM_PRETTY_FUNCTION);
  if (!dict)
    return {};

  return dict;
}

Status
ScriptedProcessPythonInterface::Attach(const ProcessAttachInfo &attach_info) {
  lldb::ProcessAttachInfoSP attach_info_sp =
      std::make_shared<ProcessAttachInfo>(attach_info);
  return GetStatusFromMethod("attach", attach_info_sp);
}

Status ScriptedProcessPythonInterface::Launch() {
  return GetStatusFromMethod("launch");
}

Status ScriptedProcessPythonInterface::Resume() {
  // When calling ScriptedProcess.Resume from lldb we should always stop.
  return GetStatusFromMethod("resume", /*should_stop=*/true);
}

std::optional<MemoryRegionInfo>
ScriptedProcessPythonInterface::GetMemoryRegionContainingAddress(
    lldb::addr_t address, Status &error) {
  llvm::Expected<std::optional<MemoryRegionInfo>> mem_region_or_err =
      Dispatch<std::optional<MemoryRegionInfo>>(
          "get_memory_region_containing_address", address);
  if (!mem_region_or_err) {
    error = Status::FromError(mem_region_or_err.takeError());
    return {};
  }

  return *mem_region_or_err;
}

StructuredData::DictionarySP ScriptedProcessPythonInterface::GetThreadsInfo() {
  StructuredData::DictionarySP dict =
      LogAndDefault(Dispatch<StructuredData::DictionarySP>("get_threads_info"),
                    LLVM_PRETTY_FUNCTION);
  if (!dict)
    return {};

  return dict;
}

bool ScriptedProcessPythonInterface::CreateBreakpoint(lldb::addr_t addr,
                                                      Status &error) {
  llvm::Expected<StructuredData::ObjectSP> obj_or_err =
      Dispatch("create_breakpoint", addr, error);
  // If there was an error on the python call, surface it to the user.
  if (!obj_or_err) {
    error = Status::FromError(obj_or_err.takeError());
    return {};
  }

  StructuredData::ObjectSP obj = *obj_or_err;
  if (!obj || !obj->IsValid())
    return {};

  return obj->GetBooleanValue();
}

lldb::DataExtractorSP ScriptedProcessPythonInterface::ReadMemoryAtAddress(
    lldb::addr_t address, size_t size, Status &error) {
  llvm::Expected<lldb::DataExtractorSP> data_or_err =
      Dispatch<lldb::DataExtractorSP>("read_memory_at_address", address, size,
                                      error);
  // If there was an error on the python call, surface it to the user.
  if (!data_or_err) {
    error = Status::FromError(data_or_err.takeError());
    return {};
  }

  return *data_or_err;
}

lldb::offset_t ScriptedProcessPythonInterface::WriteMemoryAtAddress(
    lldb::addr_t addr, lldb::DataExtractorSP data_sp, Status &error) {
  llvm::Expected<StructuredData::ObjectSP> obj_or_err =
      Dispatch("write_memory_at_address", addr, data_sp, error);
  // If there was an error on the python call, surface it to the user.
  if (!obj_or_err) {
    error = Status::FromError(obj_or_err.takeError());
    return LLDB_INVALID_OFFSET;
  }

  StructuredData::ObjectSP obj = *obj_or_err;
  if (!obj || !obj->IsValid())
    return LLDB_INVALID_OFFSET;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_OFFSET);
}

StructuredData::ArraySP ScriptedProcessPythonInterface::GetLoadedImages() {
  StructuredData::ArraySP array =
      LogAndDefault(Dispatch<StructuredData::ArraySP>("get_loaded_images"),
                    LLVM_PRETTY_FUNCTION);
  if (!array)
    return {};

  return array;
}

lldb::pid_t ScriptedProcessPythonInterface::GetProcessID() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_process_id"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return LLDB_INVALID_PROCESS_ID;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_PROCESS_ID);
}

bool ScriptedProcessPythonInterface::IsAlive() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("is_alive"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetBooleanValue();
}

std::optional<std::string>
ScriptedProcessPythonInterface::GetScriptedThreadPluginName() {
  StructuredData::ObjectSP obj = LogAndDefault(
      Dispatch("get_scripted_thread_plugin"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

lldb::ScriptedThreadInterfaceSP
ScriptedProcessPythonInterface::CreateScriptedThreadInterface() {
  return m_interpreter.CreateScriptedThreadInterface();
}

StructuredData::DictionarySP ScriptedProcessPythonInterface::GetMetadata() {
  StructuredData::DictionarySP dict = LogAndDefault(
      Dispatch<StructuredData::DictionarySP>("get_process_metadata"),
      LLVM_PRETTY_FUNCTION);
  if (!dict)
    return {};

  return dict;
}

void ScriptedProcessPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "process attach -C <script-name> [-k key -v value ...]",
      "process launch -C <script-name> [-k key -v value ...]"};
  const std::vector<llvm::StringRef> api_usages = {
      "SBAttachInfo.SetScriptedProcessClassName",
      "SBAttachInfo.SetScriptedProcessDictionary",
      "SBTarget.Attach",
      "SBLaunchInfo.SetScriptedProcessClassName",
      "SBLaunchInfo.SetScriptedProcessDictionary",
      "SBTarget.Launch"};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), llvm::StringRef("Mock process state"),
      CreateInstance, eScriptedExtensionScriptedProcess, eScriptLanguagePython,
      {ci_usages, api_usages});
}

void ScriptedProcessPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
