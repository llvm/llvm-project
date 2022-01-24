//===-- ScriptedProcessPythonInterface.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "lldb-python.h"

#include "SWIGPythonBridge.h"
#include "ScriptInterpreterPythonImpl.h"
#include "ScriptedProcessPythonInterface.h"
#include "ScriptedThreadPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedProcessPythonInterface::ScriptedProcessPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedProcessInterface(), ScriptedPythonInterface(interpreter) {}

StructuredData::GenericSP ScriptedProcessPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, ExecutionContext &exe_ctx,
    StructuredData::DictionarySP args_sp, StructuredData::Generic *script_obj) {
  if (class_name.empty())
    return {};

  TargetSP target_sp = exe_ctx.GetTargetSP();
  StructuredDataImpl args_impl(args_sp);
  std::string error_string;

  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  PythonObject ret_val = LLDBSwigPythonCreateScriptedProcess(
      class_name.str().c_str(), m_interpreter.GetDictionaryName(), target_sp,
      args_impl, error_string);

  m_object_instance_sp =
      StructuredData::GenericSP(new StructuredPythonObject(std::move(ret_val)));

  return m_object_instance_sp;
}

Status ScriptedProcessPythonInterface::Launch() {
  return GetStatusFromMethod("launch");
}

Status ScriptedProcessPythonInterface::Resume() {
  return GetStatusFromMethod("resume");
}

bool ScriptedProcessPythonInterface::ShouldStop() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("is_alive", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return {};

  return obj->GetBooleanValue();
}

Status ScriptedProcessPythonInterface::Stop() {
  return GetStatusFromMethod("stop");
}

llvm::Optional<MemoryRegionInfo>
ScriptedProcessPythonInterface::GetMemoryRegionContainingAddress(
    lldb::addr_t address, Status &error) {
  auto mem_region = Dispatch<llvm::Optional<MemoryRegionInfo>>(
      "get_memory_region_containing_address", error, address);

  if (error.Fail()) {
    return ErrorWithMessage<MemoryRegionInfo>(LLVM_PRETTY_FUNCTION,
                                              error.AsCString(), error);
  }

  return mem_region;
}

StructuredData::DictionarySP ScriptedProcessPythonInterface::GetThreadsInfo() {
  Status error;
  StructuredData::DictionarySP dict =
      Dispatch<StructuredData::DictionarySP>("get_threads_info", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, dict, error))
    return {};

  return dict;
}

StructuredData::DictionarySP
ScriptedProcessPythonInterface::GetThreadWithID(lldb::tid_t tid) {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_thread_with_id", error, tid);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return {};

  StructuredData::DictionarySP dict{obj->GetAsDictionary()};

  return dict;
}

StructuredData::DictionarySP
ScriptedProcessPythonInterface::GetRegistersForThread(lldb::tid_t tid) {
  // TODO: Implement
  return {};
}

lldb::DataExtractorSP ScriptedProcessPythonInterface::ReadMemoryAtAddress(
    lldb::addr_t address, size_t size, Status &error) {
  return Dispatch<lldb::DataExtractorSP>("read_memory_at_address", error,
                                         address, size);
}

StructuredData::DictionarySP ScriptedProcessPythonInterface::GetLoadedImages() {
  // TODO: Implement
  return {};
}

lldb::pid_t ScriptedProcessPythonInterface::GetProcessID() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_process_id", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return LLDB_INVALID_PROCESS_ID;

  return obj->GetIntegerValue(LLDB_INVALID_PROCESS_ID);
}

bool ScriptedProcessPythonInterface::IsAlive() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("is_alive", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return {};

  return obj->GetBooleanValue();
}

llvm::Optional<std::string>
ScriptedProcessPythonInterface::GetScriptedThreadPluginName() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_scripted_thread_plugin", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return {};

  return obj->GetStringValue().str();
}

lldb::ScriptedThreadInterfaceSP
ScriptedProcessPythonInterface::CreateScriptedThreadInterface() {
  return std::make_shared<ScriptedThreadPythonInterface>(m_interpreter);
}

#endif
