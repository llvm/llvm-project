//===-- ScriptedHookPythonInterface.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedHookPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;

ScriptedHookPythonInterface::ScriptedHookPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedHookInterface(), ScriptedPythonInterface(interpreter) {}

ScriptedHookInterface::SupportedHookMethods
ScriptedHookPythonInterface::GetSupportedMethods() {
  SupportedHookMethods methods;
  // Qualify through ScriptedPythonInterface to resolve the diamond
  // inheritance (both ScriptedHookInterface and ScriptedPythonInterface
  // inherit ScriptedInterface which owns m_object_instance_sp).
  auto &obj_sp = ScriptedPythonInterface::m_object_instance_sp;
  if (!obj_sp)
    return methods;

  using Locker = ScriptInterpreterPythonImpl::Locker;
  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  PythonObject implementor(PyRefType::Borrowed, (PyObject *)obj_sp->GetValue());
  if (!implementor.IsValid())
    return methods;

  methods.handle_module_loaded =
      implementor.HasAttribute("handle_module_loaded");
  methods.handle_module_unloaded =
      implementor.HasAttribute("handle_module_unloaded");
  methods.handle_stop = implementor.HasAttribute("handle_stop");
  methods.handle_resolve_addr =
      implementor.HasAttribute("handle_resolve_addr");
  return methods;
}

llvm::Expected<StructuredData::GenericSP>
ScriptedHookPythonInterface::CreatePluginObject(
    const ScriptedMetadata &scripted_metadata, lldb::TargetSP target_sp) {
  StructuredDataImpl args_sp(scripted_metadata.GetArgsSP());
  return ScriptedPythonInterface::CreatePluginObject(scripted_metadata, nullptr,
                                                     target_sp, args_sp);
}

void ScriptedHookPythonInterface::HandleModuleLoaded(
    lldb::StreamSP &output_sp) {
  Status error;
  Dispatch("handle_module_loaded", error, output_sp);
  if (error.Fail()) {
    LLDB_LOG(GetLog(LLDBLog::Script), "handle_module_loaded failed: {0}",
             error.AsCString());
  }
}

void ScriptedHookPythonInterface::HandleModuleUnloaded(
    lldb::StreamSP &output_sp) {
  Status error;
  Dispatch("handle_module_unloaded", error, output_sp);
  if (error.Fail()) {
    LLDB_LOG(GetLog(LLDBLog::Script), "handle_module_unloaded failed: {0}",
             error.AsCString());
  }
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
    LLDB_LOG(GetLog(LLDBLog::Script), "handle_stop failed: {0}",
             error.AsCString());
    if (!obj)
      return true;
    return error.ToError();
  }

  return obj->GetBooleanValue();
}

std::optional<Address>
ScriptedHookPythonInterface::HandleResolveAddress(lldb::addr_t load_addr,
                                                  lldb::StreamSP &output_sp) {
  Status error;
  Address addr =
      Dispatch<Address>("handle_resolve_addr", error, load_addr, output_sp);
  if (error.Fail()) {
    LLDB_LOG(GetLog(LLDBLog::Script), "handle_resolve_addr failed: {0}",
             error.AsCString());
  }
  if (addr.IsSectionOffset())
    return addr;
  return std::nullopt;
}

void ScriptedHookPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "target hook add -P <script-name> [-k key -v value ...]"};
  const std::vector<llvm::StringRef> api_usages = {};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      llvm::StringRef("Perform actions on target lifecycle events (module "
                      "load/unload, process stop)."),
      CreateInstance, eScriptedExtensionScriptedHook, eScriptLanguagePython,
      {ci_usages, api_usages});
}

void ScriptedHookPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
