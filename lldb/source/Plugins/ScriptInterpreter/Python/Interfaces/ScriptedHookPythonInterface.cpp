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
  // This entry point has no error channel, so a failure can only be logged.
  LogAndDefault(Dispatch("handle_module_loaded", output_sp),
                LLVM_PRETTY_FUNCTION);
}

void ScriptedHookPythonInterface::HandleModuleUnloaded(
    lldb::StreamSP &output_sp) {
  // This entry point has no error channel, so a failure can only be logged.
  LogAndDefault(Dispatch("handle_module_unloaded", output_sp),
                LLVM_PRETTY_FUNCTION);
}

llvm::Expected<bool>
ScriptedHookPythonInterface::HandleStop(ExecutionContext &exe_ctx,
                                        lldb::StreamSP &output_sp) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  llvm::Expected<StructuredData::ObjectSP> obj_or_err =
      Dispatch("handle_stop", exe_ctx_ref_sp, output_sp);
  if (!obj_or_err)
    return obj_or_err.takeError();

  // `handle_stop` is required, so a null object here means the hook returned
  // None: it expressed no preference, so stay stopped.
  StructuredData::ObjectSP obj = *obj_or_err;
  if (!obj || !obj->IsValid())
    return true;

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
      CreateInstance, eScriptedExtensionScriptedHook, eScriptLanguagePython,
      {ci_usages, api_usages});
}

void ScriptedHookPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
