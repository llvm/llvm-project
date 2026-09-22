//===-- ScriptedThreadPythonInterface.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedThreadPythonInterface.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedThreadPythonInterface::ScriptedThreadPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedThreadInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedThreadPythonInterface::CreatePluginObject(
    const ScriptedMetadata &scripted_metadata, ExecutionContext &exe_ctx,
    StructuredData::Generic *script_obj) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  return ScriptedPythonInterface::CreatePluginObject(
      scripted_metadata, script_obj, exe_ctx_ref_sp,
      scripted_metadata.GetArgsSP());
}

lldb::tid_t ScriptedThreadPythonInterface::GetThreadID() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_thread_id"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return LLDB_INVALID_THREAD_ID;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_THREAD_ID);
}

std::optional<std::string> ScriptedThreadPythonInterface::GetName() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_name"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

lldb::StateType ScriptedThreadPythonInterface::GetState() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_state"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return eStateInvalid;

  return static_cast<StateType>(obj->GetUnsignedIntegerValue(eStateInvalid));
}

std::optional<std::string> ScriptedThreadPythonInterface::GetQueue() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_queue"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

StructuredData::DictionarySP ScriptedThreadPythonInterface::GetStopReason() {
  StructuredData::DictionarySP dict =
      LogAndDefault(Dispatch<StructuredData::DictionarySP>("get_stop_reason"),
                    LLVM_PRETTY_FUNCTION);
  if (!dict)
    return {};

  return dict;
}

StructuredData::ArraySP ScriptedThreadPythonInterface::GetStackFrames() {
  StructuredData::ArraySP arr =
      LogAndDefault(Dispatch<StructuredData::ArraySP>("get_stackframes"),
                    LLVM_PRETTY_FUNCTION);
  if (!arr)
    return {};

  return arr;
}

StructuredData::DictionarySP ScriptedThreadPythonInterface::GetRegisterInfo() {
  StructuredData::DictionarySP dict =
      LogAndDefault(Dispatch<StructuredData::DictionarySP>("get_register_info"),
                    LLVM_PRETTY_FUNCTION);
  if (!dict)
    return {};

  return dict;
}

std::optional<std::string> ScriptedThreadPythonInterface::GetRegisterContext() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_register_context"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

StructuredData::ArraySP ScriptedThreadPythonInterface::GetExtendedInfo() {
  StructuredData::ArraySP arr =
      LogAndDefault(Dispatch<StructuredData::ArraySP>("get_extended_info"),
                    LLVM_PRETTY_FUNCTION);
  if (!arr)
    return {};

  return arr;
}

std::optional<std::string>
ScriptedThreadPythonInterface::GetScriptedFramePluginName() {
  StructuredData::ObjectSP obj = LogAndDefault(
      Dispatch("get_scripted_frame_plugin"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

lldb::ScriptedFrameInterfaceSP
ScriptedThreadPythonInterface::CreateScriptedFrameInterface() {
  return m_interpreter.CreateScriptedFrameInterface();
}

void ScriptedThreadPythonInterface::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "Provide thread state for a scripted process.",
      CreateInstance, eScriptedExtensionScriptedThread, eScriptLanguagePython,
      {});
}

void ScriptedThreadPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
