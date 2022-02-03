//===-- ScriptedThreadPythonInterface.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "lldb-python.h"

#include "SWIGPythonBridge.h"
#include "ScriptInterpreterPythonImpl.h"
#include "ScriptedThreadPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedThreadPythonInterface::ScriptedThreadPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedThreadInterface(), ScriptedPythonInterface(interpreter) {}

StructuredData::GenericSP ScriptedThreadPythonInterface::CreatePluginObject(
    const llvm::StringRef class_name, ExecutionContext &exe_ctx,
    StructuredData::DictionarySP args_sp, StructuredData::Generic *script_obj) {
  if (class_name.empty() && !script_obj)
    return {};

  ProcessSP process_sp = exe_ctx.GetProcessSP();
  StructuredDataImpl args_impl(args_sp);
  std::string error_string;

  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  PythonObject ret_val;

  if (!script_obj)
    ret_val = LLDBSwigPythonCreateScriptedThread(
        class_name.str().c_str(), m_interpreter.GetDictionaryName(), process_sp,
        args_impl, error_string);
  else
    ret_val = PythonObject(PyRefType::Borrowed,
                           static_cast<PyObject *>(script_obj->GetValue()));

  if (!ret_val)
    return {};

  m_object_instance_sp =
      StructuredData::GenericSP(new StructuredPythonObject(std::move(ret_val)));

  return m_object_instance_sp;
}

lldb::tid_t ScriptedThreadPythonInterface::GetThreadID() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_thread_id", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return LLDB_INVALID_THREAD_ID;

  return obj->GetIntegerValue(LLDB_INVALID_THREAD_ID);
}

llvm::Optional<std::string> ScriptedThreadPythonInterface::GetName() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_name", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return {};

  return obj->GetStringValue().str();
}

lldb::StateType ScriptedThreadPythonInterface::GetState() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_state", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return eStateInvalid;

  return static_cast<StateType>(obj->GetIntegerValue(eStateInvalid));
}

llvm::Optional<std::string> ScriptedThreadPythonInterface::GetQueue() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_queue", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return {};

  return obj->GetStringValue().str();
}

StructuredData::DictionarySP ScriptedThreadPythonInterface::GetStopReason() {
  Status error;
  StructuredData::DictionarySP dict =
      Dispatch<StructuredData::DictionarySP>("get_stop_reason", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, dict, error))
    return {};

  return dict;
}

StructuredData::ArraySP ScriptedThreadPythonInterface::GetStackFrames() {
  return nullptr;
}

StructuredData::DictionarySP ScriptedThreadPythonInterface::GetRegisterInfo() {
  Status error;
  StructuredData::DictionarySP dict =
      Dispatch<StructuredData::DictionarySP>("get_register_info", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, dict, error))
    return {};

  return dict;
}

llvm::Optional<std::string>
ScriptedThreadPythonInterface::GetRegisterContext() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_register_context", error);

  if (!CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj, error))
    return {};

  return obj->GetAsString()->GetValue().str();
}

#endif
