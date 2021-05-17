//===-- ScriptedProcessPythonInterface.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "lldb-python.h"

#include "SWIGPythonBridge.h"
#include "ScriptInterpreterPythonImpl.h"
#include "ScriptedProcessPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

StructuredData::GenericSP ScriptedProcessPythonInterface::CreatePluginObject(
    const llvm::StringRef class_name, lldb::TargetSP target_sp,
    StructuredData::DictionarySP args_sp) {
  if (class_name.empty())
    return {};

  std::string error_string;
  StructuredDataImpl *args_impl = nullptr;
  if (args_sp) {
    args_impl = new StructuredDataImpl();
    args_impl->SetObjectSP(args_sp);
  }

  void *ret_val;

  {

    Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);

    ret_val = LLDBSwigPythonCreateScriptedProcess(
        class_name.str().c_str(), m_interpreter.GetDictionaryName(), target_sp,
        args_impl, error_string);
  }

  m_object_instance_sp =
      StructuredData::GenericSP(new StructuredPythonObject(ret_val));

  return m_object_instance_sp;
}

Status ScriptedProcessPythonInterface::Launch() {
  return LaunchOrResume("launch");
}

Status ScriptedProcessPythonInterface::Resume() {
  return LaunchOrResume("resume");
}

Status
ScriptedProcessPythonInterface::LaunchOrResume(llvm::StringRef method_name) {
  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  if (!m_object_instance_sp)
    return Status("Python object ill-formed.");

  if (!m_object_instance_sp)
    return Status("Cannot convert Python object to StructuredData::Generic.");
  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)m_object_instance_sp->GetValue());

  if (!implementor.IsAllocated())
    return Status("Python implementor not allocated.");

  PythonObject pmeth(
      PyRefType::Owned,
      PyObject_GetAttrString(implementor.get(), method_name.str().c_str()));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return Status("Python method not allocated.");

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return Status("Python method not callable.");
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  PythonObject py_return(PyRefType::Owned,
                         PyObject_CallMethod(implementor.get(),
                                             method_name.str().c_str(),
                                             nullptr));

  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    return Status("Python method could not be called.");
  }

  if (PyObject *py_ret_ptr = py_return.get()) {
    lldb::SBError *sb_error =
        (lldb::SBError *)LLDBSWIGPython_CastPyObjectToSBError(py_ret_ptr);

    if (!sb_error)
      return Status("Couldn't cast lldb::SBError to lldb::Status.");

    Status status = m_interpreter.GetStatusFromSBError(*sb_error);

    if (status.Fail())
      return Status("error: %s", status.AsCString());

    return status;
  }

  return Status("Returned object is null.");
}

size_t
ScriptedProcessPythonInterface::GetGenericInteger(llvm::StringRef method_name) {
  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  if (!m_object_instance_sp)
    return LLDB_INVALID_ADDRESS;

  if (!m_object_instance_sp)
    return LLDB_INVALID_ADDRESS;
  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)m_object_instance_sp->GetValue());

  if (!implementor.IsAllocated())
    return LLDB_INVALID_ADDRESS;

  PythonObject pmeth(
      PyRefType::Owned,
      PyObject_GetAttrString(implementor.get(), method_name.str().c_str()));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return LLDB_INVALID_ADDRESS;

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return LLDB_INVALID_ADDRESS;
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  PythonObject py_return(PyRefType::Owned,
                         PyObject_CallMethod(implementor.get(),
                                             method_name.str().c_str(),
                                             nullptr));

  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }

  if (py_return.get()) {
    auto size = py_return.AsUnsignedLongLong();
    return (size) ? *size : LLDB_INVALID_ADDRESS;
  }
  return LLDB_INVALID_ADDRESS;
}

lldb::MemoryRegionInfoSP
ScriptedProcessPythonInterface::GetMemoryRegionContainingAddress(
    lldb::addr_t address) {
  // TODO: Implement
  return nullptr;
}

StructuredData::DictionarySP
ScriptedProcessPythonInterface::GetThreadWithID(lldb::tid_t tid) {
  // TODO: Implement
  return nullptr;
}

StructuredData::DictionarySP
ScriptedProcessPythonInterface::GetRegistersForThread(lldb::tid_t tid) {
  // TODO: Implement
  return nullptr;
}

lldb::DataExtractorSP ScriptedProcessPythonInterface::ReadMemoryAtAddress(
    lldb::addr_t address, size_t size, Status &error) {
  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  auto error_with_message = [&error](llvm::StringRef message) {
    error.SetErrorString(message);
    return nullptr;
  };

  static char callee_name[] = "read_memory_at_address";
  std::string param_format = GetPythonValueFormatString(address);
  param_format += GetPythonValueFormatString(size);

  if (!m_object_instance_sp)
    return error_with_message("Python object ill-formed.");

  if (!m_object_instance_sp)
    return error_with_message("Python method not callable.");

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)m_object_instance_sp->GetValue());

  if (!implementor.IsAllocated())
    return error_with_message("Python implementor not allocated.");

  PythonObject pmeth(PyRefType::Owned,
                     PyObject_GetAttrString(implementor.get(), callee_name));

  if (PyErr_Occurred())
    PyErr_Clear();

  if (!pmeth.IsAllocated())
    return error_with_message("Python method not allocated.");

  if (PyCallable_Check(pmeth.get()) == 0) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return error_with_message("Python method not callable.");
  }

  if (PyErr_Occurred())
    PyErr_Clear();

  PythonObject py_return(PyRefType::Owned,
                         PyObject_CallMethod(implementor.get(), callee_name,
                                             param_format.c_str(), address,
                                             size));

  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    return error_with_message("Python method could not be called.");
  }

  if (PyObject *py_ret_ptr = py_return.get()) {
    lldb::SBData *sb_data =
        (lldb::SBData *)LLDBSWIGPython_CastPyObjectToSBData(py_ret_ptr);

    if (!sb_data)
      return error_with_message(
          "Couldn't cast lldb::SBData to lldb::DataExtractor.");

    return m_interpreter.GetDataExtractorFromSBData(*sb_data);
  }

  return error_with_message("Returned object is null.");
}

StructuredData::DictionarySP ScriptedProcessPythonInterface::GetLoadedImages() {
  // TODO: Implement
  return nullptr;
}

lldb::pid_t ScriptedProcessPythonInterface::GetProcessID() {
  size_t pid = GetGenericInteger("get_process_id");

  return (pid >= std::numeric_limits<lldb::pid_t>::max())
             ? LLDB_INVALID_PROCESS_ID
             : pid;
}

bool ScriptedProcessPythonInterface::IsAlive() {
  return GetGenericInteger("is_alive");
  ;
}

#endif
