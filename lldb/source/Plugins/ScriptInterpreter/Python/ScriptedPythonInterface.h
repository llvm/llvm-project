//===-- ScriptedPythonInterface.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPYTHONINTERFACE_H

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "lldb/Interpreter/ScriptedInterface.h"
#include "lldb/Utility/DataBufferHeap.h"

#include "PythonDataObjects.h"
#include "SWIGPythonBridge.h"
#include "ScriptInterpreterPythonImpl.h"

namespace lldb_private {
class ScriptInterpreterPythonImpl;
class ScriptedPythonInterface : virtual public ScriptedInterface {
public:
  ScriptedPythonInterface(ScriptInterpreterPythonImpl &interpreter);
  virtual ~ScriptedPythonInterface() = default;

protected:
  template <typename T = StructuredData::ObjectSP>
  T ExtractValueFromPythonObject(python::PythonObject &p, Status &error) {
    return p.CreateStructuredObject();
  }

  template <typename T = StructuredData::ObjectSP, typename... Args>
  T Dispatch(llvm::StringRef method_name, Status &error, Args... args) {
    using namespace python;
    using Locker = ScriptInterpreterPythonImpl::Locker;

    std::string caller_signature =
        llvm::Twine(__PRETTY_FUNCTION__ + llvm::Twine(" (") +
                    llvm::Twine(method_name) + llvm::Twine(")"))
            .str();
    if (!m_object_instance_sp)
      return ErrorWithMessage<T>(caller_signature, "Python object ill-formed",
                                 error);

    Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);

    PythonObject implementor(PyRefType::Borrowed,
                             (PyObject *)m_object_instance_sp->GetValue());

    if (!implementor.IsAllocated())
      return ErrorWithMessage<T>(caller_signature,
                                 "Python implementor not allocated.", error);

    PythonObject pmeth(
        PyRefType::Owned,
        PyObject_GetAttrString(implementor.get(), method_name.str().c_str()));

    if (PyErr_Occurred())
      PyErr_Clear();

    if (!pmeth.IsAllocated())
      return ErrorWithMessage<T>(caller_signature,
                                 "Python method not allocated.", error);

    if (PyCallable_Check(pmeth.get()) == 0) {
      if (PyErr_Occurred())
        PyErr_Clear();
      return ErrorWithMessage<T>(caller_signature,
                                 "Python method not callable.", error);
    }

    if (PyErr_Occurred())
      PyErr_Clear();

    // TODO: make `const char *` when removing support for Python 2.
    char *format = nullptr;
    std::string format_buffer;

    if (sizeof...(Args) > 0) {
      FormatArgs(format_buffer, args...);
      // TODO: make `const char *` when removing support for Python 2.
      format = const_cast<char *>(format_buffer.c_str());
    }

    // TODO: make `const char *` when removing support for Python 2.
    PythonObject py_return(
        PyRefType::Owned,
        PyObject_CallMethod(implementor.get(),
                            const_cast<char *>(method_name.data()), format,
                            args...));

    if (PyErr_Occurred()) {
      PyErr_Print();
      PyErr_Clear();
      return ErrorWithMessage<T>(caller_signature,
                                 "Python method could not be called.", error);
    }

    if (!py_return.IsAllocated())
      return ErrorWithMessage<T>(caller_signature, "Returned object is null.",
                                 error);

    return ExtractValueFromPythonObject<T>(py_return, error);
  }

  Status GetStatusFromMethod(llvm::StringRef method_name);

  template <typename T, typename... Args>
  void FormatArgs(std::string &fmt, T arg, Args... args) const {
    FormatArgs(fmt, arg);
    FormatArgs(fmt, args...);
  }

  template <typename T> void FormatArgs(std::string &fmt, T arg) const {
    fmt += GetPythonValueFormatString(arg);
  }

  void FormatArgs(std::string &fmt) const {}

  // The lifetime is managed by the ScriptInterpreter
  ScriptInterpreterPythonImpl &m_interpreter;
};

template <>
StructuredData::DictionarySP
ScriptedPythonInterface::ExtractValueFromPythonObject<
    StructuredData::DictionarySP>(python::PythonObject &p, Status &error);

template <>
Status ScriptedPythonInterface::ExtractValueFromPythonObject<Status>(
    python::PythonObject &p, Status &error);

template <>
lldb::DataExtractorSP
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DataExtractorSP>(
    python::PythonObject &p, Status &error);

template <>
llvm::Optional<MemoryRegionInfo>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    llvm::Optional<MemoryRegionInfo>>(python::PythonObject &p, Status &error);

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPYTHONINTERFACE_H
