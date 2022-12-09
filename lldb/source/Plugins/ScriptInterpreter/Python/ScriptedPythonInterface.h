//===-- ScriptedPythonInterface.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPYTHONINTERFACE_H

#if LLDB_ENABLE_PYTHON

#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "lldb/Host/Config.h"
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
  ~ScriptedPythonInterface() override = default;

protected:
  template <typename T = StructuredData::ObjectSP>
  T ExtractValueFromPythonObject(python::PythonObject &p, Status &error) {
    return p.CreateStructuredObject();
  }

  template <typename T = StructuredData::ObjectSP, typename... Args>
  T Dispatch(llvm::StringRef method_name, Status &error, Args &&...args) {
    using namespace python;
    using Locker = ScriptInterpreterPythonImpl::Locker;

    std::string caller_signature =
        llvm::Twine(LLVM_PRETTY_FUNCTION + llvm::Twine(" (") +
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

    std::tuple<Args...> original_args = std::forward_as_tuple(args...);
    auto transformed_args = TransformArgs(original_args);

    llvm::Expected<PythonObject> expected_return_object =
        llvm::make_error<llvm::StringError>("Not initialized.",
                                            llvm::inconvertibleErrorCode());
    std::apply(
        [&implementor, &method_name, &expected_return_object](auto &&...args) {
          llvm::consumeError(expected_return_object.takeError());
          expected_return_object =
              implementor.CallMethod(method_name.data(), args...);
        },
        transformed_args);

    if (llvm::Error e = expected_return_object.takeError()) {
      error.SetErrorString(llvm::toString(std::move(e)).c_str());
      return ErrorWithMessage<T>(caller_signature,
                                 "Python method could not be called.", error);
    }

    PythonObject py_return = std::move(expected_return_object.get());

    if (!py_return.IsAllocated())
      return ErrorWithMessage<T>(caller_signature, "Returned object is null.",
                                 error);

    // Now that we called the python method with the transformed arguments,
    // we need to interate again over both the original and transformed
    // parameter pack, and transform back the parameter that were passed in
    // the original parameter pack as references or pointers.
    if (sizeof...(Args) > 0)
      if (!ReassignPtrsOrRefsArgs(original_args, transformed_args))
        return ErrorWithMessage<T>(
            caller_signature,
            "Couldn't re-assign reference and pointer arguments.", error);

    return ExtractValueFromPythonObject<T>(py_return, error);
  }

  Status GetStatusFromMethod(llvm::StringRef method_name);

  template <typename T> T Transform(T object) {
    // No Transformation for generic usage
    return {object};
  }

  python::PythonObject Transform(Status arg) {
    return python::ToSWIGWrapper(arg);
  }

  template <typename T, typename U>
  void ReverseTransform(T &original_arg, U transformed_arg, Status &error) {
    // If U is not a PythonObject, don't touch it!
    return;
  }

  template <typename T>
  void ReverseTransform(T &original_arg, python::PythonObject transformed_arg,
                        Status &error) {
    original_arg = ExtractValueFromPythonObject<T>(transformed_arg, error);
  }

  template <std::size_t... I, typename... Args>
  auto TransformTuple(const std::tuple<Args...> &args,
                      std::index_sequence<I...>) {
    return std::make_tuple(Transform(std::get<I>(args))...);
  }

  // This will iterate over the Dispatch parameter pack and replace in-place
  // every `lldb_private` argument that has a SB counterpart.
  template <typename... Args>
  auto TransformArgs(const std::tuple<Args...> &args) {
    return TransformTuple(args, std::make_index_sequence<sizeof...(Args)>());
  }

  template <typename T, typename U>
  void TransformBack(T &original_arg, U transformed_arg, Status &error) {
    ReverseTransform(original_arg, transformed_arg, error);
  }

  template <std::size_t... I, typename... Ts, typename... Us>
  bool ReassignPtrsOrRefsArgs(std::tuple<Ts...> &original_args,
                              std::tuple<Us...> &transformed_args,
                              std::index_sequence<I...>) {
    Status error;
    (TransformBack(std::get<I>(original_args), std::get<I>(transformed_args),
                   error),
     ...);
    return error.Success();
  }

  template <typename... Ts, typename... Us>
  bool ReassignPtrsOrRefsArgs(std::tuple<Ts...> &original_args,
                              std::tuple<Us...> &transformed_args) {
    if (sizeof...(Ts) != sizeof...(Us))
      return false;

    return ReassignPtrsOrRefsArgs(original_args, transformed_args,
                                  std::make_index_sequence<sizeof...(Ts)>());
  }

  template <typename T, typename... Args>
  void FormatArgs(std::string &fmt, T arg, Args... args) const {
    FormatArgs(fmt, arg);
    FormatArgs(fmt, args...);
  }

  template <typename T> void FormatArgs(std::string &fmt, T arg) const {
    fmt += python::PythonFormat<T>::format;
  }

  void FormatArgs(std::string &fmt) const {}

  // The lifetime is managed by the ScriptInterpreter
  ScriptInterpreterPythonImpl &m_interpreter;
};

template <>
StructuredData::ArraySP
ScriptedPythonInterface::ExtractValueFromPythonObject<StructuredData::ArraySP>(
    python::PythonObject &p, Status &error);

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
