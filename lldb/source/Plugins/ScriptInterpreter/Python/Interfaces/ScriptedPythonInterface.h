//===-- ScriptedPythonInterface.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPYTHONINTERFACE_H

#if LLDB_ENABLE_PYTHON

#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "lldb/Host/Config.h"
#include "lldb/Interpreter/Interfaces/ScriptedInterface.h"
#include "lldb/Utility/DataBufferHeap.h"

#include "../PythonDataObjects.h"
#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"

namespace lldb_private {
class ScriptInterpreterPythonImpl;
class ScriptedPythonInterface : virtual public ScriptedInterface {
public:
  ScriptedPythonInterface(ScriptInterpreterPythonImpl &interpreter);
  ~ScriptedPythonInterface() override = default;

  enum class AbstractMethodCheckerCases {
    eNotImplemented,
    eNotAllocated,
    eNotCallable,
    eUnknownArgumentCount,
    eInvalidArgumentCount,
    eValid
  };

  struct AbstrackMethodCheckerPayload {

    struct InvalidArgumentCountPayload {
      InvalidArgumentCountPayload(size_t required, size_t actual)
          : required_argument_count(required), actual_argument_count(actual) {}

      size_t required_argument_count;
      size_t actual_argument_count;
    };

    AbstractMethodCheckerCases checker_case;
    std::variant<std::monostate, InvalidArgumentCountPayload> payload;
  };

  llvm::Expected<std::map<llvm::StringLiteral, AbstrackMethodCheckerPayload>>
  CheckAbstractMethodImplementation(
      const python::PythonDictionary &class_dict) const {

    using namespace python;

    std::map<llvm::StringLiteral, AbstrackMethodCheckerPayload> checker;
#define SET_CASE_AND_CONTINUE(method_name, case)                               \
  {                                                                            \
    checker[method_name] = {case, {}};                                         \
    continue;                                                                  \
  }

    for (const AbstractMethodRequirement &requirement :
         GetAbstractMethodRequirements()) {
      llvm::StringLiteral method_name = requirement.name;
      if (!class_dict.HasKey(method_name))
        SET_CASE_AND_CONTINUE(method_name,
                              AbstractMethodCheckerCases::eNotImplemented)
      auto callable_or_err = class_dict.GetItem(method_name);
      if (!callable_or_err) {
        llvm::consumeError(callable_or_err.takeError());
        SET_CASE_AND_CONTINUE(method_name,
                              AbstractMethodCheckerCases::eNotAllocated)
      }

      PythonCallable callable = callable_or_err->AsType<PythonCallable>();
      if (!callable)
        SET_CASE_AND_CONTINUE(method_name,
                              AbstractMethodCheckerCases::eNotCallable)

      if (!requirement.min_arg_count)
        SET_CASE_AND_CONTINUE(method_name, AbstractMethodCheckerCases::eValid)

      auto arg_info_or_err = callable.GetArgInfo();
      if (!arg_info_or_err) {
        llvm::consumeError(arg_info_or_err.takeError());
        SET_CASE_AND_CONTINUE(method_name,
                              AbstractMethodCheckerCases::eUnknownArgumentCount)
      }

      PythonCallable::ArgInfo arg_info = *arg_info_or_err;
      if (requirement.min_arg_count <= arg_info.max_positional_args) {
        SET_CASE_AND_CONTINUE(method_name, AbstractMethodCheckerCases::eValid)
      } else {
        checker[method_name] = {
            AbstractMethodCheckerCases::eInvalidArgumentCount,
            AbstrackMethodCheckerPayload::InvalidArgumentCountPayload(
                requirement.min_arg_count, arg_info.max_positional_args)};
      }
    }

#undef SET_CASE_AND_CONTINUE

    return checker;
  }

  template <typename... Args>
  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name,
                     StructuredData::Generic *script_obj, Args... args) {
    using namespace python;
    using Locker = ScriptInterpreterPythonImpl::Locker;

    Log *log = GetLog(LLDBLog::Script);
    auto create_error = [](llvm::StringLiteral format, auto &&...ts) {
      return llvm::createStringError(
          llvm::formatv(format.data(), std::forward<decltype(ts)>(ts)...)
              .str());
    };

    bool has_class_name = !class_name.empty();
    bool has_interpreter_dict =
        !(llvm::StringRef(m_interpreter.GetDictionaryName()).empty());
    if (!has_class_name && !has_interpreter_dict && !script_obj) {
      if (!has_class_name)
        return create_error("Missing script class name.");
      else if (!has_interpreter_dict)
        return create_error("Invalid script interpreter dictionary.");
      else
        return create_error("Missing scripting object.");
    }

    Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);

    PythonObject result = {};

    if (script_obj) {
      result = PythonObject(PyRefType::Borrowed,
                            static_cast<PyObject *>(script_obj->GetValue()));
    } else {
      auto dict =
          PythonModule::MainModule().ResolveName<python::PythonDictionary>(
              m_interpreter.GetDictionaryName());
      if (!dict.IsAllocated())
        return create_error("Could not find interpreter dictionary: {0}",
                            m_interpreter.GetDictionaryName());

      auto init =
          PythonObject::ResolveNameWithDictionary<python::PythonCallable>(
              class_name, dict);
      if (!init.IsAllocated())
        return create_error("Could not find script class: {0}",
                            class_name.data());

      std::tuple<Args...> original_args = std::forward_as_tuple(args...);
      auto transformed_args = TransformArgs(original_args);

      std::string error_string;
      llvm::Expected<PythonCallable::ArgInfo> arg_info = init.GetArgInfo();
      if (!arg_info) {
        llvm::handleAllErrors(
            arg_info.takeError(),
            [&](PythonException &E) { error_string.append(E.ReadBacktrace()); },
            [&](const llvm::ErrorInfoBase &E) {
              error_string.append(E.message());
            });
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       error_string);
      }

      llvm::Expected<PythonObject> expected_return_object =
          create_error("Resulting object is not initialized.");

      // This relax the requirement on the number of argument for
      // initializing scripting extension if the size of the interface
      // parameter pack contains 1 less element than the extension maximum
      // number of positional arguments for this initializer.
      //
      // This addresses the cases where the embedded interpreter session
      // dictionary is passed to the extension initializer which is not used
      // most of the time.
      size_t num_args = sizeof...(Args);
      if (num_args != arg_info->max_positional_args) {
        if (num_args != arg_info->max_positional_args - 1)
          return create_error("Passed arguments ({0}) doesn't match the number "
                              "of expected arguments ({1}).",
                              num_args, arg_info->max_positional_args);

        std::apply(
            [&init, &expected_return_object](auto &&...args) {
              llvm::consumeError(expected_return_object.takeError());
              expected_return_object = init(args...);
            },
            std::tuple_cat(transformed_args, std::make_tuple(dict)));
      } else {
        std::apply(
            [&init, &expected_return_object](auto &&...args) {
              llvm::consumeError(expected_return_object.takeError());
              expected_return_object = init(args...);
            },
            transformed_args);
      }

      if (!expected_return_object)
        return expected_return_object.takeError();
      result = expected_return_object.get();
    }

    if (!result.IsValid())
      return create_error("Resulting object is not a valid Python Object.");
    if (!result.HasAttribute("__class__"))
      return create_error("Resulting object doesn't have '__class__' member.");

    PythonObject obj_class = result.GetAttributeValue("__class__");
    if (!obj_class.IsValid())
      return create_error("Resulting class object is not a valid.");
    if (!obj_class.HasAttribute("__name__"))
      return create_error(
          "Resulting object class doesn't have '__name__' member.");
    PythonString obj_class_name =
        obj_class.GetAttributeValue("__name__").AsType<PythonString>();

    PythonObject object_class_mapping_proxy =
        obj_class.GetAttributeValue("__dict__");
    if (!obj_class.HasAttribute("__dict__"))
      return create_error(
          "Resulting object class doesn't have '__dict__' member.");

    PythonCallable dict_converter = PythonModule::BuiltinsModule()
                                        .ResolveName("dict")
                                        .AsType<PythonCallable>();
    if (!dict_converter.IsAllocated())
      return create_error(
          "Python 'builtins' module doesn't have 'dict' class.");

    PythonDictionary object_class_dict =
        dict_converter(object_class_mapping_proxy).AsType<PythonDictionary>();
    if (!object_class_dict.IsAllocated())
      return create_error("Coudn't create dictionary from resulting object "
                          "class mapping proxy object.");

    auto checker_or_err = CheckAbstractMethodImplementation(object_class_dict);
    if (!checker_or_err)
      return checker_or_err.takeError();

    llvm::Error abstract_method_errors = llvm::Error::success();
    for (const auto &method_checker : *checker_or_err)
      switch (method_checker.second.checker_case) {
      case AbstractMethodCheckerCases::eNotImplemented:
        abstract_method_errors = llvm::joinErrors(
            std::move(abstract_method_errors),
            std::move(create_error("Abstract method {0}.{1} not implemented.",
                                   obj_class_name.GetString(),
                                   method_checker.first)));
        break;
      case AbstractMethodCheckerCases::eNotAllocated:
        abstract_method_errors = llvm::joinErrors(
            std::move(abstract_method_errors),
            std::move(create_error("Abstract method {0}.{1} not allocated.",
                                   obj_class_name.GetString(),
                                   method_checker.first)));
        break;
      case AbstractMethodCheckerCases::eNotCallable:
        abstract_method_errors = llvm::joinErrors(
            std::move(abstract_method_errors),
            std::move(create_error("Abstract method {0}.{1} not callable.",
                                   obj_class_name.GetString(),
                                   method_checker.first)));
        break;
      case AbstractMethodCheckerCases::eUnknownArgumentCount:
        abstract_method_errors = llvm::joinErrors(
            std::move(abstract_method_errors),
            std::move(create_error(
                "Abstract method {0}.{1} has unknown argument count.",
                obj_class_name.GetString(), method_checker.first)));
        break;
      case AbstractMethodCheckerCases::eInvalidArgumentCount: {
        auto &payload_variant = method_checker.second.payload;
        if (!std::holds_alternative<
                AbstrackMethodCheckerPayload::InvalidArgumentCountPayload>(
                payload_variant)) {
          abstract_method_errors = llvm::joinErrors(
              std::move(abstract_method_errors),
              std::move(create_error(
                  "Abstract method {0}.{1} has unexpected argument count.",
                  obj_class_name.GetString(), method_checker.first)));
        } else {
          auto payload = std::get<
              AbstrackMethodCheckerPayload::InvalidArgumentCountPayload>(
              payload_variant);
          abstract_method_errors = llvm::joinErrors(
              std::move(abstract_method_errors),
              std::move(
                  create_error("Abstract method {0}.{1} has unexpected "
                               "argument count (expected {2} but has {3}).",
                               obj_class_name.GetString(), method_checker.first,
                               payload.required_argument_count,
                               payload.actual_argument_count)));
        }
      } break;
      case AbstractMethodCheckerCases::eValid:
        LLDB_LOG(log, "Abstract method {0}.{1} implemented & valid.",
                 obj_class_name.GetString(), method_checker.first);
        break;
      }

    if (abstract_method_errors) {
      Status error = Status::FromError(std::move(abstract_method_errors));
      LLDB_LOG(log, "Abstract method error in {0}:\n{1}", class_name,
               error.AsCString());
      return error.ToError();
    }

    m_object_instance_sp = StructuredData::GenericSP(
        new StructuredPythonObject(std::move(result)));
    return m_object_instance_sp;
  }

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
      return llvm::is_contained(GetAbstractMethods(), method_name)
                 ? ErrorWithMessage<T>(caller_signature,
                                       "Python implementor not allocated.",
                                       error)
                 : T{};

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
      error = Status::FromError(std::move(e));
      return ErrorWithMessage<T>(caller_signature,
                                 "Python method could not be called.", error);
    }

    PythonObject py_return = std::move(expected_return_object.get());

    // Now that we called the python method with the transformed arguments,
    // we need to interate again over both the original and transformed
    // parameter pack, and transform back the parameter that were passed in
    // the original parameter pack as references or pointers.
    if (sizeof...(Args) > 0)
      if (!ReassignPtrsOrRefsArgs(original_args, transformed_args))
        return ErrorWithMessage<T>(
            caller_signature,
            "Couldn't re-assign reference and pointer arguments.", error);

    if (!py_return.IsAllocated())
      return {};
    return ExtractValueFromPythonObject<T>(py_return, error);
  }

  template <typename... Args>
  Status GetStatusFromMethod(llvm::StringRef method_name, Args &&...args) {
    Status error;
    Dispatch<Status>(method_name, error, std::forward<Args>(args)...);

    return error;
  }

  template <typename T> T Transform(T object) {
    // No Transformation for generic usage
    return {object};
  }

  python::PythonObject Transform(bool arg) {
    // Boolean arguments need to be turned into python objects.
    return python::PythonBoolean(arg);
  }

  python::PythonObject Transform(const Status &arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg.Clone());
  }

  python::PythonObject Transform(Status &&arg) {
    return python::SWIGBridge::ToSWIGWrapper(std::move(arg));
  }

  python::PythonObject Transform(const StructuredDataImpl &arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::ExecutionContextRefSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::TargetSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::BreakpointSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::BreakpointLocationSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::ProcessSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::ThreadPlanSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::ProcessAttachInfoSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::ProcessLaunchInfoSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(Event *arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(const SymbolContext &arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::StreamSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg.get());
  }

  python::PythonObject Transform(lldb::StackFrameSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::DataExtractorSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::DescriptionLevel arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  template <typename T, typename U>
  void ReverseTransform(T &original_arg, U transformed_arg, Status &error) {
    // If U is not a PythonObject, don't touch it!
  }

  template <typename T>
  void ReverseTransform(T &original_arg, python::PythonObject transformed_arg,
                        Status &error) {
    original_arg = ExtractValueFromPythonObject<T>(transformed_arg, error);
  }

  void ReverseTransform(bool &original_arg,
                        python::PythonObject transformed_arg, Status &error) {
    python::PythonBoolean boolean_arg = python::PythonBoolean(
        python::PyRefType::Borrowed, transformed_arg.get());
    if (boolean_arg.IsValid())
      original_arg = boolean_arg.GetValue();
    else
      error = Status::FromErrorStringWithFormatv(
          "{}: Invalid boolean argument.", LLVM_PRETTY_FUNCTION);
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
Event *ScriptedPythonInterface::ExtractValueFromPythonObject<Event *>(
    python::PythonObject &p, Status &error);

template <>
SymbolContext
ScriptedPythonInterface::ExtractValueFromPythonObject<SymbolContext>(
    python::PythonObject &p, Status &error);

template <>
lldb::StreamSP
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StreamSP>(
    python::PythonObject &p, Status &error);

template <>
lldb::StackFrameSP
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StackFrameSP>(
    python::PythonObject &p, Status &error);

template <>
lldb::BreakpointSP
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::BreakpointSP>(
    python::PythonObject &p, Status &error);

template <>
lldb::BreakpointLocationSP
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::BreakpointLocationSP>(python::PythonObject &p, Status &error);

template <>
lldb::ProcessAttachInfoSP ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ProcessAttachInfoSP>(python::PythonObject &p, Status &error);

template <>
lldb::ProcessLaunchInfoSP ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ProcessLaunchInfoSP>(python::PythonObject &p, Status &error);

template <>
lldb::DataExtractorSP
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DataExtractorSP>(
    python::PythonObject &p, Status &error);

template <>
std::optional<MemoryRegionInfo>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    std::optional<MemoryRegionInfo>>(python::PythonObject &p, Status &error);

template <>
lldb::ExecutionContextRefSP
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ExecutionContextRefSP>(python::PythonObject &p, Status &error);

template <>
lldb::DescriptionLevel
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DescriptionLevel>(
    python::PythonObject &p, Status &error);

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPYTHONINTERFACE_H
