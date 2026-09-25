//===-- ScriptedPythonInterface.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPYTHONINTERFACE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPYTHONINTERFACE_H

#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/Interpreter/Interfaces/ScriptedInterface.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Policy.h"

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

  struct AbstractMethodCheckerPayload {

    struct InvalidArgumentCountPayload {
      InvalidArgumentCountPayload(size_t required, size_t actual)
          : required_argument_count(required), actual_argument_count(actual) {}

      size_t required_argument_count;
      size_t actual_argument_count;
    };

    AbstractMethodCheckerCases checker_case;
    std::variant<std::monostate, InvalidArgumentCountPayload, std::string>
        payload;
  };

  llvm::Expected<FileSpec> GetScriptedModulePath() override {
    using namespace python;
    using Locker = ScriptInterpreterPythonImpl::Locker;

    Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);

    if (!m_object_instance_sp)
      return llvm::createStringError("scripted Interface has invalid object");

    PythonObject py_obj =
        PythonObject(PyRefType::Borrowed,
                     static_cast<PyObject *>(m_object_instance_sp->GetValue()));

    if (!py_obj.IsAllocated())
      return llvm::createStringError(
          "scripted Interface has invalid python object");

    PythonObject py_obj_class = py_obj.GetAttributeValue("__class__");
    if (!py_obj_class.IsValid())
      return llvm::createStringError(
          "scripted Interface python object is missing '__class__' attribute");

    PythonObject py_obj_module = py_obj_class.GetAttributeValue("__module__");
    if (!py_obj_module.IsValid())
      return llvm::createStringError(
          "scripted Interface python object '__class__' is missing "
          "'__module__' attribute");

    PythonString py_obj_module_str = py_obj_module.Str();
    if (!py_obj_module_str.IsValid())
      return llvm::createStringError(
          "scripted Interface python object '__class__.__module__' attribute "
          "is not a string");

    llvm::StringRef py_obj_module_str_ref = py_obj_module_str.GetString();
    PythonModule py_module = PythonModule::AddModule(py_obj_module_str_ref);
    if (!py_module.IsValid())
      return llvm::createStringError("failed to import '%s' module",
                                     py_obj_module_str_ref.data());

    PythonObject py_module_file = py_module.GetAttributeValue("__file__");
    if (!py_module_file.IsValid())
      return llvm::createStringError(
          "module '%s' is missing '__file__' attribute",
          py_obj_module_str_ref.data());

    PythonString py_module_file_str = py_module_file.Str();
    if (!py_module_file_str.IsValid())
      return llvm::createStringError(
          "module '%s.__file__' attribute is not a string",
          py_obj_module_str_ref.data());

    return FileSpec(py_module_file_str.GetString());
  }

  llvm::Expected<std::map<llvm::StringLiteral, AbstractMethodCheckerPayload>>
  CheckAbstractMethodImplementation(
      const python::PythonObject &obj_class) const {

    using namespace python;

    std::map<llvm::StringLiteral, AbstractMethodCheckerPayload> checker;
#define SET_CASE_AND_CONTINUE(method_name, case)                               \
  {                                                                            \
    checker[method_name] = {case, {}};                                         \
    continue;                                                                  \
  }

    for (const AbstractMethodRequirement &requirement :
         GetAbstractMethodRequirements()) {
      llvm::StringLiteral method_name = requirement.name;
      // Look up via attribute access so inherited methods are found; the
      // class's own __dict__ omits anything defined on a base class.
      if (!obj_class.HasAttribute(method_name))
        SET_CASE_AND_CONTINUE(method_name,
                              AbstractMethodCheckerCases::eNotImplemented)
      PythonObject attr = obj_class.GetAttributeValue(method_name);
      if (!attr.IsAllocated())
        SET_CASE_AND_CONTINUE(method_name,
                              AbstractMethodCheckerCases::eNotAllocated)

      PythonCallable callable = attr.AsType<PythonCallable>();
      if (!callable)
        SET_CASE_AND_CONTINUE(method_name,
                              AbstractMethodCheckerCases::eNotCallable)

      if (!requirement.min_arg_count)
        SET_CASE_AND_CONTINUE(method_name, AbstractMethodCheckerCases::eValid)

      auto arg_info_or_err = callable.GetArgInfo();
      if (!arg_info_or_err) {
        checker[method_name] = {
            AbstractMethodCheckerCases::eUnknownArgumentCount,
            ExtractPythonError(arg_info_or_err.takeError())};
        continue;
      }

      PythonCallable::ArgInfo arg_info = *arg_info_or_err;
      if (requirement.min_arg_count <= arg_info.max_positional_args) {
        SET_CASE_AND_CONTINUE(method_name, AbstractMethodCheckerCases::eValid)
      } else {
        checker[method_name] = {
            AbstractMethodCheckerCases::eInvalidArgumentCount,
            AbstractMethodCheckerPayload::InvalidArgumentCountPayload(
                requirement.min_arg_count, arg_info.max_positional_args)};
      }
    }

#undef SET_CASE_AND_CONTINUE

    return checker;
  }

  /// Diagnose every abstract-method violation on \a obj_class at once.
  ///
  /// \a obj_class is a class object, whether resolved by name or taken from an
  /// instance's `__class__`. Resolving it by name means this can run before any
  /// instance exists (see CreatePluginObject), which uses it to reject a
  /// malformed class without executing its `__init__`.
  llvm::Error CheckAbstractMethods(const python::PythonObject &obj_class,
                                   llvm::StringRef qualified_class_name) const {
    Log *log = GetLog(LLDBLog::Script);

    // Per-method diagnostics name the class the way Python does, unqualified.
    python::PythonString obj_class_name =
        obj_class.GetAttributeValue("__name__").AsType<python::PythonString>();
    llvm::StringRef class_name = obj_class_name.IsValid()
                                     ? obj_class_name.GetString()
                                     : qualified_class_name;
    auto create_error = [](llvm::StringLiteral format, auto &&...ts) {
      return llvm::createStringError(
          llvm::formatv(format.data(), std::forward<decltype(ts)>(ts)...)
              .str());
    };

    auto checker_or_err = CheckAbstractMethodImplementation(obj_class);
    if (!checker_or_err)
      return checker_or_err.takeError();

    llvm::Error abstract_method_errors = llvm::Error::success();
    for (const auto &method_checker : *checker_or_err)
      switch (method_checker.second.checker_case) {
      case AbstractMethodCheckerCases::eNotImplemented:
        abstract_method_errors = llvm::joinErrors(
            std::move(abstract_method_errors),
            create_error("abstract method {0}.{1} not implemented", class_name,
                         method_checker.first));
        break;
      case AbstractMethodCheckerCases::eNotAllocated:
        abstract_method_errors = llvm::joinErrors(
            std::move(abstract_method_errors),
            create_error("abstract method {0}.{1} not allocated", class_name,
                         method_checker.first));
        break;
      case AbstractMethodCheckerCases::eNotCallable:
        abstract_method_errors = llvm::joinErrors(
            std::move(abstract_method_errors),
            create_error("abstract method {0}.{1} not callable", class_name,
                         method_checker.first));
        break;
      case AbstractMethodCheckerCases::eUnknownArgumentCount: {
        const std::string *py_error =
            std::get_if<std::string>(&method_checker.second.payload);
        abstract_method_errors = llvm::joinErrors(
            std::move(abstract_method_errors),
            create_error(
                "abstract method {0}.{1} has unknown argument count: {2}",
                class_name, method_checker.first,
                py_error ? *py_error : "<no further information>"));
      } break;
      case AbstractMethodCheckerCases::eInvalidArgumentCount: {
        auto &payload_variant = method_checker.second.payload;
        if (!std::holds_alternative<
                AbstractMethodCheckerPayload::InvalidArgumentCountPayload>(
                payload_variant)) {
          abstract_method_errors = llvm::joinErrors(
              std::move(abstract_method_errors),
              create_error(
                  "abstract method {0}.{1} has unexpected argument count",
                  class_name, method_checker.first));
        } else {
          auto payload = std::get<
              AbstractMethodCheckerPayload::InvalidArgumentCountPayload>(
              payload_variant);
          abstract_method_errors = llvm::joinErrors(
              std::move(abstract_method_errors),
              create_error("abstract method {0}.{1} has unexpected "
                           "argument count (expected {2} but has {3})",
                           class_name, method_checker.first,
                           payload.required_argument_count,
                           payload.actual_argument_count));
        }
      } break;
      case AbstractMethodCheckerCases::eValid:
        LLDB_LOG(log, "Abstract method {0}.{1} implemented & valid.",
                 class_name, method_checker.first);
        break;
      }

    if (abstract_method_errors) {
      Status error = Status::FromError(std::move(abstract_method_errors));
      LLDB_LOG(log, "Abstract method error in {0}:\n{1}", qualified_class_name,
               error.AsCString());
      return error.ToError();
    }

    return llvm::Error::success();
  }

  template <typename... Args>
  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(const ScriptedMetadata &scripted_metadata,
                     StructuredData::Generic *script_obj, Args... args) {
    using namespace python;
    using Locker = ScriptInterpreterPythonImpl::Locker;

    auto create_error = [](llvm::StringLiteral format, auto &&...ts) {
      return llvm::createStringError(
          llvm::formatv(format.data(), std::forward<decltype(ts)>(ts)...)
              .str());
    };

    m_scripted_metadata = scripted_metadata;
    llvm::StringRef class_name = scripted_metadata.GetClassName();
    bool has_class_name = !class_name.empty();
    bool has_interpreter_dict =
        !(llvm::StringRef(m_interpreter.GetDictionaryName()).empty());
    if (!has_class_name && !has_interpreter_dict && !script_obj) {
      if (!has_class_name)
        return create_error("missing script class name");
      else if (!has_interpreter_dict)
        return create_error("invalid script interpreter dictionary");
      else
        return create_error("missing scripting object");
    }

    std::optional<PolicyStack::Guard> policy_guard;
    if (!UserCanRunDirectly())
      policy_guard = PolicyStack::Get().PushScriptedExtensionCall();

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
        return create_error("could not find interpreter dictionary: {0}",
                            m_interpreter.GetDictionaryName());

      auto init =
          PythonObject::ResolveNameWithDictionary<python::PythonCallable>(
              class_name, dict);
      if (!init.IsAllocated())
        return create_error("could not find script class: {0}",
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

      if (llvm::Error error = CheckAbstractMethods(init, class_name))
        return std::move(error);

      llvm::Expected<PythonObject> expected_return_object =
          create_error("resulting object is not initialized");

      // This relax the requirement on the number of argument for
      // initializing scripting extension if the size of the interface
      // parameter pack contains 1 less element than the extension maximum
      // number of positional arguments for this initializer.
      //
      // This addresses the cases where the embedded interpreter session
      // dictionary is passed to the extension initializer which is not used
      // most of the time.
      // Note, though none of our API's suggest defining the interfaces with
      // varargs, we have some extant clients that were doing that.  To keep
      // from breaking them, we just say putting a varargs in these signatures
      // turns off argument checking.
      size_t num_args = sizeof...(Args);
      if (arg_info->max_positional_args != PythonCallable::ArgInfo::UNBOUNDED &&
          num_args != arg_info->max_positional_args) {
        if (num_args != arg_info->max_positional_args - 1) {
          // `expected_return_object` starts in an error state; consume it
          // before we return with a different error, or its destructor
          // will abort.
          llvm::consumeError(expected_return_object.takeError());
          return create_error("passed arguments ({0}) doesn't match the number "
                              "of expected arguments ({1})",
                              num_args, arg_info->max_positional_args);
        }

        std::apply(
            [&init, &expected_return_object](auto &&...args) {
              if (!expected_return_object)
                llvm::consumeError(expected_return_object.takeError());
              expected_return_object = init.Call(args...);
            },
            std::tuple_cat(transformed_args, std::make_tuple(dict)));
      } else {
        std::apply(
            [&init, &expected_return_object](auto &&...args) {
              if (!expected_return_object)
                llvm::consumeError(expected_return_object.takeError());
              expected_return_object = init.Call(args...);
            },
            transformed_args);
      }

      if (!expected_return_object)
        // Drain the Python exception into a plain string while the GIL is
        // still held: `PythonException` owns raw `PyObject*` references, and
        // `py_lock` (and the GIL it holds) is released as this function
        // returns, before the caller gets a chance to touch the error.
        return llvm::createStringError(
            ExtractPythonError(expected_return_object.takeError()));
      result = expected_return_object.get();
    }

    if (!result.IsValid())
      return create_error("resulting object is not a valid Python Object");
    if (!result.HasAttribute("__class__"))
      return create_error("resulting object doesn't have '__class__' member");

    PythonObject obj_class = result.GetAttributeValue("__class__");
    if (!obj_class.IsValid())
      return create_error("resulting class object is not a valid");
    if (!obj_class.HasAttribute("__name__"))
      return create_error(
          "resulting object class doesn't have '__name__' member");
    PythonString obj_class_name =
        obj_class.GetAttributeValue("__name__").AsType<PythonString>();

    // We were handed an instance rather than building one, so there was no
    // constructor to run the check ahead of; validate it now.
    if (script_obj)
      if (llvm::Error error =
              CheckAbstractMethods(obj_class, obj_class_name.GetString()))
        return std::move(error);

    m_object_instance_sp = StructuredData::GenericSP(
        new StructuredPythonObject(std::move(result)));
    return m_object_instance_sp;
  }

  /// Call a static method on a Python class without creating an instance.
  ///
  /// This method resolves a Python class by name and calls a static method
  /// on it, returning the result. This is useful for calling class-level
  /// methods that don't require an instance.
  ///
  /// \param class_name The fully-qualified name of the Python class.
  /// \param method_name The name of the static method to call.
  /// \param args Arguments to pass to the static method.
  ///
  /// \return The return value of the static method call, or an error.
  template <typename T = StructuredData::ObjectSP, typename... Args>
  llvm::Expected<T> CallStaticMethod(llvm::StringRef class_name,
                                     llvm::StringRef method_name,
                                     Args &&...args) {
    using namespace python;
    using Locker = ScriptInterpreterPythonImpl::Locker;

    std::string caller_signature =
        llvm::Twine(LLVM_PRETTY_FUNCTION + llvm::Twine(" (") +
                    llvm::Twine(class_name) + llvm::Twine(".") +
                    llvm::Twine(method_name) + llvm::Twine(")"))
            .str();

    if (class_name.empty())
      return LogAndError(caller_signature, "missing script class name");

    std::optional<PolicyStack::Guard> policy_guard;
    if (!UserCanRunDirectly())
      policy_guard = PolicyStack::Get().PushScriptedExtensionCall();

    Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);

    // Get the interpreter dictionary.
    auto dict =
        PythonModule::MainModule().ResolveName<python::PythonDictionary>(
            m_interpreter.GetDictionaryName());
    if (!dict.IsAllocated())
      return LogAndError(caller_signature,
                         "could not find interpreter dictionary: {0}",
                         m_interpreter.GetDictionaryName());

    // Resolve the class.
    auto class_obj =
        PythonObject::ResolveNameWithDictionary<python::PythonCallable>(
            class_name, dict);
    if (!class_obj.IsAllocated())
      return LogAndError(caller_signature, "could not find script class: {0}",
                         class_name);

    // Get the static method from the class.
    if (!class_obj.HasAttribute(method_name))
      return llvm::make_error<UnimplementedError>(
          llvm::formatv("{0}.{1}", class_name, method_name).str());

    PythonCallable method =
        class_obj.GetAttributeValue(method_name).AsType<PythonCallable>();
    if (!method.IsAllocated())
      return LogAndError(caller_signature, "method {0}.{1} is not callable",
                         class_name, method_name);

    // Transform the arguments.
    std::tuple<Args...> original_args = std::forward_as_tuple(args...);
    auto transformed_args = TransformArgs(original_args);

    // Call the static method.
    llvm::Expected<PythonObject> expected_return_object =
        llvm::createStringError("not initialized");
    std::apply(
        [&method, &expected_return_object](auto &&...args) {
          if (!expected_return_object)
            llvm::consumeError(expected_return_object.takeError());
          expected_return_object = method.Call(args...);
        },
        transformed_args);

    if (llvm::Error e = expected_return_object.takeError()) {
      // TODO: Stringify `args` and include them in the message so users
      // can see what was passed to the failing call (e.g.
      // `read_memory_at_address(0x500000000, 4)`). Requires a SFINAE
      // helper that falls back to a placeholder for types without a
      // format_provider / operator<<.
      return LogAndError(
          caller_signature, "python exception in {0} method '{1}': {2}",
          class_name, method_name, ExtractPythonError(std::move(e)));
    }

    PythonObject py_return = std::move(expected_return_object.get());

    // Re-assign reference and pointer arguments if needed.
    if (sizeof...(Args) > 0)
      if (!ReassignPtrsOrRefsArgs(original_args, transformed_args))
        return LogAndError(
            caller_signature,
            "couldn't re-assign reference and pointer arguments");

    // Extract value from Python object (handles unallocated case).
    if (!py_return.IsAllocated())
      return T{};
    return ExtractValueFromPythonObject<T>(py_return);
  }

protected:
  /// Extract detailed error message including Python backtrace if available.
  ///
  /// This helper processes llvm::Error objects that may contain PythonException
  /// instances, extracting full Python backtraces when available.
  ///
  /// \param error The llvm::Error to extract information from.
  /// \return A string containing the error message, including full Python
  ///         backtrace if the error was a PythonException.
  static std::string ExtractPythonError(llvm::Error error) {
    std::string error_msg;
    llvm::handleAllErrors(
        std::move(error),
        [&](python::PythonException &E) { error_msg = E.ReadBacktrace(); },
        [&](const llvm::ErrorInfoBase &E) { error_msg = E.message(); });
    return error_msg;
  }

  /// Log \a message against \a caller_name and return it as an error.
  ///
  /// The `Expected`-returning counterpart of
  /// `ScriptedInterface::ErrorWithMessage`: it reports the failure instead of
  /// folding it into a default-constructed value.
  template <typename... Ts>
  static llvm::Error LogAndError(llvm::StringRef caller_name,
                                 const char *format, Ts &&...ts) {
    std::string message = llvm::formatv(format, std::forward<Ts>(ts)...).str();
    LLDB_LOGF(GetLog(LLDBLog::Script), "%s ERROR = %s", caller_name.data(),
              message.c_str());
    return llvm::createStringError(message);
  }

  /// Log the failure in \a value_or_err and fall back to a default-constructed
  /// \c T.
  ///
  /// For entry points whose return type cannot express failure. Prefer
  /// propagating the error wherever the signature can carry it; this is the
  /// stop-gap, and it is at least strictly better than dropping the error.
  ///
  /// Logs any failure, an unimplemented method included: a site that dispatches
  /// with Dispatch() rather than DispatchToOptional() is asserting the method
  /// should have been there.
  template <typename T>
  static T LogAndDefault(llvm::Expected<T> value_or_err,
                         llvm::StringRef caller) {
    if (value_or_err)
      return std::move(*value_or_err);

    // Consume the error before logging: LLDB_LOGF doesn't evaluate its
    // arguments when the channel is disabled, which would leave the error
    // unchecked and abort.
    std::string message = llvm::toString(value_or_err.takeError());
    LLDB_LOGF(GetLog(LLDBLog::Script), "%s failed: %s", caller.str().c_str(),
              message.c_str());
    return T{};
  }

  /// Call an optional \a method_name, reporting "the script doesn't implement
  /// it" as \c std::nullopt rather than as a failure.
  ///
  /// Use this for callbacks a script may legitimately leave out. Any other
  /// failure - in particular an exception raised inside the method -
  /// propagates; this never papers over a method that ran and failed.
  template <typename T = StructuredData::ObjectSP, typename... Args>
  llvm::Expected<std::optional<T>>
  DispatchToOptional(llvm::StringRef method_name, Args &&...args) {
    if (!llvm::is_contained(GetOptionalMethods(), method_name))
      return LogAndError(
          LLVM_PRETTY_FUNCTION,
          "method '{0}' is not declared optional: list it in "
          "GetOptionalMethods(), or dispatch it with Dispatch(), "
          "where a missing method is an error",
          method_name);

    llvm::Expected<T> value_or_err =
        Dispatch<T>(method_name, std::forward<Args>(args)...);
    if (value_or_err)
      return std::move(*value_or_err);

    if (value_or_err.template errorIsA<UnimplementedError>()) {
      llvm::consumeError(value_or_err.takeError());
      return std::nullopt;
    }
    return value_or_err.takeError();
  }

  template <typename T = StructuredData::ObjectSP>
  llvm::Expected<T> ExtractValueFromPythonObject(python::PythonObject &p) {
    return p.CreateStructuredObject();
  }

  /// Call \a method_name on the scripted object.
  ///
  /// The returned \c Expected separates the two outcomes that a plain return
  /// value cannot: a successfully extracted value (which may legitimately be
  /// empty or null, e.g. when the Python method returns \c None) and a
  /// failure to call the method at all.
  ///
  /// A class that simply doesn't implement \a method_name fails with an
  /// \c UnimplementedError, so callers of optional callbacks can recognize
  /// and ignore that case without also swallowing real exceptions.
  template <typename T = StructuredData::ObjectSP, typename... Args>
  llvm::Expected<T> Dispatch(llvm::StringRef method_name, Args &&...args) {
    using namespace python;
    using Locker = ScriptInterpreterPythonImpl::Locker;

    std::string caller_signature =
        llvm::Twine(LLVM_PRETTY_FUNCTION + llvm::Twine(" (") +
                    llvm::Twine(method_name) + llvm::Twine(")"))
            .str();
    if (!m_object_instance_sp)
      return LogAndError(caller_signature, "python object ill-formed");

    std::optional<PolicyStack::Guard> policy_guard;
    if (!UserCanRunDirectly())
      policy_guard = PolicyStack::Get().PushScriptedExtensionCall();

    Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);

    PythonObject implementor(PyRefType::Borrowed,
                             (PyObject *)m_object_instance_sp->GetValue());

    if (!implementor.IsAllocated())
      return LogAndError(caller_signature, "python implementor not allocated");

    PythonObject py_method = implementor.GetAttributeValue(method_name);
    if (!py_method.IsAllocated())
      return llvm::make_error<UnimplementedError>(
          llvm::formatv("{0}.{1}",
                        GetScriptedMetadata()
                            ? GetScriptedMetadata()->GetClassName()
                            : "<unknown>",
                        method_name)
              .str());

    std::tuple<Args...> original_args = std::forward_as_tuple(args...);
    auto transformed_args = TransformArgs(original_args);

    // Trim trailing args if the Python method accepts fewer positional
    // parameters than we're passing (e.g. `num_children(self)` vs.
    // `num_children(self, max_count)`).
    size_t call_arity = sizeof...(Args);
    if (PythonCallable callable = py_method.AsType<PythonCallable>();
        callable.IsAllocated()) {
      if (llvm::Expected<PythonCallable::ArgInfo> arg_info =
              callable.GetArgInfo()) {
        if (arg_info->max_positional_args !=
                PythonCallable::ArgInfo::UNBOUNDED &&
            arg_info->max_positional_args < call_arity)
          call_arity = arg_info->max_positional_args;
      } else {
        llvm::consumeError(arg_info.takeError());
      }
    }

    llvm::Expected<PythonObject> expected_return_object =
        llvm::createStringError("not initialized");
    CallWithArity(call_arity, transformed_args,
                  std::make_index_sequence<sizeof...(Args) + 1>{},
                  [&implementor, &method_name,
                   &expected_return_object](auto &&...call_args) {
                    if (!expected_return_object)
                      llvm::consumeError(expected_return_object.takeError());
                    expected_return_object = implementor.CallMethod(
                        method_name.data(), call_args...);
                  });

    if (llvm::Error e = expected_return_object.takeError()) {
      // TODO: Stringify `args` and include them in the message so users
      // can see what was passed to the failing call (e.g.
      // `read_memory_at_address(0x500000000, 4)`). Requires a SFINAE
      // helper that falls back to a placeholder for types without a
      // format_provider / operator<<.
      //
      // Drain the Python exception into a plain string while the GIL is
      // still held: `PythonException` owns raw `PyObject*` references, and
      // `py_lock` is released as this function returns.
      return LogAndError(
          caller_signature, "python exception in {0} method '{1}': {2}",
          GetScriptedMetadata() ? GetScriptedMetadata()->GetClassName()
                                : "<unknown>",
          method_name, ExtractPythonError(std::move(e)));
    }

    PythonObject py_return = std::move(expected_return_object.get());

    // Now that we called the python method with the transformed arguments,
    // we need to iterate again over both the original and transformed
    // parameter pack, and transform back the parameter that were passed in
    // the original parameter pack as references or pointers.
    if (sizeof...(Args) > 0)
      if (!ReassignPtrsOrRefsArgs(original_args, transformed_args))
        return LogAndError(
            caller_signature,
            "couldn't re-assign reference and pointer arguments");

    if (!py_return.IsAllocated())
      return T{};
    return ExtractValueFromPythonObject<T>(py_return);
  }

  /// Call \a method_name and fold both failure channels into one `Status`:
  /// a failure to call the method at all, and the `SBError` the method
  /// returned. The latter used to be dropped on the floor.
  template <typename... Args>
  Status GetStatusFromMethod(llvm::StringRef method_name, Args &&...args) {
    llvm::Expected<Status> status_or_err =
        Dispatch<Status>(method_name, std::forward<Args>(args)...);
    if (!status_or_err)
      return Status::FromError(status_or_err.takeError());

    return std::move(*status_or_err);
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

  template <typename T, typename = std::enable_if_t<
                            std::is_base_of_v<StructuredData::Object, T>>>
  python::PythonObject Transform(std::shared_ptr<T> arg) {
    return Transform(StructuredDataImpl(arg));
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

  python::PythonObject Transform(lldb::ThreadSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::StackFrameListSP arg) {
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

  python::PythonObject Transform(lldb::StepType arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::ValueObjectSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(lldb::DebuggerSP arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  python::PythonObject Transform(const std::vector<std::string> &arg) {
    python::PythonList list(python::PyInitialValue::Empty);
    for (const std::string &s : arg)
      list.AppendItem(python::PythonString(s));
    return list;
  }

  python::ScopedPythonObject<lldb::SBCommandReturnObject>
  Transform(CommandReturnObject *arg) {
    return python::SWIGBridge::ToSWIGWrapper(*arg);
  }

  python::PythonObject Transform(const TypeSummaryOptions &arg) {
    return python::SWIGBridge::ToSWIGWrapper(arg);
  }

  template <typename T, typename U>
  void ReverseTransform(T &original_arg, U transformed_arg, Status &error) {
    // If U is not a PythonObject, don't touch it!
  }

  template <typename T>
  void ReverseTransform(T &original_arg, python::PythonObject transformed_arg,
                        Status &error) {
    llvm::Expected<T> value_or_err =
        ExtractValueFromPythonObject<T>(transformed_arg);
    if (!value_or_err) {
      error = Status::FromError(value_or_err.takeError());
      return;
    }
    original_arg = std::move(*value_or_err);
  }

  // Read-only arguments (passed as `const T&`) have nothing to write back:
  // there's no `T` value to reassign into a const reference, and no
  // `ExtractValueFromPythonObject<T>` specialization should be required just
  // to satisfy this round-trip for a value the callee never mutates.
  template <typename T>
  void ReverseTransform(const T &original_arg,
                        python::PythonObject transformed_arg, Status &error) {}

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

  // Apply `fn` with the first `N` elements of `t`, for compile-time `N`.
  template <std::size_t N, typename Tuple, typename Fn, std::size_t... I>
  static void ApplyPrefixImpl(Tuple &&t, Fn &&fn, std::index_sequence<I...>) {
    std::forward<Fn>(fn)(std::get<I>(std::forward<Tuple>(t))...);
  }

  template <std::size_t N, typename Tuple, typename Fn>
  static void ApplyPrefix(Tuple &&t, Fn &&fn) {
    ApplyPrefixImpl<N>(std::forward<Tuple>(t), std::forward<Fn>(fn),
                       std::make_index_sequence<N>{});
  }

  // Call `fn` with a runtime-selected prefix of `t`: exactly `call_arity`
  // leading elements. `Is...` enumerates every compile-time count in
  // `[0, sizeof...(Args)]`; the runtime check picks the matching one.
  template <typename Tuple, std::size_t... Is, typename Fn>
  static void CallWithArity(size_t call_arity, Tuple &&t,
                            std::index_sequence<Is...>, Fn &&fn) {
    (void)std::initializer_list<int>{(
        Is == call_arity
            ? (ApplyPrefix<Is>(std::forward<Tuple>(t), std::forward<Fn>(fn)), 0)
            : 0)...};
  }

  template <typename T, typename U>
  void TransformBack(T &original_arg, U transformed_arg, Status &error) {
    ReverseTransform(original_arg, transformed_arg, error);
  }

  // ScopedPythonObject is non-copyable — passing it through the generic
  // TransformBack would trigger the deleted copy ctor. It manages its own
  // cleanup via the destructor when the transformed-args tuple destructs, so
  // there is nothing to reverse-transform back into the original arg.
  template <typename T, typename SB>
  void TransformBack(T &original_arg,
                     python::ScopedPythonObject<SB> &transformed_arg,
                     Status &error) {}

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
llvm::Expected<StructuredData::ArraySP>
ScriptedPythonInterface::ExtractValueFromPythonObject<StructuredData::ArraySP>(
    python::PythonObject &p);

template <>
llvm::Expected<StructuredData::DictionarySP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    StructuredData::DictionarySP>(python::PythonObject &p);

template <>
llvm::Expected<Status>
ScriptedPythonInterface::ExtractValueFromPythonObject<Status>(
    python::PythonObject &p);

template <>
llvm::Expected<Event *>
ScriptedPythonInterface::ExtractValueFromPythonObject<Event *>(
    python::PythonObject &p);

template <>
llvm::Expected<SymbolContext>
ScriptedPythonInterface::ExtractValueFromPythonObject<SymbolContext>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::StreamSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StreamSP>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::ThreadSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::ThreadSP>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::StackFrameSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StackFrameSP>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::BreakpointSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::BreakpointSP>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::BreakpointLocationSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::BreakpointLocationSP>(python::PythonObject &p);

template <>
llvm::Expected<lldb::ProcessAttachInfoSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ProcessAttachInfoSP>(python::PythonObject &p);

template <>
llvm::Expected<lldb::ProcessLaunchInfoSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ProcessLaunchInfoSP>(python::PythonObject &p);

template <>
llvm::Expected<lldb::DataExtractorSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DataExtractorSP>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::ThreadPlanSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::ThreadPlanSP>(
    python::PythonObject &p);

template <>
llvm::Expected<std::optional<MemoryRegionInfo>>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    std::optional<MemoryRegionInfo>>(python::PythonObject &p);

template <>
llvm::Expected<lldb::ExecutionContextRefSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    lldb::ExecutionContextRefSP>(python::PythonObject &p);

template <>
llvm::Expected<lldb::DescriptionLevel>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DescriptionLevel>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::StepType>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StepType>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::StackFrameListSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::StackFrameListSP>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::ValueObjectSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::ValueObjectSP>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::TargetSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::TargetSP>(
    python::PythonObject &p);

template <>
llvm::Expected<lldb::ValueObjectListSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::ValueObjectListSP>(
    python::PythonObject &p);

template <>
llvm::Expected<std::optional<lldb::ValueType>>
ScriptedPythonInterface::ExtractValueFromPythonObject<
    std::optional<lldb::ValueType>>(python::PythonObject &p);

template <>
llvm::Expected<lldb::DebuggerSP>
ScriptedPythonInterface::ExtractValueFromPythonObject<lldb::DebuggerSP>(
    python::PythonObject &p);

template <>
llvm::Expected<std::vector<std::string>>
ScriptedPythonInterface::ExtractValueFromPythonObject<std::vector<std::string>>(
    python::PythonObject &p);

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPYTHONINTERFACE_H
