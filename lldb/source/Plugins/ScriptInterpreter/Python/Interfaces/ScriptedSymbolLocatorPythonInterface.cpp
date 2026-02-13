//===-- ScriptedSymbolLocatorPythonInterface.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// clang-format off
// LLDB Python header must be included first
#include "../lldb-python.h"
// clang-format on

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedSymbolLocatorPythonInterface.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBModuleSpec.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedSymbolLocatorPythonInterface::ScriptedSymbolLocatorPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedSymbolLocatorInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedSymbolLocatorPythonInterface::CreatePluginObject(
    const llvm::StringRef class_name, ExecutionContext &exe_ctx,
    StructuredData::DictionarySP args_sp, StructuredData::Generic *script_obj) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  StructuredDataImpl sd_impl(args_sp);
  return ScriptedPythonInterface::CreatePluginObject(class_name, script_obj,
                                                     exe_ctx_ref_sp, sd_impl);
}

/// Helper to convert an internal ModuleSpec to a Python SBModuleSpec object.
/// Must be called with the GIL held.
static PythonObject ToSWIGModuleSpec(const ModuleSpec &module_spec) {
  // Build an SBModuleSpec using public API setters since the constructor
  // from ModuleSpec is private.
  SBModuleSpec sb_module_spec;

  const UUID &uuid = module_spec.GetUUID();
  if (uuid.IsValid())
    sb_module_spec.SetUUIDBytes(uuid.GetBytes().data(),
                                uuid.GetBytes().size());

  const FileSpec &file = module_spec.GetFileSpec();
  if (file)
    sb_module_spec.SetFileSpec(SBFileSpec(file.GetPath().c_str(), false));

  const FileSpec &platform_file = module_spec.GetPlatformFileSpec();
  if (platform_file)
    sb_module_spec.SetPlatformFileSpec(
        SBFileSpec(platform_file.GetPath().c_str(), false));

  const FileSpec &symbol_file = module_spec.GetSymbolFileSpec();
  if (symbol_file)
    sb_module_spec.SetSymbolFileSpec(
        SBFileSpec(symbol_file.GetPath().c_str(), false));

  const ArchSpec &arch = module_spec.GetArchitecture();
  if (arch.IsValid())
    sb_module_spec.SetTriple(arch.GetTriple().getTriple().c_str());

  ConstString object_name = module_spec.GetObjectName();
  if (object_name)
    sb_module_spec.SetObjectName(object_name.GetCString());

  sb_module_spec.SetObjectOffset(module_spec.GetObjectOffset());
  sb_module_spec.SetObjectSize(module_spec.GetObjectSize());

  return SWIGBridge::ToSWIGWrapper(
      std::make_unique<SBModuleSpec>(sb_module_spec));
}

/// Helper to convert an internal FileSpec to a Python SBFileSpec object.
/// Must be called with the GIL held.
static PythonObject ToSWIGFileSpec(const FileSpec &file_spec) {
  return SWIGBridge::ToSWIGWrapper(
      std::make_unique<SBFileSpec>(file_spec.GetPath().c_str(), false));
}

std::optional<ModuleSpec>
ScriptedSymbolLocatorPythonInterface::LocateExecutableObjectFile(
    const ModuleSpec &module_spec, Status &error) {
  if (!m_object_instance_sp)
    return {};

  // Acquire the GIL before creating any Python objects.
  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)m_object_instance_sp->GetValue());
  if (!implementor.IsAllocated())
    return {};

  PythonObject py_module_spec = ToSWIGModuleSpec(module_spec);

  auto expected = implementor.CallMethod("locate_executable_object_file",
                                         py_module_spec);
  if (!expected) {
    // Consume the PythonException while the GIL is held. Converting to string
    // forces PythonException destruction before the GIL is released.
    std::string msg = llvm::toString(expected.takeError());
    error = Status(msg);
    return {};
  }

  PythonObject py_return = std::move(*expected);
  if (!py_return.IsAllocated() || py_return.IsNone())
    return {};

  auto obj = py_return.CreateStructuredObject();
  if (!obj)
    return {};

  llvm::StringRef value = obj->GetStringValue();
  if (value.empty())
    return {};

  ModuleSpec result_spec(module_spec);
  result_spec.GetFileSpec().SetPath(value);
  return result_spec;
}

std::optional<FileSpec>
ScriptedSymbolLocatorPythonInterface::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec,
    const FileSpecList &default_search_paths, Status &error) {
  if (!m_object_instance_sp)
    return {};

  // Acquire the GIL before creating any Python objects.
  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)m_object_instance_sp->GetValue());
  if (!implementor.IsAllocated())
    return {};

  PythonObject py_module_spec = ToSWIGModuleSpec(module_spec);

  // Convert FileSpecList to a Python list of SBFileSpec.
  PythonList py_paths(default_search_paths.GetSize());
  for (size_t i = 0; i < default_search_paths.GetSize(); i++) {
    py_paths.SetItemAtIndex(
        i, ToSWIGFileSpec(default_search_paths.GetFileSpecAtIndex(i)));
  }

  auto expected = implementor.CallMethod("locate_executable_symbol_file",
                                         py_module_spec, py_paths);
  if (!expected) {
    std::string msg = llvm::toString(expected.takeError());
    error = Status(msg);
    return {};
  }

  PythonObject py_return = std::move(*expected);
  if (!py_return.IsAllocated() || py_return.IsNone())
    return {};

  auto obj = py_return.CreateStructuredObject();
  if (!obj)
    return {};

  llvm::StringRef value = obj->GetStringValue();
  if (value.empty())
    return {};

  FileSpec file_spec;
  file_spec.SetPath(value);
  return file_spec;
}

bool ScriptedSymbolLocatorPythonInterface::DownloadObjectAndSymbolFile(
    ModuleSpec &module_spec, Status &error, bool force_lookup,
    bool copy_executable) {
  if (!m_object_instance_sp)
    return false;

  // Acquire the GIL before creating any Python objects.
  Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                 Locker::FreeLock);

  PythonObject implementor(PyRefType::Borrowed,
                           (PyObject *)m_object_instance_sp->GetValue());
  if (!implementor.IsAllocated())
    return false;

  PythonObject py_module_spec = ToSWIGModuleSpec(module_spec);

  auto expected = implementor.CallMethod("download_object_and_symbol_file",
                                         py_module_spec,
                                         PythonBoolean(force_lookup),
                                         PythonBoolean(copy_executable));
  if (!expected) {
    std::string msg = llvm::toString(expected.takeError());
    error = Status(msg);
    return false;
  }

  PythonObject py_return = std::move(*expected);
  if (!py_return.IsAllocated() || py_return.IsNone())
    return false;

  auto obj = py_return.CreateStructuredObject();
  if (!obj)
    return false;

  return obj->GetBooleanValue();
}

std::optional<FileSpec>
ScriptedSymbolLocatorPythonInterface::LocateSourceFile(
    const lldb::ModuleSP &module_sp, const FileSpec &original_source_file,
    Status &error) {
  if (!m_object_instance_sp)
    return {};

  std::optional<FileSpec> result;

  {
    // Acquire the GIL before creating any Python objects. All Python objects
    // (including error objects) must be destroyed within this scope.
    Locker py_lock(&m_interpreter, Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);

    PythonObject implementor(PyRefType::Borrowed,
                             (PyObject *)m_object_instance_sp->GetValue());
    if (!implementor.IsAllocated())
      return {};

    PythonObject py_module = SWIGBridge::ToSWIGWrapper(module_sp);
    std::string source_path = original_source_file.GetPath();
    PythonString py_source_path(source_path);

    auto expected = implementor.CallMethod("locate_source_file", py_module,
                                           py_source_path);
    if (!expected) {
      // Consume the error (which may contain PythonException) while the GIL
      // is still held. Convert to string to force PythonException destruction
      // before the GIL is released.
      std::string msg = llvm::toString(expected.takeError());
      error = Status(msg);
      return {};
    }

    PythonObject py_return = std::move(*expected);
    if (py_return.IsAllocated() && !py_return.IsNone()) {
      auto obj = py_return.CreateStructuredObject();
      if (obj) {
        llvm::StringRef value = obj->GetStringValue();
        if (!value.empty()) {
          FileSpec file_spec;
          file_spec.SetPath(value);
          result = file_spec;
        }
      }
    }
  } // GIL released here, after all Python objects are destroyed.

  return result;
}

#endif // LLDB_ENABLE_PYTHON
