//===-- JITLoaderPython.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "JITLoaderPython.h"

#include "Plugins/Process/Utility/RegisterContextDummy.h"
#include "Plugins/Process/Utility/RegisterContextMemory.h"
#include "Plugins/Process/Utility/ThreadMemory.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/ValueObject/ValueObjectVariable.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(JITLoaderPython)

void JITLoaderPython::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                nullptr);
}

void JITLoaderPython::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

JITLoaderSP JITLoaderPython::CreateInstance(Process *process, bool force) {
  // Python JITLoader plug-ins must be requested by name, so force must
  // be true
  FileSpec python_os_plugin_spec(process->GetPythonJITLoaderPath());
  if (python_os_plugin_spec &&
      FileSystem::Instance().Exists(python_os_plugin_spec)) {
    std::unique_ptr<JITLoaderPython> os_up(
        new JITLoaderPython(process, python_os_plugin_spec));
    if (os_up.get() && os_up->IsValid())
      return os_up;
  }
  return nullptr;
}

llvm::StringRef JITLoaderPython::GetPluginDescriptionStatic() {
  return "JIT loader plug-in that implements a JIT loader using a python "
         "class that implements the necessary JITLoader functionality.";
}

JITLoaderPython::JITLoaderPython(lldb_private::Process *process,
                                 const FileSpec &python_module_path)
    : JITLoader(process), m_interpreter(nullptr), m_script_object_sp() {
  if (!process)
    return;
  TargetSP target_sp = process->CalculateTarget();
  if (!target_sp)
    return;
  m_interpreter = target_sp->GetDebugger().GetScriptInterpreter();
  if (!m_interpreter)
    return;

  std::string os_plugin_class_name(
      python_module_path.GetFilename().AsCString(""));
  if (os_plugin_class_name.empty())
    return;

  LoadScriptOptions options;
  char python_module_path_cstr[PATH_MAX];
  python_module_path.GetPath(python_module_path_cstr,
                             sizeof(python_module_path_cstr));
  Status error;
  if (!m_interpreter->LoadScriptingModule(python_module_path_cstr, options,
                                          error))
    return;

  // Strip the ".py" extension if there is one
  size_t py_extension_pos = os_plugin_class_name.rfind(".py");
  if (py_extension_pos != std::string::npos)
    os_plugin_class_name.erase(py_extension_pos);
  // Add ".JITLoaderPlugin" to the module name to get a string like
  // "modulename.JITLoaderPlugin"
  os_plugin_class_name += ".JITLoaderPlugin";

  JITLoaderInterfaceSP interface_sp =
      m_interpreter->CreateJITLoaderInterface();
  if (!interface_sp)
    return;

  ExecutionContext exe_ctx(process);
  auto obj_or_err = interface_sp->CreatePluginObject(
      os_plugin_class_name, exe_ctx);

  if (!obj_or_err) {
    llvm::consumeError(obj_or_err.takeError());
    return;
  }

  StructuredData::GenericSP owned_script_object_sp = *obj_or_err;
  if (!owned_script_object_sp->IsValid())
    return;

  m_script_object_sp = owned_script_object_sp;
  m_interface_sp = interface_sp;
}

JITLoaderPython::~JITLoaderPython() = default;

void JITLoaderPython::DidAttach() {
  if (m_interface_sp)
    m_interface_sp->DidAttach();
}

void JITLoaderPython::DidLaunch() {
  if (m_interface_sp)
    m_interface_sp->DidLaunch();
}

void JITLoaderPython::ModulesDidLoad(lldb_private::ModuleList &module_list) {
  if (m_interface_sp)
    m_interface_sp->ModulesDidLoad(module_list);
}

#endif // #if LLDB_ENABLE_PYTHON
