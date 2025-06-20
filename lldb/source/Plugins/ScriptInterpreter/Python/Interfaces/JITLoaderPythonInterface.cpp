//===-- JITLoaderPythonInterface.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// clang-format off
// LLDB Python header must be included first
#include "../lldb-python.h"
//clang-format on

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "JITLoaderPythonInterface.h"

using namespace lldb;
using namespace lldb_private;

JITLoaderPythonInterface::JITLoaderPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
JITLoaderPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, ExecutionContext &exe_ctx) {
  return ScriptedPythonInterface::CreatePluginObject(class_name, nullptr,
                                                     exe_ctx.GetProcessSP());
}

void JITLoaderPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "settings set target.process.python-jit-loader-plugin-path <script-path>"};
  const std::vector<llvm::StringRef> api_usages = {};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), llvm::StringRef("JIT loader python plugin"),
      CreateInstance, eScriptLanguagePython, {ci_usages, api_usages});
}

void JITLoaderPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

//------------------------------------------------------------------------------
// JITLoader API overrides
//------------------------------------------------------------------------------

void JITLoaderPythonInterface::DidAttach() {
  Status error;
  Dispatch("did_attach", error);
}

void JITLoaderPythonInterface::DidLaunch() {
  Status error;
  Dispatch("did_launch", error);
}

void JITLoaderPythonInterface::ModulesDidLoad(ModuleList &module_list) {
  Status error;
  // There is no SBModuleList, so we need to deliver each module individually 
  // to the python scripts since it uses the LLDB public API.
  module_list.ForEach([&](const lldb::ModuleSP &module_sp_ref) {
    lldb::ModuleSP module_sp(module_sp_ref);
    Dispatch("module_did_load", error, module_sp);
    return true; // Keep iterating  
  });
}

#endif
