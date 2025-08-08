//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

// LLDB Python header must be included first
#include "../lldb-python.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedBreakpointPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;

ScriptedBreakpointPythonInterface::ScriptedBreakpointPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedBreakpointInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedBreakpointPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, lldb::BreakpointSP break_sp,
    const StructuredDataImpl &args_sp) {
  return ScriptedPythonInterface::CreatePluginObject(class_name, nullptr,
                                                     break_sp, args_sp);
}

bool ScriptedBreakpointPythonInterface::ResolverCallback(
    SymbolContext sym_ctx) {
  Status error;

  StructuredData::ObjectSP obj = Dispatch("__callback__", error, sym_ctx);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error)) {
    Log *log = GetLog(LLDBLog::Script);
    LLDB_LOG(log, "Error calling __callback__ method: {1}", error);
    return true;
  }
  return obj->GetBooleanValue();
}

lldb::SearchDepth ScriptedBreakpointPythonInterface::GetDepth() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("__get_depth__", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error)) {
    return lldb::eSearchDepthModule;
  }
  uint64_t value = obj->GetUnsignedIntegerValue();
  if (value <= lldb::kLastSearchDepthKind)
    return (lldb::SearchDepth)value;
  // This is what we were doing on error before, though I'm not sure that's
  // better than returning eSearchDepthInvalid.
  return lldb::eSearchDepthModule;
}

std::optional<std::string> ScriptedBreakpointPythonInterface::GetShortHelp() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_short_help", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error)) {
    return {};
  }

  return obj->GetAsString()->GetValue().str();
}

void ScriptedBreakpointPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "breakpoint set -P classname [-k key -v value ...]"};
  const std::vector<llvm::StringRef> api_usages = {
      "SBTarget.BreakpointCreateFromScript"};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      llvm::StringRef("Create a breakpoint that chooses locations based on "
                      "user-created callbacks"),
      CreateInstance, eScriptLanguagePython, {ci_usages, api_usages});
}

void ScriptedBreakpointPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

#endif
