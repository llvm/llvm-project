//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/BreakpointResolverScripted.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

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
    const ScriptedMetadata &scripted_metadata, lldb::BreakpointSP break_sp) {
  return ScriptedPythonInterface::CreatePluginObject(
      scripted_metadata, nullptr, break_sp, scripted_metadata.GetArgsSP());
}

bool ScriptedBreakpointPythonInterface::OverridesResolver(
    Target &target, StructuredDataImpl &resolver_data) {
  TargetSP target_sp = target.shared_from_this();

  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("overrides_resolver", target_sp, resolver_data),
                    LLVM_PRETTY_FUNCTION);
  if (!obj)
    return false;

  return obj->GetBooleanValue();
}

void ScriptedBreakpointPythonInterface::SetBreakpoint(
    lldb::BreakpointSP break_sp) {
  LogAndDefault(Dispatch("set_breakpoint", break_sp), LLVM_PRETTY_FUNCTION);
}

bool ScriptedBreakpointPythonInterface::ResolverCallback(
    SymbolContext sym_ctx) {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("__callback__", sym_ctx), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return true;

  return obj->GetBooleanValue();
}

lldb::SearchDepth ScriptedBreakpointPythonInterface::GetDepth() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("__get_depth__"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return lldb::eSearchDepthModule;

  uint64_t value = obj->GetUnsignedIntegerValue();
  if (value <= lldb::kLastSearchDepthKind)
    return (lldb::SearchDepth)value;
  // This is what we were doing on error before, though I'm not sure that's
  // better than returning eSearchDepthInvalid.
  return lldb::eSearchDepthModule;
}

std::optional<std::string> ScriptedBreakpointPythonInterface::GetShortHelp() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_short_help"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

lldb::BreakpointLocationSP ScriptedBreakpointPythonInterface::WasHit(
    lldb::StackFrameSP frame_sp, lldb::BreakpointLocationSP bp_loc_sp) {
  llvm::Expected<lldb::BreakpointLocationSP> loc_or_err =
      Dispatch<lldb::BreakpointLocationSP>("was_hit", frame_sp, bp_loc_sp);
  if (!loc_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Script), loc_or_err.takeError(),
                   "Error calling was_hit method: {0}");
    return bp_loc_sp;
  }

  return *loc_or_err;
}

std::optional<std::string>
ScriptedBreakpointPythonInterface::GetLocationDescription(
    lldb::BreakpointLocationSP bp_loc_sp, lldb::DescriptionLevel level) {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_location_description", bp_loc_sp, level),
                    LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
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
      CreateInstance, eScriptedExtensionScriptedBreakpointResolver,
      eScriptLanguagePython, {ci_usages, api_usages});
}

void ScriptedBreakpointPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
