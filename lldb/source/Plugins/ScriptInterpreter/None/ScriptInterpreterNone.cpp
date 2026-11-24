//===-- ScriptInterpreterNone.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptInterpreterNone.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StringList.h"
#include "llvm/Support/ErrorExtras.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ScriptInterpreterNone)

ScriptInterpreterNone::ScriptInterpreterNone(
    Debugger &debugger, lldb::ScriptLanguage requested_language)
    : ScriptInterpreter(debugger, eScriptLanguageNone),
      m_requested_language(requested_language) {}

ScriptInterpreterNone::~ScriptInterpreterNone() = default;

static bool IsBuiltWith(lldb::ScriptLanguage language) {
  switch (language) {
  case eScriptLanguagePython:
    return LLDB_ENABLE_PYTHON;
  case eScriptLanguageLua:
    return LLDB_ENABLE_LUA;
  default:
    return false;
  }
}

llvm::Error ScriptInterpreterNone::GetUnavailableError() const {
  llvm::Error error =
      PluginManager::GetScriptInterpreterError(m_requested_language);
  // A plugin library that failed to load leaves no error behind.
  if (!error && IsBuiltWith(m_requested_language))
    error = llvm::createStringError("its plugin was not loaded");
  if (error)
    return llvm::createStringErrorV("{0} scripting is unavailable: {1}",
                                    LanguageToString(m_requested_language),
                                    llvm::toString(std::move(error)));
  return llvm::Error::success();
}

static llvm::Error CreateNoScriptingError() {
  return llvm::createStringError(
      "Embedded script interpreter unavailable. LLDB was built without "
      "scripting language support.");
}

bool ScriptInterpreterNone::ExecuteOneLine(llvm::StringRef command,
                                           CommandReturnObject *result,
                                           const ExecuteScriptOptions &) {
  llvm::Error error = GetUnavailableError();
  if (!error)
    error = CreateNoScriptingError();
  if (result)
    result->AppendError(llvm::toString(std::move(error)));
  else
    m_debugger.GetAsyncErrorStream()->Printf(
        "error: %s\n", llvm::toString(std::move(error)).c_str());
  return false;
}

llvm::Error ScriptInterpreterNone::ExecuteInterpreterLoop() {
  if (llvm::Error error = GetUnavailableError())
    return error;
  return CreateNoScriptingError();
}

bool ScriptInterpreterNone::LoadScriptingModule(
    const char *filename, const LoadScriptOptions &options, Status &error,
    StructuredData::ObjectSP *module_sp, FileSpec extra_search_dir,
    lldb::TargetSP loaded_into_target_sp) {
  if (llvm::Error unavailable = GetUnavailableError()) {
    error = Status::FromError(std::move(unavailable));
    return false;
  }
  return ScriptInterpreter::LoadScriptingModule(filename, options, error,
                                                module_sp, extra_search_dir,
                                                loaded_into_target_sp);
}

void ScriptInterpreterNone::CollectDataForBreakpointCommandCallback(
    std::vector<std::reference_wrapper<BreakpointOptions>> &options,
    CommandReturnObject &result) {
  if (llvm::Error unavailable = GetUnavailableError())
    result.AppendError(llvm::toString(std::move(unavailable)));
  else
    ScriptInterpreter::CollectDataForBreakpointCommandCallback(options, result);
}

void ScriptInterpreterNone::CollectDataForWatchpointCommandCallback(
    WatchpointOptions *wp_options, CommandReturnObject &result) {
  if (llvm::Error unavailable = GetUnavailableError())
    result.AppendError(llvm::toString(std::move(unavailable)));
  else
    ScriptInterpreter::CollectDataForWatchpointCommandCallback(wp_options,
                                                               result);
}

llvm::Error ScriptInterpreterNone::SetBreakpointCommandCallback(
    BreakpointOptions &bp_options, const char *callback_text,
    bool is_callback) {
  if (llvm::Error unavailable = GetUnavailableError())
    return unavailable;
  return ScriptInterpreter::SetBreakpointCommandCallback(
      bp_options, callback_text, is_callback);
}

llvm::Error ScriptInterpreterNone::SetBreakpointCommandCallbackFunction(
    BreakpointOptions &bp_options, const char *function_name,
    StructuredData::ObjectSP extra_args_sp) {
  if (llvm::Error unavailable = GetUnavailableError())
    return unavailable;
  return ScriptInterpreter::SetBreakpointCommandCallbackFunction(
      bp_options, function_name, extra_args_sp);
}

llvm::Error ScriptInterpreterNone::SetWatchpointCommandCallback(
    WatchpointOptions *wp_options, const char *user_input, bool is_callback) {
  if (llvm::Error unavailable = GetUnavailableError())
    return unavailable;
  return ScriptInterpreter::SetWatchpointCommandCallback(wp_options, user_input,
                                                         is_callback);
}

void ScriptInterpreterNone::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(),
                                lldb::eScriptLanguageNone, CreateInstance);
}

void ScriptInterpreterNone::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::ScriptInterpreterSP
ScriptInterpreterNone::CreateInstance(Debugger &debugger,
                                      lldb::ScriptLanguage requested_language) {
  return std::make_shared<ScriptInterpreterNone>(debugger, requested_language);
}

llvm::StringRef ScriptInterpreterNone::GetPluginDescriptionStatic() {
  return "Null script interpreter";
}
