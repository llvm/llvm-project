//===-- ScriptInterpreterNone.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_NONE_SCRIPTINTERPRETERNONE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_NONE_SCRIPTINTERPRETERNONE_H

#include "lldb/Interpreter/ScriptInterpreter.h"

namespace lldb_private {

/// Stands in for a script language without an interpreter, reporting why it
/// is unavailable.
class ScriptInterpreterNone : public ScriptInterpreter {
public:
  ScriptInterpreterNone(Debugger &debugger,
                        lldb::ScriptLanguage requested_language);

  ~ScriptInterpreterNone() override;

  bool ExecuteOneLine(
      llvm::StringRef command, CommandReturnObject *result,
      const ExecuteScriptOptions &options = ExecuteScriptOptions()) override;

  llvm::Error ExecuteInterpreterLoop() override;

  bool LoadScriptingModule(const char *filename,
                           const LoadScriptOptions &options,
                           lldb_private::Status &error,
                           StructuredData::ObjectSP *module_sp = nullptr,
                           FileSpec extra_search_dir = {},
                           lldb::TargetSP loaded_into_target_sp = {}) override;

  void CollectDataForBreakpointCommandCallback(
      std::vector<std::reference_wrapper<BreakpointOptions>> &options,
      CommandReturnObject &result) override;

  llvm::Error SetBreakpointCommandCallback(BreakpointOptions &bp_options,
                                           const char *callback_text,
                                           bool is_callback) override;

  llvm::Error SetBreakpointCommandCallbackFunction(
      BreakpointOptions &bp_options, const char *function_name,
      StructuredData::ObjectSP extra_args_sp) override;

  void
  CollectDataForWatchpointCommandCallback(WatchpointOptions *wp_options,
                                          CommandReturnObject &result) override;

  llvm::Error SetWatchpointCommandCallback(WatchpointOptions *wp_options,
                                           const char *user_input,
                                           bool is_callback) override;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static lldb::ScriptInterpreterSP
  CreateInstance(Debugger &debugger, lldb::ScriptLanguage requested_language);

  static llvm::StringRef GetPluginNameStatic() { return "script-none"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  /// Returns the error recorded for the requested language, or success.
  llvm::Error GetUnavailableError() const;

  lldb::ScriptLanguage m_requested_language;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_NONE_SCRIPTINTERPRETERNONE_H
