//===-- ScriptInterpreter.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/ScriptInterpreter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StringList.h"

using namespace lldb;
using namespace lldb_private;

ScriptInterpreter::ScriptInterpreter(Debugger &debugger,
                                     lldb::ScriptLanguage script_lang)
    : m_debugger(debugger), m_script_lang(script_lang) {}

ScriptInterpreter::~ScriptInterpreter() {}

void ScriptInterpreter::CollectDataForBreakpointCommandCallback(
    std::vector<BreakpointOptions *> &bp_options_vec,
    CommandReturnObject &result) {
  result.SetStatus(eReturnStatusFailed);
  result.AppendError(
      "This script interpreter does not support breakpoint callbacks.");
}

void ScriptInterpreter::CollectDataForWatchpointCommandCallback(
    WatchpointOptions *bp_options, CommandReturnObject &result) {
  result.SetStatus(eReturnStatusFailed);
  result.AppendError(
      "This script interpreter does not support watchpoint callbacks.");
}

bool ScriptInterpreter::LoadScriptingModule(
    const char *filename, bool init_session, lldb_private::Status &error,
    StructuredData::ObjectSP *module_sp) {
  error.SetErrorString(
      "This script interpreter does not support importing modules.");
  return false;
}

std::string ScriptInterpreter::LanguageToString(lldb::ScriptLanguage language) {
  switch (language) {
  case eScriptLanguageNone:
    return "None";
  case eScriptLanguagePython:
    return "Python";
  case eScriptLanguageLua:
    return "Lua";
  case eScriptLanguageUnknown:
    return "Unknown";
  }
  llvm_unreachable("Unhandled ScriptInterpreter!");
}

lldb::ScriptLanguage
ScriptInterpreter::StringToLanguage(const llvm::StringRef &language) {
  if (language.equals_lower(LanguageToString(eScriptLanguageNone)))
    return eScriptLanguageNone;
  if (language.equals_lower(LanguageToString(eScriptLanguagePython)))
    return eScriptLanguagePython;
  if (language.equals_lower(LanguageToString(eScriptLanguageLua)))
    return eScriptLanguageLua;
  return eScriptLanguageUnknown;
}

Status ScriptInterpreter::SetBreakpointCommandCallback(
    std::vector<BreakpointOptions *> &bp_options_vec,
    const char *callback_text) {
  Status return_error;
  for (BreakpointOptions *bp_options : bp_options_vec) {
    return_error = SetBreakpointCommandCallback(bp_options, callback_text);
    if (return_error.Success())
      break;
  }
  return return_error;
}

Status ScriptInterpreter::SetBreakpointCommandCallbackFunction(
    std::vector<BreakpointOptions *> &bp_options_vec, const char *function_name,
    StructuredData::ObjectSP extra_args_sp) {
  Status error;
  for (BreakpointOptions *bp_options : bp_options_vec) {
    error = SetBreakpointCommandCallbackFunction(bp_options, function_name,
                                                 extra_args_sp);
    if (!error.Success())
      return error;
  }
  return error;
}

std::unique_ptr<ScriptInterpreterLocker>
ScriptInterpreter::AcquireInterpreterLock() {
  return std::make_unique<ScriptInterpreterLocker>();
}
