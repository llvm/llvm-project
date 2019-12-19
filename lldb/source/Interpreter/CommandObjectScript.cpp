//===-- CommandObjectScript.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectScript.h"
#include "lldb/Core/Debugger.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Host/Config.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/Args.h"

using namespace lldb;
using namespace lldb_private;

// CommandObjectScript

CommandObjectScript::CommandObjectScript(CommandInterpreter &interpreter,
                                         ScriptLanguage script_lang)
    : CommandObjectRaw(
          interpreter, "script",
          "Invoke the script interpreter with provided code and display any "
          "results.  Start the interactive interpreter if no code is supplied.",
          "script [<script-code>]") {}

CommandObjectScript::~CommandObjectScript() {}

bool CommandObjectScript::DoExecute(llvm::StringRef command,
                                    CommandReturnObject &result) {
  if (m_interpreter.GetDebugger().GetScriptLanguage() ==
      lldb::eScriptLanguageNone) {
    result.AppendError(
        "the script-lang setting is set to none - scripting not available");
    result.SetStatus(eReturnStatusFailed);
    return false;
  }

  ScriptInterpreter *script_interpreter = GetDebugger().GetScriptInterpreter();

  if (script_interpreter == nullptr) {
    result.AppendError("no script interpreter");
    result.SetStatus(eReturnStatusFailed);
    return false;
  }

  // Script might change Python code we use for formatting. Make sure we keep
  // up to date with it.
  DataVisualization::ForceUpdate();

  if (command.empty()) {
    script_interpreter->ExecuteInterpreterLoop();
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
  }

  // We can do better when reporting the status of one-liner script execution.
  if (script_interpreter->ExecuteOneLine(command, &result))
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
  else
    result.SetStatus(eReturnStatusFailed);

  return result.Succeeded();
}
