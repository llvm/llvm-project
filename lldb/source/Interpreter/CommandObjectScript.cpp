//===-- CommandObjectScript.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectScript.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/DataVisualization.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/Args.h"

#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectScript
//-------------------------------------------------------------------------

CommandObjectScript::CommandObjectScript (CommandInterpreter &interpreter, ScriptLanguage script_lang) :
    CommandObjectRaw (interpreter, 
                      "script",
                      "Pass an expression to the script interpreter for evaluation and return the results. Drop into the interactive interpreter if no expression is given.",
                      "script [<script-expression-for-evaluation>]"),
    m_script_lang (script_lang)
{
}

CommandObjectScript::~CommandObjectScript ()
{
}

bool
CommandObjectScript::DoExecute
(
    const char *command,
    CommandReturnObject &result
)
{
    ScriptInterpreter *script_interpreter = m_interpreter.GetScriptInterpreter ();

    if (script_interpreter == NULL)
    {
        result.AppendError("no script interpreter");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    DataVisualization::ForceUpdate(); // script might change Python code we use for formatting.. make sure we keep up to date with it
    
    if (command == NULL || command[0] == '\0')
    {
        script_interpreter->ExecuteInterpreterLoop ();
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        return result.Succeeded();
    }

    // We can do better when reporting the status of one-liner script execution.
    if (script_interpreter->ExecuteOneLine (command, &result))
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
    else
        result.SetStatus(eReturnStatusFailed);

    return result.Succeeded();
}
