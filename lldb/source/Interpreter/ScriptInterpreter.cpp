//===-- ScriptInterpreter.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Interpreter/ScriptInterpreter.h"

#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#include "lldb/Utility/PseudoTerminal.h"

using namespace lldb;
using namespace lldb_private;

ScriptInterpreter::ScriptInterpreter (CommandInterpreter &interpreter, lldb::ScriptLanguage script_lang) :
    m_interpreter (interpreter),
    m_script_lang (script_lang)
{
}

ScriptInterpreter::~ScriptInterpreter ()
{
}

CommandInterpreter &
ScriptInterpreter::GetCommandInterpreter ()
{
    return m_interpreter;
}

void 
ScriptInterpreter::CollectDataForBreakpointCommandCallback 
(
    BreakpointOptions *bp_options,
    CommandReturnObject &result
)
{
    result.SetStatus (eReturnStatusFailed);
    result.AppendError ("ScriptInterpreter::GetScriptCommands(StringList &) is not implemented.");
}

void 
ScriptInterpreter::CollectDataForWatchpointCommandCallback 
(
    WatchpointOptions *bp_options,
    CommandReturnObject &result
)
{
    result.SetStatus (eReturnStatusFailed);
    result.AppendError ("ScriptInterpreter::GetScriptCommands(StringList &) is not implemented.");
}

std::string
ScriptInterpreter::LanguageToString (lldb::ScriptLanguage language)
{
    std::string return_value;

    switch (language)
    {
        case eScriptLanguageNone:
            return_value = "None";
            break;
        case eScriptLanguagePython:
            return_value = "Python";
            break;
    }

    return return_value;
}

std::unique_ptr<ScriptInterpreterLocker>
ScriptInterpreter::AcquireInterpreterLock ()
{
    return std::unique_ptr<ScriptInterpreterLocker>(new ScriptInterpreterLocker());
}

void
ScriptInterpreter::InitializeInterpreter (SWIGInitCallback python_swig_init_callback)
{
#ifndef LLDB_DISABLE_PYTHON
    ScriptInterpreterPython::InitializeInterpreter (python_swig_init_callback);
#endif // #ifndef LLDB_DISABLE_PYTHON
}

void
ScriptInterpreter::TerminateInterpreter ()
{
#ifndef LLDB_DISABLE_PYTHON
    ScriptInterpreterPython::TerminateInterpreter ();
#endif // #ifndef LLDB_DISABLE_PYTHON
}

