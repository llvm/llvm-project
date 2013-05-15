//===-- ScriptInterpreterNone.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Interpreter/ScriptInterpreterNone.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;

ScriptInterpreterNone::ScriptInterpreterNone (CommandInterpreter &interpreter) :
    ScriptInterpreter (interpreter, eScriptLanguageNone)
{
}

ScriptInterpreterNone::~ScriptInterpreterNone ()
{
}

bool
ScriptInterpreterNone::ExecuteOneLine (const char *command, CommandReturnObject *, const ExecuteScriptOptions&)
{
    m_interpreter.GetDebugger().GetErrorStream().PutCString ("error: there is no embedded script interpreter in this mode.\n");
    return false;
}

void
ScriptInterpreterNone::ExecuteInterpreterLoop ()
{
    m_interpreter.GetDebugger().GetErrorStream().PutCString ("error: there is no embedded script interpreter in this mode.\n");
}


