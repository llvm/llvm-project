//===-- ScriptInterpreterPython.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTINTERPRETERPYTHON_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTINTERPRETERPYTHON_H

#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

#include "lldb/Breakpoint/BreakpointOptions.h"
#include "lldb/Core/IOHandler.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/lldb-private.h"

#include <memory>
#include <string>
#include <vector>

namespace lldb_private {
/// Abstract interface for the Python script interpreter.
class ScriptInterpreterPython : public ScriptInterpreter,
                                public IOHandlerDelegateMultiline {
public:
  class CommandDataPython : public BreakpointOptions::CommandData {
  public:
    CommandDataPython() : BreakpointOptions::CommandData() {
      interpreter = lldb::eScriptLanguagePython;
    }
  };

  ScriptInterpreterPython(Debugger &debugger)
      : ScriptInterpreter(debugger, lldb::eScriptLanguagePython),
        IOHandlerDelegateMultiline("DONE") {}

  static void Initialize();
  static void Terminate();
  static lldb_private::ConstString GetPluginNameStatic();
  static const char *GetPluginDescriptionStatic();
  static FileSpec GetPythonDir();

protected:
  static void ComputePythonDirForApple(llvm::SmallVectorImpl<char> &path);
  static void ComputePythonDirForPosix(llvm::SmallVectorImpl<char> &path);
  static void ComputePythonDirForWindows(llvm::SmallVectorImpl<char> &path);
};
} // namespace lldb_private

#endif // LLDB_DISABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTINTERPRETERPYTHON_H
