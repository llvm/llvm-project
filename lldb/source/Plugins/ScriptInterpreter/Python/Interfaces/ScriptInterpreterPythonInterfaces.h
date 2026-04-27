//===-- ScriptInterpreterPythonInterfaces.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTINTERPRETERPYTHONINTERFACES_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTINTERPRETERPYTHONINTERFACES_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"

#include "OperatingSystemPythonInterface.h"
#include "ScriptedBreakpointPythonInterface.h"
#include "ScriptedFrameProviderPythonInterface.h"
#include "ScriptedFramePythonInterface.h"
#include "ScriptedPlatformPythonInterface.h"
#include "ScriptedProcessPythonInterface.h"
#include "ScriptedStopHookPythonInterface.h"
#include "ScriptedThreadPlanPythonInterface.h"

namespace lldb_private {
class ScriptInterpreterPythonInterfaces : public PluginInterface {
public:
  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() {
    return "script-interpreter-python-interfaces";
  }
  static llvm::StringRef GetPluginDescriptionStatic();
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTINTERPRETERPYTHONINTERFACES_H
