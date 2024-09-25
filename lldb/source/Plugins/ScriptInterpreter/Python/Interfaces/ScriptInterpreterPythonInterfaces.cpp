//===-- ScriptInterpreterPythonInterfaces.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/lldb-enumerations.h"

#if LLDB_ENABLE_PYTHON

#include "ScriptInterpreterPythonInterfaces.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ScriptInterpreterPythonInterfaces)

llvm::StringRef
ScriptInterpreterPythonInterfaces::GetPluginDescriptionStatic() {
  return "Script Interpreter Python Interfaces";
}

void ScriptInterpreterPythonInterfaces::Initialize() {
  OperatingSystemPythonInterface::Initialize();
  ScriptedPlatformPythonInterface::Initialize();
  ScriptedProcessPythonInterface::Initialize();
  ScriptedStopHookPythonInterface::Initialize();
  ScriptedThreadPlanPythonInterface::Initialize();
}

void ScriptInterpreterPythonInterfaces::Terminate() {
  OperatingSystemPythonInterface::Terminate();
  ScriptedPlatformPythonInterface::Terminate();
  ScriptedProcessPythonInterface::Terminate();
  ScriptedStopHookPythonInterface::Terminate();
  ScriptedThreadPlanPythonInterface::Terminate();
}

#endif
