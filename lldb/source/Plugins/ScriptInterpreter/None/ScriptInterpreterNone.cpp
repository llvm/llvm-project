//===-- ScriptInterpreterNone.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptInterpreterNone.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StringList.h"

#include "llvm/Support/Threading.h"

#include <mutex>

using namespace lldb;
using namespace lldb_private;

ScriptInterpreterNone::ScriptInterpreterNone(Debugger &debugger)
    : ScriptInterpreter(debugger, eScriptLanguageNone) {}

ScriptInterpreterNone::~ScriptInterpreterNone() {}

bool ScriptInterpreterNone::ExecuteOneLine(llvm::StringRef command,
                                           CommandReturnObject *,
                                           const ExecuteScriptOptions &) {
  m_debugger.GetErrorStream().PutCString(
      "error: there is no embedded script interpreter in this mode.\n");
  return false;
}

void ScriptInterpreterNone::ExecuteInterpreterLoop() {
  m_debugger.GetErrorStream().PutCString(
      "error: there is no embedded script interpreter in this mode.\n");
}

void ScriptInterpreterNone::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  lldb::eScriptLanguageNone, CreateInstance);
  });
}

void ScriptInterpreterNone::Terminate() {}

lldb::ScriptInterpreterSP
ScriptInterpreterNone::CreateInstance(Debugger &debugger) {
  return std::make_shared<ScriptInterpreterNone>(debugger);
}

lldb_private::ConstString ScriptInterpreterNone::GetPluginNameStatic() {
  static ConstString g_name("script-none");
  return g_name;
}

const char *ScriptInterpreterNone::GetPluginDescriptionStatic() {
  return "Null script interpreter";
}

lldb_private::ConstString ScriptInterpreterNone::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ScriptInterpreterNone::GetPluginVersion() { return 1; }
