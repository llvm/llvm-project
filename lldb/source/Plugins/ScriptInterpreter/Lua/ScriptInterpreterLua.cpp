//===-- ScriptInterpreterLua.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptInterpreterLua.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StringList.h"

#include "llvm/Support/Threading.h"

#include <mutex>

using namespace lldb;
using namespace lldb_private;

ScriptInterpreterLua::ScriptInterpreterLua(Debugger &debugger)
    : ScriptInterpreter(debugger, eScriptLanguageLua) {}

ScriptInterpreterLua::~ScriptInterpreterLua() {}

bool ScriptInterpreterLua::ExecuteOneLine(llvm::StringRef command,
                                          CommandReturnObject *,
                                          const ExecuteScriptOptions &) {
  m_debugger.GetErrorStream().PutCString(
      "error: the lua script interpreter is not yet implemented.\n");
  return false;
}

void ScriptInterpreterLua::ExecuteInterpreterLoop() {
  m_debugger.GetErrorStream().PutCString(
      "error: the lua script interpreter is not yet implemented.\n");
}

void ScriptInterpreterLua::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  lldb::eScriptLanguageLua, CreateInstance);
  });
}

void ScriptInterpreterLua::Terminate() {}

lldb::ScriptInterpreterSP
ScriptInterpreterLua::CreateInstance(Debugger &debugger) {
  return std::make_shared<ScriptInterpreterLua>(debugger);
}

lldb_private::ConstString ScriptInterpreterLua::GetPluginNameStatic() {
  static ConstString g_name("script-lua");
  return g_name;
}

const char *ScriptInterpreterLua::GetPluginDescriptionStatic() {
  return "Lua script interpreter";
}

lldb_private::ConstString ScriptInterpreterLua::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ScriptInterpreterLua::GetPluginVersion() { return 1; }
