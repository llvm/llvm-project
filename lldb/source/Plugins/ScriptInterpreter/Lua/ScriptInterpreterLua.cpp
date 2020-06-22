//===-- ScriptInterpreterLua.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptInterpreterLua.h"
#include "Lua.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StringList.h"
#include "lldb/Utility/Timer.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ScriptInterpreterLua)

class IOHandlerLuaInterpreter : public IOHandlerDelegate,
                                public IOHandlerEditline {
public:
  IOHandlerLuaInterpreter(Debugger &debugger,
                          ScriptInterpreterLua &script_interpreter)
      : IOHandlerEditline(debugger, IOHandler::Type::LuaInterpreter, "lua",
                          ">>> ", "..> ", true, debugger.GetUseColor(), 0,
                          *this, nullptr),
        m_script_interpreter(script_interpreter) {
    llvm::cantFail(m_script_interpreter.EnterSession(debugger.GetID()));
  }

  ~IOHandlerLuaInterpreter() override {
    llvm::cantFail(m_script_interpreter.LeaveSession());
  }

  void IOHandlerInputComplete(IOHandler &io_handler,
                              std::string &data) override {
    if (llvm::Error error = m_script_interpreter.GetLua().Run(data)) {
      *GetOutputStreamFileSP() << llvm::toString(std::move(error));
    }
  }

private:
  ScriptInterpreterLua &m_script_interpreter;
};

ScriptInterpreterLua::ScriptInterpreterLua(Debugger &debugger)
    : ScriptInterpreter(debugger, eScriptLanguageLua),
      m_lua(std::make_unique<Lua>()) {}

ScriptInterpreterLua::~ScriptInterpreterLua() {}

bool ScriptInterpreterLua::ExecuteOneLine(llvm::StringRef command,
                                          CommandReturnObject *result,
                                          const ExecuteScriptOptions &options) {
  if (llvm::Error e = m_lua->Run(command)) {
    result->AppendErrorWithFormatv(
        "lua failed attempting to evaluate '{0}': {1}\n", command,
        llvm::toString(std::move(e)));
    return false;
  }
  return true;
}

void ScriptInterpreterLua::ExecuteInterpreterLoop() {
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, LLVM_PRETTY_FUNCTION);

  // At the moment, the only time the debugger does not have an input file
  // handle is when this is called directly from lua, in which case it is
  // both dangerous and unnecessary (not to mention confusing) to try to embed
  // a running interpreter loop inside the already running lua interpreter
  // loop, so we won't do it.
  if (!m_debugger.GetInputFile().IsValid())
    return;

  IOHandlerSP io_handler_sp(new IOHandlerLuaInterpreter(m_debugger, *this));
  m_debugger.RunIOHandlerAsync(io_handler_sp);
}

bool ScriptInterpreterLua::LoadScriptingModule(
    const char *filename, bool init_session, lldb_private::Status &error,
    StructuredData::ObjectSP *module_sp) {

  FileSystem::Instance().Collect(filename);
  if (llvm::Error e = m_lua->LoadModule(filename)) {
    error.SetErrorStringWithFormatv("lua failed to import '{0}': {1}\n",
                                    filename, llvm::toString(std::move(e)));
    return false;
  }
  return true;
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

llvm::Error ScriptInterpreterLua::EnterSession(user_id_t debugger_id) {
  if (m_session_is_active)
    return llvm::Error::success();

  const char *fmt_str =
      "lldb.debugger = lldb.SBDebugger.FindDebuggerWithID({0}); "
      "lldb.target = lldb.debugger:GetSelectedTarget(); "
      "lldb.process = lldb.target:GetProcess(); "
      "lldb.thread = lldb.process:GetSelectedThread(); "
      "lldb.frame = lldb.thread:GetSelectedFrame()";
  return m_lua->Run(llvm::formatv(fmt_str, debugger_id).str());
}

llvm::Error ScriptInterpreterLua::LeaveSession() {
  if (!m_session_is_active)
    return llvm::Error::success();

  m_session_is_active = false;

  llvm::StringRef str = "lldb.debugger = nil; "
                        "lldb.target = nil; "
                        "lldb.process = nil; "
                        "lldb.thread = nil; "
                        "lldb.frame = nil";
  return m_lua->Run(str);
}

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

Lua &ScriptInterpreterLua::GetLua() { return *m_lua; }
