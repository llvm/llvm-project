//===-- ScriptInterpreterLua.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ScriptInterpreterLua_h_
#define liblldb_ScriptInterpreterLua_h_

#include "lldb/Interpreter/ScriptInterpreter.h"

namespace lldb_private {
class Lua;
class ScriptInterpreterLua : public ScriptInterpreter {
public:
  ScriptInterpreterLua(Debugger &debugger);

  ~ScriptInterpreterLua() override;

  bool ExecuteOneLine(
      llvm::StringRef command, CommandReturnObject *result,
      const ExecuteScriptOptions &options = ExecuteScriptOptions()) override;

  void ExecuteInterpreterLoop() override;

  bool
  LoadScriptingModule(const char *filename, bool init_session,
                      lldb_private::Status &error,
                      StructuredData::ObjectSP *module_sp = nullptr) override;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static lldb::ScriptInterpreterSP CreateInstance(Debugger &debugger);

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  // PluginInterface protocol
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  Lua &GetLua();

  llvm::Error EnterSession(lldb::user_id_t debugger_id);
  llvm::Error LeaveSession();

private:
  std::unique_ptr<Lua> m_lua;
  bool m_session_is_active = false;
};

} // namespace lldb_private

#endif // liblldb_ScriptInterpreterLua_h_
