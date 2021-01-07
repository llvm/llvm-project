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
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"

namespace lldb_private {
class Lua;
class ScriptInterpreterLua : public ScriptInterpreter {
public:
  class CommandDataLua : public BreakpointOptions::CommandData {
  public:
    CommandDataLua() : BreakpointOptions::CommandData() {
      interpreter = lldb::eScriptLanguageLua;
    }
  };

  ScriptInterpreterLua(Debugger &debugger);

  ~ScriptInterpreterLua() override;

  bool ExecuteOneLine(
      llvm::StringRef command, CommandReturnObject *result,
      const ExecuteScriptOptions &options = ExecuteScriptOptions()) override;

  void ExecuteInterpreterLoop() override;

  bool LoadScriptingModule(const char *filename, bool init_session,
                           lldb_private::Status &error,
                           StructuredData::ObjectSP *module_sp = nullptr,
                           FileSpec extra_search_dir = {}) override;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static lldb::ScriptInterpreterSP CreateInstance(Debugger &debugger);

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static bool BreakpointCallbackFunction(void *baton,
                                         StoppointCallbackContext *context,
                                         lldb::user_id_t break_id,
                                         lldb::user_id_t break_loc_id);

  // PluginInterface protocol
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  Lua &GetLua();

  llvm::Error EnterSession(lldb::user_id_t debugger_id);
  llvm::Error LeaveSession();

  void CollectDataForBreakpointCommandCallback(
      std::vector<BreakpointOptions *> &bp_options_vec,
      CommandReturnObject &result) override;

  Status SetBreakpointCommandCallback(BreakpointOptions *bp_options,
                                      const char *command_body_text) override;

private:
  std::unique_ptr<Lua> m_lua;
  bool m_session_is_active = false;
};

} // namespace lldb_private

#endif // liblldb_ScriptInterpreterLua_h_
