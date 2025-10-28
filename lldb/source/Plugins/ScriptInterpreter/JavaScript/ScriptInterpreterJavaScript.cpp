//===-- ScriptInterpreterJavaScript.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptInterpreterJavaScript.h"
#include "JavaScript.h"

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/IOHandler.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StringList.h"
#include "lldb/Utility/Timer.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ScriptInterpreterJavaScript)

// IOHandler for JavaScript REPL
class IOHandlerJavaScriptInterpreter : public IOHandlerDelegate,
                                       public IOHandlerEditline {
public:
  IOHandlerJavaScriptInterpreter(
      Debugger &debugger, ScriptInterpreterJavaScript &script_interpreter)
      : IOHandlerEditline(debugger, IOHandler::Type::Other, "javascript",
                          "> ",              // Prompt
                          llvm::StringRef(), // No continuation prompt
                          false,             // Single-line for now
                          debugger.GetUseColor(), 0, *this),
        m_script_interpreter(script_interpreter) {
    llvm::cantFail(m_script_interpreter.EnterSession(debugger.GetID()));
  }

  ~IOHandlerJavaScriptInterpreter() override {
    llvm::cantFail(m_script_interpreter.LeaveSession());
  }

  void IOHandlerInputComplete(IOHandler &io_handler,
                              std::string &data) override {
    if (data == "quit" || data == "exit") {
      io_handler.SetIsDone(true);
      return;
    }

    // Execute the JavaScript code
    llvm::Error error = m_script_interpreter.GetJavaScript().Run(data);

    if (error) {
      // Print error
      if (LockableStreamFileSP error_sp = io_handler.GetErrorStreamFileSP()) {
        LockedStreamFile locked_stream = error_sp->Lock();
        locked_stream << "error: " << llvm::toString(std::move(error)) << "\n";
      }
    }
  }

private:
  ScriptInterpreterJavaScript &m_script_interpreter;
};

ScriptInterpreterJavaScript::ScriptInterpreterJavaScript(Debugger &debugger)
    : ScriptInterpreter(debugger, eScriptLanguageJavaScript),
      m_javascript(std::make_unique<JavaScript>(debugger.GetOutputFileSP())) {}

ScriptInterpreterJavaScript::~ScriptInterpreterJavaScript() = default;

bool ScriptInterpreterJavaScript::ExecuteOneLine(
    llvm::StringRef command, CommandReturnObject *result,
    const ExecuteScriptOptions &options) {
  if (command.empty()) {
    if (result)
      result->AppendError("Empty command string\n");
    return false;
  }

  // Set output callback to write console.log output to the result stream
  if (result) {
    m_javascript->SetOutputCallback([result](const std::string &text) {
      result->GetOutputStream().Printf("%s", text.c_str());
    });
  }

  llvm::Error error = m_javascript->Run(command);

  // Clear the callback after execution
  m_javascript->SetOutputCallback(nullptr);

  if (error) {
    if (result)
      result->AppendError(llvm::toString(std::move(error)));
    return false;
  }

  if (result)
    result->SetStatus(eReturnStatusSuccessFinishResult);
  return true;
}

void ScriptInterpreterJavaScript::ExecuteInterpreterLoop() {
  LLDB_SCOPED_TIMER();

  if (!m_debugger.GetInputFile().IsValid())
    return;

  IOHandlerSP io_handler_sp(
      new IOHandlerJavaScriptInterpreter(m_debugger, *this));
  m_debugger.RunIOHandlerAsync(io_handler_sp);
}

bool ScriptInterpreterJavaScript::LoadScriptingModule(
    const char *filename, const LoadScriptOptions &options,
    lldb_private::Status &error, StructuredData::ObjectSP *module_sp,
    FileSpec extra_search_dir, lldb::TargetSP loaded_into_target_sp) {

  if (!filename || filename[0] == '\0') {
    error = Status::FromErrorString("Empty filename");
    return false;
  }

  llvm::Error session_error = EnterSession(m_debugger.GetID());
  if (session_error) {
    error = Status::FromErrorString(
        llvm::toString(std::move(session_error)).c_str());
    return false;
  }

  llvm::Error load_error = m_javascript->LoadModule(filename);
  if (load_error) {
    error =
        Status::FromErrorString(llvm::toString(std::move(load_error)).c_str());
    return false;
  }

  return true;
}

StructuredData::DictionarySP ScriptInterpreterJavaScript::GetInterpreterInfo() {
  auto info_dict = std::make_shared<StructuredData::Dictionary>();
  info_dict->AddStringItem("language", "javascript");
  info_dict->AddStringItem("version", "ES2020+ (V8)");
  return info_dict;
}

void ScriptInterpreterJavaScript::Initialize() {
  static llvm::once_flag g_once_flag;
  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(
        GetPluginNameStatic(), GetPluginDescriptionStatic(),
        lldb::eScriptLanguageJavaScript, CreateInstance);
  });
}

void ScriptInterpreterJavaScript::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::ScriptInterpreterSP
ScriptInterpreterJavaScript::CreateInstance(Debugger &debugger) {
  return std::make_shared<ScriptInterpreterJavaScript>(debugger);
}

llvm::StringRef ScriptInterpreterJavaScript::GetPluginDescriptionStatic() {
  return "JavaScript script interpreter";
}

JavaScript &ScriptInterpreterJavaScript::GetJavaScript() {
  return *m_javascript;
}

llvm::Error
ScriptInterpreterJavaScript::EnterSession(lldb::user_id_t debugger_id) {
  if (m_session_is_active)
    return llvm::Error::success();

  m_javascript->SetDebugger(m_debugger.shared_from_this());
  m_session_is_active = true;
  return llvm::Error::success();
}

llvm::Error ScriptInterpreterJavaScript::LeaveSession() {
  if (!m_session_is_active)
    return llvm::Error::success();

  m_session_is_active = false;
  return llvm::Error::success();
}

void ScriptInterpreterJavaScript::CollectDataForBreakpointCommandCallback(
    std::vector<std::reference_wrapper<BreakpointOptions>> &bp_options_vec,
    CommandReturnObject &result) {
  result.AppendError("Breakpoint callbacks not yet implemented for JavaScript");
}

void ScriptInterpreterJavaScript::CollectDataForWatchpointCommandCallback(
    WatchpointOptions *wp_options, CommandReturnObject &result) {
  result.AppendError("Watchpoint callbacks not yet implemented for JavaScript");
}

Status ScriptInterpreterJavaScript::SetBreakpointCommandCallback(
    BreakpointOptions &bp_options, const char *command_body_text,
    bool is_callback) {
  return Status::FromErrorString(
      "Breakpoint callbacks not yet implemented for JavaScript");
}

void ScriptInterpreterJavaScript::SetWatchpointCommandCallback(
    WatchpointOptions *wp_options, const char *command_body_text,
    bool is_callback) {
  // TODO: Implement
}

Status ScriptInterpreterJavaScript::SetBreakpointCommandCallbackFunction(
    BreakpointOptions &bp_options, const char *function_name,
    StructuredData::ObjectSP extra_args_sp) {
  const char *fmt_str = "({0})";
  std::string oneliner = llvm::formatv(fmt_str, function_name).str();

  auto data_up = std::make_unique<CommandDataJavaScript>(extra_args_sp);

  llvm::Error err =
      m_javascript->RegisterBreakpointCallback(data_up.get(), oneliner.c_str());
  if (err)
    return Status::FromError(std::move(err));

  auto baton_sp =
      std::make_shared<BreakpointOptions::CommandBaton>(std::move(data_up));
  bp_options.SetCallback(
      ScriptInterpreterJavaScript::BreakpointCallbackFunction, baton_sp);

  return Status();
}

bool ScriptInterpreterJavaScript::BreakpointCallbackFunction(
    void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id,
    lldb::user_id_t break_loc_id) {

  ExecutionContext exe_ctx(context->exe_ctx_ref);
  Target *target = exe_ctx.GetTargetPtr();
  if (!target)
    return true;

  StackFrameSP stop_frame_sp(exe_ctx.GetFrameSP());
  BreakpointSP breakpoint_sp = target->GetBreakpointByID(break_id);
  BreakpointLocationSP bp_loc_sp(breakpoint_sp->FindLocationByID(break_loc_id));

  Debugger &debugger = target->GetDebugger();
  ScriptInterpreterJavaScript *js_interpreter =
      static_cast<ScriptInterpreterJavaScript *>(
          debugger.GetScriptInterpreter(true, eScriptLanguageJavaScript));
  JavaScript &js = js_interpreter->GetJavaScript();

  CommandDataJavaScript *bp_option_data =
      static_cast<CommandDataJavaScript *>(baton);
  llvm::Expected<bool> BoolOrErr =
      js.CallBreakpointCallback(baton, stop_frame_sp, bp_loc_sp,
                                bp_option_data->m_extra_args.GetObjectSP());
  if (llvm::Error E = BoolOrErr.takeError()) {
    llvm::consumeError(std::move(E));
    return true;
  }

  return *BoolOrErr;
}

bool ScriptInterpreterJavaScript::WatchpointCallbackFunction(
    void * /*baton*/, StoppointCallbackContext * /*context*/,
    lldb::user_id_t /*watch_id*/) {
  return false;
}
