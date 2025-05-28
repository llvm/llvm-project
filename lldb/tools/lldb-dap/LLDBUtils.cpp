//===-- LLDBUtils.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBUtils.h"
#include "JSONUtils.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBThread.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <mutex>
#include <system_error>

namespace lldb_dap {

bool RunLLDBCommands(lldb::SBDebugger &debugger, llvm::StringRef prefix,
                     const llvm::ArrayRef<std::string> &commands,
                     llvm::raw_ostream &strm, bool parse_command_directives,
                     bool echo_commands) {
  if (commands.empty())
    return true;

  bool did_print_prefix = false;

  // We only need the prompt when echoing commands.
  std::string prompt_string;
  if (echo_commands) {
    prompt_string = "(lldb) ";

    // Get the current prompt from settings.
    if (const lldb::SBStructuredData prompt = debugger.GetSetting("prompt")) {
      const size_t prompt_length = prompt.GetStringValue(nullptr, 0);

      if (prompt_length != 0) {
        prompt_string.resize(prompt_length + 1);
        prompt.GetStringValue(prompt_string.data(), prompt_string.length());
      }
    }
  }

  lldb::SBCommandInterpreter interp = debugger.GetCommandInterpreter();
  for (llvm::StringRef command : commands) {
    lldb::SBCommandReturnObject result;
    bool quiet_on_success = false;
    bool check_error = false;

    while (parse_command_directives) {
      if (command.starts_with("?")) {
        command = command.drop_front();
        quiet_on_success = true;
      } else if (command.starts_with("!")) {
        command = command.drop_front();
        check_error = true;
      } else {
        break;
      }
    }

    {
      // Prevent simultaneous calls to HandleCommand, e.g. EventThreadFunction
      // may asynchronously call RunExitCommands when we are already calling
      // RunTerminateCommands.
      static std::mutex handle_command_mutex;
      std::lock_guard<std::mutex> locker(handle_command_mutex);
      interp.HandleCommand(command.str().c_str(), result,
                           /*add_to_history=*/true);
    }

    const bool got_error = !result.Succeeded();
    // The if statement below is assuming we always print out `!` prefixed
    // lines. The only time we don't print is when we have `quiet_on_success ==
    // true` and we don't have an error.
    if (quiet_on_success ? got_error : true) {
      if (!did_print_prefix && !prefix.empty()) {
        strm << prefix << "\n";
        did_print_prefix = true;
      }

      if (echo_commands)
        strm << prompt_string.c_str() << command << '\n';

      auto output_len = result.GetOutputSize();
      if (output_len) {
        const char *output = result.GetOutput();
        strm << output;
      }
      auto error_len = result.GetErrorSize();
      if (error_len) {
        const char *error = result.GetError();
        strm << error;
      }
    }
    if (check_error && got_error)
      return false; // Stop running commands.
  }
  return true;
}

std::string RunLLDBCommands(lldb::SBDebugger &debugger, llvm::StringRef prefix,
                            const llvm::ArrayRef<std::string> &commands,
                            bool &required_command_failed,
                            bool parse_command_directives, bool echo_commands) {
  required_command_failed = false;
  std::string s;
  llvm::raw_string_ostream strm(s);
  required_command_failed =
      !RunLLDBCommands(debugger, prefix, commands, strm,
                       parse_command_directives, echo_commands);
  return s;
}

bool ThreadHasStopReason(lldb::SBThread &thread) {
  switch (thread.GetStopReason()) {
  case lldb::eStopReasonTrace:
  case lldb::eStopReasonPlanComplete:
  case lldb::eStopReasonBreakpoint:
  case lldb::eStopReasonWatchpoint:
  case lldb::eStopReasonInstrumentation:
  case lldb::eStopReasonSignal:
  case lldb::eStopReasonException:
  case lldb::eStopReasonExec:
  case lldb::eStopReasonProcessorTrace:
  case lldb::eStopReasonFork:
  case lldb::eStopReasonVFork:
  case lldb::eStopReasonVForkDone:
  case lldb::eStopReasonInterrupt:
  case lldb::eStopReasonHistoryBoundary:
    return true;
  case lldb::eStopReasonThreadExiting:
  case lldb::eStopReasonInvalid:
  case lldb::eStopReasonNone:
    break;
  }
  return false;
}

static uint32_t constexpr THREAD_INDEX_SHIFT = 19;

uint32_t GetLLDBThreadIndexID(uint64_t dap_frame_id) {
  return dap_frame_id >> THREAD_INDEX_SHIFT;
}

uint32_t GetLLDBFrameID(uint64_t dap_frame_id) {
  return dap_frame_id & ((1u << THREAD_INDEX_SHIFT) - 1);
}

int64_t MakeDAPFrameID(lldb::SBFrame &frame) {
  return ((int64_t)frame.GetThread().GetIndexID() << THREAD_INDEX_SHIFT) |
         frame.GetFrameID();
}

lldb::SBEnvironment
GetEnvironmentFromArguments(const llvm::json::Object &arguments) {
  lldb::SBEnvironment envs{};
  constexpr llvm::StringRef env_key = "env";
  const llvm::json::Value *raw_json_env = arguments.get(env_key);

  if (!raw_json_env)
    return envs;

  if (raw_json_env->kind() == llvm::json::Value::Object) {
    auto env_map = GetStringMap(arguments, env_key);
    for (const auto &[key, value] : env_map)
      envs.Set(key.c_str(), value.c_str(), true);

  } else if (raw_json_env->kind() == llvm::json::Value::Array) {
    const auto envs_strings = GetStrings(&arguments, env_key);
    lldb::SBStringList entries{};
    for (const auto &env : envs_strings)
      entries.AppendString(env.c_str());

    envs.SetEntries(entries, true);
  }
  return envs;
}

lldb::StopDisassemblyType
GetStopDisassemblyDisplay(lldb::SBDebugger &debugger) {
  lldb::StopDisassemblyType result =
      lldb::StopDisassemblyType::eStopDisassemblyTypeNoDebugInfo;
  lldb::SBStructuredData string_result =
      debugger.GetSetting("stop-disassembly-display");
  const size_t result_length = string_result.GetStringValue(nullptr, 0);
  if (result_length > 0) {
    std::string result_string(result_length, '\0');
    string_result.GetStringValue(result_string.data(), result_length + 1);

    result =
        llvm::StringSwitch<lldb::StopDisassemblyType>(result_string)
            .Case("never", lldb::StopDisassemblyType::eStopDisassemblyTypeNever)
            .Case("always",
                  lldb::StopDisassemblyType::eStopDisassemblyTypeAlways)
            .Case("no-source",
                  lldb::StopDisassemblyType::eStopDisassemblyTypeNoSource)
            .Case("no-debuginfo",
                  lldb::StopDisassemblyType::eStopDisassemblyTypeNoDebugInfo)
            .Default(
                lldb::StopDisassemblyType::eStopDisassemblyTypeNoDebugInfo);
  }

  return result;
}

llvm::Error ToError(const lldb::SBError &error) {
  if (error.Success())
    return llvm::Error::success();

  return llvm::createStringError(
      std::error_code(error.GetError(), std::generic_category()),
      error.GetCString());
}

std::string GetStringValue(const lldb::SBStructuredData &data) {
  if (!data.IsValid())
    return "";

  const size_t str_length = data.GetStringValue(nullptr, 0);
  if (!str_length)
    return "";

  std::string str(str_length, 0);
  data.GetStringValue(str.data(), str_length + 1);
  return str;
}

ScopeSyncMode::ScopeSyncMode(lldb::SBDebugger &debugger)
    : m_debugger(debugger), m_async(m_debugger.GetAsync()) {
  m_debugger.SetAsync(false);
}

ScopeSyncMode::~ScopeSyncMode() { m_debugger.SetAsync(m_async); }

std::string GetSBFileSpecPath(const lldb::SBFileSpec &file_spec) {
  const auto directory_length = ::strlen(file_spec.GetDirectory());
  const auto file_name_length = ::strlen(file_spec.GetFilename());

  std::string path(directory_length + file_name_length + 1, '\0');
  file_spec.GetPath(path.data(), path.length() + 1);
  return path;
}

} // namespace lldb_dap
