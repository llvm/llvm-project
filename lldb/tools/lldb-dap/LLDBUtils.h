//===-- LLDBUtils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_LLDBUTILS_H
#define LLDB_TOOLS_LLDB_DAP_LLDBUTILS_H

#include "DAPForward.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEnvironment.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFileSpec.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <string>

namespace lldb_dap {

/// Run a list of LLDB commands in the LLDB command interpreter.
///
/// All output from every command, including the prompt + the command
/// is placed into the "strm" argument.
///
/// Each individual command can be prefixed with \b ! and/or \b ? in no
/// particular order. If \b ? is provided, then the output of that command is
/// only emitted if it fails, and if \b ! is provided, then the output is
/// emitted regardless, and \b false is returned without executing the
/// remaining commands.
///
/// \param[in] debugger
///     The debugger that will execute the lldb commands.
///
/// \param[in] prefix
///     A string that will be printed into \a strm prior to emitting
///     the prompt + command and command output. Can be NULL.
///
/// \param[in] commands
///     An array of LLDB commands to execute.
///
/// \param[in] strm
///     The stream that will receive the prefix, prompt + command and
///     all command output.
///
/// \param[in] parse_command_directives
///     If \b false, then command prefixes like \b ! or \b ? are not parsed and
///     each command is executed verbatim.
///
/// \param[in] echo_commands
///     If \b true, the command are echoed to the stream.
///
/// \return
///     \b true, unless a command prefixed with \b ! fails and parsing of
///     command directives is enabled.
bool RunLLDBCommands(lldb::SBDebugger &debugger, llvm::StringRef prefix,
                     const llvm::ArrayRef<std::string> &commands,
                     llvm::raw_ostream &strm, bool parse_command_directives,
                     bool echo_commands);

/// Run a list of LLDB commands in the LLDB command interpreter.
///
/// All output from every command, including the prompt + the command
/// is returned in the std::string return value.
///
/// \param[in] debugger
///     The debugger that will execute the lldb commands.
///
/// \param[in] prefix
///     A string that will be printed into \a strm prior to emitting
///     the prompt + command and command output. Can be NULL.
///
/// \param[in] commands
///     An array of LLDB commands to execute.
///
/// \param[out] required_command_failed
///     If parsing of command directives is enabled, this variable is set to
///     \b true if one of the commands prefixed with \b ! fails.
///
/// \param[in] parse_command_directives
///     If \b false, then command prefixes like \b ! or \b ? are not parsed and
///     each command is executed verbatim.
///
/// \param[in] echo_commands
///     If \b true, the command are echoed to the stream.
///
/// \return
///     A std::string that contains the prefix and all commands and
///     command output.
std::string RunLLDBCommands(lldb::SBDebugger &debugger, llvm::StringRef prefix,
                            const llvm::ArrayRef<std::string> &commands,
                            bool &required_command_failed,
                            bool parse_command_directives = true,
                            bool echo_commands = false);

/// Check if a thread has a stop reason.
///
/// \param[in] thread
///     The LLDB thread object to check
///
/// \return
///     \b True if the thread has a valid stop reason, \b false
///     otherwise.
bool ThreadHasStopReason(lldb::SBThread &thread);

/// Given a LLDB frame, make a frame ID that is unique to a specific
/// thread and frame.
///
/// DAP requires a Stackframe "id" to be unique, so we use the frame
/// index in the lower 32 bits and the thread index ID in the upper 32
/// bits.
///
/// \param[in] frame
///     The LLDB stack frame object generate the ID for
///
/// \return
///     A unique integer that allows us to easily find the right
///     stack frame within a thread on subsequent VS code requests.
int64_t MakeDAPFrameID(lldb::SBFrame &frame);

/// Given a DAP frame ID, convert to a LLDB thread index id.
///
/// DAP requires a Stackframe "id" to be unique, so we use the frame
/// index in the lower THREAD_INDEX_SHIFT bits and the thread index ID in
/// the upper 32 - THREAD_INDEX_SHIFT bits.
///
/// \param[in] dap_frame_id
///     The DAP frame ID to convert to a thread index ID.
///
/// \return
///     The LLDB thread index ID.
uint32_t GetLLDBThreadIndexID(uint64_t dap_frame_id);

/// Given a DAP frame ID, convert to a LLDB frame ID.
///
/// DAP requires a Stackframe "id" to be unique, so we use the frame
/// index in the lower THREAD_INDEX_SHIFT bits and the thread index ID in
/// the upper 32 - THREAD_INDEX_SHIFT bits.
///
/// \param[in] dap_frame_id
///     The DAP frame ID to convert to a frame ID.
///
/// \return
///     The LLDB frame index ID.
uint32_t GetLLDBFrameID(uint64_t dap_frame_id);

/// Gets all the environment variables from the json object depending on if the
/// kind is an object or an array.
///
/// \param[in] arguments
///     The json object with the launch options
///
/// \return
///     The environment variables stored in the env key
lldb::SBEnvironment
GetEnvironmentFromArguments(const llvm::json::Object &arguments);

/// Gets an SBFileSpec and returns its path as a string.
///
/// \param[in] file_spec
///     The file spec.
///
/// \return
///     The file path as a string.
std::string GetSBFileSpecPath(const lldb::SBFileSpec &file_spec);

/// Helper for sending telemetry to lldb server, if client-telemetry is enabled.
class TelemetryDispatcher {
public:
  TelemetryDispatcher(lldb::SBDebugger *debugger) {
    m_telemetry_json = llvm::json::Object();
    m_telemetry_json.try_emplace(
        "start_time",
        std::chrono::steady_clock::now().time_since_epoch().count());
    this->debugger = debugger;
  }

  void Set(std::string key, std::string value) {
    m_telemetry_json.try_emplace(key, value);
  }

  void Set(std::string key, int64_t value) {
    m_telemetry_json.try_emplace(key, value);
  }

  ~TelemetryDispatcher() {
    m_telemetry_json.try_emplace(
        "end_time",
        std::chrono::steady_clock::now().time_since_epoch().count());

    lldb::SBStructuredData telemetry_entry;
    llvm::json::Value val(std::move(m_telemetry_json));

    std::string string_rep = llvm::to_string(val);
    telemetry_entry.SetFromJSON(string_rep.c_str());
    debugger->DispatchClientTelemetry(telemetry_entry);
  }

private:
  llvm::json::Object m_telemetry_json;
  lldb::SBDebugger *debugger;
};

/// RAII utility to put the debugger temporarily  into synchronous mode.
class ScopeSyncMode {
public:
  ScopeSyncMode(lldb::SBDebugger &debugger);
  ~ScopeSyncMode();

private:
  lldb::SBDebugger &m_debugger;
  bool m_async;
};

/// Get the stop-disassembly-display settings
///
/// \param[in] debugger
///     The debugger that will execute the lldb commands.
///
/// \return
///     The value of the stop-disassembly-display setting
lldb::StopDisassemblyType GetStopDisassemblyDisplay(lldb::SBDebugger &debugger);

/// Take ownership of the stored error.
llvm::Error ToError(const lldb::SBError &error);

/// Provides the string value if this data structure is a string type.
std::string GetStringValue(const lldb::SBStructuredData &data);

} // namespace lldb_dap

#endif
