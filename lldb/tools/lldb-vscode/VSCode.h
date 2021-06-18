//===-- VSCode.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_VSCODE_VSCODE_H
#define LLDB_TOOLS_LLDB_VSCODE_VSCODE_H

#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX

#include <condition_variable>
#include <cstdio>
#include <iosfwd>
#include <map>
#include <set>
#include <thread>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "lldb/API/SBAttachInfo.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBCommunication.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBLanguageRuntime.h"
#include "lldb/API/SBLaunchInfo.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"

#include "ExceptionBreakpoint.h"
#include "FunctionBreakpoint.h"
#include "IOStream.h"
#include "ProgressEvent.h"
#include "RunInTerminal.h"
#include "SourceBreakpoint.h"
#include "SourceReference.h"

#define VARREF_LOCALS (int64_t)1
#define VARREF_GLOBALS (int64_t)2
#define VARREF_REGS (int64_t)3
#define VARREF_FIRST_VAR_IDX (int64_t)4
#define VARREF_IS_SCOPE(v) (VARREF_LOCALS <= 1 && v < VARREF_FIRST_VAR_IDX)
#define VARIDX_TO_VARREF(i) ((i) + VARREF_FIRST_VAR_IDX)
#define VARREF_TO_VARIDX(v) ((v)-VARREF_FIRST_VAR_IDX)
#define NO_TYPENAME "<no-type>"

namespace lldb_vscode {

typedef llvm::DenseMap<uint32_t, SourceBreakpoint> SourceBreakpointMap;
typedef llvm::StringMap<FunctionBreakpoint> FunctionBreakpointMap;
enum class OutputType { Console, Stdout, Stderr, Telemetry };

enum VSCodeBroadcasterBits {
  eBroadcastBitStopEventThread = 1u << 0,
  eBroadcastBitStopProgressThread = 1u << 1
};

typedef void (*RequestCallback)(const llvm::json::Object &command);

enum class PacketStatus {
  Success = 0,
  EndOfFile,
  JSONMalformed,
  JSONNotObject
};

struct VSCode {
  std::string debug_adaptor_path;
  InputStream input;
  OutputStream output;
  lldb::SBDebugger debugger;
  lldb::SBTarget target;
  lldb::SBValueList variables;
  lldb::SBBroadcaster broadcaster;
  int64_t num_regs;
  int64_t num_locals;
  int64_t num_globals;
  std::thread event_thread;
  std::thread progress_event_thread;
  std::unique_ptr<std::ofstream> log;
  llvm::DenseMap<lldb::addr_t, int64_t> addr_to_source_ref;
  llvm::DenseMap<int64_t, SourceReference> source_map;
  llvm::StringMap<SourceBreakpointMap> source_breakpoints;
  FunctionBreakpointMap function_breakpoints;
  std::vector<ExceptionBreakpoint> exception_breakpoints;
  std::vector<std::string> init_commands;
  std::vector<std::string> pre_run_commands;
  std::vector<std::string> exit_commands;
  std::vector<std::string> stop_commands;
  std::vector<std::string> terminate_commands;
  lldb::tid_t focus_tid;
  bool sent_terminated_event;
  bool stop_at_entry;
  bool is_attach;
  uint32_t reverse_request_seq;
  std::map<std::string, RequestCallback> request_handlers;
  bool waiting_for_run_in_terminal;
  ProgressEventReporter progress_event_reporter;
  // Keep track of the last stop thread index IDs as threads won't go away
  // unless we send a "thread" event to indicate the thread exited.
  llvm::DenseSet<lldb::tid_t> thread_ids;
  VSCode();
  ~VSCode();
  VSCode(const VSCode &rhs) = delete;
  void operator=(const VSCode &rhs) = delete;
  int64_t GetLineForPC(int64_t sourceReference, lldb::addr_t pc) const;
  ExceptionBreakpoint *GetExceptionBreakpoint(const std::string &filter);
  ExceptionBreakpoint *GetExceptionBreakpoint(const lldb::break_id_t bp_id);

  // Serialize the JSON value into a string and send the JSON packet to
  // the "out" stream.
  void SendJSON(const llvm::json::Value &json);

  std::string ReadJSON();

  void SendOutput(OutputType o, const llvm::StringRef output);

  void SendProgressEvent(uint64_t progress_id, const char *message,
                         uint64_t completed, uint64_t total);

  void __attribute__((format(printf, 3, 4)))
  SendFormattedOutput(OutputType o, const char *format, ...);

  static int64_t GetNextSourceReference();

  ExceptionBreakpoint *GetExceptionBPFromStopReason(lldb::SBThread &thread);

  lldb::SBThread GetLLDBThread(const llvm::json::Object &arguments);

  lldb::SBFrame GetLLDBFrame(const llvm::json::Object &arguments);

  llvm::json::Value CreateTopLevelScopes();

  void RunLLDBCommands(llvm::StringRef prefix,
                       const std::vector<std::string> &commands);

  void RunInitCommands();
  void RunPreRunCommands();
  void RunStopCommands();
  void RunExitCommands();
  void RunTerminateCommands();

  /// Create a new SBTarget object from the given request arguments.
  /// \param[in] arguments
  ///     Launch configuration arguments.
  ///
  /// \param[out] error
  ///     An SBError object that will contain an error description if
  ///     function failed to create the target.
  ///
  /// \return
  ///     An SBTarget object.
  lldb::SBTarget CreateTargetFromArguments(const llvm::json::Object &arguments,
                                           lldb::SBError &error);

  /// Set given target object as a current target for lldb-vscode and start
  /// listeing for its breakpoint events.
  void SetTarget(const lldb::SBTarget target);

  const std::map<std::string, RequestCallback> &GetRequestHandlers();

  PacketStatus GetNextObject(llvm::json::Object &object);
  bool HandleObject(const llvm::json::Object &object);

  /// Send a Debug Adapter Protocol reverse request to the IDE
  ///
  /// \param[in] request
  ///   The payload of the request to send.
  ///
  /// \param[out] response
  ///   The response of the IDE. It might be undefined if there was an error.
  ///
  /// \return
  ///   A \a PacketStatus object indicating the sucess or failure of the
  ///   request.
  PacketStatus SendReverseRequest(llvm::json::Object request,
                                  llvm::json::Object &response);

  /// Registers a callback handler for a Debug Adapter Protocol request
  ///
  /// \param[in] request
  ///     The name of the request following the Debug Adapter Protocol
  ///     specification.
  ///
  /// \param[in] callback
  ///     The callback to execute when the given request is triggered by the
  ///     IDE.
  void RegisterRequestCallback(std::string request, RequestCallback callback);

private:
  // Send the JSON in "json_str" to the "out" stream. Correctly send the
  // "Content-Length:" field followed by the length, followed by the raw
  // JSON bytes.
  void SendJSON(const std::string &json_str);
};

extern VSCode g_vsc;

} // namespace lldb_vscode

#endif
