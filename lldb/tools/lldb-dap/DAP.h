//===-- DAP.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_DAP_H
#define LLDB_TOOLS_LLDB_DAP_DAP_H

#include "DAPForward.h"
#include "ExceptionBreakpoint.h"
#include "FunctionBreakpoint.h"
#include "InstructionBreakpoint.h"
#include "OutputRedirector.h"
#include "ProgressEvent.h"
#include "Protocol/ProtocolBase.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "SourceBreakpoint.h"
#include "Transport.h"
#include "Variables.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFile.h"
#include "lldb/API/SBFormat.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBMutex.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Threading.h"
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#define NO_TYPENAME "<no-type>"

namespace lldb_dap {

typedef std::map<std::pair<uint32_t, uint32_t>, SourceBreakpoint>
    SourceBreakpointMap;
typedef llvm::StringMap<FunctionBreakpoint> FunctionBreakpointMap;
typedef llvm::DenseMap<lldb::addr_t, InstructionBreakpoint>
    InstructionBreakpointMap;

using AdapterFeature = protocol::AdapterFeature;
using ClientFeature = protocol::ClientFeature;

enum class OutputType { Console, Important, Stdout, Stderr, Telemetry };

/// Buffer size for handling output events.
constexpr uint64_t OutputBufferSize = (1u << 12);

enum DAPBroadcasterBits {
  eBroadcastBitStopEventThread = 1u << 0,
  eBroadcastBitStopProgressThread = 1u << 1
};

enum class ReplMode { Variable = 0, Command, Auto };

struct DAP {
  /// Path to the lldb-dap binary itself.
  static llvm::StringRef debug_adapter_path;

  Log *log;
  Transport &transport;
  lldb::SBFile in;
  OutputRedirector out;
  OutputRedirector err;

  /// Configuration specified by the launch or attach commands.
  protocol::Configuration configuration;

  /// The debugger instance for this DAP session.
  lldb::SBDebugger debugger;

  /// The target instance for this DAP session.
  lldb::SBTarget target;

  Variables variables;
  lldb::SBBroadcaster broadcaster;
  FunctionBreakpointMap function_breakpoints;
  InstructionBreakpointMap instruction_breakpoints;
  std::vector<ExceptionBreakpoint> exception_breakpoints;
  llvm::once_flag init_exception_breakpoints_flag;

  /// Map step in target id to list of function targets that user can choose.
  llvm::DenseMap<lldb::addr_t, std::string> step_in_targets;

  /// A copy of the last LaunchRequest so we can reuse its arguments if we get a
  /// RestartRequest. Restarting an AttachRequest is not supported.
  std::optional<protocol::LaunchRequestArguments> last_launch_request;

  /// The focused thread for this DAP session.
  lldb::tid_t focus_tid = LLDB_INVALID_THREAD_ID;

  bool disconnecting = false;
  llvm::once_flag terminated_event_flag;
  bool stop_at_entry = false;
  bool is_attach = false;

  /// The process event thread normally responds to process exited events by
  /// shutting down the entire adapter. When we're restarting, we keep the id of
  /// the old process here so we can detect this case and keep running.
  lldb::pid_t restarting_process_id = LLDB_INVALID_PROCESS_ID;

  /// Whether we have received the ConfigurationDone request, indicating that
  /// the client has finished initialization of the debug adapter.
  bool configuration_done;

  bool waiting_for_run_in_terminal = false;
  ProgressEventReporter progress_event_reporter;

  /// Keep track of the last stop thread index IDs as threads won't go away
  /// unless we send a "thread" event to indicate the thread exited.
  llvm::DenseSet<lldb::tid_t> thread_ids;

  uint32_t reverse_request_seq = 0;
  std::mutex call_mutex;
  llvm::SmallDenseMap<int64_t, std::unique_ptr<ResponseHandler>>
      inflight_reverse_requests;
  ReplMode repl_mode;
  lldb::SBFormat frame_format;
  lldb::SBFormat thread_format;

  /// This is used to allow request_evaluate to handle empty expressions
  /// (ie the user pressed 'return' and expects the previous expression to
  /// repeat). If the previous expression was a command, this string will be
  /// empty; if the previous expression was a variable expression, this string
  /// will contain that expression.
  std::string last_nonempty_var_expression;

  /// The set of features supported by the connected client.
  llvm::DenseSet<ClientFeature> clientFeatures;

  /// The initial thread list upon attaching.
  std::vector<protocol::Thread> initial_thread_list;

  /// Keep track of all the modules our client knows about: either through the
  /// modules request or the module events.
  /// @{
  std::mutex modules_mutex;
  llvm::StringSet<> modules;
  /// @}

  /// Number of lines of assembly code to show when no debug info is available.
  static constexpr uint32_t k_number_of_assembly_lines_for_nodebug = 32;

  /// Creates a new DAP sessions.
  ///
  /// \param[in] log
  ///     Log stream, if configured.
  /// \param[in] default_repl_mode
  ///     Default repl mode behavior, as configured by the binary.
  /// \param[in] pre_init_commands
  ///     LLDB commands to execute as soon as the debugger instance is
  ///     allocated.
  /// \param[in] transport
  ///     Transport for this debug session.
  DAP(Log *log, const ReplMode default_repl_mode,
      std::vector<std::string> pre_init_commands, Transport &transport);

  ~DAP();

  /// DAP is not copyable.
  /// @{
  DAP(const DAP &rhs) = delete;
  void operator=(const DAP &rhs) = delete;
  /// @}

  ExceptionBreakpoint *GetExceptionBreakpoint(llvm::StringRef filter);
  ExceptionBreakpoint *GetExceptionBreakpoint(const lldb::break_id_t bp_id);

  /// Redirect stdout and stderr fo the IDE's console output.
  ///
  /// Errors in this operation will be printed to the log file and the IDE's
  /// console output as well.
  llvm::Error ConfigureIO(std::FILE *overrideOut = nullptr,
                          std::FILE *overrideErr = nullptr);

  /// Stop event handler threads.
  void StopEventHandlers();

  /// Configures the debug adapter for launching/attaching.
  void SetConfiguration(const protocol::Configuration &confing, bool is_attach);

  /// Configure source maps based on the current `DAPConfiguration`.
  void ConfigureSourceMaps();

  /// Serialize the JSON value into a string and send the JSON packet to the
  /// "out" stream.
  void SendJSON(const llvm::json::Value &json);
  /// Send the given message to the client
  void Send(const protocol::Message &message);

  void SendOutput(OutputType o, const llvm::StringRef output);

  void SendProgressEvent(uint64_t progress_id, const char *message,
                         uint64_t completed, uint64_t total);

  void __attribute__((format(printf, 3, 4)))
  SendFormattedOutput(OutputType o, const char *format, ...);

  int32_t CreateSourceReference(lldb::addr_t address);

  std::optional<lldb::addr_t> GetSourceReferenceAddress(int32_t reference);

  ExceptionBreakpoint *GetExceptionBPFromStopReason(lldb::SBThread &thread);

  lldb::SBThread GetLLDBThread(lldb::tid_t id);
  lldb::SBThread GetLLDBThread(const llvm::json::Object &arguments);

  lldb::SBFrame GetLLDBFrame(uint64_t frame_id);
  /// TODO: remove this function when we finish migrating to the
  /// new protocol types.
  lldb::SBFrame GetLLDBFrame(const llvm::json::Object &arguments);

  void PopulateExceptionBreakpoints();

  /// Attempt to determine if an expression is a variable expression or
  /// lldb command using a heuristic based on the first term of the
  /// expression.
  ///
  /// \param[in] frame
  ///     The frame, used as context to detect local variable names
  /// \param[inout] expression
  ///     The expression string. Might be modified by this function to
  ///     remove the leading escape character.
  /// \param[in] partial_expression
  ///     Whether the provided `expression` is only a prefix of the
  ///     final expression. If `true`, this function might return
  ///     `ReplMode::Auto` to indicate that the expression could be
  ///     either an expression or a statement, depending on the rest of
  ///     the expression.
  /// \return the expression mode
  ReplMode DetectReplMode(lldb::SBFrame frame, std::string &expression,
                          bool partial_expression);

  /// Create a `protocol::Source` object as described in the debug adapter
  /// definition.
  ///
  /// \param[in] frame
  ///     The frame to use when populating the "Source" object.
  ///
  /// \return
  ///     A `protocol::Source` object that follows the formal JSON
  ///     definition outlined by Microsoft.
  std::optional<protocol::Source> ResolveSource(const lldb::SBFrame &frame);

  /// Create a "Source" JSON object as described in the debug adapter
  /// definition.
  ///
  /// \param[in] address
  ///     The address to use when populating out the "Source" object.
  ///
  /// \return
  ///     An optional "Source" JSON object that follows the formal JSON
  ///     definition outlined by Microsoft.
  std::optional<protocol::Source> ResolveSource(lldb::SBAddress address);

  /// Create a "Source" JSON object as described in the debug adapter
  /// definition.
  ///
  /// \param[in] address
  ///     The address to use when populating out the "Source" object.
  ///
  /// \return
  ///     An optional "Source" JSON object that follows the formal JSON
  ///     definition outlined by Microsoft.
  std::optional<protocol::Source>
  ResolveAssemblySource(lldb::SBAddress address);

  /// \return
  ///   \b false if a fatal error was found while executing these commands,
  ///   according to the rules of \a LLDBUtils::RunLLDBCommands.
  bool RunLLDBCommands(llvm::StringRef prefix,
                       llvm::ArrayRef<std::string> commands);

  llvm::Error RunAttachCommands(llvm::ArrayRef<std::string> attach_commands);
  llvm::Error RunLaunchCommands(llvm::ArrayRef<std::string> launch_commands);
  llvm::Error RunPreInitCommands();
  llvm::Error RunInitCommands();
  llvm::Error RunPreRunCommands();
  void RunPostRunCommands();
  void RunStopCommands();
  void RunExitCommands();
  void RunTerminateCommands();

  /// Create a new SBTarget object from the given request arguments.
  ///
  /// \param[out] error
  ///     An SBError object that will contain an error description if
  ///     function failed to create the target.
  ///
  /// \return
  ///     An SBTarget object.
  lldb::SBTarget CreateTarget(lldb::SBError &error);

  /// Set given target object as a current target for lldb-dap and start
  /// listeing for its breakpoint events.
  void SetTarget(const lldb::SBTarget target);

  bool HandleObject(const protocol::Message &M);

  /// Disconnect the DAP session.
  llvm::Error Disconnect();

  /// Disconnect the DAP session and optionally terminate the debuggee.
  llvm::Error Disconnect(bool terminateDebuggee);

  /// Send a "terminated" event to indicate the process is done being debugged.
  void SendTerminatedEvent();

  llvm::Error Loop();

  /// Send a Debug Adapter Protocol reverse request to the IDE.
  ///
  /// \param[in] command
  ///   The reverse request command.
  ///
  /// \param[in] arguments
  ///   The reverse request arguments.
  template <typename Handler>
  void SendReverseRequest(llvm::StringRef command,
                          llvm::json::Value arguments) {
    int64_t id;
    {
      std::lock_guard<std::mutex> locker(call_mutex);
      id = ++reverse_request_seq;
      inflight_reverse_requests[id] = std::make_unique<Handler>(command, id);
    }

    SendJSON(llvm::json::Object{
        {"type", "request"},
        {"seq", id},
        {"command", command},
        {"arguments", std::move(arguments)},
    });
  }

  /// The set of capabilities supported by this adapter.
  protocol::Capabilities GetCapabilities();

  /// Debuggee will continue from stopped state.
  void WillContinue() { variables.Clear(); }

  /// Poll the process to wait for it to reach the eStateStopped state.
  ///
  /// Wait for the process hit a stopped state. When running a launch with
  /// "launchCommands", or attach with  "attachCommands", the calls might take
  /// some time to stop at the entry point since the command is asynchronous. We
  /// need to sync up with the process and make sure it is stopped before we
  /// proceed to do anything else as we will soon be asked to set breakpoints
  /// and other things that require the process to be stopped. We must use
  /// polling because "attachCommands" or "launchCommands" may or may not send
  /// process state change events depending on if the user modifies the async
  /// setting in the debugger. Since both "attachCommands" and "launchCommands"
  /// could end up using any combination of LLDB commands, we must ensure we can
  /// also catch when the process stops, so we must poll the process to make
  /// sure we handle all cases.
  ///
  /// \param[in] seconds
  ///   The number of seconds to poll the process to wait until it is stopped.
  ///
  /// \return Error if waiting for the process fails, no error if succeeds.
  lldb::SBError WaitForProcessToStop(std::chrono::seconds seconds);

  void SetFrameFormat(llvm::StringRef format);

  void SetThreadFormat(llvm::StringRef format);

  InstructionBreakpoint *GetInstructionBreakpoint(const lldb::break_id_t bp_id);

  InstructionBreakpoint *GetInstructionBPFromStopReason(lldb::SBThread &thread);

  /// Checks if the request is cancelled.
  bool IsCancelled(const protocol::Request &);

  /// Clears the cancel request from the set of tracked cancel requests.
  void ClearCancelRequest(const protocol::CancelArguments &);

  lldb::SBMutex GetAPIMutex() const { return target.GetAPIMutex(); }

  void StartEventThread();
  void StartProgressEventThread();

  /// Sets the given protocol `breakpoints` in the given `source`, while
  /// removing any existing breakpoints in the given source if they are not in
  /// `breakpoint`.
  ///
  /// \param[in] source
  ///   The relevant source of the breakpoints.
  ///
  /// \param[in] breakpoints
  ///   The breakpoints to set.
  ///
  /// \return a vector of the breakpoints that were set.
  std::vector<protocol::Breakpoint> SetSourceBreakpoints(
      const protocol::Source &source,
      const std::optional<std::vector<protocol::SourceBreakpoint>>
          &breakpoints);

private:
  std::vector<protocol::Breakpoint> SetSourceBreakpoints(
      const protocol::Source &source,
      const std::optional<std::vector<protocol::SourceBreakpoint>> &breakpoints,
      SourceBreakpointMap &existing_breakpoints);

  /// Registration of request handler.
  /// @{
  void RegisterRequests();
  template <typename Handler> void RegisterRequest() {
    request_handlers[Handler::GetCommand()] = std::make_unique<Handler>(*this);
  }
  llvm::StringMap<std::unique_ptr<BaseRequestHandler>> request_handlers;
  /// @}

  /// Event threads.
  /// @{
  void EventThread();
  void ProgressEventThread();

  std::thread event_thread;
  std::thread progress_event_thread;
  /// @}

  /// List of addresses mapped by sourceReference.
  std::vector<lldb::addr_t> m_source_references;
  std::mutex m_source_references_mutex;

  /// Queue for all incoming messages.
  std::deque<protocol::Message> m_queue;
  std::mutex m_queue_mutex;
  std::condition_variable m_queue_cv;

  std::mutex m_cancelled_requests_mutex;
  llvm::SmallSet<int64_t, 4> m_cancelled_requests;

  std::mutex m_active_request_mutex;
  const protocol::Request *m_active_request;

  llvm::StringMap<SourceBreakpointMap> m_source_breakpoints;
  llvm::DenseMap<int64_t, SourceBreakpointMap> m_source_assembly_breakpoints;
};

} // namespace lldb_dap

#endif
