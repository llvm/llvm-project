//===-- DAP.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPLog.h"
#include "EventHelper.h"
#include "Handler/RequestHandler.h"
#include "Handler/ResponseHandler.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "OutputRedirector.h"
#include "Protocol/ProtocolBase.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "Transport.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBLanguageRuntime.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>

#if defined(_WIN32)
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace lldb_dap;
using namespace lldb_dap::protocol;

namespace {
#ifdef _WIN32
const char DEV_NULL[] = "nul";
#else
const char DEV_NULL[] = "/dev/null";
#endif
} // namespace

namespace lldb_dap {

static std::string GetStringFromStructuredData(lldb::SBStructuredData &data,
                                               const char *key) {
  lldb::SBStructuredData keyValue = data.GetValueForKey(key);
  if (!keyValue)
    return std::string();

  const size_t length = keyValue.GetStringValue(nullptr, 0);

  if (length == 0)
    return std::string();

  std::string str(length + 1, 0);
  keyValue.GetStringValue(&str[0], length + 1);
  return str;
}

static uint64_t GetUintFromStructuredData(lldb::SBStructuredData &data,
                                          const char *key) {
  lldb::SBStructuredData keyValue = data.GetValueForKey(key);

  if (!keyValue.IsValid())
    return 0;
  return keyValue.GetUnsignedIntegerValue();
}

/// Return string with first character capitalized.
static std::string capitalize(llvm::StringRef str) {
  if (str.empty())
    return "";
  return ((llvm::Twine)llvm::toUpper(str[0]) + str.drop_front()).str();
}

llvm::StringRef DAP::debug_adapter_path = "";

DAP::DAP(Log *log, const ReplMode default_repl_mode,
         std::vector<std::string> pre_init_commands, Transport &transport)
    : log(log), transport(transport), broadcaster("lldb-dap"),
      exception_breakpoints(), focus_tid(LLDB_INVALID_THREAD_ID),
      stop_at_entry(false), is_attach(false),
      restarting_process_id(LLDB_INVALID_PROCESS_ID), configuration_done(false),
      waiting_for_run_in_terminal(false),
      progress_event_reporter(
          [&](const ProgressEvent &event) { SendJSON(event.ToJSON()); }),
      reverse_request_seq(0), repl_mode(default_repl_mode) {
  configuration.preInitCommands = std::move(pre_init_commands);
  RegisterRequests();
}

DAP::~DAP() = default;

void DAP::PopulateExceptionBreakpoints() {
  llvm::call_once(init_exception_breakpoints_flag, [this]() {
    exception_breakpoints = std::vector<ExceptionBreakpoint>{};

    if (lldb::SBDebugger::SupportsLanguage(lldb::eLanguageTypeC_plus_plus)) {
      exception_breakpoints->emplace_back(*this, "cpp_catch", "C++ Catch",
                                          lldb::eLanguageTypeC_plus_plus);
      exception_breakpoints->emplace_back(*this, "cpp_throw", "C++ Throw",
                                          lldb::eLanguageTypeC_plus_plus);
    }
    if (lldb::SBDebugger::SupportsLanguage(lldb::eLanguageTypeObjC)) {
      exception_breakpoints->emplace_back(
          *this, "objc_catch", "Objective-C Catch", lldb::eLanguageTypeObjC);
      exception_breakpoints->emplace_back(
          *this, "objc_throw", "Objective-C Throw", lldb::eLanguageTypeObjC);
    }
    if (lldb::SBDebugger::SupportsLanguage(lldb::eLanguageTypeSwift)) {
      exception_breakpoints->emplace_back(*this, "swift_catch", "Swift Catch",
                                          lldb::eLanguageTypeSwift);
      exception_breakpoints->emplace_back(*this, "swift_throw", "Swift Throw",
                                          lldb::eLanguageTypeSwift);
    }
    // Besides handling the hardcoded list of languages from above, we try to
    // find any other languages that support exception breakpoints using the
    // SB API.
    for (int raw_lang = lldb::eLanguageTypeUnknown;
         raw_lang < lldb::eNumLanguageTypes; ++raw_lang) {
      lldb::LanguageType lang = static_cast<lldb::LanguageType>(raw_lang);

      // We first discard any languages already handled above.
      if (lldb::SBLanguageRuntime::LanguageIsCFamily(lang) ||
          lang == lldb::eLanguageTypeSwift)
        continue;

      if (!lldb::SBDebugger::SupportsLanguage(lang))
        continue;

      const char *name = lldb::SBLanguageRuntime::GetNameForLanguageType(lang);
      if (!name)
        continue;
      std::string raw_lang_name = name;
      std::string capitalized_lang_name = capitalize(name);

      if (lldb::SBLanguageRuntime::SupportsExceptionBreakpointsOnThrow(lang)) {
        const char *raw_throw_keyword =
            lldb::SBLanguageRuntime::GetThrowKeywordForLanguage(lang);
        std::string throw_keyword =
            raw_throw_keyword ? raw_throw_keyword : "throw";

        exception_breakpoints->emplace_back(
            *this, raw_lang_name + "_" + throw_keyword,
            capitalized_lang_name + " " + capitalize(throw_keyword), lang);
      }

      if (lldb::SBLanguageRuntime::SupportsExceptionBreakpointsOnCatch(lang)) {
        const char *raw_catch_keyword =
            lldb::SBLanguageRuntime::GetCatchKeywordForLanguage(lang);
        std::string catch_keyword =
            raw_catch_keyword ? raw_catch_keyword : "catch";

        exception_breakpoints->emplace_back(
            *this, raw_lang_name + "_" + catch_keyword,
            capitalized_lang_name + " " + capitalize(catch_keyword), lang);
      }
    }
    assert(!exception_breakpoints->empty() && "should not be empty");
  });
}

ExceptionBreakpoint *DAP::GetExceptionBreakpoint(llvm::StringRef filter) {
  // PopulateExceptionBreakpoints() is called after g_dap.debugger is created
  // in a request-initialize.
  //
  // But this GetExceptionBreakpoint() method may be called before attaching, in
  // which case, we may not have populated the filter yet.
  //
  // We also cannot call PopulateExceptionBreakpoints() in DAP::DAP() because
  // we need SBDebugger::Initialize() to have been called before this.
  //
  // So just calling PopulateExceptionBreakoints(),which does lazy-populating
  // seems easiest. Two other options include:
  //  + call g_dap.PopulateExceptionBreakpoints() in lldb-dap.cpp::main()
  //    right after the call to SBDebugger::Initialize()
  //  + Just call PopulateExceptionBreakpoints() to get a fresh list  everytime
  //    we query (a bit overkill since it's not likely to change?)
  PopulateExceptionBreakpoints();

  for (auto &bp : *exception_breakpoints) {
    if (bp.GetFilter() == filter)
      return &bp;
  }
  return nullptr;
}

ExceptionBreakpoint *DAP::GetExceptionBreakpoint(const lldb::break_id_t bp_id) {
  // See comment in the other GetExceptionBreakpoint().
  PopulateExceptionBreakpoints();

  for (auto &bp : *exception_breakpoints) {
    if (bp.GetID() == bp_id)
      return &bp;
  }
  return nullptr;
}

llvm::Error DAP::ConfigureIO(std::FILE *overrideOut, std::FILE *overrideErr) {
  in = lldb::SBFile(std::fopen(DEV_NULL, "r"), /*transfer_ownership=*/true);

  if (auto Error = out.RedirectTo(overrideOut, [this](llvm::StringRef output) {
        SendOutput(OutputType::Console, output);
      }))
    return Error;

  if (auto Error = err.RedirectTo(overrideErr, [this](llvm::StringRef output) {
        SendOutput(OutputType::Console, output);
      }))
    return Error;

  return llvm::Error::success();
}

void DAP::StopEventHandlers() {
  if (event_thread.joinable()) {
    broadcaster.BroadcastEventByType(eBroadcastBitStopEventThread);
    event_thread.join();
  }
  if (progress_event_thread.joinable()) {
    broadcaster.BroadcastEventByType(eBroadcastBitStopProgressThread);
    progress_event_thread.join();
  }
}

// Serialize the JSON value into a string and send the JSON packet to
// the "out" stream.
void DAP::SendJSON(const llvm::json::Value &json) {
  // FIXME: Instead of parsing the output message from JSON, pass the `Message`
  // as parameter to `SendJSON`.
  Message message;
  llvm::json::Path::Root root;
  if (!fromJSON(json, message, root)) {
    DAP_LOG_ERROR(log, root.getError(), "({1}) encoding failed: {0}",
                  transport.GetClientName());
    return;
  }
  Send(message);
}

void DAP::Send(const Message &message) {
  // FIXME: After all the requests have migrated from LegacyRequestHandler >
  // RequestHandler<> this should be handled in RequestHandler<>::operator().
  if (auto *resp = std::get_if<Response>(&message);
      resp && debugger.InterruptRequested()) {
    // Clear the interrupt request.
    debugger.CancelInterruptRequest();

    // If the debugger was interrupted, convert this response into a 'cancelled'
    // response because we might have a partial result.
    Response cancelled{/*request_seq=*/resp->request_seq,
                       /*command=*/resp->command,
                       /*success=*/false,
                       /*message=*/eResponseMessageCancelled,
                       /*body=*/std::nullopt};
    if (llvm::Error err = transport.Write(cancelled))
      DAP_LOG_ERROR(log, std::move(err), "({1}) write failed: {0}",
                    transport.GetClientName());
    return;
  }

  if (llvm::Error err = transport.Write(message))
    DAP_LOG_ERROR(log, std::move(err), "({1}) write failed: {0}",
                  transport.GetClientName());
}

// "OutputEvent": {
//   "allOf": [ { "$ref": "#/definitions/Event" }, {
//     "type": "object",
//     "description": "Event message for 'output' event type. The event
//                     indicates that the target has produced some output.",
//     "properties": {
//       "event": {
//         "type": "string",
//         "enum": [ "output" ]
//       },
//       "body": {
//         "type": "object",
//         "properties": {
//           "category": {
//             "type": "string",
//             "description": "The output category. If not specified,
//                             'console' is assumed.",
//             "_enum": [ "console", "stdout", "stderr", "telemetry" ]
//           },
//           "output": {
//             "type": "string",
//             "description": "The output to report."
//           },
//           "variablesReference": {
//             "type": "number",
//             "description": "If an attribute 'variablesReference' exists
//                             and its value is > 0, the output contains
//                             objects which can be retrieved by passing
//                             variablesReference to the VariablesRequest."
//           },
//           "source": {
//             "$ref": "#/definitions/Source",
//             "description": "An optional source location where the output
//                             was produced."
//           },
//           "line": {
//             "type": "integer",
//             "description": "An optional source location line where the
//                             output was produced."
//           },
//           "column": {
//             "type": "integer",
//             "description": "An optional source location column where the
//                             output was produced."
//           },
//           "data": {
//             "type":["array","boolean","integer","null","number","object",
//                     "string"],
//             "description": "Optional data to report. For the 'telemetry'
//                             category the data will be sent to telemetry, for
//                             the other categories the data is shown in JSON
//                             format."
//           }
//         },
//         "required": ["output"]
//       }
//     },
//     "required": [ "event", "body" ]
//   }]
// }
void DAP::SendOutput(OutputType o, const llvm::StringRef output) {
  if (output.empty())
    return;

  const char *category = nullptr;
  switch (o) {
  case OutputType::Console:
    category = "console";
    break;
  case OutputType::Important:
    category = "important";
    break;
  case OutputType::Stdout:
    category = "stdout";
    break;
  case OutputType::Stderr:
    category = "stderr";
    break;
  case OutputType::Telemetry:
    category = "telemetry";
    break;
  }

  // Send each line of output as an individual event, including the newline if
  // present.
  ::size_t idx = 0;
  do {
    ::size_t end = output.find('\n', idx);
    if (end == llvm::StringRef::npos)
      end = output.size() - 1;
    llvm::json::Object event(CreateEventObject("output"));
    llvm::json::Object body;
    body.try_emplace("category", category);
    EmplaceSafeString(body, "output", output.slice(idx, end + 1).str());
    event.try_emplace("body", std::move(body));
    SendJSON(llvm::json::Value(std::move(event)));
    idx = end + 1;
  } while (idx < output.size());
}

// interface ProgressStartEvent extends Event {
//   event: 'progressStart';
//
//   body: {
//     /**
//      * An ID that must be used in subsequent 'progressUpdate' and
//      'progressEnd'
//      * events to make them refer to the same progress reporting.
//      * IDs must be unique within a debug session.
//      */
//     progressId: string;
//
//     /**
//      * Mandatory (short) title of the progress reporting. Shown in the UI to
//      * describe the long running operation.
//      */
//     title: string;
//
//     /**
//      * The request ID that this progress report is related to. If specified a
//      * debug adapter is expected to emit
//      * progress events for the long running request until the request has
//      been
//      * either completed or cancelled.
//      * If the request ID is omitted, the progress report is assumed to be
//      * related to some general activity of the debug adapter.
//      */
//     requestId?: number;
//
//     /**
//      * If true, the request that reports progress may be canceled with a
//      * 'cancel' request.
//      * So this property basically controls whether the client should use UX
//      that
//      * supports cancellation.
//      * Clients that don't support cancellation are allowed to ignore the
//      * setting.
//      */
//     cancellable?: boolean;
//
//     /**
//      * Optional, more detailed progress message.
//      */
//     message?: string;
//
//     /**
//      * Optional progress percentage to display (value range: 0 to 100). If
//      * omitted no percentage will be shown.
//      */
//     percentage?: number;
//   };
// }
//
// interface ProgressUpdateEvent extends Event {
//   event: 'progressUpdate';
//
//   body: {
//     /**
//      * The ID that was introduced in the initial 'progressStart' event.
//      */
//     progressId: string;
//
//     /**
//      * Optional, more detailed progress message. If omitted, the previous
//      * message (if any) is used.
//      */
//     message?: string;
//
//     /**
//      * Optional progress percentage to display (value range: 0 to 100). If
//      * omitted no percentage will be shown.
//      */
//     percentage?: number;
//   };
// }
//
// interface ProgressEndEvent extends Event {
//   event: 'progressEnd';
//
//   body: {
//     /**
//      * The ID that was introduced in the initial 'ProgressStartEvent'.
//      */
//     progressId: string;
//
//     /**
//      * Optional, more detailed progress message. If omitted, the previous
//      * message (if any) is used.
//      */
//     message?: string;
//   };
// }

void DAP::SendProgressEvent(uint64_t progress_id, const char *message,
                            uint64_t completed, uint64_t total) {
  progress_event_reporter.Push(progress_id, message, completed, total);
}

void __attribute__((format(printf, 3, 4)))
DAP::SendFormattedOutput(OutputType o, const char *format, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, format);
  int actual_length = vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  SendOutput(
      o, llvm::StringRef(buffer, std::min<int>(actual_length, sizeof(buffer))));
}

ExceptionBreakpoint *DAP::GetExceptionBPFromStopReason(lldb::SBThread &thread) {
  const auto num = thread.GetStopReasonDataCount();
  // Check to see if have hit an exception breakpoint and change the
  // reason to "exception", but only do so if all breakpoints that were
  // hit are exception breakpoints.
  ExceptionBreakpoint *exc_bp = nullptr;
  for (size_t i = 0; i < num; i += 2) {
    // thread.GetStopReasonDataAtIndex(i) will return the bp ID and
    // thread.GetStopReasonDataAtIndex(i+1) will return the location
    // within that breakpoint. We only care about the bp ID so we can
    // see if this is an exception breakpoint that is getting hit.
    lldb::break_id_t bp_id = thread.GetStopReasonDataAtIndex(i);
    exc_bp = GetExceptionBreakpoint(bp_id);
    // If any breakpoint is not an exception breakpoint, then stop and
    // report this as a normal breakpoint
    if (exc_bp == nullptr)
      return nullptr;
  }
  return exc_bp;
}

lldb::SBThread DAP::GetLLDBThread(lldb::tid_t tid) {
  return target.GetProcess().GetThreadByID(tid);
}

lldb::SBThread DAP::GetLLDBThread(const llvm::json::Object &arguments) {
  auto tid = GetInteger<int64_t>(arguments, "threadId")
                 .value_or(LLDB_INVALID_THREAD_ID);
  return target.GetProcess().GetThreadByID(tid);
}

lldb::SBFrame DAP::GetLLDBFrame(uint64_t frame_id) {
  lldb::SBProcess process = target.GetProcess();
  // Upper 32 bits is the thread index ID
  lldb::SBThread thread =
      process.GetThreadByIndexID(GetLLDBThreadIndexID(frame_id));
  // Lower 32 bits is the frame index
  return thread.GetFrameAtIndex(GetLLDBFrameID(frame_id));
}

lldb::SBFrame DAP::GetLLDBFrame(const llvm::json::Object &arguments) {
  const auto frame_id =
      GetInteger<uint64_t>(arguments, "frameId").value_or(UINT64_MAX);
  return GetLLDBFrame(frame_id);
}

ReplMode DAP::DetectReplMode(lldb::SBFrame frame, std::string &expression,
                             bool partial_expression) {
  // Check for the escape hatch prefix.
  if (!expression.empty() &&
      llvm::StringRef(expression)
          .starts_with(configuration.commandEscapePrefix)) {
    expression = expression.substr(configuration.commandEscapePrefix.size());
    return ReplMode::Command;
  }

  switch (repl_mode) {
  case ReplMode::Variable:
    return ReplMode::Variable;
  case ReplMode::Command:
    return ReplMode::Command;
  case ReplMode::Auto:
    // To determine if the expression is a command or not, check if the first
    // term is a variable or command. If it's a variable in scope we will prefer
    // that behavior and give a warning to the user if they meant to invoke the
    // operation as a command.
    //
    // Example use case:
    //   int p and expression "p + 1" > variable
    //   int i and expression "i" > variable
    //   int var and expression "va" > command
    std::pair<llvm::StringRef, llvm::StringRef> token =
        llvm::getToken(expression);

    // If the first token is not fully finished yet, we can't
    // determine whether this will be a variable or a lldb command.
    if (partial_expression && token.second.empty())
      return ReplMode::Auto;

    std::string term = token.first.str();
    lldb::SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
    bool term_is_command = interpreter.CommandExists(term.c_str()) ||
                           interpreter.UserCommandExists(term.c_str()) ||
                           interpreter.AliasExists(term.c_str());
    bool term_is_variable = frame.FindVariable(term.c_str()).IsValid();

    // If we have both a variable and command, warn the user about the conflict.
    if (term_is_command && term_is_variable) {
      llvm::errs()
          << "Warning: Expression '" << term
          << "' is both an LLDB command and variable. It will be evaluated as "
             "a variable. To evaluate the expression as an LLDB command, use '"
          << configuration.commandEscapePrefix << "' as a prefix.\n";
    }

    // Variables take preference to commands in auto, since commands can always
    // be called using the command_escape_prefix
    return term_is_variable  ? ReplMode::Variable
           : term_is_command ? ReplMode::Command
                             : ReplMode::Variable;
  }

  llvm_unreachable("enum cases exhausted.");
}

bool DAP::RunLLDBCommands(llvm::StringRef prefix,
                          llvm::ArrayRef<std::string> commands) {
  bool required_command_failed = false;
  std::string output = ::RunLLDBCommands(
      debugger, prefix, commands, required_command_failed,
      /*parse_command_directives*/ true, /*echo_commands*/ true);
  SendOutput(OutputType::Console, output);
  return !required_command_failed;
}

static llvm::Error createRunLLDBCommandsErrorMessage(llvm::StringRef category) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      llvm::formatv(
          "Failed to run {0} commands. See the Debug Console for more details.",
          category)
          .str()
          .c_str());
}

llvm::Error
DAP::RunAttachCommands(llvm::ArrayRef<std::string> attach_commands) {
  if (!RunLLDBCommands("Running attachCommands:", attach_commands))
    return createRunLLDBCommandsErrorMessage("attach");
  return llvm::Error::success();
}

llvm::Error
DAP::RunLaunchCommands(llvm::ArrayRef<std::string> launch_commands) {
  if (!RunLLDBCommands("Running launchCommands:", launch_commands))
    return createRunLLDBCommandsErrorMessage("launch");
  return llvm::Error::success();
}

llvm::Error DAP::RunInitCommands() {
  if (!RunLLDBCommands("Running initCommands:", configuration.initCommands))
    return createRunLLDBCommandsErrorMessage("initCommands");
  return llvm::Error::success();
}

llvm::Error DAP::RunPreInitCommands() {
  if (!RunLLDBCommands("Running preInitCommands:",
                       configuration.preInitCommands))
    return createRunLLDBCommandsErrorMessage("preInitCommands");
  return llvm::Error::success();
}

llvm::Error DAP::RunPreRunCommands() {
  if (!RunLLDBCommands("Running preRunCommands:", configuration.preRunCommands))
    return createRunLLDBCommandsErrorMessage("preRunCommands");
  return llvm::Error::success();
}

void DAP::RunPostRunCommands() {
  RunLLDBCommands("Running postRunCommands:", configuration.postRunCommands);
}
void DAP::RunStopCommands() {
  RunLLDBCommands("Running stopCommands:", configuration.stopCommands);
}

void DAP::RunExitCommands() {
  RunLLDBCommands("Running exitCommands:", configuration.exitCommands);
}

void DAP::RunTerminateCommands() {
  RunLLDBCommands("Running terminateCommands:",
                  configuration.terminateCommands);
}

lldb::SBTarget DAP::CreateTarget(lldb::SBError &error) {
  // Grab the name of the program we need to debug and create a target using
  // the given program as an argument. Executable file can be a source of target
  // architecture and platform, if they differ from the host. Setting exe path
  // in launch info is useless because Target.Launch() will not change
  // architecture and platform, therefore they should be known at the target
  // creation. We also use target triple and platform from the launch
  // configuration, if given, since in some cases ELF file doesn't contain
  // enough information to determine correct arch and platform (or ELF can be
  // omitted at all), so it is good to leave the user an opportunity to specify
  // those. Any of those three can be left empty.
  auto target = this->debugger.CreateTarget(
      /*filename=*/configuration.program.data(),
      /*target_triple=*/configuration.targetTriple.data(),
      /*platform_name=*/configuration.platformName.data(),
      /*add_dependent_modules=*/true, // Add dependent modules.
      error);

  return target;
}

void DAP::SetTarget(const lldb::SBTarget target) {
  this->target = target;

  if (target.IsValid()) {
    // Configure breakpoint event listeners for the target.
    lldb::SBListener listener = this->debugger.GetListener();
    listener.StartListeningForEvents(
        this->target.GetBroadcaster(),
        lldb::SBTarget::eBroadcastBitBreakpointChanged |
            lldb::SBTarget::eBroadcastBitModulesLoaded |
            lldb::SBTarget::eBroadcastBitModulesUnloaded |
            lldb::SBTarget::eBroadcastBitSymbolsLoaded |
            lldb::SBTarget::eBroadcastBitSymbolsChanged);
    listener.StartListeningForEvents(this->broadcaster,
                                     eBroadcastBitStopEventThread);
  }
}

bool DAP::HandleObject(const Message &M) {
  TelemetryDispatcher dispatcher(&debugger);
  dispatcher.Set("client_name", transport.GetClientName().str());
  if (const auto *req = std::get_if<Request>(&M)) {
    {
      std::lock_guard<std::mutex> guard(m_active_request_mutex);
      m_active_request = req;

      // Clear the interrupt request prior to invoking a handler.
      if (debugger.InterruptRequested())
        debugger.CancelInterruptRequest();
    }

    auto cleanup = llvm::make_scope_exit([&]() {
      std::scoped_lock<std::mutex> active_request_lock(m_active_request_mutex);
      m_active_request = nullptr;
    });

    auto handler_pos = request_handlers.find(req->command);
    dispatcher.Set("client_data",
                   llvm::Twine("request_command:", req->command).str());
    if (handler_pos != request_handlers.end()) {
      handler_pos->second->Run(*req);
      return true; // Success
    }

    dispatcher.Set("error",
                   llvm::Twine("unhandled-command:" + req->command).str());
    DAP_LOG(log, "({0}) error: unhandled command '{1}'",
            transport.GetClientName(), req->command);
    return false; // Fail
  }

  if (const auto *resp = std::get_if<Response>(&M)) {
    std::unique_ptr<ResponseHandler> response_handler;
    {
      std::lock_guard<std::mutex> guard(call_mutex);
      auto inflight = inflight_reverse_requests.find(resp->request_seq);
      if (inflight != inflight_reverse_requests.end()) {
        response_handler = std::move(inflight->second);
        inflight_reverse_requests.erase(inflight);
      }
    }

    if (!response_handler)
      response_handler =
          std::make_unique<UnknownResponseHandler>("", resp->request_seq);

    // Result should be given, use null if not.
    if (resp->success) {
      (*response_handler)(resp->body);
      dispatcher.Set("client_data",
                     llvm::Twine("response_command:", resp->command).str());
    } else {
      llvm::StringRef message = "Unknown error, response failed";
      if (resp->message) {
        message =
            std::visit(llvm::makeVisitor(
                           [](const std::string &message) -> llvm::StringRef {
                             return message;
                           },
                           [](const protocol::ResponseMessage &message)
                               -> llvm::StringRef {
                             switch (message) {
                             case protocol::eResponseMessageCancelled:
                               return "cancelled";
                             case protocol::eResponseMessageNotStopped:
                               return "notStopped";
                             }
                             llvm_unreachable("unknown response message kind.");
                           }),
                       *resp->message);
      }
      dispatcher.Set("error", message.str());

      (*response_handler)(llvm::createStringError(
          std::error_code(-1, std::generic_category()), message));
    }

    return true;
  }

  dispatcher.Set("error", "Unsupported protocol message");
  DAP_LOG(log, "Unsupported protocol message");

  return false;
}

void DAP::SendTerminatedEvent() {
  // Prevent races if the process exits while we're being asked to disconnect.
  llvm::call_once(terminated_event_flag, [&] {
    RunTerminateCommands();
    // Send a "terminated" event
    llvm::json::Object event(CreateTerminatedEventObject(target));
    SendJSON(llvm::json::Value(std::move(event)));
  });
}

llvm::Error DAP::Disconnect() { return Disconnect(is_attach); }

llvm::Error DAP::Disconnect(bool terminateDebuggee) {
  lldb::SBError error;
  lldb::SBProcess process = target.GetProcess();
  auto state = process.GetState();
  switch (state) {
  case lldb::eStateInvalid:
  case lldb::eStateUnloaded:
  case lldb::eStateDetached:
  case lldb::eStateExited:
    break;
  case lldb::eStateConnected:
  case lldb::eStateAttaching:
  case lldb::eStateLaunching:
  case lldb::eStateStepping:
  case lldb::eStateCrashed:
  case lldb::eStateSuspended:
  case lldb::eStateStopped:
  case lldb::eStateRunning: {
    ScopeSyncMode scope_sync_mode(debugger);
    error = terminateDebuggee ? process.Kill() : process.Detach();
    break;
  }
  }

  SendTerminatedEvent();

  disconnecting = true;

  return ToError(error);
}

bool DAP::IsCancelled(const protocol::Request &req) {
  std::lock_guard<std::mutex> guard(m_cancelled_requests_mutex);
  return m_cancelled_requests.contains(req.seq);
}

void DAP::ClearCancelRequest(const CancelArguments &args) {
  std::lock_guard<std::mutex> guard(m_cancelled_requests_mutex);
  if (args.requestId)
    m_cancelled_requests.erase(*args.requestId);
}

template <typename T>
static std::optional<T> getArgumentsIfRequest(const Message &pm,
                                              llvm::StringLiteral command) {
  auto *const req = std::get_if<Request>(&pm);
  if (!req || req->command != command)
    return std::nullopt;

  T args;
  llvm::json::Path::Root root;
  if (!fromJSON(req->arguments, args, root)) {
    return std::nullopt;
  }

  return std::move(args);
}

llvm::Error DAP::Loop() {
  // Can't use \a std::future<llvm::Error> because it doesn't compile on
  // Windows.
  std::future<lldb::SBError> queue_reader =
      std::async(std::launch::async, [&]() -> lldb::SBError {
        llvm::set_thread_name(transport.GetClientName() + ".transport_handler");
        auto cleanup = llvm::make_scope_exit([&]() {
          // Ensure we're marked as disconnecting when the reader exits.
          disconnecting = true;
          m_queue_cv.notify_all();
        });

        while (!disconnecting) {
          llvm::Expected<Message> next =
              transport.Read(std::chrono::seconds(1));
          if (next.errorIsA<EndOfFileError>()) {
            consumeError(next.takeError());
            break;
          }

          // If the read timed out, continue to check if we should disconnect.
          if (next.errorIsA<TimeoutError>()) {
            consumeError(next.takeError());
            continue;
          }

          if (llvm::Error err = next.takeError()) {
            lldb::SBError errWrapper;
            errWrapper.SetErrorString(llvm::toString(std::move(err)).c_str());
            return errWrapper;
          }

          if (const protocol::Request *req =
                  std::get_if<protocol::Request>(&*next);
              req && req->arguments == "disconnect")
            disconnecting = true;

          const std::optional<CancelArguments> cancel_args =
              getArgumentsIfRequest<CancelArguments>(*next, "cancel");
          if (cancel_args) {
            {
              std::lock_guard<std::mutex> guard(m_cancelled_requests_mutex);
              if (cancel_args->requestId)
                m_cancelled_requests.insert(*cancel_args->requestId);
            }

            // If a cancel is requested for the active request, make a best
            // effort attempt to interrupt.
            std::lock_guard<std::mutex> guard(m_active_request_mutex);
            if (m_active_request &&
                cancel_args->requestId == m_active_request->seq) {
              DAP_LOG(
                  log,
                  "({0}) interrupting inflight request (command={1} seq={2})",
                  transport.GetClientName(), m_active_request->command,
                  m_active_request->seq);
              debugger.RequestInterrupt();
            }
          }

          {
            std::lock_guard<std::mutex> guard(m_queue_mutex);
            m_queue.push_back(std::move(*next));
          }
          m_queue_cv.notify_one();
        }

        return lldb::SBError();
      });

  auto cleanup = llvm::make_scope_exit([&]() {
    out.Stop();
    err.Stop();
    StopEventHandlers();
  });

  while (true) {
    std::unique_lock<std::mutex> lock(m_queue_mutex);
    m_queue_cv.wait(lock, [&] { return disconnecting || !m_queue.empty(); });

    if (disconnecting && m_queue.empty())
      break;

    Message next = m_queue.front();
    m_queue.pop_front();

    // Unlock while we're processing the event.
    lock.unlock();

    if (!HandleObject(next))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unhandled packet");
  }

  return ToError(queue_reader.get());
}

lldb::SBError DAP::WaitForProcessToStop(std::chrono::seconds seconds) {
  lldb::SBError error;
  lldb::SBProcess process = target.GetProcess();
  if (!process.IsValid()) {
    error.SetErrorString("invalid process");
    return error;
  }
  auto timeout_time =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  while (std::chrono::steady_clock::now() < timeout_time) {
    const auto state = process.GetState();
    switch (state) {
    case lldb::eStateUnloaded:
    case lldb::eStateAttaching:
    case lldb::eStateConnected:
    case lldb::eStateInvalid:
    case lldb::eStateLaunching:
    case lldb::eStateRunning:
    case lldb::eStateStepping:
    case lldb::eStateSuspended:
      break;
    case lldb::eStateDetached:
      error.SetErrorString("process detached during launch or attach");
      return error;
    case lldb::eStateExited:
      error.SetErrorString("process exited during launch or attach");
      return error;
    case lldb::eStateCrashed:
    case lldb::eStateStopped:
      return lldb::SBError(); // Success!
    }
    std::this_thread::sleep_for(std::chrono::microseconds(250));
  }
  error.SetErrorString(
      llvm::formatv("process failed to stop within {0}", seconds)
          .str()
          .c_str());
  return error;
}

void Variables::Clear() {
  locals.Clear();
  globals.Clear();
  registers.Clear();
  referenced_variables.clear();
}

int64_t Variables::GetNewVariableReference(bool is_permanent) {
  if (is_permanent)
    return next_permanent_var_ref++;
  return next_temporary_var_ref++;
}

bool Variables::IsPermanentVariableReference(int64_t var_ref) {
  return var_ref >= PermanentVariableStartIndex;
}

lldb::SBValue Variables::GetVariable(int64_t var_ref) const {
  if (IsPermanentVariableReference(var_ref)) {
    auto pos = referenced_permanent_variables.find(var_ref);
    if (pos != referenced_permanent_variables.end())
      return pos->second;
  } else {
    auto pos = referenced_variables.find(var_ref);
    if (pos != referenced_variables.end())
      return pos->second;
  }
  return lldb::SBValue();
}

int64_t Variables::InsertVariable(lldb::SBValue variable, bool is_permanent) {
  int64_t var_ref = GetNewVariableReference(is_permanent);
  if (is_permanent)
    referenced_permanent_variables.insert(std::make_pair(var_ref, variable));
  else
    referenced_variables.insert(std::make_pair(var_ref, variable));
  return var_ref;
}

bool StartDebuggingRequestHandler::DoExecute(
    lldb::SBDebugger debugger, char **command,
    lldb::SBCommandReturnObject &result) {
  // Command format like: `start-debugging <launch|attach> <configuration>`
  if (!command) {
    result.SetError("Invalid use of start-debugging, expected format "
                    "`start-debugging <launch|attach> <configuration>`.");
    return false;
  }

  if (!command[0] || llvm::StringRef(command[0]).empty()) {
    result.SetError("start-debugging request type missing.");
    return false;
  }

  if (!command[1] || llvm::StringRef(command[1]).empty()) {
    result.SetError("start-debugging debug configuration missing.");
    return false;
  }

  llvm::StringRef request{command[0]};
  std::string raw_configuration{command[1]};

  llvm::Expected<llvm::json::Value> configuration =
      llvm::json::parse(raw_configuration);

  if (!configuration) {
    llvm::Error err = configuration.takeError();
    std::string msg = "Failed to parse json configuration: " +
                      llvm::toString(std::move(err)) + "\n\n" +
                      raw_configuration;
    result.SetError(msg.c_str());
    return false;
  }

  dap.SendReverseRequest<LogFailureResponseHandler>(
      "startDebugging",
      llvm::json::Object{{"request", request},
                         {"configuration", std::move(*configuration)}});

  result.SetStatus(lldb::eReturnStatusSuccessFinishNoResult);

  return true;
}

bool ReplModeRequestHandler::DoExecute(lldb::SBDebugger debugger,
                                       char **command,
                                       lldb::SBCommandReturnObject &result) {
  // Command format like: `repl-mode <variable|command|auto>?`
  // If a new mode is not specified report the current mode.
  if (!command || llvm::StringRef(command[0]).empty()) {
    std::string mode;
    switch (dap.repl_mode) {
    case ReplMode::Variable:
      mode = "variable";
      break;
    case ReplMode::Command:
      mode = "command";
      break;
    case ReplMode::Auto:
      mode = "auto";
      break;
    }

    result.Printf("lldb-dap repl-mode %s.\n", mode.c_str());
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);

    return true;
  }

  llvm::StringRef new_mode{command[0]};

  if (new_mode == "variable") {
    dap.repl_mode = ReplMode::Variable;
  } else if (new_mode == "command") {
    dap.repl_mode = ReplMode::Command;
  } else if (new_mode == "auto") {
    dap.repl_mode = ReplMode::Auto;
  } else {
    lldb::SBStream error_message;
    error_message.Printf("Invalid repl-mode '%s'. Expected one of 'variable', "
                         "'command' or 'auto'.\n",
                         new_mode.data());
    result.SetError(error_message.GetData());
    return false;
  }

  result.Printf("lldb-dap repl-mode %s set.\n", new_mode.data());
  result.SetStatus(lldb::eReturnStatusSuccessFinishNoResult);
  return true;
}

// Sends a DAP event with an optional body.
//
// See
// https://code.visualstudio.com/api/references/vscode-api#debug.onDidReceiveDebugSessionCustomEvent
bool SendEventRequestHandler::DoExecute(lldb::SBDebugger debugger,
                                        char **command,
                                        lldb::SBCommandReturnObject &result) {
  // Command format like: `send-event <name> <body>?`
  if (!command || !command[0] || llvm::StringRef(command[0]).empty()) {
    result.SetError("Not enough arguments found, expected format "
                    "`lldb-dap send-event <name> <body>?`.");
    return false;
  }

  llvm::StringRef name{command[0]};
  // Events that are stateful and should be handled by lldb-dap internally.
  const std::array internal_events{"breakpoint", "capabilities", "continued",
                                   "exited",     "initialize",   "loadedSource",
                                   "module",     "process",      "stopped",
                                   "terminated", "thread"};
  if (llvm::is_contained(internal_events, name)) {
    std::string msg =
        llvm::formatv("Invalid use of lldb-dap send-event, event \"{0}\" "
                      "should be handled by lldb-dap internally.",
                      name)
            .str();
    result.SetError(msg.c_str());
    return false;
  }

  llvm::json::Object event(CreateEventObject(name));

  if (command[1] && !llvm::StringRef(command[1]).empty()) {
    // See if we have unused arguments.
    if (command[2]) {
      result.SetError(
          "Additional arguments found, expected `lldb-dap send-event "
          "<name> <body>?`.");
      return false;
    }

    llvm::StringRef raw_body{command[1]};

    llvm::Expected<llvm::json::Value> body = llvm::json::parse(raw_body);

    if (!body) {
      llvm::Error err = body.takeError();
      std::string msg = "Failed to parse custom event body: " +
                        llvm::toString(std::move(err));
      result.SetError(msg.c_str());
      return false;
    }

    event.try_emplace("body", std::move(*body));
  }

  dap.SendJSON(llvm::json::Value(std::move(event)));
  result.SetStatus(lldb::eReturnStatusSuccessFinishNoResult);
  return true;
}

void DAP::ConfigureSourceMaps() {
  if (configuration.sourceMap.empty() && configuration.sourcePath.empty())
    return;

  std::string sourceMapCommand;
  llvm::raw_string_ostream strm(sourceMapCommand);
  strm << "settings set target.source-map ";

  if (!configuration.sourceMap.empty()) {
    for (const auto &kv : configuration.sourceMap) {
      strm << "\"" << kv.first << "\" \"" << kv.second << "\" ";
    }
  } else if (!configuration.sourcePath.empty()) {
    strm << "\".\" \"" << configuration.sourcePath << "\"";
  }

  RunLLDBCommands("Setting source map:", {sourceMapCommand});
}

void DAP::SetConfiguration(const protocol::Configuration &config,
                           bool is_attach) {
  configuration = config;
  stop_at_entry = config.stopOnEntry;
  this->is_attach = is_attach;

  if (configuration.customFrameFormat)
    SetFrameFormat(*configuration.customFrameFormat);
  if (configuration.customThreadFormat)
    SetThreadFormat(*configuration.customThreadFormat);
}

void DAP::SetFrameFormat(llvm::StringRef format) {
  lldb::SBError error;
  frame_format = lldb::SBFormat(format.str().c_str(), error);
  if (error.Fail()) {
    SendOutput(OutputType::Console,
               llvm::formatv(
                   "The provided frame format '{0}' couldn't be parsed: {1}\n",
                   format, error.GetCString())
                   .str());
  }
}

void DAP::SetThreadFormat(llvm::StringRef format) {
  lldb::SBError error;
  thread_format = lldb::SBFormat(format.str().c_str(), error);
  if (error.Fail()) {
    SendOutput(OutputType::Console,
               llvm::formatv(
                   "The provided thread format '{0}' couldn't be parsed: {1}\n",
                   format, error.GetCString())
                   .str());
  }
}

InstructionBreakpoint *
DAP::GetInstructionBreakpoint(const lldb::break_id_t bp_id) {
  for (auto &bp : instruction_breakpoints) {
    if (bp.second.GetID() == bp_id)
      return &bp.second;
  }
  return nullptr;
}

InstructionBreakpoint *
DAP::GetInstructionBPFromStopReason(lldb::SBThread &thread) {
  const auto num = thread.GetStopReasonDataCount();
  InstructionBreakpoint *inst_bp = nullptr;
  for (size_t i = 0; i < num; i += 2) {
    // thread.GetStopReasonDataAtIndex(i) will return the bp ID and
    // thread.GetStopReasonDataAtIndex(i+1) will return the location
    // within that breakpoint. We only care about the bp ID so we can
    // see if this is an instruction breakpoint that is getting hit.
    lldb::break_id_t bp_id = thread.GetStopReasonDataAtIndex(i);
    inst_bp = GetInstructionBreakpoint(bp_id);
    // If any breakpoint is not an instruction breakpoint, then stop and
    // report this as a normal breakpoint
    if (inst_bp == nullptr)
      return nullptr;
  }
  return inst_bp;
}

lldb::SBValueList *Variables::GetTopLevelScope(int64_t variablesReference) {
  switch (variablesReference) {
  case VARREF_LOCALS:
    return &locals;
  case VARREF_GLOBALS:
    return &globals;
  case VARREF_REGS:
    return &registers;
  default:
    return nullptr;
  }
}

lldb::SBValue Variables::FindVariable(uint64_t variablesReference,
                                      llvm::StringRef name) {
  lldb::SBValue variable;
  if (lldb::SBValueList *top_scope = GetTopLevelScope(variablesReference)) {
    bool is_duplicated_variable_name = name.contains(" @");
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for a variable in locals or globals or registers
    int64_t end_idx = top_scope->GetSize();
    // Searching backward so that we choose the variable in closest scope
    // among variables of the same name.
    for (int64_t i = end_idx - 1; i >= 0; --i) {
      lldb::SBValue curr_variable = top_scope->GetValueAtIndex(i);
      std::string variable_name = CreateUniqueVariableNameForDisplay(
          curr_variable, is_duplicated_variable_name);
      if (variable_name == name) {
        variable = curr_variable;
        break;
      }
    }
  } else {
    // This is not under the globals or locals scope, so there are no
    // duplicated names.

    // We have a named item within an actual variable so we need to find it
    // withing the container variable by name.
    lldb::SBValue container = GetVariable(variablesReference);
    variable = container.GetChildMemberWithName(name.data());
    if (!variable.IsValid()) {
      if (name.starts_with("[")) {
        llvm::StringRef index_str(name.drop_front(1));
        uint64_t index = 0;
        if (!index_str.consumeInteger(0, index)) {
          if (index_str == "]")
            variable = container.GetChildAtIndex(index);
        }
      }
    }
  }
  return variable;
}

protocol::Capabilities DAP::GetCapabilities() {
  protocol::Capabilities capabilities;

  // Supported capabilities that are not specific to a single request.
  capabilities.supportedFeatures = {
      protocol::eAdapterFeatureLogPoints,
      protocol::eAdapterFeatureSteppingGranularity,
      protocol::eAdapterFeatureValueFormattingOptions,
  };

  // Capabilities associated with specific requests.
  for (auto &kv : request_handlers) {
    llvm::SmallDenseSet<AdapterFeature, 1> features =
        kv.second->GetSupportedFeatures();
    capabilities.supportedFeatures.insert(features.begin(), features.end());
  }

  // Available filters or options for the setExceptionBreakpoints request.
  std::vector<protocol::ExceptionBreakpointsFilter> filters;
  for (const auto &exc_bp : *exception_breakpoints)
    filters.emplace_back(CreateExceptionBreakpointFilter(exc_bp));
  capabilities.exceptionBreakpointFilters = std::move(filters);

  // FIXME: This should be registered based on the supported languages?
  std::vector<std::string> completion_characters;
  completion_characters.emplace_back(".");
  // FIXME: I wonder if we should remove this key... its very aggressive
  // triggering and accepting completions.
  completion_characters.emplace_back(" ");
  completion_characters.emplace_back("\t");
  capabilities.completionTriggerCharacters = std::move(completion_characters);

  // Put in non-DAP specification lldb specific information.
  capabilities.lldbExtVersion = debugger.GetVersionString();

  return capabilities;
}

void DAP::StartEventThread() {
  event_thread = std::thread(&DAP::EventThread, this);
}

void DAP::StartProgressEventThread() {
  progress_event_thread = std::thread(&DAP::ProgressEventThread, this);
}

void DAP::ProgressEventThread() {
  lldb::SBListener listener("lldb-dap.progress.listener");
  debugger.GetBroadcaster().AddListener(
      listener, lldb::SBDebugger::eBroadcastBitProgress |
                    lldb::SBDebugger::eBroadcastBitExternalProgress);
  broadcaster.AddListener(listener, eBroadcastBitStopProgressThread);
  lldb::SBEvent event;
  bool done = false;
  while (!done) {
    if (listener.WaitForEvent(1, event)) {
      const auto event_mask = event.GetType();
      if (event.BroadcasterMatchesRef(broadcaster)) {
        if (event_mask & eBroadcastBitStopProgressThread) {
          done = true;
        }
      } else {
        lldb::SBStructuredData data =
            lldb::SBDebugger::GetProgressDataFromEvent(event);

        const uint64_t progress_id =
            GetUintFromStructuredData(data, "progress_id");
        const uint64_t completed = GetUintFromStructuredData(data, "completed");
        const uint64_t total = GetUintFromStructuredData(data, "total");
        const std::string details =
            GetStringFromStructuredData(data, "details");

        if (completed == 0) {
          if (total == UINT64_MAX) {
            // This progress is non deterministic and won't get updated until it
            // is completed. Send the "message" which will be the combined title
            // and detail. The only other progress event for thus
            // non-deterministic progress will be the completed event So there
            // will be no need to update the detail.
            const std::string message =
                GetStringFromStructuredData(data, "message");
            SendProgressEvent(progress_id, message.c_str(), completed, total);
          } else {
            // This progress is deterministic and will receive updates,
            // on the progress creation event VSCode will save the message in
            // the create packet and use that as the title, so we send just the
            // title in the progressCreate packet followed immediately by a
            // detail packet, if there is any detail.
            const std::string title =
                GetStringFromStructuredData(data, "title");
            SendProgressEvent(progress_id, title.c_str(), completed, total);
            if (!details.empty())
              SendProgressEvent(progress_id, details.c_str(), completed, total);
          }
        } else {
          // This progress event is either the end of the progress dialog, or an
          // update with possible detail. The "detail" string we send to VS Code
          // will be appended to the progress dialog's initial text from when it
          // was created.
          SendProgressEvent(progress_id, details.c_str(), completed, total);
        }
      }
    }
  }
}

// All events from the debugger, target, process, thread and frames are
// received in this function that runs in its own thread. We are using a
// "FILE *" to output packets back to VS Code and they have mutexes in them
// them prevent multiple threads from writing simultaneously so no locking
// is required.
void DAP::EventThread() {
  llvm::set_thread_name(transport.GetClientName() + ".event_handler");
  lldb::SBEvent event;
  lldb::SBListener listener = debugger.GetListener();
  broadcaster.AddListener(listener, eBroadcastBitStopEventThread);
  debugger.GetBroadcaster().AddListener(
      listener, lldb::eBroadcastBitError | lldb::eBroadcastBitWarning);
  bool done = false;
  while (!done) {
    if (listener.WaitForEvent(1, event)) {
      const auto event_mask = event.GetType();
      if (lldb::SBProcess::EventIsProcessEvent(event)) {
        lldb::SBProcess process = lldb::SBProcess::GetProcessFromEvent(event);
        if (event_mask & lldb::SBProcess::eBroadcastBitStateChanged) {
          auto state = lldb::SBProcess::GetStateFromEvent(event);
          switch (state) {
          case lldb::eStateConnected:
          case lldb::eStateDetached:
          case lldb::eStateInvalid:
          case lldb::eStateUnloaded:
            break;
          case lldb::eStateAttaching:
          case lldb::eStateCrashed:
          case lldb::eStateLaunching:
          case lldb::eStateStopped:
          case lldb::eStateSuspended:
            // Only report a stopped event if the process was not
            // automatically restarted.
            if (!lldb::SBProcess::GetRestartedFromEvent(event)) {
              SendStdOutStdErr(*this, process);
              SendThreadStoppedEvent(*this);
            }
            break;
          case lldb::eStateRunning:
          case lldb::eStateStepping:
            WillContinue();
            SendContinuedEvent(*this);
            break;
          case lldb::eStateExited:
            lldb::SBStream stream;
            process.GetStatus(stream);
            SendOutput(OutputType::Console, stream.GetData());

            // When restarting, we can get an "exited" event for the process we
            // just killed with the old PID, or even with no PID. In that case
            // we don't have to terminate the session.
            if (process.GetProcessID() == LLDB_INVALID_PROCESS_ID ||
                process.GetProcessID() == restarting_process_id) {
              restarting_process_id = LLDB_INVALID_PROCESS_ID;
            } else {
              // Run any exit LLDB commands the user specified in the
              // launch.json
              RunExitCommands();
              SendProcessExitedEvent(*this, process);
              SendTerminatedEvent();
              done = true;
            }
            break;
          }
        } else if ((event_mask & lldb::SBProcess::eBroadcastBitSTDOUT) ||
                   (event_mask & lldb::SBProcess::eBroadcastBitSTDERR)) {
          SendStdOutStdErr(*this, process);
        }
      } else if (lldb::SBTarget::EventIsTargetEvent(event)) {
        if (event_mask & lldb::SBTarget::eBroadcastBitModulesLoaded ||
            event_mask & lldb::SBTarget::eBroadcastBitModulesUnloaded ||
            event_mask & lldb::SBTarget::eBroadcastBitSymbolsLoaded ||
            event_mask & lldb::SBTarget::eBroadcastBitSymbolsChanged) {
          const uint32_t num_modules =
              lldb::SBTarget::GetNumModulesFromEvent(event);
          std::lock_guard<std::mutex> guard(modules_mutex);
          for (uint32_t i = 0; i < num_modules; ++i) {
            lldb::SBModule module =
                lldb::SBTarget::GetModuleAtIndexFromEvent(i, event);
            if (!module.IsValid())
              continue;
            llvm::StringRef module_id = module.GetUUIDString();
            if (module_id.empty())
              continue;

            llvm::StringRef reason;
            bool id_only = false;
            if (event_mask & lldb::SBTarget::eBroadcastBitModulesLoaded) {
              modules.insert(module_id);
              reason = "new";
            } else {
              // If this is a module we've never told the client about, don't
              // send an event.
              if (!modules.contains(module_id))
                continue;

              if (event_mask & lldb::SBTarget::eBroadcastBitModulesUnloaded) {
                modules.erase(module_id);
                reason = "removed";
                id_only = true;
              } else {
                reason = "changed";
              }
            }

            llvm::json::Object body;
            body.try_emplace("reason", reason);
            body.try_emplace("module", CreateModule(target, module, id_only));
            llvm::json::Object module_event = CreateEventObject("module");
            module_event.try_emplace("body", std::move(body));
            SendJSON(llvm::json::Value(std::move(module_event)));
          }
        }
      } else if (lldb::SBBreakpoint::EventIsBreakpointEvent(event)) {
        if (event_mask & lldb::SBTarget::eBroadcastBitBreakpointChanged) {
          auto event_type =
              lldb::SBBreakpoint::GetBreakpointEventTypeFromEvent(event);
          auto bp = Breakpoint(
              *this, lldb::SBBreakpoint::GetBreakpointFromEvent(event));
          // If the breakpoint was set through DAP, it will have the
          // BreakpointBase::kDAPBreakpointLabel. Regardless of whether
          // locations were added, removed, or resolved, the breakpoint isn't
          // going away and the reason is always "changed".
          if ((event_type & lldb::eBreakpointEventTypeLocationsAdded ||
               event_type & lldb::eBreakpointEventTypeLocationsRemoved ||
               event_type & lldb::eBreakpointEventTypeLocationsResolved) &&
              bp.MatchesName(BreakpointBase::kDAPBreakpointLabel)) {
            // As the DAP client already knows the path of this breakpoint, we
            // don't need to send it back as part of the "changed" event. This
            // avoids sending paths that should be source mapped. Note that
            // CreateBreakpoint doesn't apply source mapping and certain
            // implementation ignore the source part of this event anyway.
            llvm::json::Value source_bp = bp.ToProtocolBreakpoint();
            source_bp.getAsObject()->erase("source");

            llvm::json::Object body;
            body.try_emplace("breakpoint", source_bp);
            body.try_emplace("reason", "changed");

            llvm::json::Object bp_event = CreateEventObject("breakpoint");
            bp_event.try_emplace("body", std::move(body));

            SendJSON(llvm::json::Value(std::move(bp_event)));
          }
        }
      } else if (event_mask & lldb::eBroadcastBitError ||
                 event_mask & lldb::eBroadcastBitWarning) {
        lldb::SBStructuredData data =
            lldb::SBDebugger::GetDiagnosticFromEvent(event);
        if (!data.IsValid())
          continue;
        std::string type = GetStringValue(data.GetValueForKey("type"));
        std::string message = GetStringValue(data.GetValueForKey("message"));
        SendOutput(OutputType::Important,
                   llvm::formatv("{0}: {1}", type, message).str());
      } else if (event.BroadcasterMatchesRef(broadcaster)) {
        if (event_mask & eBroadcastBitStopEventThread) {
          done = true;
        }
      }
    }
  }
}

void DAP::RegisterRequests() {
  RegisterRequest<AttachRequestHandler>();
  RegisterRequest<BreakpointLocationsRequestHandler>();
  RegisterRequest<CancelRequestHandler>();
  RegisterRequest<CompletionsRequestHandler>();
  RegisterRequest<ConfigurationDoneRequestHandler>();
  RegisterRequest<ContinueRequestHandler>();
  RegisterRequest<DataBreakpointInfoRequestHandler>();
  RegisterRequest<DisassembleRequestHandler>();
  RegisterRequest<DisconnectRequestHandler>();
  RegisterRequest<EvaluateRequestHandler>();
  RegisterRequest<ExceptionInfoRequestHandler>();
  RegisterRequest<InitializeRequestHandler>();
  RegisterRequest<LaunchRequestHandler>();
  RegisterRequest<LocationsRequestHandler>();
  RegisterRequest<NextRequestHandler>();
  RegisterRequest<PauseRequestHandler>();
  RegisterRequest<ReadMemoryRequestHandler>();
  RegisterRequest<RestartRequestHandler>();
  RegisterRequest<ScopesRequestHandler>();
  RegisterRequest<SetBreakpointsRequestHandler>();
  RegisterRequest<SetDataBreakpointsRequestHandler>();
  RegisterRequest<SetExceptionBreakpointsRequestHandler>();
  RegisterRequest<SetFunctionBreakpointsRequestHandler>();
  RegisterRequest<SetInstructionBreakpointsRequestHandler>();
  RegisterRequest<SetVariableRequestHandler>();
  RegisterRequest<SourceRequestHandler>();
  RegisterRequest<StackTraceRequestHandler>();
  RegisterRequest<StepInRequestHandler>();
  RegisterRequest<StepInTargetsRequestHandler>();
  RegisterRequest<StepOutRequestHandler>();
  RegisterRequest<ThreadsRequestHandler>();
  RegisterRequest<VariablesRequestHandler>();

  // Custom requests
  RegisterRequest<CompileUnitsRequestHandler>();
  RegisterRequest<ModulesRequestHandler>();

  // Testing requests
  RegisterRequest<TestGetTargetBreakpointsRequestHandler>();
}

} // namespace lldb_dap
