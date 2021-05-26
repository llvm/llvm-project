//===-- VSCode.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstdarg>
#include <fstream>
#include <mutex>
#include <sstream>

#include "LLDBUtils.h"
#include "VSCode.h"
#include "llvm/Support/FormatVariadic.h"

#if defined(_WIN32)
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

using namespace lldb_vscode;

namespace lldb_vscode {

VSCode g_vsc;

VSCode::VSCode()
    : variables(), broadcaster("lldb-vscode"), num_regs(0), num_locals(0),
      num_globals(0), log(),
      exception_breakpoints(
          {{"cpp_catch", "C++ Catch", lldb::eLanguageTypeC_plus_plus},
           {"cpp_throw", "C++ Throw", lldb::eLanguageTypeC_plus_plus},
           {"objc_catch", "Objective C Catch", lldb::eLanguageTypeObjC},
           {"objc_throw", "Objective C Throw", lldb::eLanguageTypeObjC},
           {"swift_catch", "Swift Catch", lldb::eLanguageTypeSwift},
           {"swift_throw", "Swift Throw", lldb::eLanguageTypeSwift}}),
      focus_tid(LLDB_INVALID_THREAD_ID), sent_terminated_event(false),
      stop_at_entry(false), is_attach(false), reverse_request_seq(0),
      waiting_for_run_in_terminal(false),
      progress_event_queue(
          [&](const ProgressEvent &event) { SendJSON(event.ToJSON()); }) {
  const char *log_file_path = getenv("LLDBVSCODE_LOG");
#if defined(_WIN32)
  // Windows opens stdout and stdin in text mode which converts \n to 13,10
  // while the value is just 10 on Darwin/Linux. Setting the file mode to binary
  // fixes this.
  int result = _setmode(fileno(stdout), _O_BINARY);
  assert(result);
  result = _setmode(fileno(stdin), _O_BINARY);
  (void)result;
  assert(result);
#endif
  if (log_file_path)
    log.reset(new std::ofstream(log_file_path));
}

VSCode::~VSCode() {}

int64_t VSCode::GetLineForPC(int64_t sourceReference, lldb::addr_t pc) const {
  auto pos = source_map.find(sourceReference);
  if (pos != source_map.end())
    return pos->second.GetLineForPC(pc);
  return 0;
}

ExceptionBreakpoint *VSCode::GetExceptionBreakpoint(const std::string &filter) {
  for (auto &bp : exception_breakpoints) {
    if (bp.filter == filter)
      return &bp;
  }
  return nullptr;
}

ExceptionBreakpoint *
VSCode::GetExceptionBreakpoint(const lldb::break_id_t bp_id) {
  for (auto &bp : exception_breakpoints) {
    if (bp.bp.GetID() == bp_id)
      return &bp;
  }
  return nullptr;
}

// Send the JSON in "json_str" to the "out" stream. Correctly send the
// "Content-Length:" field followed by the length, followed by the raw
// JSON bytes.
void VSCode::SendJSON(const std::string &json_str) {
  output.write_full("Content-Length: ");
  output.write_full(llvm::utostr(json_str.size()));
  output.write_full("\r\n\r\n");
  output.write_full(json_str);

  if (log) {
    *log << "<-- " << std::endl
         << "Content-Length: " << json_str.size() << "\r\n\r\n"
         << json_str << std::endl;
  }
}

// Serialize the JSON value into a string and send the JSON packet to
// the "out" stream.
void VSCode::SendJSON(const llvm::json::Value &json) {
  std::string s;
  llvm::raw_string_ostream strm(s);
  strm << json;
  static std::mutex mutex;
  std::lock_guard<std::mutex> locker(mutex);
  SendJSON(strm.str());
}

// Read a JSON packet from the "in" stream.
std::string VSCode::ReadJSON() {
  std::string length_str;
  std::string json_str;
  int length;

  if (!input.read_expected(log.get(), "Content-Length: "))
    return json_str;

  if (!input.read_line(log.get(), length_str))
    return json_str;

  if (!llvm::to_integer(length_str, length))
    return json_str;

  if (!input.read_expected(log.get(), "\r\n"))
    return json_str;

  if (!input.read_full(log.get(), length, json_str))
    return json_str;

  if (log) {
    *log << "--> " << std::endl
         << "Content-Length: " << length << "\r\n\r\n"
         << json_str << std::endl;
  }

  return json_str;
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
void VSCode::SendOutput(OutputType o, const llvm::StringRef output) {
  if (output.empty())
    return;

  llvm::json::Object event(CreateEventObject("output"));
  llvm::json::Object body;
  const char *category = nullptr;
  switch (o) {
  case OutputType::Console:
    category = "console";
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
  body.try_emplace("category", category);
  EmplaceSafeString(body, "output", output.str());
  event.try_emplace("body", std::move(body));
  SendJSON(llvm::json::Value(std::move(event)));
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

void VSCode::SendProgressEvent(const ProgressEvent &event) {
  progress_event_queue.Push(event);
}

void __attribute__((format(printf, 3, 4)))
VSCode::SendFormattedOutput(OutputType o, const char *format, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, format);
  int actual_length = vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  SendOutput(
      o, llvm::StringRef(buffer, std::min<int>(actual_length, sizeof(buffer))));
}

int64_t VSCode::GetNextSourceReference() {
  static int64_t ref = 0;
  return ++ref;
}

ExceptionBreakpoint *
VSCode::GetExceptionBPFromStopReason(lldb::SBThread &thread) {
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

lldb::SBThread VSCode::GetLLDBThread(const llvm::json::Object &arguments) {
  auto tid = GetSigned(arguments, "threadId", LLDB_INVALID_THREAD_ID);
  return target.GetProcess().GetThreadByID(tid);
}

lldb::SBFrame VSCode::GetLLDBFrame(const llvm::json::Object &arguments) {
  const uint64_t frame_id = GetUnsigned(arguments, "frameId", UINT64_MAX);
  lldb::SBProcess process = target.GetProcess();
  // Upper 32 bits is the thread index ID
  lldb::SBThread thread =
      process.GetThreadByIndexID(GetLLDBThreadIndexID(frame_id));
  // Lower 32 bits is the frame index
  return thread.GetFrameAtIndex(GetLLDBFrameID(frame_id));
}

llvm::json::Value VSCode::CreateTopLevelScopes() {
  llvm::json::Array scopes;
  scopes.emplace_back(CreateScope("Locals", VARREF_LOCALS, num_locals, false));
  scopes.emplace_back(
      CreateScope("Globals", VARREF_GLOBALS, num_globals, false));
  scopes.emplace_back(CreateScope("Registers", VARREF_REGS, num_regs, false));
  return llvm::json::Value(std::move(scopes));
}

void VSCode::RunLLDBCommands(llvm::StringRef prefix,
                             const std::vector<std::string> &commands) {
  SendOutput(OutputType::Console,
             llvm::StringRef(::RunLLDBCommands(prefix, commands)));
}

void VSCode::RunInitCommands() {
  RunLLDBCommands("Running initCommands:", init_commands);
}

void VSCode::RunPreRunCommands() {
  RunLLDBCommands("Running preRunCommands:", pre_run_commands);
}

void VSCode::RunStopCommands() {
  RunLLDBCommands("Running stopCommands:", stop_commands);
}

void VSCode::RunExitCommands() {
  RunLLDBCommands("Running exitCommands:", exit_commands);
}

void VSCode::RunTerminateCommands() {
  RunLLDBCommands("Running terminateCommands:", terminate_commands);
}

lldb::SBTarget
VSCode::CreateTargetFromArguments(const llvm::json::Object &arguments,
                                  lldb::SBError &error) {
  // Grab the name of the program we need to debug and create a target using
  // the given program as an argument. Executable file can be a source of target
  // architecture and platform, if they differ from the host. Setting exe path
  // in launch info is useless because Target.Launch() will not change
  // architecture and platform, therefore they should be known at the target
  // creation. We also use target triple and platform from the launch
  // configuration, if given, since in some cases ELF file doesn't contain
  // enough information to determine correct arch and platform (or ELF can be
  // omitted at all), so it is good to leave the user an apportunity to specify
  // those. Any of those three can be left empty.
  llvm::StringRef target_triple = GetString(arguments, "targetTriple");
  llvm::StringRef platform_name = GetString(arguments, "platformName");
  llvm::StringRef program = GetString(arguments, "program");
  auto target = this->debugger.CreateTarget(
      program.data(), target_triple.data(), platform_name.data(),
      true, // Add dependent modules.
      error);

  if (error.Fail()) {
    // Update message if there was an error.
    error.SetErrorStringWithFormat(
        "Could not create a target for a program '%s': %s.", program.data(),
        error.GetCString());
  }

  return target;
}

void VSCode::SetTarget(const lldb::SBTarget target) {
  this->target = target;

  if (target.IsValid()) {
    // Configure breakpoint event listeners for the target.
    lldb::SBListener listener = this->debugger.GetListener();
    listener.StartListeningForEvents(
        this->target.GetBroadcaster(),
        lldb::SBTarget::eBroadcastBitBreakpointChanged);
    listener.StartListeningForEvents(this->broadcaster,
                                     eBroadcastBitStopEventThread);
  }
}

PacketStatus VSCode::GetNextObject(llvm::json::Object &object) {
  std::string json = ReadJSON();
  if (json.empty())
    return PacketStatus::EndOfFile;

  llvm::StringRef json_sref(json);
  llvm::Expected<llvm::json::Value> json_value = llvm::json::parse(json_sref);
  if (!json_value) {
    auto error = json_value.takeError();
    if (log) {
      std::string error_str;
      llvm::raw_string_ostream strm(error_str);
      strm << error;
      strm.flush();
      *log << "error: failed to parse JSON: " << error_str << std::endl
           << json << std::endl;
    }
    return PacketStatus::JSONMalformed;
  }
  object = *json_value->getAsObject();
  if (!json_value->getAsObject()) {
    if (log)
      *log << "error: json packet isn't a object" << std::endl;
    return PacketStatus::JSONNotObject;
  }
  return PacketStatus::Success;
}

bool VSCode::HandleObject(const llvm::json::Object &object) {
  const auto packet_type = GetString(object, "type");
  if (packet_type == "request") {
    const auto command = GetString(object, "command");
    auto handler_pos = request_handlers.find(std::string(command));
    if (handler_pos != request_handlers.end()) {
      handler_pos->second(object);
      return true; // Success
    } else {
      if (log)
        *log << "error: unhandled command \"" << command.data() << std::endl;
      return false; // Fail
    }
  }
  return false;
}

PacketStatus VSCode::SendReverseRequest(llvm::json::Object request,
                                        llvm::json::Object &response) {
  request.try_emplace("seq", ++reverse_request_seq);
  SendJSON(llvm::json::Value(std::move(request)));
  while (true) {
    PacketStatus status = GetNextObject(response);
    const auto packet_type = GetString(response, "type");
    if (packet_type == "response")
      return status;
    else {
      // Not our response, we got another packet
      HandleObject(response);
    }
  }
  return PacketStatus::EndOfFile;
}

void VSCode::RegisterRequestCallback(std::string request,
                                     RequestCallback callback) {
  request_handlers[request] = callback;
}

} // namespace lldb_vscode
