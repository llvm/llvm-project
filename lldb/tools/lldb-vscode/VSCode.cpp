//===-- VSCode.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdarg.h>
#include <fstream>
#include <mutex>

#include "LLDBUtils.h"
#include "VSCode.h"
#include "llvm/Support/FormatVariadic.h"

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#endif

using namespace lldb_vscode;

namespace {
  inline bool IsEmptyLine(llvm::StringRef S) {
    return S.ltrim().empty();
  }
} // namespace

namespace lldb_vscode {

VSCode g_vsc;

VSCode::VSCode()
    : launch_info(nullptr), variables(), broadcaster("lldb-vscode"),
      num_regs(0), num_locals(0), num_globals(0), log(),
      exception_breakpoints(
          {{"cpp_catch", "C++ Catch", lldb::eLanguageTypeC_plus_plus},
           {"cpp_throw", "C++ Throw", lldb::eLanguageTypeC_plus_plus},
           {"objc_catch", "Objective C Catch", lldb::eLanguageTypeObjC},
           {"objc_throw", "Objective C Throw", lldb::eLanguageTypeObjC},
           {"swift_catch", "Swift Catch", lldb::eLanguageTypeSwift},
           {"swift_throw", "Swift Throw", lldb::eLanguageTypeSwift}}),
      focus_tid(LLDB_INVALID_THREAD_ID), sent_terminated_event(false),
      stop_at_entry(false) {
  const char *log_file_path = getenv("LLDBVSCODE_LOG");
#if defined(_WIN32)
// Windows opens stdout and stdin in text mode which converts \n to 13,10
// while the value is just 10 on Darwin/Linux. Setting the file mode to binary
// fixes this.
  assert(_setmode(fileno(stdout), _O_BINARY));
  assert(_setmode(fileno(stdin), _O_BINARY));
#endif
  if (log_file_path)
    log.reset(new std::ofstream(log_file_path));
}

VSCode::~VSCode() {
}

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

//----------------------------------------------------------------------
// Send the JSON in "json_str" to the "out" stream. Correctly send the
// "Content-Length:" field followed by the length, followed by the raw
// JSON bytes.
//----------------------------------------------------------------------
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

//----------------------------------------------------------------------
// Serialize the JSON value into a string and send the JSON packet to
// the "out" stream.
//----------------------------------------------------------------------
void VSCode::SendJSON(const llvm::json::Value &json) {
  std::string s;
  llvm::raw_string_ostream strm(s);
  strm << json;
  static std::mutex mutex;
  std::lock_guard<std::mutex> locker(mutex);
  SendJSON(strm.str());
}

//----------------------------------------------------------------------
// Read a JSON packet from the "in" stream.
//----------------------------------------------------------------------
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

  return json_str;
}

//----------------------------------------------------------------------
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
//----------------------------------------------------------------------
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

void __attribute__((format(printf, 3, 4)))
VSCode::SendFormattedOutput(OutputType o, const char *format, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, format);
  int actual_length = vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  SendOutput(o, llvm::StringRef(buffer,
                                std::min<int>(actual_length, sizeof(buffer))));
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

} // namespace lldb_vscode

