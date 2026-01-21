//===-- ExceptionInfoRequestHandler.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPError.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <utility>

using namespace lldb_dap::protocol;

namespace lldb_dap {

static std::string ThreadSummary(lldb::SBThread &thread) {
  lldb::SBStream stream;
  thread.GetDescription(stream);
  for (uint32_t idx = 0; idx < thread.GetNumFrames(); idx++) {
    lldb::SBFrame frame = thread.GetFrameAtIndex(idx);
    frame.GetDescription(stream);
  }
  return {stream.GetData(), stream.GetSize()};
}

struct RuntimeInstrumentReport {
  std::string description;
  std::string instrument;
  std::string summary;

  std::string filename;
  uint32_t column = LLDB_INVALID_COLUMN_NUMBER;
  uint32_t line = LLDB_INVALID_LINE_NUMBER;

  // keys found on UBSan
  lldb::addr_t memory = LLDB_INVALID_ADDRESS;
  lldb::tid_t tid = LLDB_INVALID_THREAD_ID;
  std::vector<lldb::user_id_t> trace;

  // keys found on MainThreadChecker
  std::string api_name;
  std::string class_name;
  std::string selector;

  // FIXME: TSan, ASan, BoundsSafety
};

static bool fromJSON(const llvm::json::Value &Params,
                     RuntimeInstrumentReport &RIR, llvm::json::Path Path) {
  llvm::json::ObjectMapper O(Params, Path);
  return O && O.mapOptional("description", RIR.description) &&
         O.mapOptional("instrumentation_class", RIR.instrument) &&
         O.mapOptional("summary", RIR.summary) &&
         O.mapOptional("filename", RIR.filename) &&
         O.mapOptional("col", RIR.column) && O.mapOptional("line", RIR.line) &&
         O.mapOptional("memory", RIR.memory) && O.mapOptional("tid", RIR.tid) &&
         O.mapOptional("trace", RIR.trace) &&
         O.mapOptional("api_name", RIR.api_name) &&
         O.mapOptional("class_name", RIR.class_name) &&
         O.mapOptional("selector", RIR.selector);
}

/// Retrieves the details of the exception that caused this event to be raised.
///
/// Clients should only call this request if the corresponding capability
/// `supportsExceptionInfoRequest` is true.
llvm::Expected<ExceptionInfoResponseBody>
ExceptionInfoRequestHandler::Run(const ExceptionInfoArguments &args) const {
  lldb::SBThread thread = dap.GetLLDBThread(args.threadId);
  if (!thread.IsValid())
    return llvm::make_error<DAPError>(
        llvm::formatv("Invalid thread id: {}", args.threadId).str());

  ExceptionInfoResponseBody body;
  llvm::raw_string_ostream OS(body.description);
  body.breakMode = eExceptionBreakModeAlways;
  const lldb::StopReason stop_reason = thread.GetStopReason();
  switch (stop_reason) {
  case lldb::eStopReasonInstrumentation:
    body.exceptionId = "runtime-instrumentation";
    break;
  case lldb::eStopReasonSignal:
    body.exceptionId = "signal";
    break;
  case lldb::eStopReasonBreakpoint: {
    const ExceptionBreakpoint *exc_bp =
        dap.GetExceptionBPFromStopReason(thread);
    if (exc_bp) {
      body.exceptionId = exc_bp->GetFilter();
      OS << exc_bp->GetLabel().str();
    } else {
      body.exceptionId = "exception";
    }
  } break;
  default:
    body.exceptionId = "exception";
  }

  lldb::SBStream stream;
  if (thread.GetStopDescription(stream))
    OS << std::string{stream.GetData(), stream.GetSize()};

  stream.Clear();
  if (thread.GetStopReasonExtendedInfoAsJSON(stream)) {
    OS << "\n";

    llvm::Expected<RuntimeInstrumentReport> report =
        llvm::json::parse<RuntimeInstrumentReport>(
            {stream.GetData(), stream.GetSize()});
    // If we failed to parse the extended stop reason info, attach it
    // unmodified.
    if (!report) {
      llvm::consumeError(report.takeError());
      OS << std::string(stream.GetData(), stream.GetSize());
    } else {
      if (!report->filename.empty()) {
        OS << report->filename;
        if (report->line != LLDB_INVALID_LINE_NUMBER) {
          OS << ":" << report->line;
          if (report->column != LLDB_INVALID_COLUMN_NUMBER)
            OS << ":" << report->column;
        }
        OS << " ";
      }

      OS << report->instrument;
      if (!report->description.empty())
        OS << ": " << report->description;
      OS << "\n";
      if (!report->summary.empty())
        OS << report->summary << "\n";

      if (!report->api_name.empty())
        OS << "API Name: " << report->api_name << "\n";
      if (!report->class_name.empty())
        OS << "Class Name: " << report->class_name << "\n";
      if (!report->selector.empty())
        OS << "Selector: " << report->selector << "\n";
    }
  }

  if (lldb::SBValue exception = thread.GetCurrentException()) {
    body.details = ExceptionDetails{};
    if (const char *name = exception.GetName())
      body.details->evaluateName = name;
    if (const char *typeName = exception.GetDisplayTypeName())
      body.details->typeName = typeName;

    stream.Clear();
    if (exception.GetDescription(stream))
      body.details->message = {stream.GetData(), stream.GetSize()};

    if (lldb::SBThread exception_backtrace =
            thread.GetCurrentExceptionBacktrace())
      body.details->stackTrace = ThreadSummary(exception_backtrace);
  }

  lldb::SBProcess process = dap.target.GetProcess();
  if (!process)
    return body;

  lldb::SBStructuredData crash_info = process.GetExtendedCrashInformation();
  stream.Clear();
  if (crash_info.IsValid() && crash_info.GetDescription(stream))
    OS << "\nExtended Crash Information:\n" +
              std::string(stream.GetData(), stream.GetSize());

  for (uint32_t idx = 0; idx < lldb::eNumInstrumentationRuntimeTypes; idx++) {
    lldb::InstrumentationRuntimeType type =
        static_cast<lldb::InstrumentationRuntimeType>(idx);
    if (!process.IsInstrumentationRuntimePresent(type))
      continue;

    lldb::SBThreadCollection threads =
        thread.GetStopReasonExtendedBacktraces(type);
    for (uint32_t tidx = 0; tidx < threads.GetSize(); tidx++) {
      auto thread = threads.GetThreadAtIndex(tidx);
      if (!thread)
        continue;
      ExceptionDetails details;
      details.stackTrace = ThreadSummary(thread);
      if (!body.details)
        body.details = std::move(details);
      else
        body.details->innerException.emplace_back(std::move(details));
    }
  }

  return body;
}

} // namespace lldb_dap
