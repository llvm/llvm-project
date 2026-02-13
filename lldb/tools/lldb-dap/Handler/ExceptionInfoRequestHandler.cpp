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
#include "SBAPIExtras.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/API/SBValue.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

namespace {

// See `InstrumentationRuntimeUBSan::RetrieveReportData`.
struct UBSanReport {
  std::string description;
  std::string summary;
  std::string filename;
  uint32_t column = LLDB_INVALID_COLUMN_NUMBER;
  uint32_t line = LLDB_INVALID_LINE_NUMBER;
};

// See `InstrumentationRuntimeMainThreadChecker::RetrieveReportData`.
struct MainThreadCheckerReport {
  std::string description;
  std::string api_name;
  std::string class_name;
  std::string selector;
};

// See `ReportRetriever::RetrieveReportData`.
struct ASanReport {
  std::string description;
  lldb::addr_t address = LLDB_INVALID_ADDRESS;
  lldb::addr_t program_counter = LLDB_INVALID_ADDRESS;
  lldb::addr_t base_pointer = LLDB_INVALID_ADDRESS;
  lldb::addr_t stack_pointer = LLDB_INVALID_ADDRESS;
  std::string stop_type;
};

// FIXME: Support TSan, BoundsSafety formatting.

using RuntimeInstrumentReport =
    std::variant<UBSanReport, MainThreadCheckerReport, ASanReport>;

static bool fromJSON(const json::Value &params, UBSanReport &report,
                     json::Path path) {
  json::ObjectMapper O(params, path);
  return O.mapOptional("description", report.description) &&
         O.mapOptional("summary", report.summary) &&
         O.mapOptional("filename", report.filename) &&
         O.mapOptional("col", report.column) &&
         O.mapOptional("line", report.line);
}

static bool fromJSON(const json::Value &params, MainThreadCheckerReport &report,
                     json::Path path) {
  json::ObjectMapper O(params, path);
  return O.mapOptional("description", report.description) &&
         O.mapOptional("api_name", report.api_name) &&
         O.mapOptional("class_name", report.class_name) &&
         O.mapOptional("selector", report.selector);
}

static bool fromJSON(const json::Value &params, ASanReport &report,
                     json::Path path) {
  json::ObjectMapper O(params, path);
  return O.mapOptional("description", report.description) &&
         O.mapOptional("address", report.address) &&
         O.mapOptional("pc", report.program_counter) &&
         O.mapOptional("bp", report.base_pointer) &&
         O.mapOptional("sp", report.stack_pointer) &&
         O.mapOptional("stop_type", report.stop_type);
}

static bool fromJSON(const json::Value &params, RuntimeInstrumentReport &report,
                     json::Path path) {
  json::ObjectMapper O(params, path);
  std::string instrumentation_class;
  if (!O || !O.map("instrumentation_class", instrumentation_class))
    return false;

  if (instrumentation_class == "UndefinedBehaviorSanitizer") {
    UBSanReport inner_report;
    bool success = fromJSON(params, inner_report, path);
    if (success)
      report = std::move(inner_report);
    return success;
  }
  if (instrumentation_class == "MainThreadChecker") {
    MainThreadCheckerReport inner_report;
    bool success = fromJSON(params, inner_report, path);
    if (success)
      report = std::move(inner_report);
    return success;
  }
  if (instrumentation_class == "AddressSanitizer") {
    ASanReport inner_report;
    bool success = fromJSON(params, inner_report, path);
    if (success)
      report = std::move(inner_report);
    return success;
  }

  // FIXME: Support additional runtime instruments with specific formatters.
  return false;
}

} // end namespace

static raw_ostream &operator<<(raw_ostream &OS, UBSanReport &report) {
  if (!report.filename.empty()) {
    OS << report.filename;
    if (report.line != LLDB_INVALID_LINE_NUMBER) {
      OS << ":" << report.line;
      if (report.column != LLDB_INVALID_COLUMN_NUMBER)
        OS << ":" << report.column;
    }
    OS << " ";
  }

  if (!report.description.empty())
    OS << report.description << "\n";

  if (!report.summary.empty())
    OS << report.summary;

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               MainThreadCheckerReport &report) {
  if (!report.description.empty())
    OS << report.description << "\n";

  if (!report.class_name.empty())
    OS << "Class Name: " << report.class_name << "\n";
  if (!report.selector.empty())
    OS << "Selector: " << report.selector << "\n";

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS, ASanReport &report) {
  if (!report.stop_type.empty())
    OS << report.stop_type << ": ";
  if (!report.description.empty())
    OS << report.description << "\n";

  if (report.address != LLDB_INVALID_ADDRESS)
    OS << "Address: 0x" << llvm::utohexstr(report.address) << "\n";
  if (report.program_counter != LLDB_INVALID_ADDRESS)
    OS << "Program counter: 0x" << llvm::utohexstr(report.program_counter)
       << "\n";
  if (report.base_pointer != LLDB_INVALID_ADDRESS)
    OS << "Base pointer: 0x" << llvm::utohexstr(report.base_pointer) << "\n";
  if (report.stack_pointer != LLDB_INVALID_ADDRESS)
    OS << "Stack pointer: 0x" << llvm::utohexstr(report.stack_pointer) << "\n";

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               RuntimeInstrumentReport &report) {
  std::visit([&](auto &r) { OS << r; }, report);
  return OS;
}

static std::string FormatExceptionId(DAP &dap, lldb::SBThread &thread) {
  const lldb::StopReason stop_reason = thread.GetStopReason();
  switch (stop_reason) {
  case lldb::eStopReasonInstrumentation:
    return "runtime-instrumentation";
  case lldb::eStopReasonSignal:
    return "signal";
  case lldb::eStopReasonBreakpoint: {
    const ExceptionBreakpoint *exc_bp =
        dap.GetExceptionBPFromStopReason(thread);
    if (exc_bp)
      return exc_bp->GetFilter().str();
  }
    LLVM_FALLTHROUGH;
  default:
    return "exception";
  }
}

static std::string FormatStopDescription(lldb::SBThread &thread) {
  lldb::SBStream stream;
  if (!thread.GetStopDescription(stream))
    return "";
  std::string desc;
  raw_string_ostream OS(desc);
  OS << stream;
  return desc;
}

static std::string FormatExtendedStopInfo(lldb::SBThread &thread) {
  lldb::SBStream stream;
  if (!thread.GetStopReasonExtendedInfoAsJSON(stream))
    return "";

  std::string stop_info;
  raw_string_ostream OS(stop_info);
  Expected<RuntimeInstrumentReport> report =
      json::parse<RuntimeInstrumentReport>(
          {stream.GetData(), stream.GetSize()});

  // Check if we can improve the formatting of the raw JSON report.
  if (report) {
    OS << *report;
  } else {
    consumeError(report.takeError());
    OS << stream;
  }

  return stop_info;
}

static std::string FormatCrashReport(lldb::SBThread &thread) {
  lldb::SBStructuredData crash_info =
      thread.GetProcess().GetExtendedCrashInformation();
  if (!crash_info)
    return "";

  std::string report;
  raw_string_ostream OS(report);
  OS << "Extended Crash Information:\n" << crash_info;

  return report;
}

static std::string FormatStackTrace(lldb::SBThread &thread) {
  std::string stack_trace;
  raw_string_ostream OS(stack_trace);

  for (auto frame : thread)
    OS << frame;

  return stack_trace;
}

static std::optional<ExceptionDetails> FormatException(lldb::SBThread &thread) {
  lldb::SBValue exception = thread.GetCurrentException();
  if (!exception)
    return {};

  ExceptionDetails details;
  raw_string_ostream OS(details.message);

  if (const char *name = exception.GetName())
    details.evaluateName = name;
  if (const char *typeName = exception.GetDisplayTypeName())
    details.typeName = typeName;

  OS << exception;

  if (lldb::SBThread exception_backtrace =
          thread.GetCurrentExceptionBacktrace())
    details.stackTrace = FormatStackTrace(exception_backtrace);

  return details;
}

static void
FormatRuntimeInstrumentStackTrace(lldb::SBThread &thread,
                                  lldb::InstrumentationRuntimeType type,
                                  std::optional<ExceptionDetails> &details) {
  lldb::SBThreadCollection threads =
      thread.GetStopReasonExtendedBacktraces(type);
  for (auto thread : threads) {
    ExceptionDetails current_details;
    current_details.stackTrace = FormatStackTrace(thread);

    if (!details)
      details = std::move(current_details);
    else
      details->innerException.emplace_back(std::move(current_details));
  }
}

/// Retrieves the details of the exception that caused this event to be raised.
///
/// Clients should only call this request if the corresponding capability
/// `supportsExceptionInfoRequest` is true.
Expected<ExceptionInfoResponseBody>
ExceptionInfoRequestHandler::Run(const ExceptionInfoArguments &args) const {
  lldb::SBThread thread = dap.GetLLDBThread(args.threadId);
  if (!thread.IsValid())
    return make_error<DAPError>(
        formatv("Invalid thread id: {}", args.threadId).str());

  ExceptionInfoResponseBody body;
  body.breakMode = eExceptionBreakModeAlways;
  body.exceptionId = FormatExceptionId(dap, thread);
  body.details = FormatException(thread);

  raw_string_ostream OS(body.description);
  OS << FormatStopDescription(thread);

  if (std::string stop_info = FormatExtendedStopInfo(thread);
      !stop_info.empty())
    OS << "\n\n" << stop_info;

  if (std::string crash_report = FormatCrashReport(thread);
      !crash_report.empty())
    OS << "\n\n" << crash_report;

  lldb::SBProcess process = thread.GetProcess();
  for (uint32_t idx = 0; idx < lldb::eNumInstrumentationRuntimeTypes; idx++) {
    lldb::InstrumentationRuntimeType type =
        static_cast<lldb::InstrumentationRuntimeType>(idx);
    if (!process.IsInstrumentationRuntimePresent(type))
      continue;

    FormatRuntimeInstrumentStackTrace(thread, type, body.details);
  }

  return body;
}
