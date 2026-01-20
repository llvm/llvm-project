//===-- ExceptionInfoRequestHandler.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "DAPError.h"
#include "DAPLog.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/lldb-enumerations.h"
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
      body.description = exc_bp->GetLabel().str() + "\n";
    } else {
      body.exceptionId = "exception";
    }
  } break;
  default:
    body.exceptionId = "exception";
  }

  lldb::SBStream stream;
  if (thread.GetStopDescription(stream))
    body.description += {stream.GetData(), stream.GetSize()};

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
    body.description += "\n\nExtended Crash Information:\n" +
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
