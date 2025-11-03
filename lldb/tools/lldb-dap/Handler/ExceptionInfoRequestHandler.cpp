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

using namespace lldb_dap::protocol;

namespace lldb_dap {

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

  ExceptionInfoResponseBody response;
  response.breakMode = eExceptionBreakModeAlways;
  const lldb::StopReason stop_reason = thread.GetStopReason();
  switch (stop_reason) {
  case lldb::eStopReasonSignal:
    response.exceptionId = "signal";
    break;
  case lldb::eStopReasonBreakpoint: {
    const ExceptionBreakpoint *exc_bp =
        dap.GetExceptionBPFromStopReason(thread);
    if (exc_bp) {
      response.exceptionId = exc_bp->GetFilter();
      response.description = exc_bp->GetLabel();
    } else {
      response.exceptionId = "exception";
    }
  } break;
  default:
    response.exceptionId = "exception";
  }

  lldb::SBStream stream;
  if (response.description.empty()) {
    if (thread.GetStopDescription(stream)) {
      response.description = {stream.GetData(), stream.GetSize()};
    }
  }

  if (lldb::SBValue exception = thread.GetCurrentException()) {
    stream.Clear();
    response.details = ExceptionDetails{};
    if (exception.GetDescription(stream)) {
      response.details->message = {stream.GetData(), stream.GetSize()};
    }

    if (lldb::SBThread exception_backtrace =
            thread.GetCurrentExceptionBacktrace()) {
      stream.Clear();
      exception_backtrace.GetDescription(stream);

      for (uint32_t idx = 0; idx < exception_backtrace.GetNumFrames(); idx++) {
        lldb::SBFrame frame = exception_backtrace.GetFrameAtIndex(idx);
        frame.GetDescription(stream);
      }
      response.details->stackTrace = {stream.GetData(), stream.GetSize()};
    }
  }
  return response;
}
} // namespace lldb_dap
