//===-- StepOutRequestHandler.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// The request resumes the given thread to step out (return) from a
/// function/method and allows all other threads to run freely by resuming
/// them.
///
/// If the debug adapter supports single thread execution (see capability
/// `supportsSingleThreadExecutionRequests`), setting the `singleThread`
/// argument to true prevents other suspended threads from resuming.
///
/// The debug adapter first sends the response and then a `stopped` event (with
/// reason `step`) after the step has completed."
Error StepOutRequestHandler::Run(const StepOutArguments &arguments) const {
  lldb::SBThread thread = dap.GetLLDBThread(arguments.threadId);
  if (!thread.IsValid())
    return make_error<DAPError>("invalid thread");

  // Remember the thread ID that caused the resume so we can set the
  // "threadCausedFocus" boolean value in the "stopped" events.
  dap.focus_tid = thread.GetThreadID();
  thread.StepOut();

  return Error::success();
}

} // namespace lldb_dap
