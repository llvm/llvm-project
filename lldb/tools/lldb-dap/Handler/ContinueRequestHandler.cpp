//===-- ContinueRequestHandler.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Handler/RequestHandler.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBProcess.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// The request resumes execution of all threads. If the debug adapter supports
/// single thread execution (see capability
/// `supportsSingleThreadExecutionRequests`), setting the `singleThread`
/// argument to true resumes only the specified thread. If not all threads were
/// resumed, the `allThreadsContinued` attribute of the response should be set
/// to false.
Expected<ContinueResponseBody>
ContinueRequestHandler::Run(const ContinueArguments &args) const {
  SBProcess process = dap.target.GetProcess();
  SBError error;

  if (!SBDebugger::StateIsStoppedState(process.GetState()))
    return make_error<NotStoppedError>();

  if (args.singleThread)
    dap.GetLLDBThread(args.threadId).Resume(error);
  else
    error = process.Continue();

  if (error.Fail())
    return ToError(error);

  ContinueResponseBody body;
  body.allThreadsContinued = !args.singleThread;
  return body;
}

} // namespace lldb_dap
