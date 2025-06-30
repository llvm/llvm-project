//===-- ThreadsRequestHandler.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "Protocol/ProtocolRequests.h"
#include "ProtocolUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDefines.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// The request retrieves a list of all threads.
Expected<ThreadsResponseBody>
ThreadsRequestHandler::Run(const ThreadsArguments &) const {
  lldb::SBProcess process = dap.target.GetProcess();
  std::vector<Thread> threads;

  // Client requests the baseline of currently existing threads after
  // a successful launch or attach by sending a 'threads' request
  // right after receiving the configurationDone response.
  // If no thread has reported to the client, it prevents something
  // like the pause request from working in the running state.
  // Return the cache of initial threads as the process might have resumed
  if (!dap.initial_thread_list.empty()) {
    threads = dap.initial_thread_list;
    dap.initial_thread_list.clear();
  } else {
    if (!lldb::SBDebugger::StateIsStoppedState(process.GetState()))
      return make_error<NotStoppedError>();

    threads = GetThreads(process, dap.thread_format);
  }

  if (threads.size() == 0)
    return make_error<DAPError>("failed to retrieve threads from process");

  return ThreadsResponseBody{threads};
}

} // namespace lldb_dap
