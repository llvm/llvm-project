//===-- SetPinnedThreadRequestHandler.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"

using namespace llvm;
using namespace lldb;

namespace lldb_dap {

/// Set the pinned thread to the specified thread. If the thread is already
/// pinned, this is a no-op. If the thread is not already pinned, the debug
/// adapter first verifies that the thread is valid and then pins it. The debug
/// adapter sends a ThreadEvent to the client to indicate that the thread is
/// pinned/unpinned. Pinned thread will be used on `SetThreadID()` for any new
/// setBreakpoints request
Error SetPinnedThreadRequestHandler::Run(
    const protocol::SetPinnedThreadArguments &args) const {
  // This indicates client wants to unpin the thread
  if (!args.threadId.has_value()) {
    dap.pinned_tid = LLDB_INVALID_THREAD_ID;
    SendThreadPinnedOrUnpinnedEvent(dap, LLDB_INVALID_THREAD_ID);
    return Error::success();
  }
  lldb::tid_t tid = args.threadId.value();
  lldb::SBThread thread = dap.GetLLDBThread(tid);
  if (!thread.IsValid())
    return make_error<DAPError>("invalid thread");

  dap.pinned_tid = tid;
  SendThreadPinnedOrUnpinnedEvent(dap, dap.pinned_tid);
  return Error::success();
}

} // namespace lldb_dap
