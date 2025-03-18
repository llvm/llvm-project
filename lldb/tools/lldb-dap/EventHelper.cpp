//===-- EventHelper.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EventHelper.h"
#include "DAP.h"
#include "Events/EventHandler.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "lldb/API/SBFileSpec.h"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>

#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#endif

namespace lldb_dap {

static void SendThreadExitedEvent(DAP &dap, lldb::tid_t tid) {
  llvm::json::Object event(CreateEventObject("thread"));
  llvm::json::Object body;
  body.try_emplace("reason", "exited");
  body.try_emplace("threadId", (int64_t)tid);
  event.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(event)));
}

// Send a thread stopped event for all threads as long as the process
// is stopped.
void SendThreadStoppedEvent(DAP &dap) {
  lldb::SBProcess process = dap.target.GetProcess();
  if (process.IsValid()) {
    auto state = process.GetState();
    if (state == lldb::eStateStopped) {
      llvm::DenseSet<lldb::tid_t> old_thread_ids;
      old_thread_ids.swap(dap.thread_ids);
      uint32_t stop_id = process.GetStopID();
      const uint32_t num_threads = process.GetNumThreads();

      // First make a pass through the threads to see if the focused thread
      // has a stop reason. In case the focus thread doesn't have a stop
      // reason, remember the first thread that has a stop reason so we can
      // set it as the focus thread if below if needed.
      lldb::tid_t first_tid_with_reason = LLDB_INVALID_THREAD_ID;
      uint32_t num_threads_with_reason = 0;
      bool focus_thread_exists = false;
      for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
        const lldb::tid_t tid = thread.GetThreadID();
        const bool has_reason = ThreadHasStopReason(thread);
        // If the focus thread doesn't have a stop reason, clear the thread ID
        if (tid == dap.focus_tid) {
          focus_thread_exists = true;
          if (!has_reason)
            dap.focus_tid = LLDB_INVALID_THREAD_ID;
        }
        if (has_reason) {
          ++num_threads_with_reason;
          if (first_tid_with_reason == LLDB_INVALID_THREAD_ID)
            first_tid_with_reason = tid;
        }
      }

      // We will have cleared dap.focus_tid if the focus thread doesn't have
      // a stop reason, so if it was cleared, or wasn't set, or doesn't exist,
      // then set the focus thread to the first thread with a stop reason.
      if (!focus_thread_exists || dap.focus_tid == LLDB_INVALID_THREAD_ID)
        dap.focus_tid = first_tid_with_reason;

      // If no threads stopped with a reason, then report the first one so
      // we at least let the UI know we stopped.
      if (num_threads_with_reason == 0) {
        lldb::SBThread thread = process.GetThreadAtIndex(0);
        dap.focus_tid = thread.GetThreadID();
        dap.SendJSON(CreateThreadStopped(dap, thread, stop_id));
      } else {
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
          lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
          dap.thread_ids.insert(thread.GetThreadID());
          if (ThreadHasStopReason(thread)) {
            dap.SendJSON(CreateThreadStopped(dap, thread, stop_id));
          }
        }
      }

      for (auto tid : old_thread_ids) {
        auto end = dap.thread_ids.end();
        auto pos = dap.thread_ids.find(tid);
        if (pos == end)
          SendThreadExitedEvent(dap, tid);
      }
    } else {
      if (dap.log)
        *dap.log << "error: SendThreadStoppedEvent() when process"
                    " isn't stopped ("
                 << lldb::SBDebugger::StateAsCString(state) << ')' << std::endl;
    }
  } else {
    if (dap.log)
      *dap.log << "error: SendThreadStoppedEvent() invalid process"
               << std::endl;
  }
  dap.RunStopCommands();
}

// Send a "terminated" event to indicate the process is done being
// debugged.
void SendTerminatedEvent(DAP &dap) {
  // Prevent races if the process exits while we're being asked to disconnect.
  llvm::call_once(dap.terminated_event_flag, [&] {
    dap.RunTerminateCommands();
    // Send a "terminated" event
    llvm::json::Object event(CreateTerminatedEventObject(dap.target));
    dap.SendJSON(llvm::json::Value(std::move(event)));
  });
}

// Grab any STDOUT and STDERR from the process and send it up to VS Code
// via an "output" event to the "stdout" and "stderr" categories.
void SendStdOutStdErr(DAP &dap, lldb::SBProcess &process) {
  char buffer[OutputBufferSize];
  size_t count;
  while ((count = process.GetSTDOUT(buffer, sizeof(buffer))) > 0)
    dap.SendOutput(llvm::StringRef(buffer, count), OutputCategory::Stdout);
  while ((count = process.GetSTDERR(buffer, sizeof(buffer))) > 0)
    dap.SendOutput(llvm::StringRef(buffer, count), OutputCategory::Stderr);
}

// Send a "continued" event to indicate the process is in the running state.
void SendContinuedEvent(DAP &dap) {
  lldb::SBProcess process = dap.target.GetProcess();
  if (!process.IsValid()) {
    return;
  }

  // If the focus thread is not set then we haven't reported any thread status
  // to the client, so nothing to report.
  if (!dap.configuration_done_sent || dap.focus_tid == LLDB_INVALID_THREAD_ID) {
    return;
  }

  llvm::json::Object event(CreateEventObject("continued"));
  llvm::json::Object body;
  body.try_emplace("threadId", (int64_t)dap.focus_tid);
  body.try_emplace("allThreadsContinued", true);
  event.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(event)));
}

} // namespace lldb_dap
