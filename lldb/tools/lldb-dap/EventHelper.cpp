//===-- EventHelper.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EventHelper.h"
#include "DAP.h"
#include "DAPError.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolEvents.h"
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBFileSpec.h"
#include "llvm/Support/Error.h"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>

#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#endif

using namespace llvm;

namespace lldb_dap {

static void SendThreadExitedEvent(DAP &dap, lldb::tid_t tid) {
  llvm::json::Object event(CreateEventObject("thread"));
  llvm::json::Object body;
  body.try_emplace("reason", "exited");
  body.try_emplace("threadId", (int64_t)tid);
  event.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(event)));
}

/// Get capabilities based on the configured target.
static llvm::DenseSet<AdapterFeature> GetTargetBasedCapabilities(DAP &dap) {
  llvm::DenseSet<AdapterFeature> capabilities;
  if (!dap.target.IsValid())
    return capabilities;

  const llvm::StringRef target_triple = dap.target.GetTriple();
  if (target_triple.starts_with("x86"))
    capabilities.insert(protocol::eAdapterFeatureStepInTargetsRequest);

  // We only support restarting launch requests not attach requests.
  if (dap.last_launch_request)
    capabilities.insert(protocol::eAdapterFeatureRestartRequest);

  return capabilities;
}

void SendExtraCapabilities(DAP &dap) {
  protocol::Capabilities capabilities = dap.GetCustomCapabilities();
  llvm::DenseSet<AdapterFeature> target_capabilities =
      GetTargetBasedCapabilities(dap);

  capabilities.supportedFeatures.insert(target_capabilities.begin(),
                                        target_capabilities.end());

  protocol::CapabilitiesEventBody body;
  body.capabilities = std::move(capabilities);

  // Only notify the client if supportedFeatures changed.
  if (!body.capabilities.supportedFeatures.empty())
    dap.Send(protocol::Event{"capabilities", std::move(body)});
}

// "ProcessEvent": {
//   "allOf": [
//     { "$ref": "#/definitions/Event" },
//     {
//       "type": "object",
//       "description": "Event message for 'process' event type. The event
//                       indicates that the debugger has begun debugging a
//                       new process. Either one that it has launched, or one
//                       that it has attached to.",
//       "properties": {
//         "event": {
//           "type": "string",
//           "enum": [ "process" ]
//         },
//         "body": {
//           "type": "object",
//           "properties": {
//             "name": {
//               "type": "string",
//               "description": "The logical name of the process. This is
//                               usually the full path to process's executable
//                               file. Example: /home/myproj/program.js."
//             },
//             "systemProcessId": {
//               "type": "integer",
//               "description": "The system process id of the debugged process.
//                               This property will be missing for non-system
//                               processes."
//             },
//             "isLocalProcess": {
//               "type": "boolean",
//               "description": "If true, the process is running on the same
//                               computer as the debug adapter."
//             },
//             "startMethod": {
//               "type": "string",
//               "enum": [ "launch", "attach", "attachForSuspendedLaunch" ],
//               "description": "Describes how the debug engine started
//                               debugging this process.",
//               "enumDescriptions": [
//                 "Process was launched under the debugger.",
//                 "Debugger attached to an existing process.",
//                 "A project launcher component has launched a new process in
//                  a suspended state and then asked the debugger to attach."
//               ]
//             }
//           },
//           "required": [ "name" ]
//         }
//       },
//       "required": [ "event", "body" ]
//     }
//   ]
// }
void SendProcessEvent(DAP &dap, LaunchMethod launch_method) {
  lldb::SBFileSpec exe_fspec = dap.target.GetExecutable();
  char exe_path[PATH_MAX];
  exe_fspec.GetPath(exe_path, sizeof(exe_path));
  llvm::json::Object event(CreateEventObject("process"));
  llvm::json::Object body;
  EmplaceSafeString(body, "name", exe_path);
  const auto pid = dap.target.GetProcess().GetProcessID();
  body.try_emplace("systemProcessId", (int64_t)pid);
  body.try_emplace("isLocalProcess", true);
  const char *startMethod = nullptr;
  switch (launch_method) {
  case Launch:
    startMethod = "launch";
    break;
  case Attach:
    startMethod = "attach";
    break;
  case AttachForSuspendedLaunch:
    startMethod = "attachForSuspendedLaunch";
    break;
  }
  body.try_emplace("startMethod", startMethod);
  event.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(event)));
}

// Send a thread stopped event for all threads as long as the process
// is stopped.
llvm::Error SendThreadStoppedEvent(DAP &dap, bool on_entry) {
  lldb::SBMutex lock = dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  lldb::SBProcess process = dap.target.GetProcess();
  if (!process.IsValid())
    return make_error<DAPError>("invalid process");

  lldb::StateType state = process.GetState();
  if (!lldb::SBDebugger::StateIsStoppedState(state))
    return make_error<NotStoppedError>();

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

  for (const auto &tid : old_thread_ids) {
    auto end = dap.thread_ids.end();
    auto pos = dap.thread_ids.find(tid);
    if (pos == end)
      SendThreadExitedEvent(dap, tid);
  }

  dap.RunStopCommands();
  return Error::success();
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
    dap.SendOutput(OutputType::Stdout, llvm::StringRef(buffer, count));
  while ((count = process.GetSTDERR(buffer, sizeof(buffer))) > 0)
    dap.SendOutput(OutputType::Stderr, llvm::StringRef(buffer, count));
}

// Send a "continued" event to indicate the process is in the running state.
void SendContinuedEvent(DAP &dap) {
  lldb::SBProcess process = dap.target.GetProcess();
  if (!process.IsValid()) {
    return;
  }

  // If the focus thread is not set then we haven't reported any thread status
  // to the client, so nothing to report.
  if (!dap.configuration_done || dap.focus_tid == LLDB_INVALID_THREAD_ID) {
    return;
  }

  llvm::json::Object event(CreateEventObject("continued"));
  llvm::json::Object body;
  body.try_emplace("threadId", (int64_t)dap.focus_tid);
  body.try_emplace("allThreadsContinued", true);
  event.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(event)));
}

// Send a "exited" event to indicate the process has exited.
void SendProcessExitedEvent(DAP &dap, lldb::SBProcess &process) {
  llvm::json::Object event(CreateEventObject("exited"));
  llvm::json::Object body;
  body.try_emplace("exitCode", (int64_t)process.GetExitStatus());
  event.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(event)));
}

} // namespace lldb_dap
