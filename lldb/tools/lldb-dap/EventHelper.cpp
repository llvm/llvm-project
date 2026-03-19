//===-- EventHelper.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EventHelper.h"
#include "Breakpoint.h"
#include "BreakpointBase.h"
#include "DAP.h"
#include "DAPError.h"
#include "DAPLog.h"
#include "DAPSessionManager.h"
#include "Handler/ResponseHandler.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolEvents.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "ProtocolUtils.h"
#include "SBAPIExtras.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBPlatform.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBThread.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>
#include <utility>

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
//       "description": "The event indicates that the debugger has begun
//       debugging a new process. Either one that it has launched, or one that
//       it has attached to.", "properties": {
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
//               usually the full path to process's executable file. Example:
//               /home/example/myproj/program.js."
//             },
//             "systemProcessId": {
//               "type": "integer",
//               "description": "The process ID of the debugged process, as
//               assigned by the operating system. This property should be
//               omitted for logical processes that do not map to operating
//               system processes on the machine."
//             },
//             "isLocalProcess": {
//               "type": "boolean",
//               "description": "If true, the process is running on the same
//               computer as the debug adapter."
//             },
//             "startMethod": {
//               "type": "string",
//               "enum": [ "launch", "attach", "attachForSuspendedLaunch" ],
//               "description": "Describes how the debug engine started
//               debugging this process.", "enumDescriptions": [
//                 "Process was launched under the debugger.",
//                 "Debugger attached to an existing process.",
//                 "A project launcher component has launched a new process in a
//                 suspended state and then asked the debugger to attach."
//               ]
//             },
//             "pointerSize": {
//               "type": "integer",
//               "description": "The size of a pointer or address for this
//               process, in bits. This value may be used by clients when
//               formatting addresses for display."
//             }
//           },
//           "required": [ "name" ]
//         }
//       },
//       "required": [ "event", "body" ]
//     }
//   ]
// },
void SendProcessEvent(DAP &dap, LaunchMethod launch_method) {
  lldb::SBFileSpec exe_fspec = dap.target.GetExecutable();
  char exe_path[PATH_MAX];
  exe_fspec.GetPath(exe_path, sizeof(exe_path));
  llvm::json::Object event(CreateEventObject("process"));
  llvm::json::Object body;
  EmplaceSafeString(body, "name", exe_path);
  const auto pid = dap.target.GetProcess().GetProcessID();
  body.try_emplace("systemProcessId", (int64_t)pid);
  body.try_emplace("isLocalProcess", dap.target.GetPlatform().IsHost());
  body.try_emplace("pointerSize", dap.target.GetAddressByteSize() * 8);
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

static void SendStoppedEvent(DAP &dap, lldb::SBThread &thread, bool on_entry,
                             bool all_threads_stopped, bool preserve_focus) {
  protocol::StoppedEventBody body;
  body.reason = protocol::eStoppedReasonPause;
  if (on_entry) {
    body.reason = protocol::eStoppedReasonEntry;
  } else if (thread.IsValid()) {
    switch (thread.GetStopReason()) {
    case lldb::eStopReasonTrace:
    case lldb::eStopReasonPlanComplete:
    case lldb::eStopReasonProcessorTrace:
    case lldb::eStopReasonHistoryBoundary:
      body.reason = protocol::eStoppedReasonStep;
      break;
    case lldb::eStopReasonBreakpoint: {
      ExceptionBreakpoint *exc_bp = dap.GetExceptionBPFromStopReason(thread);
      if (exc_bp) {
        body.reason = protocol::eStoppedReasonException;
        body.text = exc_bp->GetLabel();
      } else {
        InstructionBreakpoint *inst_bp =
            dap.GetInstructionBPFromStopReason(thread);
        body.reason = inst_bp ? protocol::eStoppedReasonInstructionBreakpoint
                              : protocol::eStoppedReasonBreakpoint;

        llvm::raw_string_ostream OS(body.text);
        OS << "breakpoint";
        for (size_t idx = 0; idx < thread.GetStopReasonDataCount(); idx += 2) {
          lldb::break_id_t bp_id = thread.GetStopReasonDataAtIndex(idx);
          lldb::break_id_t bp_loc_id = thread.GetStopReasonDataAtIndex(idx + 1);
          body.hitBreakpointIds.push_back(bp_id);
          OS << " " << bp_id << "." << bp_loc_id;
        }
      }
    } break;
    case lldb::eStopReasonWatchpoint: {
      body.reason = protocol::eStoppedReasonDataBreakpoint;
      lldb::break_id_t bp_id = thread.GetStopReasonDataAtIndex(0);
      body.hitBreakpointIds.push_back(bp_id);
      body.text = llvm::formatv("data breakpoint {0}", bp_id).str();
    } break;
    case lldb::eStopReasonSignal:
    case lldb::eStopReasonException:
    case lldb::eStopReasonInstrumentation:
      body.reason = protocol::eStoppedReasonException;
      break;
    case lldb::eStopReasonExec:
    case lldb::eStopReasonFork:
    case lldb::eStopReasonVFork:
    case lldb::eStopReasonVForkDone:
      body.reason = protocol::eStoppedReasonEntry;
      break;
    case lldb::eStopReasonInterrupt:
      body.reason = protocol::eStoppedReasonPause;
      break;
    case lldb::eStopReasonThreadExiting:
    case lldb::eStopReasonInvalid:
    case lldb::eStopReasonNone:
      break;
    }

    lldb::SBStream description;
    thread.GetStopDescription(description);
    body.description = {description.GetData(), description.GetSize()};
  }
  lldb::tid_t tid = thread.GetThreadID();
  body.threadId = tid;
  body.allThreadsStopped = all_threads_stopped;
  body.preserveFocusHint = preserve_focus;

  dap.Send(protocol::Event{"stopped", std::move(body)});
}

// Send a thread stopped event for the first stopped thread as the process is
// stopped.
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

  lldb::SBThread focused_thread;
  std::vector<lldb::SBThread> stopped_threads;
  for (auto thread : process) {
    // Collect all known thread ids for sending thread events.
    dap.thread_ids.insert(thread.GetThreadID());

    if (!ThreadHasStopReason(thread))
      continue;

    // Focus on the first stopped thread
    if (!focused_thread.IsValid())
      focused_thread = thread;
    else
      stopped_threads.push_back(thread);
  }

  // If no stopped threads were detected, fallback to the selected thread.
  if (!focused_thread)
    focused_thread = process.GetSelectedThread();

  if (!focused_thread)
    return make_error<DAPError>("no stopped threads");

  // Send stopped events for each thread thats stopped.
  for (auto thread : stopped_threads)
    SendStoppedEvent(dap, thread, on_entry, /*all_threads_stopped=*/false,
                     /*preserve_focus=*/true);

  // Notify the focused thread last to ensure the UI is focused correctly.
  SendStoppedEvent(dap, focused_thread, on_entry, /*all_threads_stopped=*/true,
                   /*preserve_focus=*/false);

  // Update focused thread.
  dap.focus_tid = focused_thread.GetThreadID();

  for (const auto &tid : old_thread_ids)
    if (!dap.thread_ids.contains(tid))
      SendThreadExitedEvent(dap, tid);

  dap.RunStopCommands();

  return Error::success();
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

void SendInvalidatedEvent(
    DAP &dap, llvm::ArrayRef<protocol::InvalidatedEventBody::Area> areas,
    lldb::tid_t tid) {
  if (!dap.clientFeatures.contains(protocol::eClientFeatureInvalidatedEvent))
    return;
  protocol::InvalidatedEventBody body;
  body.areas = areas;

  if (tid != LLDB_INVALID_THREAD_ID)
    body.threadId = tid;

  dap.Send(protocol::Event{"invalidated", std::move(body)});
}

void SendMemoryEvent(DAP &dap, lldb::SBValue variable) {
  if (!dap.clientFeatures.contains(protocol::eClientFeatureMemoryEvent))
    return;
  protocol::MemoryEventBody body;
  body.memoryReference = variable.GetLoadAddress();
  body.count = variable.GetByteSize();
  if (body.memoryReference == LLDB_INVALID_ADDRESS)
    return;
  dap.Send(protocol::Event{"memory", std::move(body)});
}

// Event handler functions that are called by EventThread.
// These handlers extract the necessary objects from events and find the
// appropriate DAP instance to handle them, maintaining compatibility with
// the original DAP::Handle*Event pattern while supporting multi-session
// debugging.

static void HandleProcessEvent(const lldb::SBEvent &event, bool &process_exited,
                               Log &log) {
  lldb::SBProcess process = lldb::SBProcess::GetProcessFromEvent(event);

  // Find the DAP instance that owns this process's target.
  DAP *dap = DAPSessionManager::FindDAP(process.GetTarget());
  if (!dap) {
    DAP_LOG(log, "Unable to find DAP instance for process {0}",
            process.GetProcessID());
    return;
  }

  const uint32_t event_mask = event.GetType();

  if (event_mask & lldb::SBProcess::eBroadcastBitStateChanged) {
    auto state = lldb::SBProcess::GetStateFromEvent(event);
    switch (state) {
    case lldb::eStateConnected:
    case lldb::eStateDetached:
    case lldb::eStateInvalid:
    case lldb::eStateUnloaded:
      break;
    case lldb::eStateAttaching:
    case lldb::eStateCrashed:
    case lldb::eStateLaunching:
    case lldb::eStateStopped:
    case lldb::eStateSuspended:
      // Only report a stopped event if the process was not
      // automatically restarted.
      if (!lldb::SBProcess::GetRestartedFromEvent(event)) {
        SendStdOutStdErr(*dap, process);
        if (llvm::Error err = SendThreadStoppedEvent(*dap))
          DAP_LOG_ERROR(dap->log, std::move(err),
                        "({1}) reporting thread stopped: {0}",
                        dap->GetClientName());
      }
      break;
    case lldb::eStateRunning:
    case lldb::eStateStepping:
      dap->WillContinue();
      SendContinuedEvent(*dap);
      break;
    case lldb::eStateExited:
      lldb::SBStream stream;
      process.GetStatus(stream);
      dap->SendOutput(OutputType::Console, stream.GetData());

      // When restarting, we can get an "exited" event for the process we
      // just killed with the old PID, or even with no PID. In that case
      // we don't have to terminate the session.
      if (process.GetProcessID() == LLDB_INVALID_PROCESS_ID ||
          process.GetProcessID() == dap->restarting_process_id) {
        dap->restarting_process_id = LLDB_INVALID_PROCESS_ID;
      } else {
        // Run any exit LLDB commands the user specified in the
        // launch.json
        dap->RunExitCommands();
        SendProcessExitedEvent(*dap, process);
        dap->SendTerminatedEvent();
        process_exited = true;
      }
      break;
    }
  } else if ((event_mask & lldb::SBProcess::eBroadcastBitSTDOUT) ||
             (event_mask & lldb::SBProcess::eBroadcastBitSTDERR)) {
    SendStdOutStdErr(*dap, process);
  }
}

static void HandleTargetEvent(const lldb::SBEvent &event, Log &log) {
  lldb::SBTarget target = lldb::SBTarget::GetTargetFromEvent(event);

  // Find the DAP instance that owns this target.
  DAP *dap = DAPSessionManager::FindDAP(target);
  if (!dap) {
    DAP_LOG(log, "Unable to find DAP instance for target");
    return;
  }

  const uint32_t event_mask = event.GetType();
  if (event_mask & lldb::SBTarget::eBroadcastBitModulesLoaded ||
      event_mask & lldb::SBTarget::eBroadcastBitModulesUnloaded ||
      event_mask & lldb::SBTarget::eBroadcastBitSymbolsLoaded ||
      event_mask & lldb::SBTarget::eBroadcastBitSymbolsChanged) {
    const uint32_t num_modules = lldb::SBTarget::GetNumModulesFromEvent(event);
    const bool remove_module =
        event_mask & lldb::SBTarget::eBroadcastBitModulesUnloaded;

    // NOTE: Both mutexes must be acquired to prevent deadlock when
    // handling `modules_request`, which also requires both locks.
    lldb::SBMutex api_mutex = dap->GetAPIMutex();
    const std::scoped_lock<lldb::SBMutex, std::mutex> guard(api_mutex,
                                                            dap->modules_mutex);
    for (uint32_t i = 0; i < num_modules; ++i) {
      lldb::SBModule module =
          lldb::SBTarget::GetModuleAtIndexFromEvent(i, event);

      std::optional<protocol::Module> p_module =
          CreateModule(dap->target, module, remove_module);
      if (!p_module)
        continue;

      llvm::StringRef module_id = p_module->id;

      const bool module_exists = dap->modules.contains(module_id);
      if (remove_module && module_exists) {
        dap->modules.erase(module_id);
        dap->Send(protocol::Event{
            "module", protocol::ModuleEventBody{
                          std::move(p_module).value(),
                          protocol::ModuleEventBody::eReasonRemoved}});
      } else if (module_exists) {
        dap->Send(protocol::Event{
            "module", protocol::ModuleEventBody{
                          std::move(p_module).value(),
                          protocol::ModuleEventBody::eReasonChanged}});
      } else if (!remove_module) {
        dap->modules.insert(module_id);
        dap->Send(protocol::Event{
            "module",
            protocol::ModuleEventBody{std::move(p_module).value(),
                                      protocol::ModuleEventBody::eReasonNew}});
      }
    }
  } else if (event_mask & lldb::SBTarget::eBroadcastBitNewTargetCreated) {
    // For NewTargetCreated events, GetTargetFromEvent returns the parent
    // target, and GetCreatedTargetFromEvent returns the newly created target.
    lldb::SBTarget created_target =
        lldb::SBTarget::GetCreatedTargetFromEvent(event);

    if (!target.IsValid() || !created_target.IsValid()) {
      DAP_LOG(log, "Received NewTargetCreated event but parent or "
                   "created target is invalid");
      return;
    }

    // Send a startDebugging reverse request with the debugger and target
    // IDs. The new DAP instance will use these IDs to find the existing
    // debugger and target via FindDebuggerWithID and
    // FindTargetByGloballyUniqueID.
    llvm::json::Object configuration;
    configuration.try_emplace("type", "lldb");
    configuration.try_emplace("name", created_target.GetTargetSessionName());

    json::Object session{{"targetId", created_target.GetGloballyUniqueID()},
                         {"debuggerId", created_target.GetDebugger().GetID()}};
    configuration.try_emplace("session", std::move(session));

    llvm::json::Object request;
    request.try_emplace("request", "attach");
    request.try_emplace("configuration", std::move(configuration));

    dap->SendReverseRequest<LogFailureResponseHandler>("startDebugging",
                                                       std::move(request));
  }
}

static void HandleBreakpointEvent(const lldb::SBEvent &event, Log &log) {
  const uint32_t event_mask = event.GetType();
  if (!(event_mask & lldb::SBTarget::eBroadcastBitBreakpointChanged))
    return;

  lldb::SBBreakpoint bp = lldb::SBBreakpoint::GetBreakpointFromEvent(event);
  if (!bp.IsValid())
    return;

  // Find the DAP instance that owns this breakpoint's target.
  DAP *dap = DAPSessionManager::FindDAP(bp.GetTarget());
  if (!dap) {
    DAP_LOG(log, "Unable to find DAP instance for breakpoint");
    return;
  }

  auto event_type = lldb::SBBreakpoint::GetBreakpointEventTypeFromEvent(event);
  auto breakpoint = Breakpoint(*dap, bp);
  // If the breakpoint was set through DAP, it will have the
  // BreakpointBase::kDAPBreakpointLabel. Regardless of whether
  // locations were added, removed, or resolved, the breakpoint isn't
  // going away and the reason is always "changed".
  if ((event_type & lldb::eBreakpointEventTypeLocationsAdded ||
       event_type & lldb::eBreakpointEventTypeLocationsRemoved ||
       event_type & lldb::eBreakpointEventTypeLocationsResolved) &&
      breakpoint.MatchesName(BreakpointBase::kDAPBreakpointLabel)) {
    // As the DAP client already knows the path of this breakpoint, we
    // don't need to send it back as part of the "changed" event. This
    // avoids sending paths that should be source mapped. Note that
    // CreateBreakpoint doesn't apply source mapping and certain
    // implementation ignore the source part of this event anyway.
    protocol::Breakpoint protocol_bp = breakpoint.ToProtocolBreakpoint();

    // "source" is not needed here, unless we add adapter data to be
    // saved by the client.
    if (protocol_bp.source && !protocol_bp.source->adapterData)
      protocol_bp.source = std::nullopt;

    llvm::json::Object body;
    body.try_emplace("breakpoint", protocol_bp);
    body.try_emplace("reason", "changed");

    llvm::json::Object bp_event = CreateEventObject("breakpoint");
    bp_event.try_emplace("body", std::move(body));

    dap->SendJSON(llvm::json::Value(std::move(bp_event)));
  }
}

static void HandleThreadEvent(const lldb::SBEvent &event, Log &log) {
  uint32_t event_type = event.GetType();

  if (!(event_type & lldb::SBThread::eBroadcastBitStackChanged))
    return;

  lldb::SBThread thread = lldb::SBThread::GetThreadFromEvent(event);
  if (!thread.IsValid())
    return;

  // Find the DAP instance that owns this thread's process/target.
  DAP *dap = DAPSessionManager::FindDAP(thread.GetProcess().GetTarget());
  if (!dap) {
    DAP_LOG(log, "Unable to find DAP instance for thread");
    return;
  }

  SendInvalidatedEvent(*dap, {protocol::InvalidatedEventBody::eAreaStacks},
                       thread.GetThreadID());
}

static void HandleDiagnosticEvent(const lldb::SBEvent &event, Log &log) {
  // Global debugger events - send to all DAP instances.
  std::vector<DAP *> active_instances =
      DAPSessionManager::GetInstance().GetActiveSessions();
  for (DAP *dap_instance : active_instances) {
    if (!dap_instance)
      continue;

    lldb::SBStructuredData data =
        lldb::SBDebugger::GetDiagnosticFromEvent(event);
    if (!data.IsValid())
      continue;

    std::string type = GetStringValue(data.GetValueForKey("type"));
    std::string message = GetStringValue(data.GetValueForKey("message"));
    dap_instance->SendOutput(OutputType::Important,
                             llvm::formatv("{0}: {1}\n", type, message).str());
  }
}

// Note: EventThread() is architecturally different from the other functions in
// this file. While the functions above are event helpers that operate on a
// single DAP instance (taking `DAP &dap` as a parameter), EventThread() is a
// shared event processing loop that:
// 1. Listens to events from a shared debugger instance
// 2. Dispatches events to the appropriate handler, which internally finds the
//    DAP instance using DAPSessionManager::FindDAP()
// 3. Handles events for multiple different DAP sessions
// This allows multiple DAP sessions to share a single debugger and event
// thread, which is essential for the target handoff mechanism where child
// processes/targets are debugged in separate DAP sessions.
//
// All events from the debugger, target, process, thread and frames are
// received in this function that runs in its own thread. We are using a
// "FILE *" to output packets back to VS Code and they have mutexes in them
// them prevent multiple threads from writing simultaneously so no locking
// is required.
void EventThread(lldb::SBDebugger debugger, lldb::SBBroadcaster broadcaster,
                 llvm::StringRef client_name, Log &log) {
  std::string thread_name =
      llvm::formatv("lldb.DAP.client.{}.event_handler", client_name);
  if (thread_name.length() > llvm::get_max_thread_name_length())
    thread_name = llvm::formatv("DAP.{}.evt", client_name);
  llvm::set_thread_name(thread_name);

  lldb::SBListener listener = debugger.GetListener();
  broadcaster.AddListener(listener, eBroadcastBitStopEventThread);
  debugger.GetBroadcaster().AddListener(
      listener, lldb::eBroadcastBitError | lldb::eBroadcastBitWarning);

  // listen for thread events.
  listener.StartListeningForEventClass(
      debugger, lldb::SBThread::GetBroadcasterClassName(),
      lldb::SBThread::eBroadcastBitStackChanged);

  lldb::SBEvent event;
  bool done = false;
  while (!done) {
    if (!listener.WaitForEvent(UINT32_MAX, event))
      continue;

    const uint32_t event_mask = event.GetType();
    if (lldb::SBProcess::EventIsProcessEvent(event)) {
      HandleProcessEvent(event, /*&process_exited=*/done, log);
    } else if (lldb::SBTarget::EventIsTargetEvent(event)) {
      HandleTargetEvent(event, log);
    } else if (lldb::SBBreakpoint::EventIsBreakpointEvent(event)) {
      HandleBreakpointEvent(event, log);
    } else if (lldb::SBThread::EventIsThreadEvent(event)) {
      HandleThreadEvent(event, log);
    } else if (event_mask & lldb::eBroadcastBitError ||
               event_mask & lldb::eBroadcastBitWarning) {
      HandleDiagnosticEvent(event, log);
    } else if (event.BroadcasterMatchesRef(broadcaster)) {
      if (event_mask & eBroadcastBitStopEventThread) {
        done = true;
      }
    }
  }
}

} // namespace lldb_dap
