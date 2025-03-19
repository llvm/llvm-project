//===-- InitializeRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBStream.h"

using namespace lldb;

namespace lldb_dap {

static std::string GetStringFromStructuredData(lldb::SBStructuredData &data,
                                               const char *key) {
  lldb::SBStructuredData keyValue = data.GetValueForKey(key);
  if (!keyValue)
    return std::string();

  const size_t length = keyValue.GetStringValue(nullptr, 0);

  if (length == 0)
    return std::string();

  std::string str(length + 1, 0);
  keyValue.GetStringValue(&str[0], length + 1);
  return str;
}

static uint64_t GetUintFromStructuredData(lldb::SBStructuredData &data,
                                          const char *key) {
  lldb::SBStructuredData keyValue = data.GetValueForKey(key);

  if (!keyValue.IsValid())
    return 0;
  return keyValue.GetUnsignedIntegerValue();
}

void ProgressEventThreadFunction(DAP &dap) {
  lldb::SBListener listener("lldb-dap.progress.listener");
  dap.debugger.GetBroadcaster().AddListener(
      listener, lldb::SBDebugger::eBroadcastBitProgress |
                    lldb::SBDebugger::eBroadcastBitExternalProgress);
  dap.broadcaster.AddListener(listener, eBroadcastBitStopProgressThread);
  lldb::SBEvent event;
  bool done = false;
  while (!done) {
    if (listener.WaitForEvent(1, event)) {
      const auto event_mask = event.GetType();
      if (event.BroadcasterMatchesRef(dap.broadcaster)) {
        if (event_mask & eBroadcastBitStopProgressThread) {
          done = true;
        }
      } else {
        lldb::SBStructuredData data =
            lldb::SBDebugger::GetProgressDataFromEvent(event);

        const uint64_t progress_id =
            GetUintFromStructuredData(data, "progress_id");
        const uint64_t completed = GetUintFromStructuredData(data, "completed");
        const uint64_t total = GetUintFromStructuredData(data, "total");
        const std::string details =
            GetStringFromStructuredData(data, "details");

        if (completed == 0) {
          if (total == UINT64_MAX) {
            // This progress is non deterministic and won't get updated until it
            // is completed. Send the "message" which will be the combined title
            // and detail. The only other progress event for thus
            // non-deterministic progress will be the completed event So there
            // will be no need to update the detail.
            const std::string message =
                GetStringFromStructuredData(data, "message");
            dap.SendProgressEvent(progress_id, message.c_str(), completed,
                                  total);
          } else {
            // This progress is deterministic and will receive updates,
            // on the progress creation event VSCode will save the message in
            // the create packet and use that as the title, so we send just the
            // title in the progressCreate packet followed immediately by a
            // detail packet, if there is any detail.
            const std::string title =
                GetStringFromStructuredData(data, "title");
            dap.SendProgressEvent(progress_id, title.c_str(), completed, total);
            if (!details.empty())
              dap.SendProgressEvent(progress_id, details.c_str(), completed,
                                    total);
          }
        } else {
          // This progress event is either the end of the progress dialog, or an
          // update with possible detail. The "detail" string we send to VS Code
          // will be appended to the progress dialog's initial text from when it
          // was created.
          dap.SendProgressEvent(progress_id, details.c_str(), completed, total);
        }
      }
    }
  }
}

// All events from the debugger, target, process, thread and frames are
// received in this function that runs in its own thread. We are using a
// "FILE *" to output packets back to VS Code and they have mutexes in them
// them prevent multiple threads from writing simultaneously so no locking
// is required.
static void EventThreadFunction(DAP &dap) {
  llvm::set_thread_name(dap.transport.GetClientName() + ".event_handler");
  lldb::SBEvent event;
  lldb::SBListener listener = dap.debugger.GetListener();
  dap.broadcaster.AddListener(listener, eBroadcastBitStopEventThread);
  bool done = false;
  while (!done) {
    if (listener.WaitForEvent(1, event)) {
      const auto event_mask = event.GetType();
      if (lldb::SBProcess::EventIsProcessEvent(event)) {
        lldb::SBProcess process = lldb::SBProcess::GetProcessFromEvent(event);
        if (event_mask & lldb::SBProcess::eBroadcastBitStateChanged) {
          auto state = lldb::SBProcess::GetStateFromEvent(event);
          switch (state) {
          case lldb::eStateInvalid:
            // Not a state event
            break;
          case lldb::eStateUnloaded:
            break;
          case lldb::eStateConnected:
            break;
          case lldb::eStateAttaching:
            break;
          case lldb::eStateLaunching:
            break;
          case lldb::eStateStepping:
            break;
          case lldb::eStateCrashed:
            break;
          case lldb::eStateDetached:
            break;
          case lldb::eStateSuspended:
            break;
          case lldb::eStateStopped:
            // We launch and attach in synchronous mode then the first stop
            // event will not be delivered. If we use "launchCommands" during a
            // launch or "attachCommands" during an attach we might some process
            // stop events which we do not want to send an event for. We will
            // manually send a stopped event in request_configurationDone(...)
            // so don't send any before then.
            if (dap.configuration_done_sent) {
              // Only report a stopped event if the process was not
              // automatically restarted.
              if (!lldb::SBProcess::GetRestartedFromEvent(event)) {
                SendStdOutStdErr(dap, process);
                SendThreadStoppedEvent(dap);
              }
            }
            break;
          case lldb::eStateRunning:
            dap.WillContinue();
            SendContinuedEvent(dap);
            break;
          case lldb::eStateExited:
            lldb::SBStream stream;
            process.GetStatus(stream);
            dap.SendOutput(OutputType::Console, stream.GetData());

            // When restarting, we can get an "exited" event for the process we
            // just killed with the old PID, or even with no PID. In that case
            // we don't have to terminate the session.
            if (process.GetProcessID() == LLDB_INVALID_PROCESS_ID ||
                process.GetProcessID() == dap.restarting_process_id) {
              dap.restarting_process_id = LLDB_INVALID_PROCESS_ID;
            } else {
              // Run any exit LLDB commands the user specified in the
              // launch.json
              dap.RunExitCommands();
              SendProcessExitedEvent(dap, process);
              dap.SendTerminatedEvent();
              done = true;
            }
            break;
          }
        } else if ((event_mask & lldb::SBProcess::eBroadcastBitSTDOUT) ||
                   (event_mask & lldb::SBProcess::eBroadcastBitSTDERR)) {
          SendStdOutStdErr(dap, process);
        }
      } else if (lldb::SBBreakpoint::EventIsBreakpointEvent(event)) {
        if (event_mask & lldb::SBTarget::eBroadcastBitBreakpointChanged) {
          auto event_type =
              lldb::SBBreakpoint::GetBreakpointEventTypeFromEvent(event);
          auto bp = Breakpoint(
              dap, lldb::SBBreakpoint::GetBreakpointFromEvent(event));
          // If the breakpoint was set through DAP, it will have the
          // BreakpointBase::GetBreakpointLabel() label. Regardless
          // of whether  locations were added, removed, or resolved, the
          // breakpoint isn't going away and the reason is always "changed".
          if ((event_type & lldb::eBreakpointEventTypeLocationsAdded ||
               event_type & lldb::eBreakpointEventTypeLocationsRemoved ||
               event_type & lldb::eBreakpointEventTypeLocationsResolved) &&
              bp.MatchesName(BreakpointBase::GetBreakpointLabel())) {
            // As the DAP client already knows the path of this breakpoint, we
            // don't need to send it back as part of the "changed" event. This
            // avoids sending paths that should be source mapped. Note that
            // CreateBreakpoint doesn't apply source mapping and certain
            // implementation ignore the source part of this event anyway.
            llvm::json::Value source_bp = CreateBreakpoint(&bp);
            source_bp.getAsObject()->erase("source");

            llvm::json::Object body;
            body.try_emplace("breakpoint", source_bp);
            body.try_emplace("reason", "changed");

            llvm::json::Object bp_event = CreateEventObject("breakpoint");
            bp_event.try_emplace("body", std::move(body));

            dap.SendJSON(llvm::json::Value(std::move(bp_event)));
          }
        }
      } else if (event.BroadcasterMatchesRef(dap.broadcaster)) {
        if (event_mask & eBroadcastBitStopEventThread) {
          done = true;
        }
      }
    }
  }
}

// "InitializeRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Initialize request; value of command field is
//                     'initialize'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "initialize" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/InitializeRequestArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "InitializeRequestArguments": {
//   "type": "object",
//   "description": "Arguments for 'initialize' request.",
//   "properties": {
//     "clientID": {
//       "type": "string",
//       "description": "The ID of the (frontend) client using this adapter."
//     },
//     "adapterID": {
//       "type": "string",
//       "description": "The ID of the debug adapter."
//     },
//     "locale": {
//       "type": "string",
//       "description": "The ISO-639 locale of the (frontend) client using
//                       this adapter, e.g. en-US or de-CH."
//     },
//     "linesStartAt1": {
//       "type": "boolean",
//       "description": "If true all line numbers are 1-based (default)."
//     },
//     "columnsStartAt1": {
//       "type": "boolean",
//       "description": "If true all column numbers are 1-based (default)."
//     },
//     "pathFormat": {
//       "type": "string",
//       "_enum": [ "path", "uri" ],
//       "description": "Determines in what format paths are specified. The
//                       default is 'path', which is the native format."
//     },
//     "supportsVariableType": {
//       "type": "boolean",
//       "description": "Client supports the optional type attribute for
//                       variables."
//     },
//     "supportsVariablePaging": {
//       "type": "boolean",
//       "description": "Client supports the paging of variables."
//     },
//     "supportsRunInTerminalRequest": {
//       "type": "boolean",
//       "description": "Client supports the runInTerminal request."
//     }
//   },
//   "required": [ "adapterID" ]
// },
// "InitializeResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'initialize' request.",
//     "properties": {
//       "body": {
//         "$ref": "#/definitions/Capabilities",
//         "description": "The capabilities of this debug adapter."
//       }
//     }
//   }]
// }
void InitializeRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;

  const auto *arguments = request.getObject("arguments");
  // sourceInitFile option is not from formal DAP specification. It is only
  // used by unit tests to prevent sourcing .lldbinit files from environment
  // which may affect the outcome of tests.
  bool source_init_file =
      GetBoolean(arguments, "sourceInitFile").value_or(true);

  // Do not source init files until in/out/err are configured.
  dap.debugger = lldb::SBDebugger::Create(false);
  dap.debugger.SetInputFile(dap.in);
  auto out_fd = dap.out.GetWriteFileDescriptor();
  if (llvm::Error err = out_fd.takeError()) {
    response["success"] = false;
    EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }
  dap.debugger.SetOutputFile(lldb::SBFile(*out_fd, "w", false));
  auto err_fd = dap.err.GetWriteFileDescriptor();
  if (llvm::Error err = err_fd.takeError()) {
    response["success"] = false;
    EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }
  dap.debugger.SetErrorFile(lldb::SBFile(*err_fd, "w", false));

  auto interp = dap.debugger.GetCommandInterpreter();

  if (source_init_file) {
    dap.debugger.SkipLLDBInitFiles(false);
    dap.debugger.SkipAppInitFiles(false);
    lldb::SBCommandReturnObject init;
    interp.SourceInitFileInGlobalDirectory(init);
    interp.SourceInitFileInHomeDirectory(init);
  }

  if (llvm::Error err = dap.RunPreInitCommands()) {
    response["success"] = false;
    EmplaceSafeString(response, "message", llvm::toString(std::move(err)));
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  dap.PopulateExceptionBreakpoints();
  auto cmd = dap.debugger.GetCommandInterpreter().AddMultiwordCommand(
      "lldb-dap", "Commands for managing lldb-dap.");
  if (GetBoolean(arguments, "supportsStartDebuggingRequest").value_or(false)) {
    cmd.AddCommand(
        "start-debugging", new StartDebuggingRequestHandler(dap),
        "Sends a startDebugging request from the debug adapter to the client "
        "to start a child debug session of the same type as the caller.");
  }
  cmd.AddCommand(
      "repl-mode", new ReplModeRequestHandler(dap),
      "Get or set the repl behavior of lldb-dap evaluation requests.");
  cmd.AddCommand("send-event", new SendEventRequestHandler(dap),
                 "Sends an DAP event to the client.");

  dap.progress_event_thread =
      std::thread(ProgressEventThreadFunction, std::ref(dap));

  // Start our event thread so we can receive events from the debugger, target,
  // process and more.
  dap.event_thread = std::thread(EventThreadFunction, std::ref(dap));

  llvm::StringMap<bool> capabilities = dap.GetCapabilities();
  for (auto &kv : capabilities)
    body.try_emplace(kv.getKey(), kv.getValue());

  // Available filters or options for the setExceptionBreakpoints request.
  llvm::json::Array filters;
  for (const auto &exc_bp : *dap.exception_breakpoints)
    filters.emplace_back(CreateExceptionBreakpointFilter(exc_bp));
  body.try_emplace("exceptionBreakpointFilters", std::move(filters));

  llvm::json::Array completion_characters;
  completion_characters.emplace_back(".");
  completion_characters.emplace_back(" ");
  completion_characters.emplace_back("\t");
  body.try_emplace("completionTriggerCharacters",
                   std::move(completion_characters));

  // Put in non-DAP specification lldb specific information.
  llvm::json::Object lldb_json;
  lldb_json.try_emplace("version", dap.debugger.GetVersionString());
  body.try_emplace("__lldb", std::move(lldb_json));

  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
