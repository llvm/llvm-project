//===-- lldb-vscode.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(_WIN32)
// We need to #define NOMINMAX in order to skip `min()` and `max()` macro
// definitions that conflict with other system headers.
// We also need to #undef GetObject (which is defined to GetObjectW) because
// the JSON code we use also has methods named `GetObject()` and we conflict
// against these.
#define NOMINMAX
#include <windows.h>
#undef GetObject
#include <io.h>
#else
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "VSCode.h"

#if defined(_WIN32)
#define PATH_MAX MAX_PATH
typedef int socklen_t;
constexpr const char *dev_null_path = "nul";

#else
constexpr const char *dev_null_path = "/dev/null";

#endif

using namespace lldb_vscode;

namespace {

typedef void (*RequestCallback)(const llvm::json::Object &command);

enum LaunchMethod { Launch, Attach, AttachForSuspendedLaunch };

enum VSCodeBroadcasterBits { eBroadcastBitStopEventThread = 1u << 0 };

SOCKET AcceptConnection(int portno) {
  // Accept a socket connection from any host on "portno".
  SOCKET newsockfd = -1;
  struct sockaddr_in serv_addr, cli_addr;
  SOCKET sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    if (g_vsc.log)
      *g_vsc.log << "error: opening socket (" << strerror(errno) << ")"
                 << std::endl;
  } else {
    memset((char *)&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    // serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      if (g_vsc.log)
        *g_vsc.log << "error: binding socket (" << strerror(errno) << ")"
                   << std::endl;
    } else {
      listen(sockfd, 5);
      socklen_t clilen = sizeof(cli_addr);
      newsockfd = accept(sockfd, (struct sockaddr *)&cli_addr, &clilen);
      if (newsockfd < 0)
        if (g_vsc.log)
          *g_vsc.log << "error: accept (" << strerror(errno) << ")"
                     << std::endl;
    }
#if defined(_WIN32)
    closesocket(sockfd);
#else
    close(sockfd);
#endif
  }
  return newsockfd;
}

std::vector<const char *> MakeArgv(const llvm::ArrayRef<std::string> &strs) {
  // Create and return an array of "const char *", one for each C string in
  // "strs" and terminate the list with a NULL. This can be used for argument
  // vectors (argv) or environment vectors (envp) like those passed to the
  // "main" function in C programs.
  std::vector<const char *> argv;
  for (const auto &s : strs)
    argv.push_back(s.c_str());
  argv.push_back(nullptr);
  return argv;
}

//----------------------------------------------------------------------
// Send a "exited" event to indicate the process has exited.
//----------------------------------------------------------------------
void SendProcessExitedEvent(lldb::SBProcess &process) {
  llvm::json::Object event(CreateEventObject("exited"));
  llvm::json::Object body;
  body.try_emplace("exitCode", (int64_t)process.GetExitStatus());
  event.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(event)));
}

void SendThreadExitedEvent(lldb::tid_t tid) {
  llvm::json::Object event(CreateEventObject("thread"));
  llvm::json::Object body;
  body.try_emplace("reason", "exited");
  body.try_emplace("threadId", (int64_t)tid);
  event.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(event)));
}

//----------------------------------------------------------------------
// Send a "terminated" event to indicate the process is done being
// debugged.
//----------------------------------------------------------------------
void SendTerminatedEvent() {
  if (!g_vsc.sent_terminated_event) {
    g_vsc.sent_terminated_event = true;
    // Send a "terminated" event
    llvm::json::Object event(CreateEventObject("terminated"));
    g_vsc.SendJSON(llvm::json::Value(std::move(event)));
  }
}

//----------------------------------------------------------------------
// Send a thread stopped event for all threads as long as the process
// is stopped.
//----------------------------------------------------------------------
void SendThreadStoppedEvent() {
  lldb::SBProcess process = g_vsc.target.GetProcess();
  if (process.IsValid()) {
    auto state = process.GetState();
    if (state == lldb::eStateStopped) {
      llvm::DenseSet<lldb::tid_t> old_thread_ids;
      old_thread_ids.swap(g_vsc.thread_ids);
      uint32_t stop_id = process.GetStopID();
      const uint32_t num_threads = process.GetNumThreads();

      // First make a pass through the threads to see if the focused thread
      // has a stop reason. In case the focus thread doesn't have a stop
      // reason, remember the first thread that has a stop reason so we can
      // set it as the focus thread if below if needed.
      lldb::tid_t first_tid_with_reason = LLDB_INVALID_THREAD_ID;
      uint32_t num_threads_with_reason = 0;
      for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
        const lldb::tid_t tid = thread.GetThreadID();
        const bool has_reason = ThreadHasStopReason(thread);
        // If the focus thread doesn't have a stop reason, clear the thread ID
        if (tid == g_vsc.focus_tid && !has_reason)
          g_vsc.focus_tid = LLDB_INVALID_THREAD_ID;
        if (has_reason) {
          ++num_threads_with_reason;
          if (first_tid_with_reason == LLDB_INVALID_THREAD_ID)
            first_tid_with_reason = tid;
        }
      }

      // We will have cleared g_vsc.focus_tid if he focus thread doesn't
      // have a stop reason, so if it was cleared, or wasn't set, then set the
      // focus thread to the first thread with a stop reason.
      if (g_vsc.focus_tid == LLDB_INVALID_THREAD_ID)
        g_vsc.focus_tid = first_tid_with_reason;

      // If no threads stopped with a reason, then report the first one so
      // we at least let the UI know we stopped.
      if (num_threads_with_reason == 0) {
        lldb::SBThread thread = process.GetThreadAtIndex(0);
        g_vsc.SendJSON(CreateThreadStopped(thread, stop_id));
      } else {
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
          lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
          g_vsc.thread_ids.insert(thread.GetThreadID());
          if (ThreadHasStopReason(thread)) {
            g_vsc.SendJSON(CreateThreadStopped(thread, stop_id));
          }
        }
      }

      for (auto tid : old_thread_ids) {
        auto end = g_vsc.thread_ids.end();
        auto pos = g_vsc.thread_ids.find(tid);
        if (pos == end)
          SendThreadExitedEvent(tid);
      }
    } else {
      if (g_vsc.log)
        *g_vsc.log << "error: SendThreadStoppedEvent() when process"
                      " isn't stopped ("
                   << lldb::SBDebugger::StateAsCString(state) << ')'
                   << std::endl;
    }
  } else {
    if (g_vsc.log)
      *g_vsc.log << "error: SendThreadStoppedEvent() invalid process"
                 << std::endl;
  }
  g_vsc.RunStopCommands();
}

//----------------------------------------------------------------------
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
//----------------------------------------------------------------------
void SendProcessEvent(LaunchMethod launch_method) {
  lldb::SBFileSpec exe_fspec = g_vsc.target.GetExecutable();
  char exe_path[PATH_MAX];
  exe_fspec.GetPath(exe_path, sizeof(exe_path));
  llvm::json::Object event(CreateEventObject("process"));
  llvm::json::Object body;
  EmplaceSafeString(body, "name", std::string(exe_path));
  const auto pid = g_vsc.target.GetProcess().GetProcessID();
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
  g_vsc.SendJSON(llvm::json::Value(std::move(event)));
}

//----------------------------------------------------------------------
// Grab any STDOUT and STDERR from the process and send it up to VS Code
// via an "output" event to the "stdout" and "stderr" categories.
//----------------------------------------------------------------------
void SendStdOutStdErr(lldb::SBProcess &process) {
  char buffer[1024];
  size_t count;
  while ((count = process.GetSTDOUT(buffer, sizeof(buffer))) > 0)
  g_vsc.SendOutput(OutputType::Stdout, llvm::StringRef(buffer, count));
  while ((count = process.GetSTDERR(buffer, sizeof(buffer))) > 0)
    g_vsc.SendOutput(OutputType::Stderr, llvm::StringRef(buffer, count));
}

//----------------------------------------------------------------------
// All events from the debugger, target, process, thread and frames are
// received in this function that runs in its own thread. We are using a
// "FILE *" to output packets back to VS Code and they have mutexes in them
// them prevent multiple threads from writing simultaneously so no locking
// is required.
//----------------------------------------------------------------------
void EventThreadFunction() {
  lldb::SBEvent event;
  lldb::SBListener listener = g_vsc.debugger.GetListener();
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
            // Only report a stopped event if the process was not restarted.
            if (!lldb::SBProcess::GetRestartedFromEvent(event)) {
              SendStdOutStdErr(process);
              SendThreadStoppedEvent();
            }
            break;
          case lldb::eStateRunning:
            break;
          case lldb::eStateExited: {
            // Run any exit LLDB commands the user specified in the
            // launch.json
            g_vsc.RunExitCommands();
            SendProcessExitedEvent(process);
            SendTerminatedEvent();
            done = true;
          } break;
          }
        } else if ((event_mask & lldb::SBProcess::eBroadcastBitSTDOUT) ||
                   (event_mask & lldb::SBProcess::eBroadcastBitSTDERR)) {
          SendStdOutStdErr(process);
        }
      } else if (lldb::SBBreakpoint::EventIsBreakpointEvent(event)) {
        if (event_mask & lldb::SBTarget::eBroadcastBitBreakpointChanged) {
          auto event_type =
              lldb::SBBreakpoint::GetBreakpointEventTypeFromEvent(event);
          const auto num_locs =
              lldb::SBBreakpoint::GetNumBreakpointLocationsFromEvent(event);
          auto bp = lldb::SBBreakpoint::GetBreakpointFromEvent(event);
          bool added = event_type & lldb::eBreakpointEventTypeLocationsAdded;
          bool removed =
              event_type & lldb::eBreakpointEventTypeLocationsRemoved;
          if (added || removed) {
            for (size_t i = 0; i < num_locs; ++i) {
              auto bp_loc =
                  lldb::SBBreakpoint::GetBreakpointLocationAtIndexFromEvent(
                      event, i);
              auto bp_event = CreateEventObject("breakpoint");
              llvm::json::Object body;
              body.try_emplace("breakpoint", CreateBreakpoint(bp_loc));
              if (added)
                body.try_emplace("reason", "new");
              else
                body.try_emplace("reason", "removed");
              bp_event.try_emplace("body", std::move(body));
              g_vsc.SendJSON(llvm::json::Value(std::move(bp_event)));
            }
          }
        }
      } else if (event.BroadcasterMatchesRef(g_vsc.broadcaster)) {
        if (event_mask & eBroadcastBitStopEventThread) {
          done = true;
        }
      }
    }
  }
}

//----------------------------------------------------------------------
// Both attach and launch take a either a sourcePath or sourceMap
// argument (or neither), from which we need to set the target.source-map.
//----------------------------------------------------------------------
void SetSourceMapFromArguments(const llvm::json::Object &arguments) {
  const char *sourceMapHelp =
      "source must be be an array of two-element arrays, "
      "each containing a source and replacement path string.\n";

  std::string sourceMapCommand;
  llvm::raw_string_ostream strm(sourceMapCommand);
  strm << "settings set target.source-map ";
  auto sourcePath = GetString(arguments, "sourcePath");

  // sourceMap is the new, more general form of sourcePath and overrides it.
  auto sourceMap = arguments.getArray("sourceMap");
  if (sourceMap) {
    for (const auto &value : *sourceMap) {
      auto mapping = value.getAsArray();
      if (mapping == nullptr || mapping->size() != 2 ||
          (*mapping)[0].kind() != llvm::json::Value::String ||
          (*mapping)[1].kind() != llvm::json::Value::String) {
        g_vsc.SendOutput(OutputType::Console, llvm::StringRef(sourceMapHelp));
        return;
      }
      auto mapFrom = GetAsString((*mapping)[0]);
      auto mapTo = GetAsString((*mapping)[1]);
      strm << "\"" << mapFrom << "\" \"" << mapTo << "\"";
    }
  } else {
    if (ObjectContainsKey(arguments, "sourceMap")) {
      g_vsc.SendOutput(OutputType::Console, llvm::StringRef(sourceMapHelp));
      return;
    }
    if (sourcePath.empty())
      return;
    // Do any source remapping needed before we create our targets
    strm << "\".\" \"" << sourcePath << "\"";
  }
  strm.flush();
  if (!sourceMapCommand.empty()) {
    g_vsc.RunLLDBCommands("Setting source map:", {sourceMapCommand});
  }
}

//----------------------------------------------------------------------
// "AttachRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Attach request; value of command field is 'attach'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "attach" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/AttachRequestArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "AttachRequestArguments": {
//   "type": "object",
//   "description": "Arguments for 'attach' request.\nThe attach request has no
//   standardized attributes."
// },
// "AttachResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'attach' request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// }
//----------------------------------------------------------------------
void request_attach(const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  const lldb::pid_t pid =
      GetUnsigned(arguments, "pid", LLDB_INVALID_PROCESS_ID);
  if (pid != LLDB_INVALID_PROCESS_ID)
    g_vsc.attach_info.SetProcessID(pid);
  const auto wait_for = GetBoolean(arguments, "waitFor", false);
  g_vsc.attach_info.SetWaitForLaunch(wait_for, false /*async*/);
  g_vsc.init_commands = GetStrings(arguments, "initCommands");
  g_vsc.pre_run_commands = GetStrings(arguments, "preRunCommands");
  g_vsc.stop_commands = GetStrings(arguments, "stopCommands");
  g_vsc.exit_commands = GetStrings(arguments, "exitCommands");
  auto attachCommands = GetStrings(arguments, "attachCommands");
  g_vsc.stop_at_entry = GetBoolean(arguments, "stopOnEntry", false);
  const auto debuggerRoot = GetString(arguments, "debuggerRoot");

  // This is a hack for loading DWARF in .o files on Mac where the .o files
  // in the debug map of the main executable have relative paths which require
  // the lldb-vscode binary to have its working directory set to that relative
  // root for the .o files in order to be able to load debug info.
  if (!debuggerRoot.empty()) {
    llvm::sys::fs::set_current_path(debuggerRoot.data());
  }

  // Run any initialize LLDB commands the user specified in the launch.json
  g_vsc.RunInitCommands();

  // Grab the name of the program we need to debug and set it as the first
  // argument that will be passed to the program we will debug.
  const auto program = GetString(arguments, "program");
  if (!program.empty()) {
    lldb::SBFileSpec program_fspec(program.data(), true /*resolve_path*/);

    g_vsc.launch_info.SetExecutableFile(program_fspec,
                                        false /*add_as_first_arg*/);
    const char *target_triple = nullptr;
    const char *uuid_cstr = nullptr;
    // Stand alone debug info file if different from executable
    const char *symfile = nullptr;
    g_vsc.target.AddModule(program.data(), target_triple, uuid_cstr, symfile);
    if (error.Fail()) {
      response["success"] = llvm::json::Value(false);
      EmplaceSafeString(response, "message", std::string(error.GetCString()));
      g_vsc.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }
  }

  const bool detatchOnError = GetBoolean(arguments, "detachOnError", false);
  g_vsc.launch_info.SetDetachOnError(detatchOnError);

  // Run any pre run LLDB commands the user specified in the launch.json
  g_vsc.RunPreRunCommands();

  if (pid == LLDB_INVALID_PROCESS_ID && wait_for) {
    char attach_info[256];
    auto attach_info_len =
        snprintf(attach_info, sizeof(attach_info),
                 "Waiting to attach to \"%s\"...", program.data());
    g_vsc.SendOutput(OutputType::Console, llvm::StringRef(attach_info,
                                                          attach_info_len));
  }
  if (attachCommands.empty()) {
    // No "attachCommands", just attach normally.
    // Disable async events so the attach will be successful when we return from
    // the launch call and the launch will happen synchronously
    g_vsc.debugger.SetAsync(false);
    g_vsc.target.Attach(g_vsc.attach_info, error);
    // Reenable async events
    g_vsc.debugger.SetAsync(true);
  } else {
    // We have "attachCommands" that are a set of commands that are expected
    // to execute the commands after which a process should be created. If there
    // is no valid process after running these commands, we have failed.
    g_vsc.RunLLDBCommands("Running attachCommands:", attachCommands);
    // The custom commands might have created a new target so we should use the
    // selected target after these commands are run.
    g_vsc.target = g_vsc.debugger.GetSelectedTarget();
  }

  SetSourceMapFromArguments(*arguments);

  if (error.Success()) {
    auto attached_pid = g_vsc.target.GetProcess().GetProcessID();
    if (attached_pid == LLDB_INVALID_PROCESS_ID) {
      if (attachCommands.empty())
        error.SetErrorString("failed to attach to a process");
      else
        error.SetErrorString("attachCommands failed to attach to a process");
    }
  }

  if (error.Fail()) {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message", std::string(error.GetCString()));
  }
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
  if (error.Success()) {
    SendProcessEvent(Attach);
    g_vsc.SendJSON(CreateEventObject("initialized"));
    // SendThreadStoppedEvent();
  }
}

//----------------------------------------------------------------------
// "ContinueRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Continue request; value of command field is 'continue'.
//                     The request starts the debuggee to run again.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "continue" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/ContinueArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "ContinueArguments": {
//   "type": "object",
//   "description": "Arguments for 'continue' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Continue execution for the specified thread (if
//                       possible). If the backend cannot continue on a single
//                       thread but will continue on all threads, it should
//                       set the allThreadsContinued attribute in the response
//                       to true."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "ContinueResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'continue' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "allThreadsContinued": {
//             "type": "boolean",
//             "description": "If true, the continue request has ignored the
//                             specified thread and continued all threads
//                             instead. If this attribute is missing a value
//                             of 'true' is assumed for backward
//                             compatibility."
//           }
//         }
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
//----------------------------------------------------------------------
void request_continue(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  lldb::SBProcess process = g_vsc.target.GetProcess();
  auto arguments = request.getObject("arguments");
  // Remember the thread ID that caused the resume so we can set the
  // "threadCausedFocus" boolean value in the "stopped" events.
  g_vsc.focus_tid = GetUnsigned(arguments, "threadId", LLDB_INVALID_THREAD_ID);
  lldb::SBError error = process.Continue();
  llvm::json::Object body;
  body.try_emplace("allThreadsContinued", true);
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "ConfigurationDoneRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//             "type": "object",
//             "description": "ConfigurationDone request; value of command field
//             is 'configurationDone'.\nThe client of the debug protocol must
//             send this request at the end of the sequence of configuration
//             requests (which was started by the InitializedEvent).",
//             "properties": {
//             "command": {
//             "type": "string",
//             "enum": [ "configurationDone" ]
//             },
//             "arguments": {
//             "$ref": "#/definitions/ConfigurationDoneArguments"
//             }
//             },
//             "required": [ "command" ]
//             }]
// },
// "ConfigurationDoneArguments": {
//   "type": "object",
//   "description": "Arguments for 'configurationDone' request.\nThe
//   configurationDone request has no standardized attributes."
// },
// "ConfigurationDoneResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//             "type": "object",
//             "description": "Response to 'configurationDone' request. This is
//             just an acknowledgement, so no body field is required."
//             }]
// },
//----------------------------------------------------------------------
void request_configurationDone(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
  if (g_vsc.stop_at_entry)
    SendThreadStoppedEvent();
  else
    g_vsc.target.GetProcess().Continue();
}

//----------------------------------------------------------------------
// "DisconnectRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Disconnect request; value of command field is
//                     'disconnect'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "disconnect" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/DisconnectArguments"
//       }
//     },
//     "required": [ "command" ]
//   }]
// },
// "DisconnectArguments": {
//   "type": "object",
//   "description": "Arguments for 'disconnect' request.",
//   "properties": {
//     "terminateDebuggee": {
//       "type": "boolean",
//       "description": "Indicates whether the debuggee should be terminated
//                       when the debugger is disconnected. If unspecified,
//                       the debug adapter is free to do whatever it thinks
//                       is best. A client can only rely on this attribute
//                       being properly honored if a debug adapter returns
//                       true for the 'supportTerminateDebuggee' capability."
//     },
//     "restart": {
//       "type": "boolean",
//       "description": "Indicates whether the debuggee should be restart
//                       the process."
//     }
//   }
// },
// "DisconnectResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'disconnect' request. This is just an
//                     acknowledgement, so no body field is required."
//   }]
// }
//----------------------------------------------------------------------
void request_disconnect(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");

  bool terminateDebuggee = GetBoolean(arguments, "terminateDebuggee", false);
  lldb::SBProcess process = g_vsc.target.GetProcess();
  auto state = process.GetState();

  switch (state) {
  case lldb::eStateInvalid:
  case lldb::eStateUnloaded:
  case lldb::eStateDetached:
  case lldb::eStateExited:
    break;
  case lldb::eStateConnected:
  case lldb::eStateAttaching:
  case lldb::eStateLaunching:
  case lldb::eStateStepping:
  case lldb::eStateCrashed:
  case lldb::eStateSuspended:
  case lldb::eStateStopped:
  case lldb::eStateRunning:
    g_vsc.debugger.SetAsync(false);
    if (terminateDebuggee)
      process.Kill();
    else
      process.Detach();
    g_vsc.debugger.SetAsync(true);
    break;
  }
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
  SendTerminatedEvent();
  if (g_vsc.event_thread.joinable()) {
    g_vsc.broadcaster.BroadcastEventByType(eBroadcastBitStopEventThread);
    g_vsc.event_thread.join();
  }
}

void request_exceptionInfo(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  llvm::json::Object body;
  lldb::SBThread thread = g_vsc.GetLLDBThread(*arguments);
  if (thread.IsValid()) {
    auto stopReason = thread.GetStopReason();
    if (stopReason == lldb::eStopReasonSignal)
      body.try_emplace("exceptionId", "signal");
    else if (stopReason == lldb::eStopReasonBreakpoint) {
      ExceptionBreakpoint *exc_bp = g_vsc.GetExceptionBPFromStopReason(thread);
      if (exc_bp) {
        EmplaceSafeString(body, "exceptionId", exc_bp->filter);
        EmplaceSafeString(body, "description", exc_bp->label);
      } else {
        body.try_emplace("exceptionId", "exception");
      }
    } else {
      body.try_emplace("exceptionId", "exception");
    }
    if (!ObjectContainsKey(body, "description")) {
      char description[1024];
      if (thread.GetStopDescription(description, sizeof(description))) {
        EmplaceSafeString(body, "description", std::string(description));
      }
    }
    body.try_emplace("breakMode", "always");
    // auto excInfoCount = thread.GetStopReasonDataCount();
    // for (auto i=0; i<excInfoCount; ++i) {
    //   uint64_t exc_data = thread.GetStopReasonDataAtIndex(i);
    // }
  } else {
    response["success"] = llvm::json::Value(false);
  }
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
//  "EvaluateRequest": {
//    "allOf": [ { "$ref": "#/definitions/Request" }, {
//      "type": "object",
//      "description": "Evaluate request; value of command field is 'evaluate'.
//                      Evaluates the given expression in the context of the
//                      top most stack frame. The expression has access to any
//                      variables and arguments that are in scope.",
//      "properties": {
//        "command": {
//          "type": "string",
//          "enum": [ "evaluate" ]
//        },
//        "arguments": {
//          "$ref": "#/definitions/EvaluateArguments"
//        }
//      },
//      "required": [ "command", "arguments"  ]
//    }]
//  },
//  "EvaluateArguments": {
//    "type": "object",
//    "description": "Arguments for 'evaluate' request.",
//    "properties": {
//      "expression": {
//        "type": "string",
//        "description": "The expression to evaluate."
//      },
//      "frameId": {
//        "type": "integer",
//        "description": "Evaluate the expression in the scope of this stack
//                        frame. If not specified, the expression is evaluated
//                        in the global scope."
//      },
//      "context": {
//        "type": "string",
//        "_enum": [ "watch", "repl", "hover" ],
//        "enumDescriptions": [
//          "evaluate is run in a watch.",
//          "evaluate is run from REPL console.",
//          "evaluate is run from a data hover."
//        ],
//        "description": "The context in which the evaluate request is run."
//      },
//      "format": {
//        "$ref": "#/definitions/ValueFormat",
//        "description": "Specifies details on how to format the Evaluate
//                        result."
//      }
//    },
//    "required": [ "expression" ]
//  },
//  "EvaluateResponse": {
//    "allOf": [ { "$ref": "#/definitions/Response" }, {
//      "type": "object",
//      "description": "Response to 'evaluate' request.",
//      "properties": {
//        "body": {
//          "type": "object",
//          "properties": {
//            "result": {
//              "type": "string",
//              "description": "The result of the evaluate request."
//            },
//            "type": {
//              "type": "string",
//              "description": "The optional type of the evaluate result."
//            },
//            "presentationHint": {
//              "$ref": "#/definitions/VariablePresentationHint",
//              "description": "Properties of a evaluate result that can be
//                              used to determine how to render the result in
//                              the UI."
//            },
//            "variablesReference": {
//              "type": "number",
//              "description": "If variablesReference is > 0, the evaluate
//                              result is structured and its children can be
//                              retrieved by passing variablesReference to the
//                              VariablesRequest."
//            },
//            "namedVariables": {
//              "type": "number",
//              "description": "The number of named child variables. The
//                              client can use this optional information to
//                              present the variables in a paged UI and fetch
//                              them in chunks."
//            },
//            "indexedVariables": {
//              "type": "number",
//              "description": "The number of indexed child variables. The
//                              client can use this optional information to
//                              present the variables in a paged UI and fetch
//                              them in chunks."
//            }
//          },
//          "required": [ "result", "variablesReference" ]
//        }
//      },
//      "required": [ "body" ]
//    }]
//  }
//----------------------------------------------------------------------
void request_evaluate(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  auto arguments = request.getObject("arguments");
  lldb::SBFrame frame = g_vsc.GetLLDBFrame(*arguments);
  const auto expression = GetString(arguments, "expression");

  if (!expression.empty() && expression[0] == '`') {
    auto result = RunLLDBCommands(llvm::StringRef(),
                                     {expression.substr(1)});
    EmplaceSafeString(body, "result", result);
    body.try_emplace("variablesReference", (int64_t)0);
  } else {
    // Always try to get the answer from the local variables if possible. If
    // this fails, then actually evaluate an expression using the expression
    // parser. "frame variable" is more reliable than the expression parser in
    // many cases and it is faster.
    lldb::SBValue value = frame.GetValueForVariablePath(
        expression.data(), lldb::eDynamicDontRunTarget);
    if (value.GetError().Fail())
      value = frame.EvaluateExpression(expression.data());
    if (value.GetError().Fail()) {
      response["success"] = llvm::json::Value(false);
      const char *error_cstr = value.GetError().GetCString();
      if (error_cstr && error_cstr[0])
        EmplaceSafeString(response, "message", std::string(error_cstr));
      else
        EmplaceSafeString(response, "message", "evaluate failed");
    } else {
      SetValueForKey(value, body, "result");
      auto value_typename = value.GetType().GetDisplayTypeName();
      EmplaceSafeString(body, "type", value_typename ? value_typename : NO_TYPENAME);
      if (value.MightHaveChildren()) {
        auto variablesReference = VARIDX_TO_VARREF(g_vsc.variables.GetSize());
        g_vsc.variables.Append(value);
        body.try_emplace("variablesReference", variablesReference);
      } else {
        body.try_emplace("variablesReference", (int64_t)0);
      }
    }
  }
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
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
//----------------------------------------------------------------------
void request_initialize(const llvm::json::Object &request) {
  g_vsc.debugger = lldb::SBDebugger::Create(true /*source_init_files*/);
  // Create an empty target right away since we might get breakpoint requests
  // before we are given an executable to launch in a "launch" request, or a
  // executable when attaching to a process by process ID in a "attach"
  // request.
  FILE *out = fopen(dev_null_path, "w");
  if (out) {
    // Set the output and error file handles to redirect into nothing otherwise
    // if any code in LLDB prints to the debugger file handles, the output and
    // error file handles are initialized to STDOUT and STDERR and any output
    // will kill our debug session.
    g_vsc.debugger.SetOutputFileHandle(out, true);
    g_vsc.debugger.SetErrorFileHandle(out, false);
  }

  g_vsc.target = g_vsc.debugger.CreateTarget(nullptr);
  lldb::SBListener listener = g_vsc.debugger.GetListener();
  listener.StartListeningForEvents(
      g_vsc.target.GetBroadcaster(),
      lldb::SBTarget::eBroadcastBitBreakpointChanged);
  listener.StartListeningForEvents(g_vsc.broadcaster,
                                   eBroadcastBitStopEventThread);
  // Start our event thread so we can receive events from the debugger, target,
  // process and more.
  g_vsc.event_thread = std::thread(EventThreadFunction);

  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  // The debug adapter supports the configurationDoneRequest.
  body.try_emplace("supportsConfigurationDoneRequest", true);
  // The debug adapter supports function breakpoints.
  body.try_emplace("supportsFunctionBreakpoints", true);
  // The debug adapter supports conditional breakpoints.
  body.try_emplace("supportsConditionalBreakpoints", true);
  // The debug adapter supports breakpoints that break execution after a
  // specified number of hits.
  body.try_emplace("supportsHitConditionalBreakpoints", true);
  // The debug adapter supports a (side effect free) evaluate request for
  // data hovers.
  body.try_emplace("supportsEvaluateForHovers", true);
  // Available filters or options for the setExceptionBreakpoints request.
  llvm::json::Array filters;
  for (const auto &exc_bp : g_vsc.exception_breakpoints) {
    filters.emplace_back(CreateExceptionBreakpointFilter(exc_bp));
  }
  body.try_emplace("exceptionBreakpointFilters", std::move(filters));
  // The debug adapter supports stepping back via the stepBack and
  // reverseContinue requests.
  body.try_emplace("supportsStepBack", false);
  // The debug adapter supports setting a variable to a value.
  body.try_emplace("supportsSetVariable", true);
  // The debug adapter supports restarting a frame.
  body.try_emplace("supportsRestartFrame", false);
  // The debug adapter supports the gotoTargetsRequest.
  body.try_emplace("supportsGotoTargetsRequest", false);
  // The debug adapter supports the stepInTargetsRequest.
  body.try_emplace("supportsStepInTargetsRequest", false);
  // The debug adapter supports the completionsRequest.
  body.try_emplace("supportsCompletionsRequest", false);
  // The debug adapter supports the modules request.
  body.try_emplace("supportsModulesRequest", false);
  // The set of additional module information exposed by the debug adapter.
  //   body.try_emplace("additionalModuleColumns"] = ColumnDescriptor
  // Checksum algorithms supported by the debug adapter.
  //   body.try_emplace("supportedChecksumAlgorithms"] = ChecksumAlgorithm
  // The debug adapter supports the RestartRequest. In this case a client
  // should not implement 'restart' by terminating and relaunching the adapter
  // but by calling the RestartRequest.
  body.try_emplace("supportsRestartRequest", false);
  // The debug adapter supports 'exceptionOptions' on the
  // setExceptionBreakpoints request.
  body.try_emplace("supportsExceptionOptions", true);
  // The debug adapter supports a 'format' attribute on the stackTraceRequest,
  // variablesRequest, and evaluateRequest.
  body.try_emplace("supportsValueFormattingOptions", true);
  // The debug adapter supports the exceptionInfo request.
  body.try_emplace("supportsExceptionInfoRequest", true);
  // The debug adapter supports the 'terminateDebuggee' attribute on the
  // 'disconnect' request.
  body.try_emplace("supportTerminateDebuggee", true);
  // The debug adapter supports the delayed loading of parts of the stack,
  // which requires that both the 'startFrame' and 'levels' arguments and the
  // 'totalFrames' result of the 'StackTrace' request are supported.
  body.try_emplace("supportsDelayedStackTraceLoading", true);
  // The debug adapter supports the 'loadedSources' request.
  body.try_emplace("supportsLoadedSourcesRequest", false);

  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "LaunchRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Launch request; value of command field is 'launch'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "launch" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/LaunchRequestArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "LaunchRequestArguments": {
//   "type": "object",
//   "description": "Arguments for 'launch' request.",
//   "properties": {
//     "noDebug": {
//       "type": "boolean",
//       "description": "If noDebug is true the launch request should launch
//                       the program without enabling debugging."
//     }
//   }
// },
// "LaunchResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'launch' request. This is just an
//                     acknowledgement, so no body field is required."
//   }]
// }
//----------------------------------------------------------------------
void request_launch(const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  g_vsc.init_commands = GetStrings(arguments, "initCommands");
  g_vsc.pre_run_commands = GetStrings(arguments, "preRunCommands");
  g_vsc.stop_commands = GetStrings(arguments, "stopCommands");
  g_vsc.exit_commands = GetStrings(arguments, "exitCommands");
  g_vsc.stop_at_entry = GetBoolean(arguments, "stopOnEntry", false);
  const auto debuggerRoot = GetString(arguments, "debuggerRoot");

  // This is a hack for loading DWARF in .o files on Mac where the .o files
  // in the debug map of the main executable have relative paths which require
  // the lldb-vscode binary to have its working directory set to that relative
  // root for the .o files in order to be able to load debug info.
  if (!debuggerRoot.empty()) {
    llvm::sys::fs::set_current_path(debuggerRoot.data());
  }

  SetSourceMapFromArguments(*arguments);

  // Run any initialize LLDB commands the user specified in the launch.json
  g_vsc.RunInitCommands();

  // Grab the current working directory if there is one and set it in the
  // launch info.
  const auto cwd = GetString(arguments, "cwd");
  if (!cwd.empty())
    g_vsc.launch_info.SetWorkingDirectory(cwd.data());

  // Grab the name of the program we need to debug and set it as the first
  // argument that will be passed to the program we will debug.
  llvm::StringRef program = GetString(arguments, "program");
  if (!program.empty()) {
    lldb::SBFileSpec program_fspec(program.data(), true /*resolve_path*/);
    g_vsc.launch_info.SetExecutableFile(program_fspec,
                                        true /*add_as_first_arg*/);
    const char *target_triple = nullptr;
    const char *uuid_cstr = nullptr;
    // Stand alone debug info file if different from executable
    const char *symfile = nullptr;
    lldb::SBModule module = g_vsc.target.AddModule(
        program.data(), target_triple, uuid_cstr, symfile);
    if (!module.IsValid()) {
      response["success"] = llvm::json::Value(false);

      EmplaceSafeString(
          response, "message",
          llvm::formatv("Could not load program '{0}'.", program).str());
      g_vsc.SendJSON(llvm::json::Value(std::move(response)));
    }
  }

  // Extract any extra arguments and append them to our program arguments for
  // when we launch
  auto args = GetStrings(arguments, "args");
  if (!args.empty())
    g_vsc.launch_info.SetArguments(MakeArgv(args).data(), true);

  // Pass any environment variables along that the user specified.
  auto envs = GetStrings(arguments, "env");
  if (!envs.empty())
    g_vsc.launch_info.SetEnvironmentEntries(MakeArgv(envs).data(), true);

  auto flags = g_vsc.launch_info.GetLaunchFlags();

  if (GetBoolean(arguments, "disableASLR", true))
    flags |= lldb::eLaunchFlagDisableASLR;
  if (GetBoolean(arguments, "disableSTDIO", false))
    flags |= lldb::eLaunchFlagDisableSTDIO;
  if (GetBoolean(arguments, "shellExpandArguments", false))
    flags |= lldb::eLaunchFlagShellExpandArguments;
  const bool detatchOnError = GetBoolean(arguments, "detachOnError", false);
  g_vsc.launch_info.SetDetachOnError(detatchOnError);
  g_vsc.launch_info.SetLaunchFlags(flags | lldb::eLaunchFlagDebug |
                                   lldb::eLaunchFlagStopAtEntry);

  // Run any pre run LLDB commands the user specified in the launch.json
  g_vsc.RunPreRunCommands();

  // Disable async events so the launch will be successful when we return from
  // the launch call and the launch will happen synchronously
  g_vsc.debugger.SetAsync(false);
  g_vsc.target.Launch(g_vsc.launch_info, error);
  if (error.Fail()) {
    response["success"] = llvm::json::Value(false);
    EmplaceSafeString(response, "message", std::string(error.GetCString()));
  }
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));

  SendProcessEvent(Launch);
  g_vsc.SendJSON(llvm::json::Value(CreateEventObject("initialized")));
  // Reenable async events and start the event thread to catch async events.
  g_vsc.debugger.SetAsync(true);
}

//----------------------------------------------------------------------
// "NextRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Next request; value of command field is 'next'. The
//                     request starts the debuggee to run again for one step.
//                     The debug adapter first sends the NextResponse and then
//                     a StoppedEvent (event type 'step') after the step has
//                     completed.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "next" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/NextArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "NextArguments": {
//   "type": "object",
//   "description": "Arguments for 'next' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Execute 'next' for this thread."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "NextResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'next' request. This is just an
//                     acknowledgement, so no body field is required."
//   }]
// }
//----------------------------------------------------------------------
void request_next(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  lldb::SBThread thread = g_vsc.GetLLDBThread(*arguments);
  if (thread.IsValid()) {
    // Remember the thread ID that caused the resume so we can set the
    // "threadCausedFocus" boolean value in the "stopped" events.
    g_vsc.focus_tid = thread.GetThreadID();
    thread.StepOver();
  } else {
    response["success"] = llvm::json::Value(false);
  }
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "PauseRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Pause request; value of command field is 'pause'. The
//     request suspenses the debuggee. The debug adapter first sends the
//     PauseResponse and then a StoppedEvent (event type 'pause') after the
//     thread has been paused successfully.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "pause" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/PauseArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "PauseArguments": {
//   "type": "object",
//   "description": "Arguments for 'pause' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Pause execution for this thread."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "PauseResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'pause' request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// }
//----------------------------------------------------------------------
void request_pause(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  lldb::SBProcess process = g_vsc.target.GetProcess();
  lldb::SBError error = process.Stop();
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "ScopesRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Scopes request; value of command field is 'scopes'. The
//     request returns the variable scopes for a given stackframe ID.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "scopes" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/ScopesArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "ScopesArguments": {
//   "type": "object",
//   "description": "Arguments for 'scopes' request.",
//   "properties": {
//     "frameId": {
//       "type": "integer",
//       "description": "Retrieve the scopes for this stackframe."
//     }
//   },
//   "required": [ "frameId" ]
// },
// "ScopesResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'scopes' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "scopes": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Scope"
//             },
//             "description": "The scopes of the stackframe. If the array has
//             length zero, there are no scopes available."
//           }
//         },
//         "required": [ "scopes" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
//----------------------------------------------------------------------
void request_scopes(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  auto arguments = request.getObject("arguments");
  lldb::SBFrame frame = g_vsc.GetLLDBFrame(*arguments);
  g_vsc.variables.Clear();
  g_vsc.variables.Append(frame.GetVariables(true,   // arguments
                                            true,   // locals
                                            false,  // statics
                                            true)); // in_scope_only
  g_vsc.num_locals = g_vsc.variables.GetSize();
  g_vsc.variables.Append(frame.GetVariables(false,  // arguments
                                            false,  // locals
                                            true,   // statics
                                            true)); // in_scope_only
  g_vsc.num_globals = g_vsc.variables.GetSize() - (g_vsc.num_locals);
  g_vsc.variables.Append(frame.GetRegisters());
  g_vsc.num_regs =
      g_vsc.variables.GetSize() - (g_vsc.num_locals + g_vsc.num_globals);
  body.try_emplace("scopes", g_vsc.CreateTopLevelScopes());
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "SetBreakpointsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "SetBreakpoints request; value of command field is
//     'setBreakpoints'. Sets multiple breakpoints for a single source and
//     clears all previous breakpoints in that source. To clear all breakpoint
//     for a source, specify an empty array. When a breakpoint is hit, a
//     StoppedEvent (event type 'breakpoint') is generated.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "setBreakpoints" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SetBreakpointsArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SetBreakpointsArguments": {
//   "type": "object",
//   "description": "Arguments for 'setBreakpoints' request.",
//   "properties": {
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "The source location of the breakpoints; either
//       source.path or source.reference must be specified."
//     },
//     "breakpoints": {
//       "type": "array",
//       "items": {
//         "$ref": "#/definitions/SourceBreakpoint"
//       },
//       "description": "The code locations of the breakpoints."
//     },
//     "lines": {
//       "type": "array",
//       "items": {
//         "type": "integer"
//       },
//       "description": "Deprecated: The code locations of the breakpoints."
//     },
//     "sourceModified": {
//       "type": "boolean",
//       "description": "A value of true indicates that the underlying source
//       has been modified which results in new breakpoint locations."
//     }
//   },
//   "required": [ "source" ]
// },
// "SetBreakpointsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'setBreakpoints' request. Returned is
//     information about each breakpoint created by this request. This includes
//     the actual code location and whether the breakpoint could be verified.
//     The breakpoints returned are in the same order as the elements of the
//     'breakpoints' (or the deprecated 'lines') in the
//     SetBreakpointsArguments.", "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "breakpoints": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Breakpoint"
//             },
//             "description": "Information about the breakpoints. The array
//             elements are in the same order as the elements of the
//             'breakpoints' (or the deprecated 'lines') in the
//             SetBreakpointsArguments."
//           }
//         },
//         "required": [ "breakpoints" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// },
// "SourceBreakpoint": {
//   "type": "object",
//   "description": "Properties of a breakpoint or logpoint passed to the
//   setBreakpoints request.", "properties": {
//     "line": {
//       "type": "integer",
//       "description": "The source line of the breakpoint or logpoint."
//     },
//     "column": {
//       "type": "integer",
//       "description": "An optional source column of the breakpoint."
//     },
//     "condition": {
//       "type": "string",
//       "description": "An optional expression for conditional breakpoints."
//     },
//     "hitCondition": {
//       "type": "string",
//       "description": "An optional expression that controls how many hits of
//       the breakpoint are ignored. The backend is expected to interpret the
//       expression as needed."
//     },
//     "logMessage": {
//       "type": "string",
//       "description": "If this attribute exists and is non-empty, the backend
//       must not 'break' (stop) but log the message instead. Expressions within
//       {} are interpolated."
//     }
//   },
//   "required": [ "line" ]
// }
//----------------------------------------------------------------------
void request_setBreakpoints(const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  auto source = arguments->getObject("source");
  const auto path = GetString(source, "path");
  auto breakpoints = arguments->getArray("breakpoints");
  llvm::json::Array response_breakpoints;
  // Decode the source breakpoint infos for this "setBreakpoints" request
  SourceBreakpointMap request_bps;
  for (const auto &bp : *breakpoints) {
    auto bp_obj = bp.getAsObject();
    if (bp_obj) {
      SourceBreakpoint src_bp(*bp_obj);
      request_bps[src_bp.line] = std::move(src_bp);
    }
  }

  // See if we already have breakpoints set for this source file from a
  // previous "setBreakpoints" request
  auto old_src_bp_pos = g_vsc.source_breakpoints.find(path);
  if (old_src_bp_pos != g_vsc.source_breakpoints.end()) {

    // We have already set breakpoints in this source file and they are giving
    // use a new list of lines to set breakpoints on. Some breakpoints might
    // already be set, and some might not. We need to remove any breakpoints
    // whose lines are not contained in the any breakpoints lines in in the
    // "breakpoints" array.

    // Delete any breakpoints in this source file that aren't in the
    // request_bps set. There is no call to remove breakpoints other than
    // calling this function with a smaller or empty "breakpoints" list.
    std::vector<uint32_t> remove_lines;
    for (auto &pair: old_src_bp_pos->second) {
      auto request_pos = request_bps.find(pair.first);
      if (request_pos == request_bps.end()) {
        // This breakpoint no longer exists in this source file, delete it
        g_vsc.target.BreakpointDelete(pair.second.bp.GetID());
        remove_lines.push_back(pair.first);
      } else {
        pair.second.UpdateBreakpoint(request_pos->second);
        // Remove this breakpoint from the request breakpoints since we have
        // handled it here and we don't need to set a new breakpoint below.
        request_bps.erase(request_pos);
        // Add this breakpoint info to the response
        AppendBreakpoint(pair.second.bp, response_breakpoints);
      }
    }
    // Remove any lines from this existing source breakpoint map
    for (auto line: remove_lines)
     old_src_bp_pos->second.erase(line);

    // Now add any breakpoint infos left over in request_bps are the
    // breakpoints that weren't set in this source file yet. We need to update
    // thread source breakpoint info for the source file in the variable
    // "old_src_bp_pos->second" so the info for this source file is up to date.
    for (auto &pair : request_bps) {
      pair.second.SetBreakpoint(path.data());
      // Add this breakpoint info to the response
      AppendBreakpoint(pair.second.bp, response_breakpoints);
      old_src_bp_pos->second[pair.first] = std::move(pair.second);
    }
  } else {
    // No breakpoints were set for this source file yet. Set all breakpoints
    // for each line and add them to the response and create an entry in
    // g_vsc.source_breakpoints for this source file.
    for (auto &pair : request_bps) {
      pair.second.SetBreakpoint(path.data());
      // Add this breakpoint info to the response
      AppendBreakpoint(pair.second.bp, response_breakpoints);
    }
    g_vsc.source_breakpoints[path] = std::move(request_bps);
  }

  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "SetExceptionBreakpointsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "SetExceptionBreakpoints request; value of command field
//     is 'setExceptionBreakpoints'. The request configures the debuggers
//     response to thrown exceptions. If an exception is configured to break, a
//     StoppedEvent is fired (event type 'exception').", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "setExceptionBreakpoints" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SetExceptionBreakpointsArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SetExceptionBreakpointsArguments": {
//   "type": "object",
//   "description": "Arguments for 'setExceptionBreakpoints' request.",
//   "properties": {
//     "filters": {
//       "type": "array",
//       "items": {
//         "type": "string"
//       },
//       "description": "IDs of checked exception options. The set of IDs is
//       returned via the 'exceptionBreakpointFilters' capability."
//     },
//     "exceptionOptions": {
//       "type": "array",
//       "items": {
//         "$ref": "#/definitions/ExceptionOptions"
//       },
//       "description": "Configuration options for selected exceptions."
//     }
//   },
//   "required": [ "filters" ]
// },
// "SetExceptionBreakpointsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'setExceptionBreakpoints' request. This is
//     just an acknowledgement, so no body field is required."
//   }]
// }
//----------------------------------------------------------------------
void request_setExceptionBreakpoints(const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  auto filters = arguments->getArray("filters");
  // Keep a list of any exception breakpoint filter names that weren't set
  // so we can clear any exception breakpoints if needed.
  std::set<std::string> unset_filters;
  for (const auto &bp : g_vsc.exception_breakpoints)
    unset_filters.insert(bp.filter);

  for (const auto &value : *filters) {
    const auto filter = GetAsString(value);
    auto exc_bp = g_vsc.GetExceptionBreakpoint(filter);
    if (exc_bp) {
      exc_bp->SetBreakpoint();
      unset_filters.erase(filter);
    }
  }
  for (const auto &filter : unset_filters) {
    auto exc_bp = g_vsc.GetExceptionBreakpoint(filter);
    if (exc_bp)
      exc_bp->ClearBreakpoint();
  }
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "SetFunctionBreakpointsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "SetFunctionBreakpoints request; value of command field is
//     'setFunctionBreakpoints'. Sets multiple function breakpoints and clears
//     all previous function breakpoints. To clear all function breakpoint,
//     specify an empty array. When a function breakpoint is hit, a StoppedEvent
//     (event type 'function breakpoint') is generated.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "setFunctionBreakpoints" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SetFunctionBreakpointsArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SetFunctionBreakpointsArguments": {
//   "type": "object",
//   "description": "Arguments for 'setFunctionBreakpoints' request.",
//   "properties": {
//     "breakpoints": {
//       "type": "array",
//       "items": {
//         "$ref": "#/definitions/FunctionBreakpoint"
//       },
//       "description": "The function names of the breakpoints."
//     }
//   },
//   "required": [ "breakpoints" ]
// },
// "FunctionBreakpoint": {
//   "type": "object",
//   "description": "Properties of a breakpoint passed to the
//   setFunctionBreakpoints request.", "properties": {
//     "name": {
//       "type": "string",
//       "description": "The name of the function."
//     },
//     "condition": {
//       "type": "string",
//       "description": "An optional expression for conditional breakpoints."
//     },
//     "hitCondition": {
//       "type": "string",
//       "description": "An optional expression that controls how many hits of
//       the breakpoint are ignored. The backend is expected to interpret the
//       expression as needed."
//     }
//   },
//   "required": [ "name" ]
// },
// "SetFunctionBreakpointsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'setFunctionBreakpoints' request. Returned is
//     information about each breakpoint created by this request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "breakpoints": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Breakpoint"
//             },
//             "description": "Information about the breakpoints. The array
//             elements correspond to the elements of the 'breakpoints' array."
//           }
//         },
//         "required": [ "breakpoints" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
//----------------------------------------------------------------------
void request_setFunctionBreakpoints(const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  auto breakpoints = arguments->getArray("breakpoints");
  FunctionBreakpointMap request_bps;
  llvm::json::Array response_breakpoints;
  for (const auto &value : *breakpoints) {
    auto bp_obj = value.getAsObject();
    if (bp_obj == nullptr)
      continue;
    FunctionBreakpoint func_bp(*bp_obj);
    request_bps[func_bp.functionName] = std::move(func_bp);
  }

  std::vector<llvm::StringRef> remove_names;
  // Disable any function breakpoints that aren't in the request_bps.
  // There is no call to remove function breakpoints other than calling this
  // function with a smaller or empty "breakpoints" list.
  for (auto &pair: g_vsc.function_breakpoints) {
    auto request_pos = request_bps.find(pair.first());
    if (request_pos == request_bps.end()) {
      // This function breakpoint no longer exists delete it from LLDB
      g_vsc.target.BreakpointDelete(pair.second.bp.GetID());
      remove_names.push_back(pair.first());
    } else {
      // Update the existing breakpoint as any setting withing the function
      // breakpoint might have changed.
      pair.second.UpdateBreakpoint(request_pos->second);
      // Remove this breakpoint from the request breakpoints since we have
      // handled it here and we don't need to set a new breakpoint below.
      request_bps.erase(request_pos);
      // Add this breakpoint info to the response
      AppendBreakpoint(pair.second.bp, response_breakpoints);
    }
  }
  // Remove any breakpoints that are no longer in our list
  for (const auto &name: remove_names)
    g_vsc.function_breakpoints.erase(name);

  // Any breakpoints that are left in "request_bps" are breakpoints that
  // need to be set.
  for (auto &pair : request_bps) {
    pair.second.SetBreakpoint();
    // Add this breakpoint info to the response
    AppendBreakpoint(pair.second.bp, response_breakpoints);
    g_vsc.function_breakpoints[pair.first()] = std::move(pair.second);
  }

  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "SourceRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Source request; value of command field is 'source'. The
//     request retrieves the source code for a given source reference.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "source" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SourceArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SourceArguments": {
//   "type": "object",
//   "description": "Arguments for 'source' request.",
//   "properties": {
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "Specifies the source content to load. Either
//       source.path or source.sourceReference must be specified."
//     },
//     "sourceReference": {
//       "type": "integer",
//       "description": "The reference to the source. This is the same as
//       source.sourceReference. This is provided for backward compatibility
//       since old backends do not understand the 'source' attribute."
//     }
//   },
//   "required": [ "sourceReference" ]
// },
// "SourceResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'source' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "content": {
//             "type": "string",
//             "description": "Content of the source reference."
//           },
//           "mimeType": {
//             "type": "string",
//             "description": "Optional content type (mime type) of the source."
//           }
//         },
//         "required": [ "content" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
//----------------------------------------------------------------------
void request_source(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;

  auto arguments = request.getObject("arguments");
  auto source = arguments->getObject("source");
  auto sourceReference = GetSigned(source, "sourceReference", -1);
  auto pos = g_vsc.source_map.find((lldb::addr_t)sourceReference);
  if (pos != g_vsc.source_map.end()) {
    EmplaceSafeString(body, "content", pos->second.content);
  } else {
    response["success"] = llvm::json::Value(false);
  }
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "StackTraceRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "StackTrace request; value of command field is
//     'stackTrace'. The request returns a stacktrace from the current execution
//     state.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "stackTrace" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/StackTraceArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "StackTraceArguments": {
//   "type": "object",
//   "description": "Arguments for 'stackTrace' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Retrieve the stacktrace for this thread."
//     },
//     "startFrame": {
//       "type": "integer",
//       "description": "The index of the first frame to return; if omitted
//       frames start at 0."
//     },
//     "levels": {
//       "type": "integer",
//       "description": "The maximum number of frames to return. If levels is
//       not specified or 0, all frames are returned."
//     },
//     "format": {
//       "$ref": "#/definitions/StackFrameFormat",
//       "description": "Specifies details on how to format the stack frames."
//     }
//  },
//   "required": [ "threadId" ]
// },
// "StackTraceResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'stackTrace' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "stackFrames": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/StackFrame"
//             },
//             "description": "The frames of the stackframe. If the array has
//             length zero, there are no stackframes available. This means that
//             there is no location information available."
//           },
//           "totalFrames": {
//             "type": "integer",
//             "description": "The total number of frames available."
//           }
//         },
//         "required": [ "stackFrames" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
//----------------------------------------------------------------------
void request_stackTrace(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  lldb::SBError error;
  auto arguments = request.getObject("arguments");
  lldb::SBThread thread = g_vsc.GetLLDBThread(*arguments);
  llvm::json::Array stackFrames;
  llvm::json::Object body;

  if (thread.IsValid()) {
    const auto startFrame = GetUnsigned(arguments, "startFrame", 0);
    const auto levels = GetUnsigned(arguments, "levels", 0);
    const auto endFrame = (levels == 0) ? INT64_MAX : (startFrame + levels);
    for (uint32_t i = startFrame; i < endFrame; ++i) {
      auto frame = thread.GetFrameAtIndex(i);
      if (!frame.IsValid())
        break;
      stackFrames.emplace_back(CreateStackFrame(frame));
    }
  }
  body.try_emplace("stackFrames", std::move(stackFrames));
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "StepInRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "StepIn request; value of command field is 'stepIn'. The
//     request starts the debuggee to step into a function/method if possible.
//     If it cannot step into a target, 'stepIn' behaves like 'next'. The debug
//     adapter first sends the StepInResponse and then a StoppedEvent (event
//     type 'step') after the step has completed. If there are multiple
//     function/method calls (or other targets) on the source line, the optional
//     argument 'targetId' can be used to control into which target the 'stepIn'
//     should occur. The list of possible targets for a given source line can be
//     retrieved via the 'stepInTargets' request.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "stepIn" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/StepInArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "StepInArguments": {
//   "type": "object",
//   "description": "Arguments for 'stepIn' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Execute 'stepIn' for this thread."
//     },
//     "targetId": {
//       "type": "integer",
//       "description": "Optional id of the target to step into."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "StepInResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'stepIn' request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// }
//----------------------------------------------------------------------
void request_stepIn(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  lldb::SBThread thread = g_vsc.GetLLDBThread(*arguments);
  if (thread.IsValid()) {
    // Remember the thread ID that caused the resume so we can set the
    // "threadCausedFocus" boolean value in the "stopped" events.
    g_vsc.focus_tid = thread.GetThreadID();
    thread.StepInto();
  } else {
    response["success"] = llvm::json::Value(false);
  }
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "StepOutRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "StepOut request; value of command field is 'stepOut'. The
//     request starts the debuggee to run again for one step. The debug adapter
//     first sends the StepOutResponse and then a StoppedEvent (event type
//     'step') after the step has completed.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "stepOut" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/StepOutArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "StepOutArguments": {
//   "type": "object",
//   "description": "Arguments for 'stepOut' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Execute 'stepOut' for this thread."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "StepOutResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'stepOut' request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// }
//----------------------------------------------------------------------
void request_stepOut(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  auto arguments = request.getObject("arguments");
  lldb::SBThread thread = g_vsc.GetLLDBThread(*arguments);
  if (thread.IsValid()) {
    // Remember the thread ID that caused the resume so we can set the
    // "threadCausedFocus" boolean value in the "stopped" events.
    g_vsc.focus_tid = thread.GetThreadID();
    thread.StepOut();
  } else {
    response["success"] = llvm::json::Value(false);
  }
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "ThreadsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Thread request; value of command field is 'threads'. The
//     request retrieves a list of all threads.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "threads" ]
//       }
//     },
//     "required": [ "command" ]
//   }]
// },
// "ThreadsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'threads' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "threads": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Thread"
//             },
//             "description": "All threads."
//           }
//         },
//         "required": [ "threads" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
//----------------------------------------------------------------------
void request_threads(const llvm::json::Object &request) {

  lldb::SBProcess process = g_vsc.target.GetProcess();
  llvm::json::Object response;
  FillResponse(request, response);

  const uint32_t num_threads = process.GetNumThreads();
  llvm::json::Array threads;
  for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
    threads.emplace_back(CreateThread(thread));
  }
  if (threads.size() == 0) {
    response["success"] = llvm::json::Value(false);
  }
  llvm::json::Object body;
  body.try_emplace("threads", std::move(threads));
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "SetVariableRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "setVariable request; value of command field is
//     'setVariable'. Set the variable with the given name in the variable
//     container to a new value.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "setVariable" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SetVariableArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SetVariableArguments": {
//   "type": "object",
//   "description": "Arguments for 'setVariable' request.",
//   "properties": {
//     "variablesReference": {
//       "type": "integer",
//       "description": "The reference of the variable container."
//     },
//     "name": {
//       "type": "string",
//       "description": "The name of the variable."
//     },
//     "value": {
//       "type": "string",
//       "description": "The value of the variable."
//     },
//     "format": {
//       "$ref": "#/definitions/ValueFormat",
//       "description": "Specifies details on how to format the response value."
//     }
//   },
//   "required": [ "variablesReference", "name", "value" ]
// },
// "SetVariableResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'setVariable' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "value": {
//             "type": "string",
//             "description": "The new value of the variable."
//           },
//           "type": {
//             "type": "string",
//             "description": "The type of the new value. Typically shown in the
//             UI when hovering over the value."
//           },
//           "variablesReference": {
//             "type": "number",
//             "description": "If variablesReference is > 0, the new value is
//             structured and its children can be retrieved by passing
//             variablesReference to the VariablesRequest."
//           },
//           "namedVariables": {
//             "type": "number",
//             "description": "The number of named child variables. The client
//             can use this optional information to present the variables in a
//             paged UI and fetch them in chunks."
//           },
//           "indexedVariables": {
//             "type": "number",
//             "description": "The number of indexed child variables. The client
//             can use this optional information to present the variables in a
//             paged UI and fetch them in chunks."
//           }
//         },
//         "required": [ "value" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
//----------------------------------------------------------------------
void request_setVariable(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array variables;
  llvm::json::Object body;
  auto arguments = request.getObject("arguments");
  // This is a reference to the containing variable/scope
  const auto variablesReference =
      GetUnsigned(arguments, "variablesReference", 0);
  const auto name = GetString(arguments, "name");
  const auto value = GetString(arguments, "value");
  // Set success to false just in case we don't find the variable by name
  response.try_emplace("success", false);

  lldb::SBValue variable;
  int64_t newVariablesReference = 0;

  // The "id" is the unique integer ID that is unique within the enclosing
  // variablesReference. It is optionally added to any "interface Variable"
  // objects to uniquely identify a variable within an enclosing
  // variablesReference. It helps to disambiguate between two variables that
  // have the same name within the same scope since the "setVariables" request
  // only specifies the variable reference of the enclosing scope/variable, and
  // the name of the variable. We could have two shadowed variables with the
  // same name in "Locals" or "Globals". In our case the "id" absolute index
  // of the variable within the g_vsc.variables list.
  const auto id_value = GetUnsigned(arguments, "id", UINT64_MAX);
  if (id_value != UINT64_MAX) {
    variable = g_vsc.variables.GetValueAtIndex(id_value);
  } else if (VARREF_IS_SCOPE(variablesReference)) {
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for a variable in locals or globals or registers
    int64_t start_idx = 0;
    int64_t end_idx = 0;
    switch (variablesReference) {
    case VARREF_LOCALS:
      start_idx = 0;
      end_idx = start_idx + g_vsc.num_locals;
      break;
    case VARREF_GLOBALS:
      start_idx = g_vsc.num_locals;
      end_idx = start_idx + g_vsc.num_globals;
      break;
    case VARREF_REGS:
      start_idx = g_vsc.num_locals + g_vsc.num_globals;
      end_idx = start_idx + g_vsc.num_regs;
      break;
    default:
      break;
    }

    // Find the variable by name in the correct scope and hope we don't have
    // multiple variables with the same name. We search backwards because
    // the list of variables has the top most variables first and variables
    // in deeper scopes are last. This means we will catch the deepest
    // variable whose name matches which is probably what the user wants.
    for (int64_t i = end_idx - 1; i >= start_idx; --i) {
      auto curr_variable = g_vsc.variables.GetValueAtIndex(i);
      llvm::StringRef variable_name(curr_variable.GetName());
      if (variable_name == name) {
        variable = curr_variable;
        if (curr_variable.MightHaveChildren())
          newVariablesReference = i;
        break;
      }
    }
  } else {
    // We have a named item within an actual variable so we need to find it
    // withing the container variable by name.
    const int64_t var_idx = VARREF_TO_VARIDX(variablesReference);
    lldb::SBValue container = g_vsc.variables.GetValueAtIndex(var_idx);
    variable = container.GetChildMemberWithName(name.data());
    if (!variable.IsValid()) {
      if (name.startswith("[")) {
        llvm::StringRef index_str(name.drop_front(1));
        uint64_t index = 0;
        if (!index_str.consumeInteger(0, index)) {
          if (index_str == "]")
            variable = container.GetChildAtIndex(index);
        }
      }
    }

    // We don't know the index of the variable in our g_vsc.variables
    if (variable.IsValid()) {
      if (variable.MightHaveChildren()) {
        newVariablesReference = VARIDX_TO_VARREF(g_vsc.variables.GetSize());
        g_vsc.variables.Append(variable);
      }
    }
  }

  if (variable.IsValid()) {
    lldb::SBError error;
    bool success = variable.SetValueFromCString(value.data(), error);
    if (success) {
      SetValueForKey(variable, body, "value");
      EmplaceSafeString(body, "type", variable.GetType().GetDisplayTypeName());
      body.try_emplace("variablesReference", newVariablesReference);
    } else {
      EmplaceSafeString(body, "message", std::string(error.GetCString()));
    }
    response["success"] = llvm::json::Value(success);
  }

  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

//----------------------------------------------------------------------
// "VariablesRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Variables request; value of command field is 'variables'.
//     Retrieves all child variables for the given variable reference. An
//     optional filter can be used to limit the fetched children to either named
//     or indexed children.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "variables" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/VariablesArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "VariablesArguments": {
//   "type": "object",
//   "description": "Arguments for 'variables' request.",
//   "properties": {
//     "variablesReference": {
//       "type": "integer",
//       "description": "The Variable reference."
//     },
//     "filter": {
//       "type": "string",
//       "enum": [ "indexed", "named" ],
//       "description": "Optional filter to limit the child variables to either
//       named or indexed. If ommited, both types are fetched."
//     },
//     "start": {
//       "type": "integer",
//       "description": "The index of the first variable to return; if omitted
//       children start at 0."
//     },
//     "count": {
//       "type": "integer",
//       "description": "The number of variables to return. If count is missing
//       or 0, all variables are returned."
//     },
//     "format": {
//       "$ref": "#/definitions/ValueFormat",
//       "description": "Specifies details on how to format the Variable
//       values."
//     }
//   },
//   "required": [ "variablesReference" ]
// },
// "VariablesResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'variables' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "variables": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Variable"
//             },
//             "description": "All (or a range) of variables for the given
//             variable reference."
//           }
//         },
//         "required": [ "variables" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
//----------------------------------------------------------------------
void request_variables(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array variables;
  auto arguments = request.getObject("arguments");
  const auto variablesReference =
      GetUnsigned(arguments, "variablesReference", 0);
  const int64_t start = GetSigned(arguments, "start", 0);
  const int64_t count = GetSigned(arguments, "count", 0);
  bool hex = false;
  auto format = arguments->getObject("format");
  if (format)
    hex = GetBoolean(format, "hex", false);

  if (VARREF_IS_SCOPE(variablesReference)) {
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for the list of args, locals or globals.
    int64_t start_idx = 0;
    int64_t num_children = 0;
    switch (variablesReference) {
    case VARREF_LOCALS:
      start_idx = start;
      num_children = g_vsc.num_locals;
      break;
    case VARREF_GLOBALS:
      start_idx = start + g_vsc.num_locals + start;
      num_children = g_vsc.num_globals;
      break;
    case VARREF_REGS:
      start_idx = start + g_vsc.num_locals + g_vsc.num_globals;
      num_children = g_vsc.num_regs;
      break;
    default:
      break;
    }
    const int64_t end_idx = start_idx + ((count == 0) ? num_children : count);
    for (auto i = start_idx; i < end_idx; ++i) {
      lldb::SBValue variable = g_vsc.variables.GetValueAtIndex(i);
      if (!variable.IsValid())
        break;
      variables.emplace_back(
          CreateVariable(variable, VARIDX_TO_VARREF(i), i, hex));
    }
  } else {
    // We are expanding a variable that has children, so we will return its
    // children.
    const int64_t var_idx = VARREF_TO_VARIDX(variablesReference);
    lldb::SBValue variable = g_vsc.variables.GetValueAtIndex(var_idx);
    if (variable.IsValid()) {
      const auto num_children = variable.GetNumChildren();
      const int64_t end_idx = start + ((count == 0) ? num_children : count);
      for (auto i = start; i < end_idx; ++i) {
        lldb::SBValue child = variable.GetChildAtIndex(i);
        if (!child.IsValid())
          break;
        if (child.MightHaveChildren()) {
          const int64_t var_idx = g_vsc.variables.GetSize();
          auto childVariablesReferences = VARIDX_TO_VARREF(var_idx);
          variables.emplace_back(
              CreateVariable(child, childVariablesReferences, var_idx, hex));
          g_vsc.variables.Append(child);
        } else {
          variables.emplace_back(CreateVariable(child, 0, INT64_MAX, hex));
        }
      }
    }
  }
  llvm::json::Object body;
  body.try_emplace("variables", std::move(variables));
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

// A request used in testing to get the details on all breakpoints that are
// currently set in the target. This helps us to test "setBreakpoints" and
// "setFunctionBreakpoints" requests to verify we have the correct set of
// breakpoints currently set in LLDB.
void request__testGetTargetBreakpoints(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array response_breakpoints;
  for (uint32_t i = 0; g_vsc.target.GetBreakpointAtIndex(i).IsValid(); ++i) {
    auto bp = g_vsc.target.GetBreakpointAtIndex(i);
    AppendBreakpoint(bp, response_breakpoints);
  }
  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  g_vsc.SendJSON(llvm::json::Value(std::move(response)));
}

const std::map<std::string, RequestCallback> &GetRequestHandlers() {
#define REQUEST_CALLBACK(name)                                                 \
  { #name, request_##name }
  static std::map<std::string, RequestCallback> g_request_handlers = {
      // VSCode Debug Adaptor requests
      REQUEST_CALLBACK(attach),
      REQUEST_CALLBACK(continue),
      REQUEST_CALLBACK(configurationDone),
      REQUEST_CALLBACK(disconnect),
      REQUEST_CALLBACK(evaluate),
      REQUEST_CALLBACK(exceptionInfo),
      REQUEST_CALLBACK(initialize),
      REQUEST_CALLBACK(launch),
      REQUEST_CALLBACK(next),
      REQUEST_CALLBACK(pause),
      REQUEST_CALLBACK(scopes),
      REQUEST_CALLBACK(setBreakpoints),
      REQUEST_CALLBACK(setExceptionBreakpoints),
      REQUEST_CALLBACK(setFunctionBreakpoints),
      REQUEST_CALLBACK(setVariable),
      REQUEST_CALLBACK(source),
      REQUEST_CALLBACK(stackTrace),
      REQUEST_CALLBACK(stepIn),
      REQUEST_CALLBACK(stepOut),
      REQUEST_CALLBACK(threads),
      REQUEST_CALLBACK(variables),
      // Testing requests
      REQUEST_CALLBACK(_testGetTargetBreakpoints),
  };
#undef REQUEST_CALLBACK
  return g_request_handlers;
}
  
} // anonymous namespace

int main(int argc, char *argv[]) {

  // Initialize LLDB first before we do anything.
  lldb::SBDebugger::Initialize();

  if (argc == 2) {
    const char *arg = argv[1];
#if !defined(_WIN32)
    if (strcmp(arg, "-g") == 0) {
      printf("Paused waiting for debugger to attach (pid = %i)...\n", getpid());
      pause();
    } else {
#else
    {
#endif
      int portno = atoi(arg);
      printf("Listening on port %i...\n", portno);
      SOCKET socket_fd = AcceptConnection(portno);
      if (socket_fd >= 0) {
        g_vsc.input.descriptor = StreamDescriptor::from_socket(socket_fd, true);
        g_vsc.output.descriptor =
            StreamDescriptor::from_socket(socket_fd, false);
      } else {
        exit(1);
      }
    }
  } else {
    g_vsc.input.descriptor = StreamDescriptor::from_file(fileno(stdin), false);
    g_vsc.output.descriptor =
        StreamDescriptor::from_file(fileno(stdout), false);
  }
  auto request_handlers = GetRequestHandlers();
  uint32_t packet_idx = 0;
  while (true) {
    std::string json = g_vsc.ReadJSON();
    if (json.empty())
      break;

    llvm::StringRef json_sref(json);
    llvm::Expected<llvm::json::Value> json_value = llvm::json::parse(json_sref);
    if (!json_value) {
      auto error = json_value.takeError();
      if (g_vsc.log) {
        std::string error_str;
        llvm::raw_string_ostream strm(error_str);
        strm << error;
        strm.flush();

        *g_vsc.log << "error: failed to parse JSON: " << error_str << std::endl
                   << json << std::endl;
      }
      return 1;
    }

    auto object = json_value->getAsObject();
    if (!object) {
      if (g_vsc.log)
        *g_vsc.log << "error: json packet isn't a object" << std::endl;
      return 1;
    }

    const auto packet_type = GetString(object, "type");
    if (packet_type == "request") {
      const auto command = GetString(object, "command");
      auto handler_pos = request_handlers.find(command);
      if (handler_pos != request_handlers.end()) {
        handler_pos->second(*object);
      } else {
        if (g_vsc.log)
          *g_vsc.log << "error: unhandled command \"" << command.data() << std::endl;
        return 1;
      }
    }
    ++packet_idx;
  }

  // We must terminate the debugger in a thread before the C++ destructor
  // chain messes everything up.
  lldb::SBDebugger::Terminate();
  return 0;
}
