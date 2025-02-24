//===-- lldb-dap.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "FifoFiles.h"
#include "Handler/RequestHandler.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "RunInTerminal.h"
#include "Watchpoint.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFile.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "lldb/API/SBStream.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/Socket.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UriParser.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>

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
typedef int socklen_t;
#else
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/prctl.h>
#endif

using namespace lldb_dap;
using lldb_private::NativeSocket;
using lldb_private::Socket;
using lldb_private::Status;

namespace {
using namespace llvm::opt;

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

static constexpr llvm::opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};
class LLDBDAPOptTable : public llvm::opt::GenericOptTable {
public:
  LLDBDAPOptTable()
      : llvm::opt::GenericOptTable(OptionStrTable, OptionPrefixesTable,
                                   InfoTable, true) {}
};

typedef void (*RequestCallback)(const llvm::json::Object &command);

/// Page size used for reporting addtional frames in the 'stackTrace' request.
constexpr int StackPageSize = 20;

lldb::SBValueList *GetTopLevelScope(DAP &dap, int64_t variablesReference) {
  switch (variablesReference) {
  case VARREF_LOCALS:
    return &dap.variables.locals;
  case VARREF_GLOBALS:
    return &dap.variables.globals;
  case VARREF_REGS:
    return &dap.variables.registers;
  default:
    return nullptr;
  }
}

lldb::SBValue FindVariable(DAP &dap, uint64_t variablesReference,
                           llvm::StringRef name) {
  lldb::SBValue variable;
  if (lldb::SBValueList *top_scope =
          GetTopLevelScope(dap, variablesReference)) {
    bool is_duplicated_variable_name = name.contains(" @");
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for a variable in locals or globals or registers
    int64_t end_idx = top_scope->GetSize();
    // Searching backward so that we choose the variable in closest scope
    // among variables of the same name.
    for (int64_t i = end_idx - 1; i >= 0; --i) {
      lldb::SBValue curr_variable = top_scope->GetValueAtIndex(i);
      std::string variable_name = CreateUniqueVariableNameForDisplay(
          curr_variable, is_duplicated_variable_name);
      if (variable_name == name) {
        variable = curr_variable;
        break;
      }
    }
  } else {
    // This is not under the globals or locals scope, so there are no duplicated
    // names.

    // We have a named item within an actual variable so we need to find it
    // withing the container variable by name.
    lldb::SBValue container = dap.variables.GetVariable(variablesReference);
    variable = container.GetChildMemberWithName(name.data());
    if (!variable.IsValid()) {
      if (name.starts_with("[")) {
        llvm::StringRef index_str(name.drop_front(1));
        uint64_t index = 0;
        if (!index_str.consumeInteger(0, index)) {
          if (index_str == "]")
            variable = container.GetChildAtIndex(index);
        }
      }
    }
  }
  return variable;
}

// Fill in the stack frames of the thread.
//
// Threads stacks may contain runtime specific extended backtraces, when
// constructing a stack trace first report the full thread stack trace then
// perform a breadth first traversal of any extended backtrace frames.
//
// For example:
//
// Thread (id=th0) stack=[s0, s1, s2, s3]
//   \ Extended backtrace "libdispatch" Thread (id=th1) stack=[s0, s1]
//     \ Extended backtrace "libdispatch" Thread (id=th2) stack=[s0, s1]
//   \ Extended backtrace "Application Specific Backtrace" Thread (id=th3)
//   stack=[s0, s1, s2]
//
// Which will flatten into:
//
//  0. th0->s0
//  1. th0->s1
//  2. th0->s2
//  3. th0->s3
//  4. label - Enqueued from th1, sf=-1, i=-4
//  5. th1->s0
//  6. th1->s1
//  7. label - Enqueued from th2
//  8. th2->s0
//  9. th2->s1
// 10. label - Application Specific Backtrace
// 11. th3->s0
// 12. th3->s1
// 13. th3->s2
//
// s=3,l=3 = [th0->s3, label1, th1->s0]
bool FillStackFrames(DAP &dap, lldb::SBThread &thread,
                     llvm::json::Array &stack_frames, int64_t &offset,
                     const int64_t start_frame, const int64_t levels) {
  bool reached_end_of_stack = false;
  for (int64_t i = start_frame;
       static_cast<int64_t>(stack_frames.size()) < levels; i++) {
    if (i == -1) {
      stack_frames.emplace_back(
          CreateExtendedStackFrameLabel(thread, dap.frame_format));
      continue;
    }

    lldb::SBFrame frame = thread.GetFrameAtIndex(i);
    if (!frame.IsValid()) {
      offset += thread.GetNumFrames() + 1 /* label between threads */;
      reached_end_of_stack = true;
      break;
    }

    stack_frames.emplace_back(CreateStackFrame(frame, dap.frame_format));
  }

  if (dap.display_extended_backtrace && reached_end_of_stack) {
    // Check for any extended backtraces.
    for (uint32_t bt = 0;
         bt < thread.GetProcess().GetNumExtendedBacktraceTypes(); bt++) {
      lldb::SBThread backtrace = thread.GetExtendedBacktraceThread(
          thread.GetProcess().GetExtendedBacktraceTypeAtIndex(bt));
      if (!backtrace.IsValid())
        continue;

      reached_end_of_stack = FillStackFrames(
          dap, backtrace, stack_frames, offset,
          (start_frame - offset) > 0 ? start_frame - offset : -1, levels);
      if (static_cast<int64_t>(stack_frames.size()) >= levels)
        break;
    }
  }

  return reached_end_of_stack;
}

// "compileUnitsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Compile Unit request; value of command field is
//                     'compileUnits'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "compileUnits" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/compileUnitRequestArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "compileUnitsRequestArguments": {
//   "type": "object",
//   "description": "Arguments for 'compileUnits' request.",
//   "properties": {
//     "moduleId": {
//       "type": "string",
//       "description": "The ID of the module."
//     }
//   },
//   "required": [ "moduleId" ]
// },
// "compileUnitsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'compileUnits' request.",
//     "properties": {
//       "body": {
//         "description": "Response to 'compileUnits' request. Array of
//                         paths of compile units."
//       }
//     }
//   }]
// }
void request_compileUnits(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  llvm::json::Array units;
  const auto *arguments = request.getObject("arguments");
  std::string module_id = std::string(GetString(arguments, "moduleId"));
  int num_modules = dap.target.GetNumModules();
  for (int i = 0; i < num_modules; i++) {
    auto curr_module = dap.target.GetModuleAtIndex(i);
    if (module_id == curr_module.GetUUIDString()) {
      int num_units = curr_module.GetNumCompileUnits();
      for (int j = 0; j < num_units; j++) {
        auto curr_unit = curr_module.GetCompileUnitAtIndex(j);
        units.emplace_back(CreateCompileUnit(curr_unit));
      }
      body.try_emplace("compileUnits", std::move(units));
      break;
    }
  }
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

// "modulesRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Modules request; value of command field is
//                     'modules'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "modules" ]
//       },
//     },
//     "required": [ "command" ]
//   }]
// },
// "modulesResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'modules' request.",
//     "properties": {
//       "body": {
//         "description": "Response to 'modules' request. Array of
//                         module objects."
//       }
//     }
//   }]
// }
void request_modules(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);

  llvm::json::Array modules;
  for (size_t i = 0; i < dap.target.GetNumModules(); i++) {
    lldb::SBModule module = dap.target.GetModuleAtIndex(i);
    modules.emplace_back(CreateModule(dap.target, module));
  }

  llvm::json::Object body;
  body.try_emplace("modules", std::move(modules));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
void request_pause(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  lldb::SBProcess process = dap.target.GetProcess();
  lldb::SBError error = process.Stop();
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
void request_scopes(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  const auto *arguments = request.getObject("arguments");
  lldb::SBFrame frame = dap.GetLLDBFrame(*arguments);
  // As the user selects different stack frames in the GUI, a "scopes" request
  // will be sent to the DAP. This is the only way we know that the user has
  // selected a frame in a thread. There are no other notifications that are
  // sent and VS code doesn't allow multiple frames to show variables
  // concurrently. If we select the thread and frame as the "scopes" requests
  // are sent, this allows users to type commands in the debugger console
  // with a backtick character to run lldb commands and these lldb commands
  // will now have the right context selected as they are run. If the user
  // types "`bt" into the debugger console and we had another thread selected
  // in the LLDB library, we would show the wrong thing to the user. If the
  // users switches threads with a lldb command like "`thread select 14", the
  // GUI will not update as there are no "event" notification packets that
  // allow us to change the currently selected thread or frame in the GUI that
  // I am aware of.
  if (frame.IsValid()) {
    frame.GetThread().GetProcess().SetSelectedThread(frame.GetThread());
    frame.GetThread().SetSelectedFrame(frame.GetFrameID());
  }

  dap.variables.locals = frame.GetVariables(/*arguments=*/true,
                                            /*locals=*/true,
                                            /*statics=*/false,
                                            /*in_scope_only=*/true);
  dap.variables.globals = frame.GetVariables(/*arguments=*/false,
                                             /*locals=*/false,
                                             /*statics=*/true,
                                             /*in_scope_only=*/true);
  dap.variables.registers = frame.GetRegisters();
  body.try_emplace("scopes", dap.CreateTopLevelScopes());
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
void request_setBreakpoints(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  const auto *source = arguments->getObject("source");
  const auto path = GetString(source, "path");
  const auto *breakpoints = arguments->getArray("breakpoints");
  llvm::json::Array response_breakpoints;

  // Decode the source breakpoint infos for this "setBreakpoints" request
  SourceBreakpointMap request_bps;
  // "breakpoints" may be unset, in which case we treat it the same as being set
  // to an empty array.
  if (breakpoints) {
    for (const auto &bp : *breakpoints) {
      const auto *bp_obj = bp.getAsObject();
      if (bp_obj) {
        SourceBreakpoint src_bp(dap, *bp_obj);
        std::pair<uint32_t, uint32_t> bp_pos(src_bp.line, src_bp.column);
        request_bps.try_emplace(bp_pos, src_bp);
        const auto [iv, inserted] =
            dap.source_breakpoints[path].try_emplace(bp_pos, src_bp);
        // We check if this breakpoint already exists to update it
        if (inserted)
          iv->getSecond().SetBreakpoint(path.data());
        else
          iv->getSecond().UpdateBreakpoint(src_bp);
        AppendBreakpoint(&iv->getSecond(), response_breakpoints, path,
                         src_bp.line);
      }
    }
  }

  // Delete any breakpoints in this source file that aren't in the
  // request_bps set. There is no call to remove breakpoints other than
  // calling this function with a smaller or empty "breakpoints" list.
  auto old_src_bp_pos = dap.source_breakpoints.find(path);
  if (old_src_bp_pos != dap.source_breakpoints.end()) {
    for (auto &old_bp : old_src_bp_pos->second) {
      auto request_pos = request_bps.find(old_bp.first);
      if (request_pos == request_bps.end()) {
        // This breakpoint no longer exists in this source file, delete it
        dap.target.BreakpointDelete(old_bp.second.bp.GetID());
        old_src_bp_pos->second.erase(old_bp.first);
      }
    }
  }

  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
void request_setExceptionBreakpoints(DAP &dap,
                                     const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  const auto *filters = arguments->getArray("filters");
  // Keep a list of any exception breakpoint filter names that weren't set
  // so we can clear any exception breakpoints if needed.
  std::set<std::string> unset_filters;
  for (const auto &bp : *dap.exception_breakpoints)
    unset_filters.insert(bp.filter);

  for (const auto &value : *filters) {
    const auto filter = GetAsString(value);
    auto *exc_bp = dap.GetExceptionBreakpoint(std::string(filter));
    if (exc_bp) {
      exc_bp->SetBreakpoint();
      unset_filters.erase(std::string(filter));
    }
  }
  for (const auto &filter : unset_filters) {
    auto *exc_bp = dap.GetExceptionBreakpoint(filter);
    if (exc_bp)
      exc_bp->ClearBreakpoint();
  }
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
void request_setFunctionBreakpoints(DAP &dap,
                                    const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  const auto *breakpoints = arguments->getArray("breakpoints");
  llvm::json::Array response_breakpoints;

  // Disable any function breakpoints that aren't in this request.
  // There is no call to remove function breakpoints other than calling this
  // function with a smaller or empty "breakpoints" list.
  const auto name_iter = dap.function_breakpoints.keys();
  llvm::DenseSet<llvm::StringRef> seen(name_iter.begin(), name_iter.end());
  for (const auto &value : *breakpoints) {
    const auto *bp_obj = value.getAsObject();
    if (!bp_obj)
      continue;
    FunctionBreakpoint fn_bp(dap, *bp_obj);
    const auto [it, inserted] =
        dap.function_breakpoints.try_emplace(fn_bp.functionName, dap, *bp_obj);
    if (inserted)
      it->second.SetBreakpoint();
    else
      it->second.UpdateBreakpoint(fn_bp);

    AppendBreakpoint(&it->second, response_breakpoints);
    seen.erase(fn_bp.functionName);
  }

  // Remove any breakpoints that are no longer in our list
  for (const auto &name : seen) {
    auto fn_bp = dap.function_breakpoints.find(name);
    if (fn_bp == dap.function_breakpoints.end())
      continue;
    dap.target.BreakpointDelete(fn_bp->second.bp.GetID());
    dap.function_breakpoints.erase(name);
  }

  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

// "DataBreakpointInfoRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Obtains information on a possible data breakpoint that
//     could be set on an expression or variable.\nClients should only call this
//     request if the corresponding capability `supportsDataBreakpoints` is
//     true.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "dataBreakpointInfo" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/DataBreakpointInfoArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "DataBreakpointInfoArguments": {
//   "type": "object",
//   "description": "Arguments for `dataBreakpointInfo` request.",
//   "properties": {
//     "variablesReference": {
//       "type": "integer",
//       "description": "Reference to the variable container if the data
//       breakpoint is requested for a child of the container. The
//       `variablesReference` must have been obtained in the current suspended
//       state. See 'Lifetime of Object References' in the Overview section for
//       details."
//     },
//     "name": {
//       "type": "string",
//       "description": "The name of the variable's child to obtain data
//       breakpoint information for.\nIf `variablesReference` isn't specified,
//       this can be an expression."
//     },
//     "frameId": {
//       "type": "integer",
//       "description": "When `name` is an expression, evaluate it in the scope
//       of this stack frame. If not specified, the expression is evaluated in
//       the global scope. When `variablesReference` is specified, this property
//       has no effect."
//     }
//   },
//   "required": [ "name" ]
// },
// "DataBreakpointInfoResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `dataBreakpointInfo` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "dataId": {
//             "type": [ "string", "null" ],
//             "description": "An identifier for the data on which a data
//             breakpoint can be registered with the `setDataBreakpoints`
//             request or null if no data breakpoint is available. If a
//             `variablesReference` or `frameId` is passed, the `dataId` is
//             valid in the current suspended state, otherwise it's valid
//             indefinitely. See 'Lifetime of Object References' in the Overview
//             section for details. Breakpoints set using the `dataId` in the
//             `setDataBreakpoints` request may outlive the lifetime of the
//             associated `dataId`."
//           },
//           "description": {
//             "type": "string",
//             "description": "UI string that describes on what data the
//             breakpoint is set on or why a data breakpoint is not available."
//           },
//           "accessTypes": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/DataBreakpointAccessType"
//             },
//             "description": "Attribute lists the available access types for a
//             potential data breakpoint. A UI client could surface this
//             information."
//           },
//           "canPersist": {
//             "type": "boolean",
//             "description": "Attribute indicates that a potential data
//             breakpoint could be persisted across sessions."
//           }
//         },
//         "required": [ "dataId", "description" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void request_dataBreakpointInfo(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  lldb::SBError error;
  llvm::json::Array accessTypes{"read", "write", "readWrite"};
  const auto *arguments = request.getObject("arguments");
  const auto variablesReference =
      GetUnsigned(arguments, "variablesReference", 0);
  llvm::StringRef name = GetString(arguments, "name");
  lldb::SBFrame frame = dap.GetLLDBFrame(*arguments);
  lldb::SBValue variable = FindVariable(dap, variablesReference, name);
  std::string addr, size;

  if (variable.IsValid()) {
    lldb::addr_t load_addr = variable.GetLoadAddress();
    size_t byte_size = variable.GetByteSize();
    if (load_addr == LLDB_INVALID_ADDRESS) {
      body.try_emplace("dataId", nullptr);
      body.try_emplace("description",
                       "does not exist in memory, its location is " +
                           std::string(variable.GetLocation()));
    } else if (byte_size == 0) {
      body.try_emplace("dataId", nullptr);
      body.try_emplace("description", "variable size is 0");
    } else {
      addr = llvm::utohexstr(load_addr);
      size = llvm::utostr(byte_size);
    }
  } else if (variablesReference == 0 && frame.IsValid()) {
    lldb::SBValue value = frame.EvaluateExpression(name.data());
    if (value.GetError().Fail()) {
      lldb::SBError error = value.GetError();
      const char *error_cstr = error.GetCString();
      body.try_emplace("dataId", nullptr);
      body.try_emplace("description", error_cstr && error_cstr[0]
                                          ? std::string(error_cstr)
                                          : "evaluation failed");
    } else {
      uint64_t load_addr = value.GetValueAsUnsigned();
      lldb::SBData data = value.GetPointeeData();
      if (data.IsValid()) {
        size = llvm::utostr(data.GetByteSize());
        addr = llvm::utohexstr(load_addr);
        lldb::SBMemoryRegionInfo region;
        lldb::SBError err =
            dap.target.GetProcess().GetMemoryRegionInfo(load_addr, region);
        // Only lldb-server supports "qMemoryRegionInfo". So, don't fail this
        // request if SBProcess::GetMemoryRegionInfo returns error.
        if (err.Success()) {
          if (!(region.IsReadable() || region.IsWritable())) {
            body.try_emplace("dataId", nullptr);
            body.try_emplace("description",
                             "memory region for address " + addr +
                                 " has no read or write permissions");
          }
        }
      } else {
        body.try_emplace("dataId", nullptr);
        body.try_emplace("description",
                         "unable to get byte size for expression: " +
                             name.str());
      }
    }
  } else {
    body.try_emplace("dataId", nullptr);
    body.try_emplace("description", "variable not found: " + name.str());
  }

  if (!body.getObject("dataId")) {
    body.try_emplace("dataId", addr + "/" + size);
    body.try_emplace("accessTypes", std::move(accessTypes));
    body.try_emplace("description",
                     size + " bytes at " + addr + " " + name.str());
  }
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

// "SetDataBreakpointsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Replaces all existing data breakpoints with new data
//     breakpoints.\nTo clear all data breakpoints, specify an empty
//     array.\nWhen a data breakpoint is hit, a `stopped` event (with reason
//     `data breakpoint`) is generated.\nClients should only call this request
//     if the corresponding capability `supportsDataBreakpoints` is true.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "setDataBreakpoints" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SetDataBreakpointsArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SetDataBreakpointsArguments": {
//   "type": "object",
//   "description": "Arguments for `setDataBreakpoints` request.",
//   "properties": {
//     "breakpoints": {
//       "type": "array",
//       "items": {
//         "$ref": "#/definitions/DataBreakpoint"
//       },
//       "description": "The contents of this array replaces all existing data
//       breakpoints. An empty array clears all data breakpoints."
//     }
//   },
//   "required": [ "breakpoints" ]
// },
// "SetDataBreakpointsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `setDataBreakpoints` request.\nReturned is
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
//             "description": "Information about the data breakpoints. The array
//             elements correspond to the elements of the input argument
//             `breakpoints` array."
//           }
//         },
//         "required": [ "breakpoints" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void request_setDataBreakpoints(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  const auto *breakpoints = arguments->getArray("breakpoints");
  llvm::json::Array response_breakpoints;
  dap.target.DeleteAllWatchpoints();
  std::vector<Watchpoint> watchpoints;
  if (breakpoints) {
    for (const auto &bp : *breakpoints) {
      const auto *bp_obj = bp.getAsObject();
      if (bp_obj)
        watchpoints.emplace_back(dap, *bp_obj);
    }
  }
  // If two watchpoints start at the same address, the latter overwrite the
  // former. So, we only enable those at first-seen addresses when iterating
  // backward.
  std::set<lldb::addr_t> addresses;
  for (auto iter = watchpoints.rbegin(); iter != watchpoints.rend(); ++iter) {
    if (addresses.count(iter->addr) == 0) {
      iter->SetWatchpoint();
      addresses.insert(iter->addr);
    }
  }
  for (auto wp : watchpoints)
    AppendBreakpoint(&wp, response_breakpoints);

  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
void request_source(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body{{"content", ""}};
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
//       "description": "Specifies details on how to format the stack frames.
//       The attribute is only honored by a debug adapter if the corresponding
//       capability `supportsValueFormattingOptions` is true."
//     }
//  },
//   "required": [ "threadId" ]
// },
// "StackTraceResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `stackTrace` request.",
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
//             "description": "The total number of frames available in the
//             stack. If omitted or if `totalFrames` is larger than the
//             available frames, a client is expected to request frames until
//             a request returns less frames than requested (which indicates
//             the end of the stack). Returning monotonically increasing
//             `totalFrames` values for subsequent requests can be used to
//             enforce paging in the client."
//           }
//         },
//         "required": [ "stackFrames" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void request_stackTrace(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  lldb::SBError error;
  const auto *arguments = request.getObject("arguments");
  lldb::SBThread thread = dap.GetLLDBThread(*arguments);
  llvm::json::Array stack_frames;
  llvm::json::Object body;

  if (thread.IsValid()) {
    const auto start_frame = GetUnsigned(arguments, "startFrame", 0);
    const auto levels = GetUnsigned(arguments, "levels", 0);
    int64_t offset = 0;
    bool reached_end_of_stack =
        FillStackFrames(dap, thread, stack_frames, offset, start_frame,
                        levels == 0 ? INT64_MAX : levels);
    body.try_emplace("totalFrames",
                     start_frame + stack_frames.size() +
                         (reached_end_of_stack ? 0 : StackPageSize));
  }

  body.try_emplace("stackFrames", std::move(stack_frames));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
void request_threads(DAP &dap, const llvm::json::Object &request) {
  lldb::SBProcess process = dap.target.GetProcess();
  llvm::json::Object response;
  FillResponse(request, response);

  const uint32_t num_threads = process.GetNumThreads();
  llvm::json::Array threads;
  for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
    threads.emplace_back(CreateThread(thread, dap.thread_format));
  }
  if (threads.size() == 0) {
    response["success"] = llvm::json::Value(false);
  }
  llvm::json::Object body;
  body.try_emplace("threads", std::move(threads));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
//           },
//           "valueLocationReference": {
//             "type": "integer",
//             "description": "A reference that allows the client to request the
//             location where the new value is declared. For example, if the new
//             value is function pointer, the adapter may be able to look up the
//             function's location. This should be present only if the adapter
//             is likely to be able to resolve the location.\n\nThis reference
//             shares the same lifetime as the `variablesReference`. See
//             'Lifetime of Object References' in the Overview section for
//             details."
//           }
//         },
//         "required": [ "value" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void request_setVariable(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array variables;
  llvm::json::Object body;
  const auto *arguments = request.getObject("arguments");
  // This is a reference to the containing variable/scope
  const auto variablesReference =
      GetUnsigned(arguments, "variablesReference", 0);
  llvm::StringRef name = GetString(arguments, "name");

  const auto value = GetString(arguments, "value");
  // Set success to false just in case we don't find the variable by name
  response.try_emplace("success", false);

  lldb::SBValue variable;

  // The "id" is the unique integer ID that is unique within the enclosing
  // variablesReference. It is optionally added to any "interface Variable"
  // objects to uniquely identify a variable within an enclosing
  // variablesReference. It helps to disambiguate between two variables that
  // have the same name within the same scope since the "setVariables" request
  // only specifies the variable reference of the enclosing scope/variable, and
  // the name of the variable. We could have two shadowed variables with the
  // same name in "Locals" or "Globals". In our case the "id" absolute index
  // of the variable within the dap.variables list.
  const auto id_value = GetUnsigned(arguments, "id", UINT64_MAX);
  if (id_value != UINT64_MAX) {
    variable = dap.variables.GetVariable(id_value);
  } else {
    variable = FindVariable(dap, variablesReference, name);
  }

  if (variable.IsValid()) {
    lldb::SBError error;
    bool success = variable.SetValueFromCString(value.data(), error);
    if (success) {
      VariableDescription desc(variable, dap.enable_auto_variable_summaries);
      EmplaceSafeString(body, "result", desc.display_value);
      EmplaceSafeString(body, "type", desc.display_type_name);

      // We don't know the index of the variable in our dap.variables
      // so always insert a new one to get its variablesReference.
      // is_permanent is false because debug console does not support
      // setVariable request.
      int64_t new_var_ref =
          dap.variables.InsertVariable(variable, /*is_permanent=*/false);
      if (variable.MightHaveChildren())
        body.try_emplace("variablesReference", new_var_ref);
      else
        body.try_emplace("variablesReference", 0);
      if (lldb::addr_t addr = variable.GetLoadAddress();
          addr != LLDB_INVALID_ADDRESS)
        body.try_emplace("memoryReference", EncodeMemoryReference(addr));
      if (ValuePointsToCode(variable))
        body.try_emplace("valueLocationReference", new_var_ref);
    } else {
      EmplaceSafeString(body, "message", std::string(error.GetCString()));
    }
    response["success"] = llvm::json::Value(success);
  } else {
    response["success"] = llvm::json::Value(false);
  }

  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

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
void request_variables(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array variables;
  const auto *arguments = request.getObject("arguments");
  const auto variablesReference =
      GetUnsigned(arguments, "variablesReference", 0);
  const int64_t start = GetSigned(arguments, "start", 0);
  const int64_t count = GetSigned(arguments, "count", 0);
  bool hex = false;
  const auto *format = arguments->getObject("format");
  if (format)
    hex = GetBoolean(format, "hex", false);

  if (lldb::SBValueList *top_scope =
          GetTopLevelScope(dap, variablesReference)) {
    // variablesReference is one of our scopes, not an actual variable it is
    // asking for the list of args, locals or globals.
    int64_t start_idx = 0;
    int64_t num_children = 0;

    if (variablesReference == VARREF_REGS) {
      // Change the default format of any pointer sized registers in the first
      // register set to be the lldb::eFormatAddressInfo so we show the pointer
      // and resolve what the pointer resolves to. Only change the format if the
      // format was set to the default format or if it was hex as some registers
      // have formats set for them.
      const uint32_t addr_size = dap.target.GetProcess().GetAddressByteSize();
      lldb::SBValue reg_set = dap.variables.registers.GetValueAtIndex(0);
      const uint32_t num_regs = reg_set.GetNumChildren();
      for (uint32_t reg_idx = 0; reg_idx < num_regs; ++reg_idx) {
        lldb::SBValue reg = reg_set.GetChildAtIndex(reg_idx);
        const lldb::Format format = reg.GetFormat();
        if (format == lldb::eFormatDefault || format == lldb::eFormatHex) {
          if (reg.GetByteSize() == addr_size)
            reg.SetFormat(lldb::eFormatAddressInfo);
        }
      }
    }

    num_children = top_scope->GetSize();
    if (num_children == 0 && variablesReference == VARREF_LOCALS) {
      // Check for an error in the SBValueList that might explain why we don't
      // have locals. If we have an error display it as the sole value in the
      // the locals.

      // "error" owns the error string so we must keep it alive as long as we
      // want to use the returns "const char *"
      lldb::SBError error = top_scope->GetError();
      const char *var_err = error.GetCString();
      if (var_err) {
        // Create a fake variable named "error" to explain why variables were
        // not available. This new error will help let users know when there was
        // a problem that kept variables from being available for display and
        // allow users to fix this issue instead of seeing no variables. The
        // errors are only set when there is a problem that the user could
        // fix, so no error will show up when you have no debug info, only when
        // we do have debug info and something that is fixable can be done.
        llvm::json::Object object;
        EmplaceSafeString(object, "name", "<error>");
        EmplaceSafeString(object, "type", "const char *");
        EmplaceSafeString(object, "value", var_err);
        object.try_emplace("variablesReference", (int64_t)0);
        variables.emplace_back(std::move(object));
      }
    }
    const int64_t end_idx = start_idx + ((count == 0) ? num_children : count);

    // We first find out which variable names are duplicated
    std::map<std::string, int> variable_name_counts;
    for (auto i = start_idx; i < end_idx; ++i) {
      lldb::SBValue variable = top_scope->GetValueAtIndex(i);
      if (!variable.IsValid())
        break;
      variable_name_counts[GetNonNullVariableName(variable)]++;
    }

    // Now we construct the result with unique display variable names
    for (auto i = start_idx; i < end_idx; ++i) {
      lldb::SBValue variable = top_scope->GetValueAtIndex(i);

      if (!variable.IsValid())
        break;

      int64_t var_ref =
          dap.variables.InsertVariable(variable, /*is_permanent=*/false);
      variables.emplace_back(CreateVariable(
          variable, var_ref, hex, dap.enable_auto_variable_summaries,
          dap.enable_synthetic_child_debugging,
          variable_name_counts[GetNonNullVariableName(variable)] > 1));
    }
  } else {
    // We are expanding a variable that has children, so we will return its
    // children.
    lldb::SBValue variable = dap.variables.GetVariable(variablesReference);
    if (variable.IsValid()) {
      auto addChild = [&](lldb::SBValue child,
                          std::optional<std::string> custom_name = {}) {
        if (!child.IsValid())
          return;
        bool is_permanent =
            dap.variables.IsPermanentVariableReference(variablesReference);
        int64_t var_ref = dap.variables.InsertVariable(child, is_permanent);
        variables.emplace_back(CreateVariable(
            child, var_ref, hex, dap.enable_auto_variable_summaries,
            dap.enable_synthetic_child_debugging,
            /*is_name_duplicated=*/false, custom_name));
      };
      const int64_t num_children = variable.GetNumChildren();
      int64_t end_idx = start + ((count == 0) ? num_children : count);
      int64_t i = start;
      for (; i < end_idx && i < num_children; ++i)
        addChild(variable.GetChildAtIndex(i));

      // If we haven't filled the count quota from the request, we insert a new
      // "[raw]" child that can be used to inspect the raw version of a
      // synthetic member. That eliminates the need for the user to go to the
      // debug console and type `frame var <variable> to get these values.
      if (dap.enable_synthetic_child_debugging && variable.IsSynthetic() &&
          i == num_children)
        addChild(variable.GetNonSyntheticValue(), "[raw]");
    }
  }
  llvm::json::Object body;
  body.try_emplace("variables", std::move(variables));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

// "LocationsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Looks up information about a location reference
//                     previously returned by the debug adapter.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "locations" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/LocationsArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "LocationsArguments": {
//   "type": "object",
//   "description": "Arguments for `locations` request.",
//   "properties": {
//     "locationReference": {
//       "type": "integer",
//       "description": "Location reference to resolve."
//     }
//   },
//   "required": [ "locationReference" ]
// },
// "LocationsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `locations` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "source": {
//             "$ref": "#/definitions/Source",
//             "description": "The source containing the location; either
//                             `source.path` or `source.sourceReference` must be
//                             specified."
//           },
//           "line": {
//             "type": "integer",
//             "description": "The line number of the location. The client
//                             capability `linesStartAt1` determines whether it
//                             is 0- or 1-based."
//           },
//           "column": {
//             "type": "integer",
//             "description": "Position of the location within the `line`. It is
//                             measured in UTF-16 code units and the client
//                             capability `columnsStartAt1` determines whether
//                             it is 0- or 1-based. If no column is given, the
//                             first position in the start line is assumed."
//           },
//           "endLine": {
//             "type": "integer",
//             "description": "End line of the location, present if the location
//                             refers to a range.  The client capability
//                             `linesStartAt1` determines whether it is 0- or
//                             1-based."
//           },
//           "endColumn": {
//             "type": "integer",
//             "description": "End position of the location within `endLine`,
//                             present if the location refers to a range. It is
//                             measured in UTF-16 code units and the client
//                             capability `columnsStartAt1` determines whether
//                             it is 0- or 1-based."
//           }
//         },
//         "required": [ "source", "line" ]
//       }
//     }
//   }]
// },
void request_locations(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  auto *arguments = request.getObject("arguments");

  uint64_t location_id = GetUnsigned(arguments, "locationReference", 0);
  // We use the lowest bit to distinguish between value location and declaration
  // location
  auto [var_ref, is_value_location] = UnpackLocation(location_id);
  lldb::SBValue variable = dap.variables.GetVariable(var_ref);
  if (!variable.IsValid()) {
    response["success"] = false;
    response["message"] = "Invalid variable reference";
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  llvm::json::Object body;
  if (is_value_location) {
    // Get the value location
    if (!variable.GetType().IsPointerType() &&
        !variable.GetType().IsReferenceType()) {
      response["success"] = false;
      response["message"] =
          "Value locations are only available for pointers and references";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    lldb::addr_t addr = variable.GetValueAsAddress();
    lldb::SBLineEntry line_entry =
        dap.target.ResolveLoadAddress(addr).GetLineEntry();

    if (!line_entry.IsValid()) {
      response["success"] = false;
      response["message"] = "Failed to resolve line entry for location";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    body.try_emplace("source", CreateSource(line_entry.GetFileSpec()));
    if (int line = line_entry.GetLine())
      body.try_emplace("line", line);
    if (int column = line_entry.GetColumn())
      body.try_emplace("column", column);
  } else {
    // Get the declaration location
    lldb::SBDeclaration decl = variable.GetDeclaration();
    if (!decl.IsValid()) {
      response["success"] = false;
      response["message"] = "No declaration location available";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    body.try_emplace("source", CreateSource(decl.GetFileSpec()));
    if (int line = decl.GetLine())
      body.try_emplace("line", line);
    if (int column = decl.GetColumn())
      body.try_emplace("column", column);
  }

  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

// "DisassembleRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Disassembles code stored at the provided
//     location.\nClients should only call this request if the corresponding
//     capability `supportsDisassembleRequest` is true.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "disassemble" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/DisassembleArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "DisassembleArguments": {
//   "type": "object",
//   "description": "Arguments for `disassemble` request.",
//   "properties": {
//     "memoryReference": {
//       "type": "string",
//       "description": "Memory reference to the base location containing the
//       instructions to disassemble."
//     },
//     "offset": {
//       "type": "integer",
//       "description": "Offset (in bytes) to be applied to the reference
//       location before disassembling. Can be negative."
//     },
//     "instructionOffset": {
//       "type": "integer",
//       "description": "Offset (in instructions) to be applied after the byte
//       offset (if any) before disassembling. Can be negative."
//     },
//     "instructionCount": {
//       "type": "integer",
//       "description": "Number of instructions to disassemble starting at the
//       specified location and offset.\nAn adapter must return exactly this
//       number of instructions - any unavailable instructions should be
//       replaced with an implementation-defined 'invalid instruction' value."
//     },
//     "resolveSymbols": {
//       "type": "boolean",
//       "description": "If true, the adapter should attempt to resolve memory
//       addresses and other values to symbolic names."
//     }
//   },
//   "required": [ "memoryReference", "instructionCount" ]
// },
// "DisassembleResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `disassemble` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "instructions": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/DisassembledInstruction"
//             },
//             "description": "The list of disassembled instructions."
//           }
//         },
//         "required": [ "instructions" ]
//       }
//     }
//   }]
// }
void request_disassemble(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  auto *arguments = request.getObject("arguments");

  llvm::StringRef memoryReference = GetString(arguments, "memoryReference");
  auto addr_opt = DecodeMemoryReference(memoryReference);
  if (!addr_opt.has_value()) {
    response["success"] = false;
    response["message"] =
        "Malformed memory reference: " + memoryReference.str();
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }
  lldb::addr_t addr_ptr = *addr_opt;

  addr_ptr += GetSigned(arguments, "instructionOffset", 0);
  lldb::SBAddress addr(addr_ptr, dap.target);
  if (!addr.IsValid()) {
    response["success"] = false;
    response["message"] = "Memory reference not found in the current binary.";
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  const auto inst_count = GetUnsigned(arguments, "instructionCount", 0);
  lldb::SBInstructionList insts = dap.target.ReadInstructions(addr, inst_count);

  if (!insts.IsValid()) {
    response["success"] = false;
    response["message"] = "Failed to find instructions for memory address.";
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  const bool resolveSymbols = GetBoolean(arguments, "resolveSymbols", false);
  llvm::json::Array instructions;
  const auto num_insts = insts.GetSize();
  for (size_t i = 0; i < num_insts; ++i) {
    lldb::SBInstruction inst = insts.GetInstructionAtIndex(i);
    auto addr = inst.GetAddress();
    const auto inst_addr = addr.GetLoadAddress(dap.target);
    const char *m = inst.GetMnemonic(dap.target);
    const char *o = inst.GetOperands(dap.target);
    const char *c = inst.GetComment(dap.target);
    auto d = inst.GetData(dap.target);

    std::string bytes;
    llvm::raw_string_ostream sb(bytes);
    for (unsigned i = 0; i < inst.GetByteSize(); i++) {
      lldb::SBError error;
      uint8_t b = d.GetUnsignedInt8(error, i);
      if (error.Success()) {
        sb << llvm::format("%2.2x ", b);
      }
    }

    llvm::json::Object disassembled_inst{
        {"address", "0x" + llvm::utohexstr(inst_addr)},
        {"instructionBytes",
         bytes.size() > 0 ? bytes.substr(0, bytes.size() - 1) : ""},
    };

    std::string instruction;
    llvm::raw_string_ostream si(instruction);

    lldb::SBSymbol symbol = addr.GetSymbol();
    // Only add the symbol on the first line of the function.
    if (symbol.IsValid() && symbol.GetStartAddress() == addr) {
      // If we have a valid symbol, append it as a label prefix for the first
      // instruction. This is so you can see the start of a function/callsite
      // in the assembly, at the moment VS Code (1.80) does not visualize the
      // symbol associated with the assembly instruction.
      si << (symbol.GetMangledName() != nullptr ? symbol.GetMangledName()
                                                : symbol.GetName())
         << ": ";

      if (resolveSymbols) {
        disassembled_inst.try_emplace("symbol", symbol.GetDisplayName());
      }
    }

    si << llvm::formatv("{0,7} {1,12}", m, o);
    if (c && c[0]) {
      si << " ; " << c;
    }

    disassembled_inst.try_emplace("instruction", instruction);

    auto line_entry = addr.GetLineEntry();
    // If the line number is 0 then the entry represents a compiler generated
    // location.
    if (line_entry.GetStartAddress() == addr && line_entry.IsValid() &&
        line_entry.GetFileSpec().IsValid() && line_entry.GetLine() != 0) {
      auto source = CreateSource(line_entry);
      disassembled_inst.try_emplace("location", source);

      const auto line = line_entry.GetLine();
      if (line && line != LLDB_INVALID_LINE_NUMBER) {
        disassembled_inst.try_emplace("line", line);
      }
      const auto column = line_entry.GetColumn();
      if (column && column != LLDB_INVALID_COLUMN_NUMBER) {
        disassembled_inst.try_emplace("column", column);
      }

      auto end_line_entry = line_entry.GetEndAddress().GetLineEntry();
      if (end_line_entry.IsValid() &&
          end_line_entry.GetFileSpec() == line_entry.GetFileSpec()) {
        const auto end_line = end_line_entry.GetLine();
        if (end_line && end_line != LLDB_INVALID_LINE_NUMBER &&
            end_line != line) {
          disassembled_inst.try_emplace("endLine", end_line);

          const auto end_column = end_line_entry.GetColumn();
          if (end_column && end_column != LLDB_INVALID_COLUMN_NUMBER &&
              end_column != column) {
            disassembled_inst.try_emplace("endColumn", end_column - 1);
          }
        }
      }
    }

    instructions.emplace_back(std::move(disassembled_inst));
  }

  llvm::json::Object body;
  body.try_emplace("instructions", std::move(instructions));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

// "ReadMemoryRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Reads bytes from memory at the provided location. Clients
//                     should only call this request if the corresponding
//                     capability `supportsReadMemoryRequest` is true.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "readMemory" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/ReadMemoryArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "ReadMemoryArguments": {
//   "type": "object",
//   "description": "Arguments for `readMemory` request.",
//   "properties": {
//     "memoryReference": {
//       "type": "string",
//       "description": "Memory reference to the base location from which data
//                       should be read."
//     },
//     "offset": {
//       "type": "integer",
//       "description": "Offset (in bytes) to be applied to the reference
//                       location before reading data. Can be negative."
//     },
//     "count": {
//       "type": "integer",
//       "description": "Number of bytes to read at the specified location and
//                       offset."
//     }
//   },
//   "required": [ "memoryReference", "count" ]
// },
// "ReadMemoryResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `readMemory` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "address": {
//             "type": "string",
//             "description": "The address of the first byte of data returned.
//                             Treated as a hex value if prefixed with `0x`, or
//                             as a decimal value otherwise."
//           },
//           "unreadableBytes": {
//             "type": "integer",
//             "description": "The number of unreadable bytes encountered after
//                             the last successfully read byte.\nThis can be
//                             used to determine the number of bytes that should
//                             be skipped before a subsequent
//             `readMemory` request succeeds."
//           },
//           "data": {
//             "type": "string",
//             "description": "The bytes read from memory, encoded using base64.
//                             If the decoded length of `data` is less than the
//                             requested `count` in the original `readMemory`
//                             request, and `unreadableBytes` is zero or
//                             omitted, then the client should assume it's
//                             reached the end of readable memory."
//           }
//         },
//         "required": [ "address" ]
//       }
//     }
//   }]
// },
void request_readMemory(DAP &dap, const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  auto *arguments = request.getObject("arguments");

  llvm::StringRef memoryReference = GetString(arguments, "memoryReference");
  auto addr_opt = DecodeMemoryReference(memoryReference);
  if (!addr_opt.has_value()) {
    response["success"] = false;
    response["message"] =
        "Malformed memory reference: " + memoryReference.str();
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }
  lldb::addr_t addr_int = *addr_opt;
  addr_int += GetSigned(arguments, "offset", 0);
  const uint64_t count_requested = GetUnsigned(arguments, "count", 0);

  // We also need support reading 0 bytes
  // VS Code sends those requests to check if a `memoryReference`
  // can be dereferenced.
  const uint64_t count_read = std::max<uint64_t>(count_requested, 1);
  std::vector<uint8_t> buf;
  buf.resize(count_read);
  lldb::SBError error;
  lldb::SBAddress addr{addr_int, dap.target};
  size_t count_result =
      dap.target.ReadMemory(addr, buf.data(), count_read, error);
  if (count_result == 0) {
    response["success"] = false;
    EmplaceSafeString(response, "message", error.GetCString());
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }
  buf.resize(std::min<size_t>(count_result, count_requested));

  llvm::json::Object body;
  std::string formatted_addr = "0x" + llvm::utohexstr(addr_int);
  body.try_emplace("address", formatted_addr);
  body.try_emplace("data", llvm::encodeBase64(buf));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

// A request used in testing to get the details on all breakpoints that are
// currently set in the target. This helps us to test "setBreakpoints" and
// "setFunctionBreakpoints" requests to verify we have the correct set of
// breakpoints currently set in LLDB.
void request__testGetTargetBreakpoints(DAP &dap,
                                       const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array response_breakpoints;
  for (uint32_t i = 0; dap.target.GetBreakpointAtIndex(i).IsValid(); ++i) {
    auto bp = Breakpoint(dap, dap.target.GetBreakpointAtIndex(i));
    AppendBreakpoint(&bp, response_breakpoints);
  }
  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

// "SetInstructionBreakpointsRequest": {
//   "allOf": [
//     {"$ref": "#/definitions/Request"},
//     {
//       "type": "object",
//       "description" :
//           "Replaces all existing instruction breakpoints. Typically, "
//           "instruction breakpoints would be set from a disassembly window. "
//           "\nTo clear all instruction breakpoints, specify an empty "
//           "array.\nWhen an instruction breakpoint is hit, a `stopped` event "
//           "(with reason `instruction breakpoint`) is generated.\nClients "
//           "should only call this request if the corresponding capability "
//           "`supportsInstructionBreakpoints` is true.",
//       "properties": {
//         "command": { "type": "string", "enum": ["setInstructionBreakpoints"]
//         }, "arguments": {"$ref":
//         "#/definitions/SetInstructionBreakpointsArguments"}
//       },
//       "required": [ "command", "arguments" ]
//     }
//   ]
// },
// "SetInstructionBreakpointsArguments": {
//   "type": "object",
//   "description": "Arguments for `setInstructionBreakpoints` request",
//   "properties": {
//     "breakpoints": {
//       "type": "array",
//       "items": {"$ref": "#/definitions/InstructionBreakpoint"},
//       "description": "The instruction references of the breakpoints"
//     }
//   },
//   "required": ["breakpoints"]
// },
// "SetInstructionBreakpointsResponse": {
//   "allOf": [
//     {"$ref": "#/definitions/Response"},
//     {
//       "type": "object",
//       "description": "Response to `setInstructionBreakpoints` request",
//       "properties": {
//         "body": {
//           "type": "object",
//           "properties": {
//             "breakpoints": {
//               "type": "array",
//               "items": {"$ref": "#/definitions/Breakpoint"},
//               "description":
//                   "Information about the breakpoints. The array elements
//                   " "correspond to the elements of the `breakpoints`
//                   array."
//             }
//           },
//           "required": ["breakpoints"]
//         }
//       },
//       "required": ["body"]
//     }
//   ]
// },
// "InstructionBreakpoint": {
//   "type": "object",
//   "description": "Properties of a breakpoint passed to the "
//                   "`setInstructionBreakpoints` request",
//   "properties": {
//     "instructionReference": {
//       "type": "string",
//       "description" :
//           "The instruction reference of the breakpoint.\nThis should be a "
//           "memory or instruction pointer reference from an
//           `EvaluateResponse`, "
//           "`Variable`, `StackFrame`, `GotoTarget`, or `Breakpoint`."
//     },
//     "offset": {
//       "type": "integer",
//       "description": "The offset from the instruction reference in "
//                       "bytes.\nThis can be negative."
//     },
//     "condition": {
//       "type": "string",
//       "description": "An expression for conditional breakpoints.\nIt is only
//       "
//                       "honored by a debug adapter if the corresponding "
//                       "capability `supportsConditionalBreakpoints` is true."
//     },
//     "hitCondition": {
//       "type": "string",
//       "description": "An expression that controls how many hits of the "
//                       "breakpoint are ignored.\nThe debug adapter is expected
//                       " "to interpret the expression as needed.\nThe
//                       attribute " "is only honored by a debug adapter if the
//                       corresponding " "capability
//                       `supportsHitConditionalBreakpoints` is true."
//     },
//     "mode": {
//       "type": "string",
//       "description": "The mode of this breakpoint. If defined, this must be
//       "
//                       "one of the `breakpointModes` the debug adapter "
//                       "advertised in its `Capabilities`."
//     }
//   },
//   "required": ["instructionReference"]
// },
// "Breakpoint": {
//   "type": "object",
//   "description" :
//       "Information about a breakpoint created in `setBreakpoints`, "
//       "`setFunctionBreakpoints`, `setInstructionBreakpoints`, or "
//       "`setDataBreakpoints` requests.",
//   "properties": {
//     "id": {
//       "type": "integer",
//       "description" :
//           "The identifier for the breakpoint. It is needed if breakpoint
//           " "events are used to update or remove breakpoints."
//     },
//     "verified": {
//       "type": "boolean",
//       "description": "If true, the breakpoint could be set (but not "
//                       "necessarily at the desired location)."
//     },
//     "message": {
//       "type": "string",
//       "description": "A message about the state of the breakpoint.\nThis
//       "
//                       "is shown to the user and can be used to explain
//                       why " "a breakpoint could not be verified."
//     },
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "The source where the breakpoint is located."
//     },
//     "line": {
//       "type": "integer",
//       "description" :
//           "The start line of the actual range covered by the breakpoint."
//     },
//     "column": {
//       "type": "integer",
//       "description" :
//           "Start position of the source range covered by the breakpoint.
//           " "It is measured in UTF-16 code units and the client
//           capability "
//           "`columnsStartAt1` determines whether it is 0- or 1-based."
//     },
//     "endLine": {
//       "type": "integer",
//       "description" :
//           "The end line of the actual range covered by the breakpoint."
//     },
//     "endColumn": {
//       "type": "integer",
//       "description" :
//           "End position of the source range covered by the breakpoint. It
//           " "is measured in UTF-16 code units and the client capability "
//           "`columnsStartAt1` determines whether it is 0- or 1-based.\nIf
//           " "no end line is given, then the end column is assumed to be
//           in " "the start line."
//     },
//     "instructionReference": {
//       "type": "string",
//       "description": "A memory reference to where the breakpoint is
//       set."
//     },
//     "offset": {
//       "type": "integer",
//       "description": "The offset from the instruction reference.\nThis "
//                       "can be negative."
//     },
//     "reason": {
//       "type": "string",
//       "description" :
//           "A machine-readable explanation of why a breakpoint may not be
//           " "verified. If a breakpoint is verified or a specific reason
//           is " "not known, the adapter should omit this property.
//           Possible " "values include:\n\n- `pending`: Indicates a
//           breakpoint might be " "verified in the future, but the adapter
//           cannot verify it in the " "current state.\n - `failed`:
//           Indicates a breakpoint was not " "able to be verified, and the
//           adapter does not believe it can be " "verified without
//           intervention.",
//       "enum": [ "pending", "failed" ]
//     }
//   },
//   "required": ["verified"]
// },
void request_setInstructionBreakpoints(DAP &dap,
                                       const llvm::json::Object &request) {
  llvm::json::Object response;
  llvm::json::Array response_breakpoints;
  llvm::json::Object body;
  FillResponse(request, response);

  const auto *arguments = request.getObject("arguments");
  const auto *breakpoints = arguments->getArray("breakpoints");

  // Disable any instruction breakpoints that aren't in this request.
  // There is no call to remove instruction breakpoints other than calling this
  // function with a smaller or empty "breakpoints" list.
  llvm::DenseSet<lldb::addr_t> seen;
  for (const auto &addr : dap.instruction_breakpoints)
    seen.insert(addr.first);

  for (const auto &bp : *breakpoints) {
    const auto *bp_obj = bp.getAsObject();
    if (!bp_obj)
      continue;
    // Read instruction breakpoint request.
    InstructionBreakpoint inst_bp(dap, *bp_obj);
    const auto [iv, inserted] = dap.instruction_breakpoints.try_emplace(
        inst_bp.instructionAddressReference, dap, *bp_obj);
    if (inserted)
      iv->second.SetBreakpoint();
    else
      iv->second.UpdateBreakpoint(inst_bp);
    AppendBreakpoint(&iv->second, response_breakpoints);
    seen.erase(inst_bp.instructionAddressReference);
  }

  for (const auto &addr : seen) {
    auto inst_bp = dap.instruction_breakpoints.find(addr);
    if (inst_bp == dap.instruction_breakpoints.end())
      continue;
    dap.target.BreakpointDelete(inst_bp->second.bp.GetID());
    dap.instruction_breakpoints.erase(addr);
  }

  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

void RegisterRequestCallbacks(DAP &dap) {
  dap.RegisterRequest<AttachRequestHandler>();
  dap.RegisterRequest<BreakpointLocationsRequestHandler>();
  dap.RegisterRequest<CompletionsRequestHandler>();
  dap.RegisterRequest<ConfigurationDoneRequestHandler>();
  dap.RegisterRequest<ContinueRequestHandler>();
  dap.RegisterRequest<DisconnectRequestHandler>();
  dap.RegisterRequest<EvaluateRequestHandler>();
  dap.RegisterRequest<ExceptionInfoRequestHandler>();
  dap.RegisterRequest<InitializeRequestHandler>();
  dap.RegisterRequest<LaunchRequestHandler>();
  dap.RegisterRequest<NextRequestHandler>();
  dap.RegisterRequest<RestartRequestHandler>();
  dap.RegisterRequest<StepInRequestHandler>();
  dap.RegisterRequest<StepInTargetsRequestHandler>();
  dap.RegisterRequest<StepOutRequestHandler>();

  dap.RegisterRequestCallback("pause", request_pause);
  dap.RegisterRequestCallback("scopes", request_scopes);
  dap.RegisterRequestCallback("setBreakpoints", request_setBreakpoints);
  dap.RegisterRequestCallback("setExceptionBreakpoints",
                              request_setExceptionBreakpoints);
  dap.RegisterRequestCallback("setFunctionBreakpoints",
                              request_setFunctionBreakpoints);
  dap.RegisterRequestCallback("dataBreakpointInfo", request_dataBreakpointInfo);
  dap.RegisterRequestCallback("setDataBreakpoints", request_setDataBreakpoints);
  dap.RegisterRequestCallback("setVariable", request_setVariable);
  dap.RegisterRequestCallback("source", request_source);
  dap.RegisterRequestCallback("stackTrace", request_stackTrace);
  dap.RegisterRequestCallback("threads", request_threads);
  dap.RegisterRequestCallback("variables", request_variables);
  dap.RegisterRequestCallback("locations", request_locations);
  dap.RegisterRequestCallback("disassemble", request_disassemble);
  dap.RegisterRequestCallback("readMemory", request_readMemory);
  dap.RegisterRequestCallback("setInstructionBreakpoints",
                              request_setInstructionBreakpoints);
  // Custom requests
  dap.RegisterRequestCallback("compileUnits", request_compileUnits);
  dap.RegisterRequestCallback("modules", request_modules);
  // Testing requests
  dap.RegisterRequestCallback("_testGetTargetBreakpoints",
                              request__testGetTargetBreakpoints);
}

} // anonymous namespace

static void printHelp(LLDBDAPOptTable &table, llvm::StringRef tool_name) {
  std::string usage_str = tool_name.str() + " options";
  table.printHelp(llvm::outs(), usage_str.c_str(), "LLDB DAP", false);

  std::string examples = R"___(
EXAMPLES:
  The debug adapter can be started in two modes.

  Running lldb-dap without any arguments will start communicating with the
  parent over stdio. Passing a --connection URI will cause lldb-dap to listen
  for a connection in the specified mode.

    lldb-dap --connection connection://localhost:<port>

  Passing --wait-for-debugger will pause the process at startup and wait for a
  debugger to attach to the process.

    lldb-dap -g
)___";
  llvm::outs() << examples;
}

// If --launch-target is provided, this instance of lldb-dap becomes a
// runInTerminal launcher. It will ultimately launch the program specified in
// the --launch-target argument, which is the original program the user wanted
// to debug. This is done in such a way that the actual debug adaptor can
// place breakpoints at the beginning of the program.
//
// The launcher will communicate with the debug adaptor using a fifo file in the
// directory specified in the --comm-file argument.
//
// Regarding the actual flow, this launcher will first notify the debug adaptor
// of its pid. Then, the launcher will be in a pending state waiting to be
// attached by the adaptor.
//
// Once attached and resumed, the launcher will exec and become the program
// specified by --launch-target, which is the original target the
// user wanted to run.
//
// In case of errors launching the target, a suitable error message will be
// emitted to the debug adaptor.
static void LaunchRunInTerminalTarget(llvm::opt::Arg &target_arg,
                                      llvm::StringRef comm_file,
                                      lldb::pid_t debugger_pid, char *argv[]) {
#if defined(_WIN32)
  llvm::errs() << "runInTerminal is only supported on POSIX systems\n";
  exit(EXIT_FAILURE);
#else

  // On Linux with the Yama security module enabled, a process can only attach
  // to its descendants by default. In the runInTerminal case the target
  // process is launched by the client so we need to allow tracing explicitly.
#if defined(__linux__)
  if (debugger_pid != LLDB_INVALID_PROCESS_ID)
    (void)prctl(PR_SET_PTRACER, debugger_pid, 0, 0, 0);
#endif

  RunInTerminalLauncherCommChannel comm_channel(comm_file);
  if (llvm::Error err = comm_channel.NotifyPid()) {
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
    exit(EXIT_FAILURE);
  }

  // We will wait to be attached with a timeout. We don't wait indefinitely
  // using a signal to prevent being paused forever.

  // This env var should be used only for tests.
  const char *timeout_env_var = getenv("LLDB_DAP_RIT_TIMEOUT_IN_MS");
  int timeout_in_ms =
      timeout_env_var != nullptr ? atoi(timeout_env_var) : 20000;
  if (llvm::Error err = comm_channel.WaitUntilDebugAdaptorAttaches(
          std::chrono::milliseconds(timeout_in_ms))) {
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
    exit(EXIT_FAILURE);
  }

  const char *target = target_arg.getValue();
  execvp(target, argv);

  std::string error = std::strerror(errno);
  comm_channel.NotifyError(error);
  llvm::errs() << error << "\n";
  exit(EXIT_FAILURE);
#endif
}

/// used only by TestVSCode_redirection_to_console.py
static void redirection_test() {
  printf("stdout message\n");
  fprintf(stderr, "stderr message\n");
  fflush(stdout);
  fflush(stderr);
}

/// Duplicates a file descriptor, setting FD_CLOEXEC if applicable.
static int DuplicateFileDescriptor(int fd) {
#if defined(F_DUPFD_CLOEXEC)
  // Ensure FD_CLOEXEC is set.
  return ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
#else
  return ::dup(fd);
#endif
}

static llvm::Expected<std::pair<Socket::SocketProtocol, std::string>>
validateConnection(llvm::StringRef conn) {
  auto uri = lldb_private::URI::Parse(conn);

  if (uri && (uri->scheme == "tcp" || uri->scheme == "connect" ||
              !uri->hostname.empty() || uri->port)) {
    return std::make_pair(
        Socket::ProtocolTcp,
        formatv("[{0}]:{1}", uri->hostname.empty() ? "0.0.0.0" : uri->hostname,
                uri->port.value_or(0)));
  }

  if (uri && (uri->scheme == "unix" || uri->scheme == "unix-connect" ||
              uri->path != "/")) {
    return std::make_pair(Socket::ProtocolUnixDomain, uri->path.str());
  }

  return llvm::createStringError(
      "Unsupported connection specifier, expected 'unix-connect:///path' or "
      "'connect://[host]:port', got '%s'.",
      conn.str().c_str());
}

static llvm::Error
serveConnection(const Socket::SocketProtocol &protocol, const std::string &name,
                std::ofstream *log, llvm::StringRef program_path,
                const ReplMode default_repl_mode,
                const std::vector<std::string> &pre_init_commands) {
  Status status;
  static std::unique_ptr<Socket> listener = Socket::Create(protocol, status);
  if (status.Fail()) {
    return status.takeError();
  }

  status = listener->Listen(name, /*backlog=*/5);
  if (status.Fail()) {
    return status.takeError();
  }

  std::string address = llvm::join(listener->GetListeningConnectionURI(), ", ");
  if (log)
    *log << "started with connection listeners " << address << "\n";

  llvm::outs() << "Listening for: " << address << "\n";
  // Ensure listening address are flushed for calles to retrieve the resolve
  // address.
  llvm::outs().flush();

  static lldb_private::MainLoop g_loop;
  llvm::sys::SetInterruptFunction([]() {
    g_loop.AddPendingCallback(
        [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
  });
  std::condition_variable dap_sessions_condition;
  std::mutex dap_sessions_mutex;
  std::map<Socket *, DAP *> dap_sessions;
  unsigned int clientCount = 0;
  auto handle = listener->Accept(g_loop, [=, &dap_sessions_condition,
                                          &dap_sessions_mutex, &dap_sessions,
                                          &clientCount](
                                             std::unique_ptr<Socket> sock) {
    std::string name = llvm::formatv("client_{0}", clientCount++).str();
    if (log) {
      auto now = std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch());
      *log << llvm::formatv("{0:f9}", now.count()).str()
           << " client connected: " << name << "\n";
    }

    // Move the client into a background thread to unblock accepting the next
    // client.
    std::thread client([=, &dap_sessions_condition, &dap_sessions_mutex,
                        &dap_sessions, sock = std::move(sock)]() {
      llvm::set_thread_name(name + ".runloop");
      StreamDescriptor input =
          StreamDescriptor::from_socket(sock->GetNativeSocket(), false);
      // Close the output last for the best chance at error reporting.
      StreamDescriptor output =
          StreamDescriptor::from_socket(sock->GetNativeSocket(), false);
      DAP dap = DAP(name, program_path, log, std::move(input),
                    std::move(output), default_repl_mode, pre_init_commands);

      if (auto Err = dap.ConfigureIO()) {
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                    "Failed to configure stdout redirect: ");
        return;
      }

      RegisterRequestCallbacks(dap);

      {
        std::scoped_lock<std::mutex> lock(dap_sessions_mutex);
        dap_sessions[sock.get()] = &dap;
      }

      if (auto Err = dap.Loop()) {
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                    "DAP session error: ");
      }

      if (log) {
        auto now = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch());
        *log << llvm::formatv("{0:f9}", now.count()).str()
             << " client closed: " << name << "\n";
      }

      std::unique_lock<std::mutex> lock(dap_sessions_mutex);
      dap_sessions.erase(sock.get());
      std::notify_all_at_thread_exit(dap_sessions_condition, std::move(lock));
    });
    client.detach();
  });

  if (auto Err = handle.takeError()) {
    return Err;
  }

  status = g_loop.Run();
  if (status.Fail()) {
    return status.takeError();
  }

  if (log)
    *log << "lldb-dap server shutdown requested, disconnecting remaining "
            "clients...\n";

  bool client_failed = false;
  {
    std::scoped_lock<std::mutex> lock(dap_sessions_mutex);
    for (auto [sock, dap] : dap_sessions) {
      auto error = dap->Disconnect();
      if (error.Fail()) {
        client_failed = true;
        llvm::errs() << "DAP client " << dap->name
                     << " disconnected failed: " << error.GetCString() << "\n";
      }
      // Close the socket to ensure the DAP::Loop read finishes.
      sock->Close();
    }
  }

  // Wait for all clients to finish disconnecting.
  std::unique_lock<std::mutex> lock(dap_sessions_mutex);
  dap_sessions_condition.wait(lock, [&] { return dap_sessions.empty(); });

  if (client_failed)
    return llvm::make_error<llvm::StringError>(
        "disconnecting all clients failed", llvm::inconvertibleErrorCode());

  return llvm::Error::success();
}

int main(int argc, char *argv[]) {
  llvm::InitLLVM IL(argc, argv, /*InstallPipeSignalExitHandler=*/false);
#if !defined(__APPLE__)
  llvm::setBugReportMsg("PLEASE submit a bug report to " LLDB_BUG_REPORT_URL
                        " and include the crash backtrace.\n");
#else
  llvm::setBugReportMsg("PLEASE submit a bug report to " LLDB_BUG_REPORT_URL
                        " and include the crash report from "
                        "~/Library/Logs/DiagnosticReports/.\n");
#endif

  llvm::SmallString<256> program_path(argv[0]);
  llvm::sys::fs::make_absolute(program_path);

  LLDBDAPOptTable T;
  unsigned MAI, MAC;
  llvm::ArrayRef<const char *> ArgsArr = llvm::ArrayRef(argv + 1, argc);
  llvm::opt::InputArgList input_args = T.ParseArgs(ArgsArr, MAI, MAC);

  if (input_args.hasArg(OPT_help)) {
    printHelp(T, llvm::sys::path::filename(argv[0]));
    return EXIT_SUCCESS;
  }

  ReplMode default_repl_mode = ReplMode::Auto;
  if (input_args.hasArg(OPT_repl_mode)) {
    llvm::opt::Arg *repl_mode = input_args.getLastArg(OPT_repl_mode);
    llvm::StringRef repl_mode_value = repl_mode->getValue();
    if (repl_mode_value == "auto") {
      default_repl_mode = ReplMode::Auto;
    } else if (repl_mode_value == "variable") {
      default_repl_mode = ReplMode::Variable;
    } else if (repl_mode_value == "command") {
      default_repl_mode = ReplMode::Command;
    } else {
      llvm::errs() << "'" << repl_mode_value
                   << "' is not a valid option, use 'variable', 'command' or "
                      "'auto'.\n";
      return EXIT_FAILURE;
    }
  }

  if (llvm::opt::Arg *target_arg = input_args.getLastArg(OPT_launch_target)) {
    if (llvm::opt::Arg *comm_file = input_args.getLastArg(OPT_comm_file)) {
      lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
      llvm::opt::Arg *debugger_pid = input_args.getLastArg(OPT_debugger_pid);
      if (debugger_pid) {
        llvm::StringRef debugger_pid_value = debugger_pid->getValue();
        if (debugger_pid_value.getAsInteger(10, pid)) {
          llvm::errs() << "'" << debugger_pid_value
                       << "' is not a valid "
                          "PID\n";
          return EXIT_FAILURE;
        }
      }
      int target_args_pos = argc;
      for (int i = 0; i < argc; i++)
        if (strcmp(argv[i], "--launch-target") == 0) {
          target_args_pos = i + 1;
          break;
        }
      LaunchRunInTerminalTarget(*target_arg, comm_file->getValue(), pid,
                                argv + target_args_pos);
    } else {
      llvm::errs() << "\"--launch-target\" requires \"--comm-file\" to be "
                      "specified\n";
      return EXIT_FAILURE;
    }
  }

  std::string connection;
  if (auto *arg = input_args.getLastArg(OPT_connection)) {
    const auto *path = arg->getValue();
    connection.assign(path);
  }

#if !defined(_WIN32)
  if (input_args.hasArg(OPT_wait_for_debugger)) {
    printf("Paused waiting for debugger to attach (pid = %i)...\n", getpid());
    pause();
  }
#endif

  std::unique_ptr<std::ofstream> log = nullptr;
  const char *log_file_path = getenv("LLDBDAP_LOG");
  if (log_file_path)
    log = std::make_unique<std::ofstream>(log_file_path);

  // Initialize LLDB first before we do anything.
  lldb::SBError error = lldb::SBDebugger::InitializeWithErrorHandling();
  if (error.Fail()) {
    lldb::SBStream os;
    error.GetDescription(os);
    llvm::errs() << "lldb initialize failed: " << os.GetData() << "\n";
    return EXIT_FAILURE;
  }

  // Terminate the debugger before the C++ destructor chain kicks in.
  auto terminate_debugger =
      llvm::make_scope_exit([] { lldb::SBDebugger::Terminate(); });

  std::vector<std::string> pre_init_commands;
  for (const std::string &arg :
       input_args.getAllArgValues(OPT_pre_init_command)) {
    pre_init_commands.push_back(arg);
  }

  if (!connection.empty()) {
    auto maybeProtoclAndName = validateConnection(connection);
    if (auto Err = maybeProtoclAndName.takeError()) {
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Invalid connection: ");
      return EXIT_FAILURE;
    }

    Socket::SocketProtocol protocol;
    std::string name;
    std::tie(protocol, name) = *maybeProtoclAndName;
    if (auto Err = serveConnection(protocol, name, log.get(), program_path,
                                   default_repl_mode, pre_init_commands)) {
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Connection failed: ");
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

#if defined(_WIN32)
  // Windows opens stdout and stdin in text mode which converts \n to 13,10
  // while the value is just 10 on Darwin/Linux. Setting the file mode to
  // binary fixes this.
  int result = _setmode(fileno(stdout), _O_BINARY);
  assert(result);
  result = _setmode(fileno(stdin), _O_BINARY);
  UNUSED_IF_ASSERT_DISABLED(result);
  assert(result);
#endif

  int stdout_fd = DuplicateFileDescriptor(fileno(stdout));
  if (stdout_fd == -1) {
    llvm::logAllUnhandledErrors(
        llvm::errorCodeToError(llvm::errnoAsErrorCode()), llvm::errs(),
        "Failed to configure stdout redirect: ");
    return EXIT_FAILURE;
  }

  StreamDescriptor input = StreamDescriptor::from_file(fileno(stdin), false);
  StreamDescriptor output = StreamDescriptor::from_file(stdout_fd, false);

  DAP dap = DAP("stdin/stdout", program_path, log.get(), std::move(input),
                std::move(output), default_repl_mode, pre_init_commands);

  // stdout/stderr redirection to the IDE's console
  if (auto Err = dap.ConfigureIO(stdout, stderr)) {
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                "Failed to configure stdout redirect: ");
    return EXIT_FAILURE;
  }

  RegisterRequestCallbacks(dap);

  // used only by TestVSCode_redirection_to_console.py
  if (getenv("LLDB_DAP_TEST_STDOUT_STDERR_REDIRECTION") != nullptr)
    redirection_test();

  if (auto Err = dap.Loop()) {
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                "DAP session error: ");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
