//===-- JSONUtils.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONUtils.h"
#include "DAP.h"
#include "ExceptionBreakpoint.h"
#include "LLDBUtils.h"
#include "ProtocolUtils.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBEnvironment.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBFunction.h"
#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBQueue.h"
#include "lldb/API/SBSection.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBValue.h"
#include "lldb/Host/PosixApi.h" // IWYU pragma: keep
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lldb_dap {

void EmplaceSafeString(llvm::json::Object &obj, llvm::StringRef key,
                       llvm::StringRef str) {
  if (LLVM_LIKELY(llvm::json::isUTF8(str)))
    obj.try_emplace(key, str.str());
  else
    obj.try_emplace(key, llvm::json::fixUTF8(str));
}

llvm::StringRef GetAsString(const llvm::json::Value &value) {
  if (auto s = value.getAsString())
    return *s;
  return llvm::StringRef();
}

// Gets a string from a JSON object using the key, or returns an empty string.
std::optional<llvm::StringRef> GetString(const llvm::json::Object &obj,
                                         llvm::StringRef key) {
  return obj.getString(key);
}

std::optional<llvm::StringRef> GetString(const llvm::json::Object *obj,
                                         llvm::StringRef key) {
  if (obj == nullptr)
    return std::nullopt;

  return GetString(*obj, key);
}

std::optional<bool> GetBoolean(const llvm::json::Object &obj,
                               llvm::StringRef key) {
  if (auto value = obj.getBoolean(key))
    return *value;
  if (auto value = obj.getInteger(key))
    return *value != 0;
  return std::nullopt;
}

std::optional<bool> GetBoolean(const llvm::json::Object *obj,
                               llvm::StringRef key) {
  if (obj != nullptr)
    return GetBoolean(*obj, key);
  return std::nullopt;
}

bool ObjectContainsKey(const llvm::json::Object &obj, llvm::StringRef key) {
  return obj.find(key) != obj.end();
}

std::string EncodeMemoryReference(lldb::addr_t addr) {
  return "0x" + llvm::utohexstr(addr);
}

std::optional<lldb::addr_t>
DecodeMemoryReference(llvm::StringRef memoryReference) {
  if (!memoryReference.starts_with("0x"))
    return std::nullopt;

  lldb::addr_t addr;
  if (memoryReference.consumeInteger(0, addr))
    return std::nullopt;

  return addr;
}

bool DecodeMemoryReference(const llvm::json::Value &v, llvm::StringLiteral key,
                           lldb::addr_t &out, llvm::json::Path path,
                           bool required) {
  const llvm::json::Object *v_obj = v.getAsObject();
  if (!v_obj) {
    path.report("expected object");
    return false;
  }

  const llvm::json::Value *mem_ref_value = v_obj->get(key);
  if (!mem_ref_value) {
    if (!required)
      return true;

    path.field(key).report("missing value");
    return false;
  }

  const std::optional<llvm::StringRef> mem_ref_str =
      mem_ref_value->getAsString();
  if (!mem_ref_str) {
    path.field(key).report("expected string");
    return false;
  }

  const std::optional<lldb::addr_t> addr_opt =
      DecodeMemoryReference(*mem_ref_str);
  if (!addr_opt) {
    path.field(key).report("malformed memory reference");
    return false;
  }

  out = *addr_opt;
  return true;
}

std::vector<std::string> GetStrings(const llvm::json::Object *obj,
                                    llvm::StringRef key) {
  std::vector<std::string> strs;
  const auto *json_array = obj->getArray(key);
  if (!json_array)
    return strs;
  for (const auto &value : *json_array) {
    switch (value.kind()) {
    case llvm::json::Value::String:
      strs.push_back(value.getAsString()->str());
      break;
    case llvm::json::Value::Number:
    case llvm::json::Value::Boolean:
      strs.push_back(llvm::to_string(value));
      break;
    case llvm::json::Value::Null:
    case llvm::json::Value::Object:
    case llvm::json::Value::Array:
      break;
    }
  }
  return strs;
}

std::unordered_map<std::string, std::string>
GetStringMap(const llvm::json::Object &obj, llvm::StringRef key) {
  std::unordered_map<std::string, std::string> strs;
  const auto *const json_object = obj.getObject(key);
  if (!json_object)
    return strs;

  for (const auto &[key, value] : *json_object) {
    switch (value.kind()) {
    case llvm::json::Value::String:
      strs.emplace(key.str(), value.getAsString()->str());
      break;
    case llvm::json::Value::Number:
    case llvm::json::Value::Boolean:
      strs.emplace(key.str(), llvm::to_string(value));
      break;
    case llvm::json::Value::Null:
    case llvm::json::Value::Object:
    case llvm::json::Value::Array:
      break;
    }
  }
  return strs;
}

static bool IsClassStructOrUnionType(lldb::SBType t) {
  return (t.GetTypeClass() & (lldb::eTypeClassUnion | lldb::eTypeClassStruct |
                              lldb::eTypeClassArray)) != 0;
}

/// Create a short summary for a container that contains the summary of its
/// first children, so that the user can get a glimpse of its contents at a
/// glance.
static std::optional<std::string>
TryCreateAutoSummaryForContainer(lldb::SBValue &v) {
  if (!v.MightHaveChildren())
    return std::nullopt;
  /// As this operation can be potentially slow, we limit the total time spent
  /// fetching children to a few ms.
  const auto max_evaluation_time = std::chrono::milliseconds(10);
  /// We don't want to generate a extremely long summary string, so we limit its
  /// length.
  const size_t max_length = 32;

  auto start = std::chrono::steady_clock::now();
  std::string summary;
  llvm::raw_string_ostream os(summary);
  os << "{";

  llvm::StringRef separator = "";

  for (size_t i = 0, e = v.GetNumChildren(); i < e; ++i) {
    // If we reached the time limit or exceeded the number of characters, we
    // dump `...` to signal that there are more elements in the collection.
    if (summary.size() > max_length ||
        (std::chrono::steady_clock::now() - start) > max_evaluation_time) {
      os << separator << "...";
      break;
    }
    lldb::SBValue child = v.GetChildAtIndex(i);

    if (llvm::StringRef name = child.GetName(); !name.empty()) {
      llvm::StringRef desc;
      if (llvm::StringRef summary = child.GetSummary(); !summary.empty())
        desc = summary;
      else if (llvm::StringRef value = child.GetValue(); !value.empty())
        desc = value;
      else if (IsClassStructOrUnionType(child.GetType()))
        desc = "{...}";
      else
        continue;

      // If the child is an indexed entry, we don't show its index to save
      // characters.
      if (name.starts_with("["))
        os << separator << desc;
      else
        os << separator << name << ":" << desc;
      separator = ", ";
    }
  }
  os << "}";

  if (summary == "{...}" || summary == "{}")
    return std::nullopt;
  return summary;
}

/// Try to create a summary string for the given value that doesn't have a
/// summary of its own.
static std::optional<std::string> TryCreateAutoSummary(lldb::SBValue &value) {
  // We use the dereferenced value for generating the summary.
  if (value.GetType().IsPointerType() || value.GetType().IsReferenceType())
    value = value.Dereference();

  // We only support auto summaries for containers.
  return TryCreateAutoSummaryForContainer(value);
}

void FillResponse(const llvm::json::Object &request,
                  llvm::json::Object &response) {
  // Fill in all of the needed response fields to a "request" and set "success"
  // to true by default.
  response.try_emplace("type", "response");
  response.try_emplace("seq", (int64_t)0);
  EmplaceSafeString(response, "command",
                    GetString(request, "command").value_or(""));
  const uint64_t seq = GetInteger<uint64_t>(request, "seq").value_or(0);
  response.try_emplace("request_seq", seq);
  response.try_emplace("success", true);
}

// "Scope": {
//   "type": "object",
//   "description": "A Scope is a named container for variables. Optionally
//                   a scope can map to a source or a range within a source.",
//   "properties": {
//     "name": {
//       "type": "string",
//       "description": "Name of the scope such as 'Arguments', 'Locals'."
//     },
//     "presentationHint": {
//       "type": "string",
//       "description": "An optional hint for how to present this scope in the
//                       UI. If this attribute is missing, the scope is shown
//                       with a generic UI.",
//       "_enum": [ "arguments", "locals", "registers" ],
//     },
//     "variablesReference": {
//       "type": "integer",
//       "description": "The variables of this scope can be retrieved by
//                       passing the value of variablesReference to the
//                       VariablesRequest."
//     },
//     "namedVariables": {
//       "type": "integer",
//       "description": "The number of named variables in this scope. The
//                       client can use this optional information to present
//                       the variables in a paged UI and fetch them in chunks."
//     },
//     "indexedVariables": {
//       "type": "integer",
//       "description": "The number of indexed variables in this scope. The
//                       client can use this optional information to present
//                       the variables in a paged UI and fetch them in chunks."
//     },
//     "expensive": {
//       "type": "boolean",
//       "description": "If true, the number of variables in this scope is
//                       large or expensive to retrieve."
//     },
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "Optional source for this scope."
//     },
//     "line": {
//       "type": "integer",
//       "description": "Optional start line of the range covered by this
//                       scope."
//     },
//     "column": {
//       "type": "integer",
//       "description": "Optional start column of the range covered by this
//                       scope."
//     },
//     "endLine": {
//       "type": "integer",
//       "description": "Optional end line of the range covered by this scope."
//     },
//     "endColumn": {
//       "type": "integer",
//       "description": "Optional end column of the range covered by this
//                       scope."
//     }
//   },
//   "required": [ "name", "variablesReference", "expensive" ]
// }
llvm::json::Value CreateScope(const llvm::StringRef name,
                              int64_t variablesReference,
                              int64_t namedVariables, bool expensive) {
  llvm::json::Object object;
  EmplaceSafeString(object, "name", name.str());

  // TODO: Support "arguments" scope. At the moment lldb-dap includes the
  // arguments into the "locals" scope.
  if (variablesReference == VARREF_LOCALS) {
    object.try_emplace("presentationHint", "locals");
  } else if (variablesReference == VARREF_REGS) {
    object.try_emplace("presentationHint", "registers");
  }

  object.try_emplace("variablesReference", variablesReference);
  object.try_emplace("expensive", expensive);
  object.try_emplace("namedVariables", namedVariables);
  return llvm::json::Value(std::move(object));
}

// "Event": {
//   "allOf": [ { "$ref": "#/definitions/ProtocolMessage" }, {
//     "type": "object",
//     "description": "Server-initiated event.",
//     "properties": {
//       "type": {
//         "type": "string",
//         "enum": [ "event" ]
//       },
//       "event": {
//         "type": "string",
//         "description": "Type of event."
//       },
//       "body": {
//         "type": [ "array", "boolean", "integer", "null", "number" ,
//                   "object", "string" ],
//         "description": "Event-specific information."
//       }
//     },
//     "required": [ "type", "event" ]
//   }]
// },
// "ProtocolMessage": {
//   "type": "object",
//   "description": "Base class of requests, responses, and events.",
//   "properties": {
//         "seq": {
//           "type": "integer",
//           "description": "Sequence number."
//         },
//         "type": {
//           "type": "string",
//           "description": "Message type.",
//           "_enum": [ "request", "response", "event" ]
//         }
//   },
//   "required": [ "seq", "type" ]
// }
llvm::json::Object CreateEventObject(const llvm::StringRef event_name) {
  llvm::json::Object event;
  event.try_emplace("seq", 0);
  event.try_emplace("type", "event");
  EmplaceSafeString(event, "event", event_name);
  return event;
}

// "StackFrame": {
//   "type": "object",
//   "description": "A Stackframe contains the source location.",
//   "properties": {
//     "id": {
//       "type": "integer",
//       "description": "An identifier for the stack frame. It must be unique
//                       across all threads. This id can be used to retrieve
//                       the scopes of the frame with the 'scopesRequest' or
//                       to restart the execution of a stackframe."
//     },
//     "name": {
//       "type": "string",
//       "description": "The name of the stack frame, typically a method name."
//     },
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "The optional source of the frame."
//     },
//     "line": {
//       "type": "integer",
//       "description": "The line within the file of the frame. If source is
//                       null or doesn't exist, line is 0 and must be ignored."
//     },
//     "column": {
//       "type": "integer",
//       "description": "The column within the line. If source is null or
//                       doesn't exist, column is 0 and must be ignored."
//     },
//     "endLine": {
//       "type": "integer",
//       "description": "An optional end line of the range covered by the
//                       stack frame."
//     },
//     "endColumn": {
//       "type": "integer",
//       "description": "An optional end column of the range covered by the
//                       stack frame."
//     },
//     "instructionPointerReference": {
// 	     "type": "string",
// 	     "description": "A memory reference for the current instruction
//                         pointer in this frame."
//     },
//     "moduleId": {
//       "type": ["integer", "string"],
//       "description": "The module associated with this frame, if any."
//     },
//     "presentationHint": {
//       "type": "string",
//       "enum": [ "normal", "label", "subtle" ],
//       "description": "An optional hint for how to present this frame in
//                       the UI. A value of 'label' can be used to indicate
//                       that the frame is an artificial frame that is used
//                       as a visual label or separator. A value of 'subtle'
//                       can be used to change the appearance of a frame in
//                       a 'subtle' way."
//     }
//   },
//   "required": [ "id", "name", "line", "column" ]
// }
llvm::json::Value CreateStackFrame(DAP &dap, lldb::SBFrame &frame,
                                   lldb::SBFormat &format) {
  llvm::json::Object object;
  int64_t frame_id = MakeDAPFrameID(frame);
  object.try_emplace("id", frame_id);

  std::string frame_name;
  lldb::SBStream stream;
  if (format && frame.GetDescriptionWithFormat(format, stream).Success()) {
    frame_name = stream.GetData();

    // `function_name` can be a nullptr, which throws an error when assigned to
    // an `std::string`.
  } else if (const char *name = frame.GetDisplayFunctionName()) {
    frame_name = name;
  }

  if (frame_name.empty()) {
    // If the function name is unavailable, display the pc address as a 16-digit
    // hex string, e.g. "0x0000000000012345"
    frame_name = GetLoadAddressString(frame.GetPC());
  }

  // We only include `[opt]` if a custom frame format is not specified.
  if (!format && frame.GetFunction().GetIsOptimized())
    frame_name += " [opt]";

  EmplaceSafeString(object, "name", frame_name);

  std::optional<protocol::Source> source = dap.ResolveSource(frame);

  if (source && !IsAssemblySource(*source)) {
    // This is a normal source with a valid line entry.
    auto line_entry = frame.GetLineEntry();
    object.try_emplace("line", line_entry.GetLine());
    auto column = line_entry.GetColumn();
    object.try_emplace("column", column);
  } else if (frame.GetSymbol().IsValid()) {
    // This is a source where the disassembly is used, but there is a valid
    // symbol. Calculate the line of the current PC from the start of the
    // current symbol.
    lldb::SBInstructionList inst_list = dap.target.ReadInstructions(
        frame.GetSymbol().GetStartAddress(), frame.GetPCAddress(), nullptr);
    size_t inst_line = inst_list.GetSize();

    // Line numbers are 1-based.
    object.try_emplace("line", inst_line + 1);
    object.try_emplace("column", 1);
  } else {
    // No valid line entry or symbol.
    object.try_emplace("line", 1);
    object.try_emplace("column", 1);
  }

  if (source)
    object.try_emplace("source", std::move(source).value());

  const auto pc = frame.GetPC();
  if (pc != LLDB_INVALID_ADDRESS) {
    std::string formatted_addr = "0x" + llvm::utohexstr(pc);
    object.try_emplace("instructionPointerReference", formatted_addr);
  }

  if (frame.IsArtificial() || frame.IsHidden())
    object.try_emplace("presentationHint", "subtle");

  lldb::SBModule module = frame.GetModule();
  if (module.IsValid()) {
    std::string uuid = module.GetUUIDString();
    if (!uuid.empty())
      object.try_emplace("moduleId", uuid);
  }

  return llvm::json::Value(std::move(object));
}

llvm::json::Value CreateExtendedStackFrameLabel(lldb::SBThread &thread,
                                                lldb::SBFormat &format) {
  std::string name;
  lldb::SBStream stream;
  if (format && thread.GetDescriptionWithFormat(format, stream).Success()) {
    name = stream.GetData();
  } else {
    const uint32_t thread_idx = thread.GetExtendedBacktraceOriginatingIndexID();
    const char *queue_name = thread.GetQueueName();
    if (queue_name != nullptr) {
      name = llvm::formatv("Enqueued from {0} (Thread {1})", queue_name,
                           thread_idx);
    } else {
      name = llvm::formatv("Thread {0}", thread_idx);
    }
  }

  return llvm::json::Value(llvm::json::Object{{"id", thread.GetThreadID() + 1},
                                              {"name", name},
                                              {"presentationHint", "label"}});
}

// "StoppedEvent": {
//   "allOf": [ { "$ref": "#/definitions/Event" }, {
//     "type": "object",
//     "description": "Event message for 'stopped' event type. The event
//                     indicates that the execution of the debuggee has stopped
//                     due to some condition. This can be caused by a break
//                     point previously set, a stepping action has completed,
//                     by executing a debugger statement etc.",
//     "properties": {
//       "event": {
//         "type": "string",
//         "enum": [ "stopped" ]
//       },
//       "body": {
//         "type": "object",
//         "properties": {
//           "reason": {
//             "type": "string",
//             "description": "The reason for the event. For backward
//                             compatibility this string is shown in the UI if
//                             the 'description' attribute is missing (but it
//                             must not be translated).",
//             "_enum": [ "step", "breakpoint", "exception", "pause", "entry" ]
//           },
//           "description": {
//             "type": "string",
//             "description": "The full reason for the event, e.g. 'Paused
//                             on exception'. This string is shown in the UI
//                             as is."
//           },
//           "threadId": {
//             "type": "integer",
//             "description": "The thread which was stopped."
//           },
//           "text": {
//             "type": "string",
//             "description": "Additional information. E.g. if reason is
//                             'exception', text contains the exception name.
//                             This string is shown in the UI."
//           },
//           "allThreadsStopped": {
//             "type": "boolean",
//             "description": "If allThreadsStopped is true, a debug adapter
//                             can announce that all threads have stopped.
//                             The client should use this information to
//                             enable that all threads can be expanded to
//                             access their stacktraces. If the attribute
//                             is missing or false, only the thread with the
//                             given threadId can be expanded."
//           }
//         },
//         "required": [ "reason" ]
//       }
//     },
//     "required": [ "event", "body" ]
//   }]
// }
llvm::json::Value CreateThreadStopped(DAP &dap, lldb::SBThread &thread,
                                      uint32_t stop_id) {
  llvm::json::Object event(CreateEventObject("stopped"));
  llvm::json::Object body;
  switch (thread.GetStopReason()) {
  case lldb::eStopReasonTrace:
  case lldb::eStopReasonPlanComplete:
    body.try_emplace("reason", "step");
    break;
  case lldb::eStopReasonBreakpoint: {
    ExceptionBreakpoint *exc_bp = dap.GetExceptionBPFromStopReason(thread);
    if (exc_bp) {
      body.try_emplace("reason", "exception");
      EmplaceSafeString(body, "description", exc_bp->GetLabel());
    } else {
      InstructionBreakpoint *inst_bp =
          dap.GetInstructionBPFromStopReason(thread);
      if (inst_bp) {
        body.try_emplace("reason", "instruction breakpoint");
      } else {
        body.try_emplace("reason", "breakpoint");
      }
      std::vector<lldb::break_id_t> bp_ids;
      std::ostringstream desc_sstream;
      desc_sstream << "breakpoint";
      for (size_t idx = 0; idx < thread.GetStopReasonDataCount(); idx += 2) {
        lldb::break_id_t bp_id = thread.GetStopReasonDataAtIndex(idx);
        lldb::break_id_t bp_loc_id = thread.GetStopReasonDataAtIndex(idx + 1);
        bp_ids.push_back(bp_id);
        desc_sstream << " " << bp_id << "." << bp_loc_id;
      }
      std::string desc_str = desc_sstream.str();
      body.try_emplace("hitBreakpointIds", llvm::json::Array(bp_ids));
      EmplaceSafeString(body, "description", desc_str);
    }
  } break;
  case lldb::eStopReasonWatchpoint:
  case lldb::eStopReasonInstrumentation:
    body.try_emplace("reason", "breakpoint");
    break;
  case lldb::eStopReasonProcessorTrace:
    body.try_emplace("reason", "processor trace");
    break;
  case lldb::eStopReasonHistoryBoundary:
    body.try_emplace("reason", "history boundary");
    break;
  case lldb::eStopReasonSignal:
  case lldb::eStopReasonException:
    body.try_emplace("reason", "exception");
    break;
  case lldb::eStopReasonExec:
    body.try_emplace("reason", "entry");
    break;
  case lldb::eStopReasonFork:
    body.try_emplace("reason", "fork");
    break;
  case lldb::eStopReasonVFork:
    body.try_emplace("reason", "vfork");
    break;
  case lldb::eStopReasonVForkDone:
    body.try_emplace("reason", "vforkdone");
    break;
  case lldb::eStopReasonInterrupt:
    body.try_emplace("reason", "async interrupt");
    break;
  case lldb::eStopReasonThreadExiting:
  case lldb::eStopReasonInvalid:
  case lldb::eStopReasonNone:
    break;
  }
  if (stop_id == 0)
    body.try_emplace("reason", "entry");
  const lldb::tid_t tid = thread.GetThreadID();
  body.try_emplace("threadId", (int64_t)tid);
  // If no description has been set, then set it to the default thread stopped
  // description. If we have breakpoints that get hit and shouldn't be reported
  // as breakpoints, then they will set the description above.
  if (!ObjectContainsKey(body, "description")) {
    char description[1024];
    if (thread.GetStopDescription(description, sizeof(description))) {
      EmplaceSafeString(body, "description", description);
    }
  }
  // "threadCausedFocus" is used in tests to validate breaking behavior.
  if (tid == dap.focus_tid) {
    body.try_emplace("threadCausedFocus", true);
  }
  body.try_emplace("preserveFocusHint", tid != dap.focus_tid);
  body.try_emplace("allThreadsStopped", true);
  event.try_emplace("body", std::move(body));
  return llvm::json::Value(std::move(event));
}

const char *GetNonNullVariableName(lldb::SBValue &v) {
  const char *name = v.GetName();
  return name ? name : "<null>";
}

std::string CreateUniqueVariableNameForDisplay(lldb::SBValue &v,
                                               bool is_name_duplicated) {
  lldb::SBStream name_builder;
  name_builder.Print(GetNonNullVariableName(v));
  if (is_name_duplicated) {
    lldb::SBDeclaration declaration = v.GetDeclaration();
    const char *file_name = declaration.GetFileSpec().GetFilename();
    const uint32_t line = declaration.GetLine();

    if (file_name != nullptr && line > 0)
      name_builder.Printf(" @ %s:%u", file_name, line);
    else if (const char *location = v.GetLocation())
      name_builder.Printf(" @ %s", location);
  }
  return name_builder.GetData();
}

VariableDescription::VariableDescription(lldb::SBValue v,
                                         bool auto_variable_summaries,
                                         bool format_hex,
                                         bool is_name_duplicated,
                                         std::optional<std::string> custom_name)
    : v(v) {
  name = custom_name
             ? *custom_name
             : CreateUniqueVariableNameForDisplay(v, is_name_duplicated);

  type_obj = v.GetType();
  std::string raw_display_type_name =
      llvm::StringRef(type_obj.GetDisplayTypeName()).str();
  display_type_name =
      !raw_display_type_name.empty() ? raw_display_type_name : NO_TYPENAME;

  // Only format hex/default if there is no existing special format.
  if (v.GetFormat() == lldb::eFormatDefault ||
      v.GetFormat() == lldb::eFormatHex) {
    if (format_hex)
      v.SetFormat(lldb::eFormatHex);
    else
      v.SetFormat(lldb::eFormatDefault);
  }

  llvm::raw_string_ostream os_display_value(display_value);

  if (lldb::SBError sb_error = v.GetError(); sb_error.Fail()) {
    error = sb_error.GetCString();
    os_display_value << "<error: " << error << ">";
  } else {
    value = llvm::StringRef(v.GetValue()).str();
    summary = llvm::StringRef(v.GetSummary()).str();
    if (summary.empty() && auto_variable_summaries)
      auto_summary = TryCreateAutoSummary(v);

    std::optional<std::string> effective_summary =
        !summary.empty() ? summary : auto_summary;

    if (!value.empty()) {
      os_display_value << value;
      if (effective_summary)
        os_display_value << " " << *effective_summary;
    } else if (effective_summary) {
      os_display_value << *effective_summary;

      // As last resort, we print its type and address if available.
    } else {
      if (!raw_display_type_name.empty()) {
        os_display_value << raw_display_type_name;
        lldb::addr_t address = v.GetLoadAddress();
        if (address != LLDB_INVALID_ADDRESS)
          os_display_value << " @ " << llvm::format_hex(address, 0);
      }
    }
  }

  lldb::SBStream evaluateStream;
  v.GetExpressionPath(evaluateStream);
  evaluate_name = llvm::StringRef(evaluateStream.GetData()).str();
}

std::string VariableDescription::GetResult(llvm::StringRef context) {
  // In repl context, the results can be displayed as multiple lines so more
  // detailed descriptions can be returned.
  if (context != "repl")
    return display_value;

  if (!v.IsValid())
    return display_value;

  // Try the SBValue::GetDescription(), which may call into language runtime
  // specific formatters (see ValueObjectPrinter).
  lldb::SBStream stream;
  v.GetDescription(stream);
  llvm::StringRef description = stream.GetData();
  return description.trim().str();
}

bool ValuePointsToCode(lldb::SBValue v) {
  if (!v.GetType().GetPointeeType().IsFunctionType())
    return false;

  lldb::addr_t addr = v.GetValueAsAddress();
  lldb::SBLineEntry line_entry =
      v.GetTarget().ResolveLoadAddress(addr).GetLineEntry();

  return line_entry.IsValid();
}

int64_t PackLocation(int64_t var_ref, bool is_value_location) {
  return var_ref << 1 | is_value_location;
}

std::pair<int64_t, bool> UnpackLocation(int64_t location_id) {
  return std::pair{location_id >> 1, location_id & 1};
}

llvm::json::Value CreateCompileUnit(lldb::SBCompileUnit &unit) {
  llvm::json::Object object;
  char unit_path_arr[PATH_MAX];
  unit.GetFileSpec().GetPath(unit_path_arr, sizeof(unit_path_arr));
  std::string unit_path(unit_path_arr);
  object.try_emplace("compileUnitPath", unit_path);
  return llvm::json::Value(std::move(object));
}

/// See
/// https://microsoft.github.io/debug-adapter-protocol/specification#Reverse_Requests_RunInTerminal
llvm::json::Object CreateRunInTerminalReverseRequest(
    llvm::StringRef program, const std::vector<std::string> &args,
    const llvm::StringMap<std::string> &env, llvm::StringRef cwd,
    llvm::StringRef comm_file, lldb::pid_t debugger_pid,
    const std::vector<std::optional<std::string>> &stdio, bool external) {
  llvm::json::Object run_in_terminal_args;
  if (external) {
    // This indicates the IDE to open an external terminal window.
    run_in_terminal_args.try_emplace("kind", "external");
  } else {
    // This indicates the IDE to open an embedded terminal, instead of opening
    // the terminal in a new window.
    run_in_terminal_args.try_emplace("kind", "integrated");
  }
  // The program path must be the first entry in the "args" field
  std::vector<std::string> req_args = {DAP::debug_adapter_path.str(),
                                       "--comm-file", comm_file.str()};
  if (debugger_pid != LLDB_INVALID_PROCESS_ID) {
    req_args.push_back("--debugger-pid");
    req_args.push_back(std::to_string(debugger_pid));
  }
  req_args.push_back("--launch-target");
  req_args.push_back(program.str());
  if (!stdio.empty()) {
    req_args.push_back("--stdio");
    std::stringstream ss;
    for (const std::optional<std::string> &file : stdio) {
      if (file)
        ss << *file;
      ss << ":";
    }
    std::string files = ss.str();
    files.pop_back();
    req_args.push_back(std::move(files));
  }
  req_args.insert(req_args.end(), args.begin(), args.end());
  run_in_terminal_args.try_emplace("args", req_args);

  if (!cwd.empty())
    run_in_terminal_args.try_emplace("cwd", cwd);

  if (!env.empty()) {
    llvm::json::Object env_json;
    for (const auto &kv : env) {
      if (!kv.first().empty())
        env_json.try_emplace(kv.first(), kv.second);
    }
    run_in_terminal_args.try_emplace("env",
                                     llvm::json::Value(std::move(env_json)));
  }

  return run_in_terminal_args;
}

// Keep all the top level items from the statistics dump, except for the
// "modules" array. It can be huge and cause delay
// Array and dictionary value will return as <key, JSON string> pairs
static void FilterAndGetValueForKey(const lldb::SBStructuredData data,
                                    const char *key, llvm::json::Object &out) {
  lldb::SBStructuredData value = data.GetValueForKey(key);
  std::string key_utf8 = llvm::json::fixUTF8(key);
  if (llvm::StringRef(key) == "modules")
    return;
  switch (value.GetType()) {
  case lldb::eStructuredDataTypeFloat:
    out.try_emplace(key_utf8, value.GetFloatValue());
    break;
  case lldb::eStructuredDataTypeUnsignedInteger:
    out.try_emplace(key_utf8, value.GetIntegerValue((uint64_t)0));
    break;
  case lldb::eStructuredDataTypeSignedInteger:
    out.try_emplace(key_utf8, value.GetIntegerValue((int64_t)0));
    break;
  case lldb::eStructuredDataTypeArray: {
    lldb::SBStream contents;
    value.GetAsJSON(contents);
    out.try_emplace(key_utf8, llvm::json::fixUTF8(contents.GetData()));
  } break;
  case lldb::eStructuredDataTypeBoolean:
    out.try_emplace(key_utf8, value.GetBooleanValue());
    break;
  case lldb::eStructuredDataTypeString: {
    // Get the string size before reading
    const size_t str_length = value.GetStringValue(nullptr, 0);
    std::string str(str_length + 1, 0);
    value.GetStringValue(&str[0], str_length);
    out.try_emplace(key_utf8, llvm::json::fixUTF8(str));
  } break;
  case lldb::eStructuredDataTypeDictionary: {
    lldb::SBStream contents;
    value.GetAsJSON(contents);
    out.try_emplace(key_utf8, llvm::json::fixUTF8(contents.GetData()));
  } break;
  case lldb::eStructuredDataTypeNull:
  case lldb::eStructuredDataTypeGeneric:
  case lldb::eStructuredDataTypeInvalid:
    break;
  }
}

static void addStatistic(lldb::SBTarget &target, llvm::json::Object &event) {
  lldb::SBStructuredData statistics = target.GetStatistics();
  bool is_dictionary =
      statistics.GetType() == lldb::eStructuredDataTypeDictionary;
  if (!is_dictionary)
    return;
  llvm::json::Object stats_body;

  lldb::SBStringList keys;
  if (!statistics.GetKeys(keys))
    return;
  for (size_t i = 0; i < keys.GetSize(); i++) {
    const char *key = keys.GetStringAtIndex(i);
    FilterAndGetValueForKey(statistics, key, stats_body);
  }
  llvm::json::Object body{{"$__lldb_statistics", std::move(stats_body)}};
  event.try_emplace("body", std::move(body));
}

llvm::json::Object CreateTerminatedEventObject(lldb::SBTarget &target) {
  llvm::json::Object event(CreateEventObject("terminated"));
  addStatistic(target, event);
  return event;
}

std::string JSONToString(const llvm::json::Value &json) {
  std::string data;
  llvm::raw_string_ostream os(data);
  os << json;
  return data;
}

} // namespace lldb_dap
