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

static uint64_t GetDebugInfoSizeInSection(lldb::SBSection section) {
  uint64_t debug_info_size = 0;
  llvm::StringRef section_name(section.GetName());
  if (section_name.starts_with(".debug") ||
      section_name.starts_with("__debug") ||
      section_name.starts_with(".apple") || section_name.starts_with("__apple"))
    debug_info_size += section.GetFileByteSize();
  size_t num_sub_sections = section.GetNumSubSections();
  for (size_t i = 0; i < num_sub_sections; i++) {
    debug_info_size +=
        GetDebugInfoSizeInSection(section.GetSubSectionAtIndex(i));
  }
  return debug_info_size;
}

static uint64_t GetDebugInfoSize(lldb::SBModule module) {
  uint64_t debug_info_size = 0;
  size_t num_sections = module.GetNumSections();
  for (size_t i = 0; i < num_sections; i++) {
    debug_info_size += GetDebugInfoSizeInSection(module.GetSectionAtIndex(i));
  }
  return debug_info_size;
}

static std::string ConvertDebugInfoSizeToString(uint64_t debug_info) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);
  if (debug_info < 1024) {
    oss << debug_info << "B";
  } else if (debug_info < 1024 * 1024) {
    double kb = double(debug_info) / 1024.0;
    oss << kb << "KB";
  } else if (debug_info < 1024 * 1024 * 1024) {
    double mb = double(debug_info) / (1024.0 * 1024.0);
    oss << mb << "MB";
  } else {
    double gb = double(debug_info) / (1024.0 * 1024.0 * 1024.0);
    oss << gb << "GB";
  }
  return oss.str();
}

llvm::json::Value CreateModule(lldb::SBTarget &target, lldb::SBModule &module,
                               bool id_only) {
  llvm::json::Object object;
  if (!target.IsValid() || !module.IsValid())
    return llvm::json::Value(std::move(object));

  const char *uuid = module.GetUUIDString();
  object.try_emplace("id", uuid ? std::string(uuid) : std::string(""));

  if (id_only)
    return llvm::json::Value(std::move(object));

  object.try_emplace("name", std::string(module.GetFileSpec().GetFilename()));
  char module_path_arr[PATH_MAX];
  module.GetFileSpec().GetPath(module_path_arr, sizeof(module_path_arr));
  std::string module_path(module_path_arr);
  object.try_emplace("path", module_path);
  if (module.GetNumCompileUnits() > 0) {
    std::string symbol_str = "Symbols loaded.";
    std::string debug_info_size;
    uint64_t debug_info = GetDebugInfoSize(module);
    if (debug_info > 0) {
      debug_info_size = ConvertDebugInfoSizeToString(debug_info);
    }
    object.try_emplace("symbolStatus", symbol_str);
    object.try_emplace("debugInfoSize", debug_info_size);
    char symbol_path_arr[PATH_MAX];
    module.GetSymbolFileSpec().GetPath(symbol_path_arr,
                                       sizeof(symbol_path_arr));
    std::string symbol_path(symbol_path_arr);
    object.try_emplace("symbolFilePath", symbol_path);
  } else {
    object.try_emplace("symbolStatus", "Symbols not found.");
  }
  std::string load_address =
      llvm::formatv("{0:x}",
                    module.GetObjectFileHeaderAddress().GetLoadAddress(target))
          .str();
  object.try_emplace("addressRange", load_address);
  std::string version_str;
  uint32_t version_nums[3];
  uint32_t num_versions =
      module.GetVersion(version_nums, sizeof(version_nums) / sizeof(uint32_t));
  for (uint32_t i = 0; i < num_versions; ++i) {
    if (!version_str.empty())
      version_str += ".";
    version_str += std::to_string(version_nums[i]);
  }
  if (!version_str.empty())
    object.try_emplace("version", version_str);
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

protocol::ExceptionBreakpointsFilter
CreateExceptionBreakpointFilter(const ExceptionBreakpoint &bp) {
  protocol::ExceptionBreakpointsFilter filter;
  filter.filter = bp.GetFilter();
  filter.label = bp.GetLabel();
  filter.defaultState = ExceptionBreakpoint::kDefaultValue;
  return filter;
}

static std::string GetLoadAddressString(const lldb::addr_t addr) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << llvm::format_hex(addr, 18);
  return result;
}

protocol::Source CreateSource(const lldb::SBFileSpec &file) {
  protocol::Source source;
  if (file.IsValid()) {
    const char *name = file.GetFilename();
    if (name)
      source.name = name;
    char path[PATH_MAX] = "";
    if (file.GetPath(path, sizeof(path)) &&
        lldb::SBFileSpec::ResolvePath(path, path, PATH_MAX))
      source.path = path;
  }
  return source;
}

protocol::Source CreateSource(const lldb::SBLineEntry &line_entry) {
  return CreateSource(line_entry.GetFileSpec());
}

protocol::Source CreateSource(llvm::StringRef source_path) {
  protocol::Source source;
  llvm::StringRef name = llvm::sys::path::filename(source_path);
  source.name = name;
  source.path = source_path;
  return source;
}

protocol::Source CreateAssemblySource(const lldb::SBTarget &target,
                                      lldb::SBAddress &address) {
  protocol::Source source;

  auto symbol = address.GetSymbol();
  std::string name;
  if (symbol.IsValid()) {
    source.sourceReference = symbol.GetStartAddress().GetLoadAddress(target);
    name = symbol.GetName();
  } else {
    const auto load_addr = address.GetLoadAddress(target);
    source.sourceReference = load_addr;
    name = GetLoadAddressString(load_addr);
  }

  lldb::SBModule module = address.GetModule();
  if (module.IsValid()) {
    lldb::SBFileSpec file_spec = module.GetFileSpec();
    if (file_spec.IsValid()) {
      std::string path = GetSBFileSpecPath(file_spec);
      if (!path.empty())
        source.path = path + '`' + name;
    }
  }

  source.name = std::move(name);

  // Mark the source as deemphasized since users will only be able to view
  // assembly for these frames.
  source.presentationHint =
      protocol::Source::PresentationHint::eSourcePresentationHintDeemphasize;

  return source;
}

bool ShouldDisplayAssemblySource(
    const lldb::SBLineEntry &line_entry,
    lldb::StopDisassemblyType stop_disassembly_display) {
  if (stop_disassembly_display == lldb::eStopDisassemblyTypeNever)
    return false;

  if (stop_disassembly_display == lldb::eStopDisassemblyTypeAlways)
    return true;

  // A line entry of 0 indicates the line is compiler generated i.e. no source
  // file is associated with the frame.
  auto file_spec = line_entry.GetFileSpec();
  if (!file_spec.IsValid() || line_entry.GetLine() == 0 ||
      line_entry.GetLine() == LLDB_INVALID_LINE_NUMBER)
    return true;

  if (stop_disassembly_display == lldb::eStopDisassemblyTypeNoSource &&
      !file_spec.Exists()) {
    return true;
  }

  return false;
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
llvm::json::Value
CreateStackFrame(lldb::SBFrame &frame, lldb::SBFormat &format,
                 lldb::StopDisassemblyType stop_disassembly_display) {
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

  auto line_entry = frame.GetLineEntry();
  if (!ShouldDisplayAssemblySource(line_entry, stop_disassembly_display)) {
    object.try_emplace("source", CreateSource(line_entry));
    object.try_emplace("line", line_entry.GetLine());
    auto column = line_entry.GetColumn();
    object.try_emplace("column", column);
  } else if (frame.GetSymbol().IsValid()) {
    // If no source is associated with the frame, use the DAPFrameID to track
    // the 'source' and generate assembly.
    auto frame_address = frame.GetPCAddress();
    object.try_emplace("source", CreateAssemblySource(
                                     frame.GetThread().GetProcess().GetTarget(),
                                     frame_address));

    // Calculate the line of the current PC from the start of the current
    // symbol.
    lldb::SBTarget target = frame.GetThread().GetProcess().GetTarget();
    lldb::SBInstructionList inst_list = target.ReadInstructions(
        frame.GetSymbol().GetStartAddress(), frame.GetPCAddress(), nullptr);
    size_t inst_line = inst_list.GetSize();

    // Line numbers are 1-based.
    object.try_emplace("line", inst_line + 1);
    object.try_emplace("column", 1);
  } else {
    // No valid line entry or symbol.
    auto frame_address = frame.GetPCAddress();
    object.try_emplace("source", CreateAssemblySource(
                                     frame.GetThread().GetProcess().GetTarget(),
                                     frame_address));
    object.try_emplace("line", 1);
    object.try_emplace("column", 1);
  }

  const auto pc = frame.GetPC();
  if (pc != LLDB_INVALID_ADDRESS) {
    std::string formatted_addr = "0x" + llvm::utohexstr(pc);
    object.try_emplace("instructionPointerReference", formatted_addr);
  }

  if (frame.IsArtificial() || frame.IsHidden())
    object.try_emplace("presentationHint", "subtle");

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

// "Thread": {
//   "type": "object",
//   "description": "A Thread",
//   "properties": {
//     "id": {
//       "type": "integer",
//       "description": "Unique identifier for the thread."
//     },
//     "name": {
//       "type": "string",
//       "description": "A name of the thread."
//     }
//   },
//   "required": [ "id", "name" ]
// }
llvm::json::Value CreateThread(lldb::SBThread &thread, lldb::SBFormat &format) {
  llvm::json::Object object;
  object.try_emplace("id", (int64_t)thread.GetThreadID());
  std::string thread_str;
  lldb::SBStream stream;
  if (format && thread.GetDescriptionWithFormat(format, stream).Success()) {
    thread_str = stream.GetData();
  } else {
    const char *thread_name = thread.GetName();
    const char *queue_name = thread.GetQueueName();

    if (thread_name) {
      thread_str = std::string(thread_name);
    } else if (queue_name) {
      auto kind = thread.GetQueue().GetKind();
      std::string queue_kind_label = "";
      if (kind == lldb::eQueueKindSerial) {
        queue_kind_label = " (serial)";
      } else if (kind == lldb::eQueueKindConcurrent) {
        queue_kind_label = " (concurrent)";
      }

      thread_str =
          llvm::formatv("Thread {0} Queue: {1}{2}", thread.GetIndexID(),
                        queue_name, queue_kind_label)
              .str();
    } else {
      thread_str = llvm::formatv("Thread {0}", thread.GetIndexID()).str();
    }
  }

  EmplaceSafeString(object, "name", thread_str);

  return llvm::json::Value(std::move(object));
}

llvm::json::Array GetThreads(lldb::SBProcess process, lldb::SBFormat &format) {
  lldb::SBMutex lock = process.GetTarget().GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  llvm::json::Array threads;
  const uint32_t num_threads = process.GetNumThreads();
  for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
    threads.emplace_back(CreateThread(thread, format));
  }
  return threads;
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
      lldb::break_id_t bp_id = thread.GetStopReasonDataAtIndex(0);
      lldb::break_id_t bp_loc_id = thread.GetStopReasonDataAtIndex(1);
      std::string desc_str =
          llvm::formatv("breakpoint {0}.{1}", bp_id, bp_loc_id);
      body.try_emplace("hitBreakpointIds",
                       llvm::json::Array{llvm::json::Value(bp_id)});
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

llvm::json::Object VariableDescription::GetVariableExtensionsJSON() {
  llvm::json::Object extensions;
  if (error)
    EmplaceSafeString(extensions, "error", *error);
  if (!value.empty())
    EmplaceSafeString(extensions, "value", value);
  if (!summary.empty())
    EmplaceSafeString(extensions, "summary", summary);
  if (auto_summary)
    EmplaceSafeString(extensions, "autoSummary", *auto_summary);

  if (lldb::SBDeclaration decl = v.GetDeclaration(); decl.IsValid()) {
    llvm::json::Object decl_obj;
    if (lldb::SBFileSpec file = decl.GetFileSpec(); file.IsValid()) {
      char path[PATH_MAX] = "";
      if (file.GetPath(path, sizeof(path)) &&
          lldb::SBFileSpec::ResolvePath(path, path, PATH_MAX)) {
        decl_obj.try_emplace("path", std::string(path));
      }
    }

    if (int line = decl.GetLine())
      decl_obj.try_emplace("line", line);
    if (int column = decl.GetColumn())
      decl_obj.try_emplace("column", column);

    if (!decl_obj.empty())
      extensions.try_emplace("declaration", std::move(decl_obj));
  }
  return extensions;
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

// "Variable": {
//   "type": "object",
//   "description": "A Variable is a name/value pair. Optionally a variable
//                   can have a 'type' that is shown if space permits or when
//                   hovering over the variable's name. An optional 'kind' is
//                   used to render additional properties of the variable,
//                   e.g. different icons can be used to indicate that a
//                   variable is public or private. If the value is
//                   structured (has children), a handle is provided to
//                   retrieve the children with the VariablesRequest. If
//                   the number of named or indexed children is large, the
//                   numbers should be returned via the optional
//                   'namedVariables' and 'indexedVariables' attributes. The
//                   client can use this optional information to present the
//                   children in a paged UI and fetch them in chunks.",
//   "properties": {
//     "name": {
//       "type": "string",
//       "description": "The variable's name."
//     },
//     "value": {
//       "type": "string",
//       "description": "The variable's value. This can be a multi-line text,
//                       e.g. for a function the body of a function."
//     },
//     "type": {
//       "type": "string",
//       "description": "The type of the variable's value. Typically shown in
//                       the UI when hovering over the value."
//     },
//     "presentationHint": {
//       "$ref": "#/definitions/VariablePresentationHint",
//       "description": "Properties of a variable that can be used to determine
//                       how to render the variable in the UI."
//     },
//     "evaluateName": {
//       "type": "string",
//       "description": "Optional evaluatable name of this variable which can
//                       be passed to the 'EvaluateRequest' to fetch the
//                       variable's value."
//     },
//     "variablesReference": {
//       "type": "integer",
//       "description": "If variablesReference is > 0, the variable is
//                       structured and its children can be retrieved by
//                       passing variablesReference to the VariablesRequest."
//     },
//     "namedVariables": {
//       "type": "integer",
//       "description": "The number of named child variables. The client can
//                       use this optional information to present the children
//                       in a paged UI and fetch them in chunks."
//     },
//     "indexedVariables": {
//       "type": "integer",
//       "description": "The number of indexed child variables. The client
//                       can use this optional information to present the
//                       children in a paged UI and fetch them in chunks."
//     },
//     "memoryReference": {
//        "type": "string",
//        "description": "A memory reference associated with this variable.
//                        For pointer type variables, this is generally a
//                        reference to the memory address contained in the
//                        pointer. For executable data, this reference may later
//                        be used in a `disassemble` request. This attribute may
//                        be returned by a debug adapter if corresponding
//                        capability `supportsMemoryReferences` is true."
//     },
//     "declarationLocationReference": {
//       "type": "integer",
//       "description": "A reference that allows the client to request the
//                       location where the variable is declared. This should be
//                       present only if the adapter is likely to be able to
//                       resolve the location.\n\nThis reference shares the same
//                       lifetime as the `variablesReference`. See 'Lifetime of
//                       Object References' in the Overview section for
//                       details."
//     },
//     "valueLocationReference": {
//       "type": "integer",
//       "description": "A reference that allows the client to request the
//                       location where the variable's value is declared. For
//                       example, if the variable contains a function pointer,
//                       the adapter may be able to look up the function's
//                       location. This should be present only if the adapter
//                       is likely to be able to resolve the location.\n\nThis
//                       reference shares the same lifetime as the
//                       `variablesReference`. See 'Lifetime of Object
//                       References' in the Overview section for details."
//     },
//
//     "$__lldb_extensions": {
//       "description": "Unofficial extensions to the protocol",
//       "properties": {
//         "declaration": {
//           "type": "object",
//           "description": "The source location where the variable was
//                           declared. This value won't be present if no
//                           declaration is available.
//                           Superseded by `declarationLocationReference`",
//           "properties": {
//             "path": {
//               "type": "string",
//               "description": "The source file path where the variable was
//                              declared."
//             },
//             "line": {
//               "type": "number",
//               "description": "The 1-indexed source line where the variable
//                               was declared."
//             },
//             "column": {
//               "type": "number",
//               "description": "The 1-indexed source column where the variable
//                               was declared."
//             }
//           }
//         },
//         "value": {
//           "type": "string",
//           "description": "The internal value of the variable as returned by
//                            This is effectively SBValue.GetValue(). The other
//                            `value` entry in the top-level variable response
//                            is, on the other hand, just a display string for
//                            the variable."
//         },
//         "summary": {
//           "type": "string",
//           "description": "The summary string of the variable. This is
//                           effectively SBValue.GetSummary()."
//         },
//         "autoSummary": {
//           "type": "string",
//           "description": "The auto generated summary if using
//                           `enableAutoVariableSummaries`."
//         },
//         "error": {
//           "type": "string",
//           "description": "An error message generated if LLDB couldn't inspect
//                           the variable."
//         }
//       }
//     }
//   },
//   "required": [ "name", "value", "variablesReference" ]
// }
llvm::json::Value CreateVariable(lldb::SBValue v, int64_t var_ref,
                                 bool format_hex, bool auto_variable_summaries,
                                 bool synthetic_child_debugging,
                                 bool is_name_duplicated,
                                 std::optional<std::string> custom_name) {
  VariableDescription desc(v, auto_variable_summaries, format_hex,
                           is_name_duplicated, custom_name);
  llvm::json::Object object;
  EmplaceSafeString(object, "name", desc.name);
  EmplaceSafeString(object, "value", desc.display_value);

  if (!desc.evaluate_name.empty())
    EmplaceSafeString(object, "evaluateName", desc.evaluate_name);

  // If we have a type with many children, we would like to be able to
  // give a hint to the IDE that the type has indexed children so that the
  // request can be broken up in grabbing only a few children at a time. We
  // want to be careful and only call "v.GetNumChildren()" if we have an array
  // type or if we have a synthetic child provider producing indexed children.
  // We don't want to call "v.GetNumChildren()" on all objects as class, struct
  // and union types don't need to be completed if they are never expanded. So
  // we want to avoid calling this to only cases where we it makes sense to keep
  // performance high during normal debugging.

  // If we have an array type, say that it is indexed and provide the number
  // of children in case we have a huge array. If we don't do this, then we
  // might take a while to produce all children at onces which can delay your
  // debug session.
  if (desc.type_obj.IsArrayType()) {
    object.try_emplace("indexedVariables", v.GetNumChildren());
  } else if (v.IsSynthetic()) {
    // For a type with a synthetic child provider, the SBType of "v" won't tell
    // us anything about what might be displayed. Instead, we check if the first
    // child's name is "[0]" and then say it is indexed. We call
    // GetNumChildren() only if the child name matches to avoid a potentially
    // expensive operation.
    if (lldb::SBValue first_child = v.GetChildAtIndex(0)) {
      llvm::StringRef first_child_name = first_child.GetName();
      if (first_child_name == "[0]") {
        size_t num_children = v.GetNumChildren();
        // If we are creating a "[raw]" fake child for each synthetic type, we
        // have to account for it when returning indexed variables.
        if (synthetic_child_debugging)
          ++num_children;
        object.try_emplace("indexedVariables", num_children);
      }
    }
  }
  EmplaceSafeString(object, "type", desc.display_type_name);

  // A unique variable identifier to help in properly identifying variables with
  // the same name. This is an extension to the VS protocol.
  object.try_emplace("id", var_ref);

  if (v.MightHaveChildren())
    object.try_emplace("variablesReference", var_ref);
  else
    object.try_emplace("variablesReference", 0);

  if (v.GetDeclaration().IsValid())
    object.try_emplace("declarationLocationReference",
                       PackLocation(var_ref, false));

  if (ValuePointsToCode(v))
    object.try_emplace("valueLocationReference", PackLocation(var_ref, true));

  if (lldb::addr_t addr = v.GetLoadAddress(); addr != LLDB_INVALID_ADDRESS)
    object.try_emplace("memoryReference", EncodeMemoryReference(addr));

  object.try_emplace("$__lldb_extensions", desc.GetVariableExtensionsJSON());
  return llvm::json::Value(std::move(object));
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
    const llvm::StringMap<std::string> env, llvm::StringRef cwd,
    llvm::StringRef comm_file, lldb::pid_t debugger_pid) {
  llvm::json::Object run_in_terminal_args;
  // This indicates the IDE to open an embedded terminal, instead of opening
  // the terminal in a new window.
  run_in_terminal_args.try_emplace("kind", "integrated");

  // The program path must be the first entry in the "args" field
  std::vector<std::string> req_args = {DAP::debug_adapter_path.str(),
                                       "--comm-file", comm_file.str()};
  if (debugger_pid != LLDB_INVALID_PROCESS_ID) {
    req_args.push_back("--debugger-pid");
    req_args.push_back(std::to_string(debugger_pid));
  }
  req_args.push_back("--launch-target");
  req_args.push_back(program.str());
  req_args.insert(req_args.end(), args.begin(), args.end());
  run_in_terminal_args.try_emplace("args", args);

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
