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
#include "Protocol/ProtocolBase.h"
#include "Protocol/ProtocolRequests.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBLineEntry.h"
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstddef>
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
                           bool required, bool allow_empty) {
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

  if (allow_empty && mem_ref_str->empty()) {
    out = LLDB_INVALID_ADDRESS;
    return true;
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
  response.try_emplace("seq", protocol::kCalculateSeq);
  EmplaceSafeString(response, "command",
                    request.getString("command").value_or(""));
  const uint64_t seq = GetInteger<uint64_t>(request, "seq").value_or(0);
  response.try_emplace("request_seq", seq);
  response.try_emplace("success", true);
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
  event.try_emplace("seq", protocol::kCalculateSeq);
  event.try_emplace("type", "event");
  EmplaceSafeString(event, "event", event_name);
  return event;
}

llvm::StringRef GetNonNullVariableName(lldb::SBValue &v) {
  const llvm::StringRef name = v.GetName();
  return !name.empty() ? name : "<null>";
}

std::string CreateUniqueVariableNameForDisplay(lldb::SBValue &v,
                                               bool is_name_duplicated) {
  std::string unique_name{};
  llvm::raw_string_ostream name_builder(unique_name);
  name_builder << GetNonNullVariableName(v);
  if (is_name_duplicated) {
    const lldb::SBDeclaration declaration = v.GetDeclaration();
    const llvm::StringRef file_name = declaration.GetFileSpec().GetFilename();
    const uint32_t line = declaration.GetLine();

    if (!file_name.empty() && line != 0 && line != LLDB_INVALID_LINE_NUMBER)
      name_builder << llvm::formatv(" @ {}:{}", file_name, line);
    else if (llvm::StringRef location = v.GetLocation(); !location.empty())
      name_builder << llvm::formatv(" @ {}", location);
  }
  return unique_name;
}

VariableDescription::VariableDescription(
    lldb::SBValue val, bool auto_variable_summaries, bool format_hex,
    bool is_name_duplicated, std::optional<llvm::StringRef> custom_name)
    : val(val) {
  name = custom_name.value_or(
      CreateUniqueVariableNameForDisplay(val, is_name_duplicated));

  type_obj = val.GetType();
  const llvm::StringRef type_name = type_obj.GetDisplayTypeName();
  display_type_name = type_name.empty() ? NO_TYPENAME : type_name;

  // Only format hex/default if there is no existing special format.
  if (const lldb::Format current_format = val.GetFormat();
      current_format == lldb::eFormatDefault ||
      current_format == lldb::eFormatHex) {

    val.SetFormat(format_hex ? lldb::eFormatHex : lldb::eFormatDefault);
  }

  llvm::raw_string_ostream os_display_value(display_value);

  if (lldb::SBError sb_error = val.GetError(); sb_error.Fail()) {
    error = sb_error.GetCString();
    os_display_value << "<error: " << error << ">";
  } else {
    value = val.GetValue();
    summary = val.GetSummary();
    if (summary.empty() && auto_variable_summaries)
      auto_summary = TryCreateAutoSummary(val);

    llvm::StringRef display_summary = auto_summary ? *auto_summary : summary;
    const bool has_summary = !display_summary.empty();

    if (!value.empty()) {
      os_display_value << value;
      if (has_summary)
        os_display_value << " " << display_summary;
    } else if (has_summary) {
      os_display_value << display_summary;

    } else if (!type_name.empty()) {
      // As last resort, we print its type if available.
      os_display_value << type_name;
    }
  }

  lldb::SBStream evaluateStream;
  val.GetExpressionPath(evaluateStream);
  evaluate_name = llvm::StringRef(evaluateStream.GetData()).str();
}

std::string VariableDescription::GetResult(protocol::EvaluateContext context) {
  // In repl and clipboard contexts, the results can be displayed as multiple
  // lines so more detailed descriptions can be returned.
  if (context != protocol::eEvaluateContextRepl &&
      context != protocol::eEvaluateContextClipboard)
    return display_value;

  if (!val.IsValid())
    return display_value;

  // Try the SBValue::GetDescription(), which may call into language runtime
  // specific formatters (see ValueObjectPrinter).
  lldb::SBStream stream;
  if (context == protocol::eEvaluateContextRepl)
    val.GetDescription(stream, lldb::eDescriptionLevelFull);
  else
    val.GetDescription(stream, lldb::eDescriptionLevelBrief);
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

  if (!stdio.empty()) {
    req_args.emplace_back("--stdio");

    std::stringstream ss;
    std::string_view delimiter;
    for (const std::optional<std::string> &file : stdio) {
      ss << std::exchange(delimiter, ":");
      if (file)
        ss << *file;
    }
    req_args.push_back(ss.str());
  }

  // WARNING: Any argument added after `launch-target` is passed to to the
  // target.
  req_args.emplace_back("--launch-target");
  req_args.push_back(program.str());
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

llvm::json::Object CreateInitializedEventObject(lldb::SBTarget &target) {
  llvm::json::Object event(CreateEventObject("initialized"));
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
