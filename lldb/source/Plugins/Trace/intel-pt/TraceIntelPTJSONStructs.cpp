//===-- TraceIntelPTJSONStructs.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTJSONStructs.h"

#include "llvm/Support/JSON.h"
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

namespace llvm {
namespace json {

bool fromJSON(const Value &value, JSONTraceIntelPTSettings &plugin_settings,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("cpuInfo", plugin_settings.cpuInfo) &&
         fromJSON(value, (JSONTracePluginSettings &)plugin_settings, path);
}

bool fromJSON(const json::Value &value, JSONTraceIntelPTCPUInfo &cpu_info,
              Path path) {
  ObjectMapper o(value, path);
  return o && o.map("vendor", cpu_info.vendor) &&
         o.map("family", cpu_info.family) && o.map("model", cpu_info.model) &&
         o.map("stepping", cpu_info.stepping);
}

Value toJSON(const JSONTraceIntelPTCPUInfo &cpu_info) {
  return Value(Object{{"family", cpu_info.family},
                      {"model", cpu_info.model},
                      {"stepping", cpu_info.stepping},
                      {"vendor", cpu_info.vendor}});
}

llvm::json::Value toJSON(const JSONTraceIntelPTTrace &trace) {
  llvm::json::Object json_trace;
  json_trace["type"] = trace.type;
  json_trace["cpuInfo"] = toJSON(trace.cpuInfo);
  return std::move(json_trace);
}

llvm::json::Value toJSON(const JSONTraceIntelPTSession &session) {
  llvm::json::Object json_session;
  json_session["trace"] = toJSON(session.ipt_trace);
  json_session["processes"] = toJSON(session.session_base);
  return std::move(json_session);
}

} // namespace json
} // namespace llvm
