//===-- TraceIntelPTJSONStructs.h -----------------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H

#include "../common/TraceJSONStructs.h"
#include <intel-pt.h>

namespace lldb_private {
namespace trace_intel_pt {

struct JSONTraceIntelPTCPUInfo {
  JSONTraceIntelPTCPUInfo() = default;

  JSONTraceIntelPTCPUInfo(pt_cpu cpu_info) {
    family = static_cast<int64_t>(cpu_info.family);
    model = static_cast<int64_t>(cpu_info.model);
    stepping = static_cast<int64_t>(cpu_info.stepping);
    vendor = cpu_info.vendor == pcv_intel ? "intel" : "Unknown";
  }

  int64_t family;
  int64_t model;
  int64_t stepping;
  std::string vendor;
};

struct JSONTraceIntelPTTrace {
  std::string type;
  JSONTraceIntelPTCPUInfo cpuInfo;
};

struct JSONTraceIntelPTSession {
  JSONTraceIntelPTTrace ipt_trace;
  JSONTraceSessionBase session_base;
};

struct JSONTraceIntelPTSettings : JSONTracePluginSettings {
  JSONTraceIntelPTCPUInfo cpuInfo;
};

} // namespace trace_intel_pt
} // namespace lldb_private

namespace llvm {
namespace json {

bool fromJSON(
    const Value &value,
    lldb_private::trace_intel_pt::JSONTraceIntelPTSettings &plugin_settings,
    Path path);

bool fromJSON(const llvm::json::Value &value,
              lldb_private::trace_intel_pt::JSONTraceIntelPTCPUInfo &packet,
              llvm::json::Path path);

llvm::json::Value
toJSON(const lldb_private::trace_intel_pt::JSONTraceIntelPTCPUInfo &cpu_info);

llvm::json::Value
toJSON(const lldb_private::trace_intel_pt::JSONTraceIntelPTTrace &trace);

llvm::json::Value
toJSON(const lldb_private::trace_intel_pt::JSONTraceIntelPTSession &session);

} // namespace json
} // namespace llvm

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTJSONSTRUCTS_H
