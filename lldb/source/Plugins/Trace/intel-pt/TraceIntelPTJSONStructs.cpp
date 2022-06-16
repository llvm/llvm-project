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
using namespace llvm::json;

namespace lldb_private {
namespace trace_intel_pt {

Optional<std::vector<lldb::core_id_t>> JSONTraceSession::GetCoreIds() {
  if (!cores)
    return None;
  std::vector<lldb::core_id_t> core_ids;
  for (const JSONCore &core : *cores)
    core_ids.push_back(core.core_id);
  return core_ids;
}

json::Value toJSON(const JSONModule &module) {
  json::Object json_module;
  json_module["systemPath"] = module.system_path;
  if (module.file)
    json_module["file"] = *module.file;
  json_module["loadAddress"] = module.load_address;
  if (module.uuid)
    json_module["uuid"] = *module.uuid;
  return std::move(json_module);
}

bool fromJSON(const json::Value &value, JSONModule &module, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("systemPath", module.system_path) &&
         o.map("file", module.file) &&
         o.map("loadAddress", module.load_address) &&
         o.map("uuid", module.uuid);
}

json::Value toJSON(const JSONThread &thread) {
  return json::Object{{"tid", thread.tid},
                      {"traceBuffer", thread.trace_buffer}};
}

bool fromJSON(const json::Value &value, JSONThread &thread, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("tid", thread.tid) &&
         o.map("traceBuffer", thread.trace_buffer);
}

json::Value toJSON(const JSONProcess &process) {
  return Object{
      {"pid", process.pid},
      {"triple", process.triple},
      {"threads", process.threads},
      {"modules", process.modules},
  };
}

bool fromJSON(const json::Value &value, JSONProcess &process, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("pid", process.pid) && o.map("triple", process.triple) &&
         o.map("threads", process.threads) && o.map("modules", process.modules);
}

json::Value toJSON(const JSONCore &core) {
  return Object{
      {"coreId", core.core_id},
      {"traceBuffer", core.trace_buffer},
      {"contextSwitchTrace", core.context_switch_trace},
  };
}

bool fromJSON(const json::Value &value, JSONCore &core, Path path) {
  ObjectMapper o(value, path);
  uint64_t core_id;
  if (!o || !o.map("coreId", core_id) ||
      !o.map("traceBuffer", core.trace_buffer) ||
      !o.map("contextSwitchTrace", core.context_switch_trace))
    return false;
  core.core_id = core_id;
  return true;
}

json::Value toJSON(const pt_cpu &cpu_info) {
  return Object{
      {"vendor", cpu_info.vendor == pcv_intel ? "GenuineIntel" : "Unknown"},
      {"family", cpu_info.family},
      {"model", cpu_info.model},
      {"stepping", cpu_info.stepping},
  };
}

bool fromJSON(const json::Value &value, pt_cpu &cpu_info, Path path) {
  ObjectMapper o(value, path);
  std::string vendor;
  uint64_t family, model, stepping;
  if (!o || !o.map("vendor", vendor) || !o.map("family", family) ||
      !o.map("model", model) || !o.map("stepping", stepping))
    return false;
  cpu_info.vendor = vendor == "GenuineIntel" ? pcv_intel : pcv_unknown;
  cpu_info.family = family;
  cpu_info.model = model;
  cpu_info.stepping = stepping;
  return true;
}

json::Value toJSON(const JSONTraceSession &session) {
  return Object{{"type", session.type},
                {"processes", session.processes},
                // We have to do this because the compiler fails at doing it
                // automatically because pt_cpu is not in a namespace
                {"cpuInfo", toJSON(session.cpu_info)},
                {"cores", session.cores},
                {"tscPerfZeroConversion", session.tsc_perf_zero_conversion}};
}

bool fromJSON(const json::Value &value, JSONTraceSession &session, Path path) {
  ObjectMapper o(value, path);
  if (!o || !o.map("processes", session.processes) ||
      !o.map("type", session.type) || !o.map("cores", session.cores) ||
      !o.map("tscPerfZeroConversion", session.tsc_perf_zero_conversion))
    return false;
  // We have to do this because the compiler fails at doing it automatically
  // because pt_cpu is not in a namespace
  if (!fromJSON(*value.getAsObject()->get("cpuInfo"), session.cpu_info,
                path.field("cpuInfo")))
    return false;
  return true;
}

} // namespace trace_intel_pt
} // namespace lldb_private
