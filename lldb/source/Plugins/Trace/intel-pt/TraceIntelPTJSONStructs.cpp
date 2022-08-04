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

Optional<std::vector<lldb::cpu_id_t>> JSONTraceBundleDescription::GetCpuIds() {
  if (!cpus)
    return None;
  std::vector<lldb::cpu_id_t> cpu_ids;
  for (const JSONCpu &cpu : *cpus)
    cpu_ids.push_back(cpu.id);
  return cpu_ids;
}

json::Value toJSON(const JSONModule &module) {
  json::Object json_module;
  json_module["systemPath"] = module.system_path;
  if (module.file)
    json_module["file"] = *module.file;
  json_module["loadAddress"] = toJSON(module.load_address, true);
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
  json::Object obj{{"tid", thread.tid}};
  if (thread.ipt_trace)
    obj["iptTrace"] = *thread.ipt_trace;
  return obj;
}

bool fromJSON(const json::Value &value, JSONThread &thread, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("tid", thread.tid) && o.map("iptTrace", thread.ipt_trace);
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

json::Value toJSON(const JSONCpu &cpu) {
  return Object{
      {"id", cpu.id},
      {"iptTrace", cpu.ipt_trace},
      {"contextSwitchTrace", cpu.context_switch_trace},
  };
}

bool fromJSON(const json::Value &value, JSONCpu &cpu, Path path) {
  ObjectMapper o(value, path);
  uint64_t cpu_id;
  if (!(o && o.map("id", cpu_id) && o.map("iptTrace", cpu.ipt_trace) &&
        o.map("contextSwitchTrace", cpu.context_switch_trace)))
    return false;
  cpu.id = cpu_id;
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
  if (!(o && o.map("vendor", vendor) && o.map("family", family) &&
        o.map("model", model) && o.map("stepping", stepping)))
    return false;
  cpu_info.vendor = vendor == "GenuineIntel" ? pcv_intel : pcv_unknown;
  cpu_info.family = family;
  cpu_info.model = model;
  cpu_info.stepping = stepping;
  return true;
}

json::Value toJSON(const JSONTraceBundleDescription &bundle_description) {
  return Object{
      {"type", bundle_description.type},
      {"processes", bundle_description.processes},
      // We have to do this because the compiler fails at doing it
      // automatically because pt_cpu is not in a namespace
      {"cpuInfo", toJSON(bundle_description.cpu_info)},
      {"cpus", bundle_description.cpus},
      {"tscPerfZeroConversion", bundle_description.tsc_perf_zero_conversion}};
}

bool fromJSON(const json::Value &value,
              JSONTraceBundleDescription &bundle_description, Path path) {
  ObjectMapper o(value, path);
  if (!(o && o.map("processes", bundle_description.processes) &&
        o.map("type", bundle_description.type) &&
        o.map("cpus", bundle_description.cpus) &&
        o.map("tscPerfZeroConversion",
              bundle_description.tsc_perf_zero_conversion)))
    return false;
  if (bundle_description.cpus && !bundle_description.tsc_perf_zero_conversion) {
    path.report(
        "\"tscPerfZeroConversion\" is required when \"cpus\" is provided");
    return false;
  }
  // We have to do this because the compiler fails at doing it automatically
  // because pt_cpu is not in a namespace
  if (!fromJSON(*value.getAsObject()->get("cpuInfo"),
                bundle_description.cpu_info, path.field("cpuInfo")))
    return false;
  return true;
}

} // namespace trace_intel_pt
} // namespace lldb_private
