//===-- TraceIntelPTSessionFileParser.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTSessionFileParser.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Target/ThreadTrace.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

StringRef TraceIntelPTSessionFileParser::GetSchema() {
  static std::string schema;
  if (schema.empty()) {
    schema = TraceSessionFileParser::BuildSchema(R"({
    "type": "intel-pt",
    "pt_cpu": {
      "vendor": "intel" | "unknown",
      "family": integer,
      "model": integer,
      "stepping": integer
    }
  })");
  }
  return schema;
}

pt_cpu TraceIntelPTSessionFileParser::ParsePTCPU(const JSONPTCPU &pt_cpu) {
  return {pt_cpu.vendor.compare("intel") == 0 ? pcv_intel : pcv_unknown,
          static_cast<uint16_t>(pt_cpu.family),
          static_cast<uint8_t>(pt_cpu.model),
          static_cast<uint8_t>(pt_cpu.stepping)};
}

TraceSP TraceIntelPTSessionFileParser::CreateTraceIntelPTInstance(
    const pt_cpu &pt_cpu, std::vector<ParsedProcess> &parsed_processes) {
  std::vector<ThreadTraceSP> threads;
  for (const ParsedProcess &parsed_process : parsed_processes)
    threads.insert(threads.end(), parsed_process.threads.begin(),
                   parsed_process.threads.end());

  TraceSP trace_instance(new TraceIntelPT(pt_cpu, threads));
  for (const ParsedProcess &parsed_process : parsed_processes)
    parsed_process.target_sp->SetTrace(trace_instance);

  return trace_instance;
}

Expected<TraceSP> TraceIntelPTSessionFileParser::Parse() {
  json::Path::Root root("traceSession");
  TraceSessionFileParser::JSONTraceSession<JSONTraceIntelPTSettings> session;
  if (!json::fromJSON(m_trace_session_file, session, root))
    return CreateJSONError(root, m_trace_session_file);

  if (Expected<std::vector<ParsedProcess>> parsed_processes =
          ParseCommonSessionFile(session))
    return CreateTraceIntelPTInstance(ParsePTCPU(session.trace.pt_cpu),
                                      *parsed_processes);
  else
    return parsed_processes.takeError();
}

namespace llvm {
namespace json {

bool fromJSON(const Value &value,
              TraceIntelPTSessionFileParser::JSONPTCPU &pt_cpu, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("vendor", pt_cpu.vendor) &&
         o.map("family", pt_cpu.family) && o.map("model", pt_cpu.model) &&
         o.map("stepping", pt_cpu.stepping);
}

bool fromJSON(
    const Value &value,
    TraceIntelPTSessionFileParser::JSONTraceIntelPTSettings &plugin_settings,
    Path path) {
  ObjectMapper o(value, path);
  return o && o.map("pt_cpu", plugin_settings.pt_cpu) &&
         fromJSON(
             value,
             (TraceSessionFileParser::JSONTracePluginSettings &)plugin_settings,
             path);
}

} // namespace json
} // namespace llvm
