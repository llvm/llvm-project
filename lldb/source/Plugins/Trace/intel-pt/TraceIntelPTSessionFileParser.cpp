//===-- TraceIntelPTSessionFileParser.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTSessionFileParser.h"

#include "../common/ThreadPostMortemTrace.h"
#include "TraceIntelPT.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

StringRef TraceIntelPTSessionFileParser::GetSchema() {
  static std::string schema;
  if (schema.empty()) {
    schema = TraceSessionFileParser::BuildSchema(R"({
    "type": "intel-pt",
    "cpuInfo": {
      "vendor": "intel" | "unknown",
      "family": integer,
      "model": integer,
      "stepping": integer
    }
  })");
  }
  return schema;
}

pt_cpu TraceIntelPTSessionFileParser::ParsePTCPU(
    const JSONTraceIntelPTCPUInfo &cpu_info) {
  return {cpu_info.vendor.compare("intel") == 0 ? pcv_intel : pcv_unknown,
          static_cast<uint16_t>(cpu_info.family),
          static_cast<uint8_t>(cpu_info.model),
          static_cast<uint8_t>(cpu_info.stepping)};
}

TraceSP TraceIntelPTSessionFileParser::CreateTraceIntelPTInstance(
    const pt_cpu &cpu_info, std::vector<ParsedProcess> &parsed_processes) {
  std::vector<ThreadPostMortemTraceSP> threads;
  for (const ParsedProcess &parsed_process : parsed_processes)
    threads.insert(threads.end(), parsed_process.threads.begin(),
                   parsed_process.threads.end());

  TraceSP trace_instance(new TraceIntelPT(cpu_info, threads));
  for (const ParsedProcess &parsed_process : parsed_processes)
    parsed_process.target_sp->SetTrace(trace_instance);

  return trace_instance;
}

Expected<TraceSP> TraceIntelPTSessionFileParser::Parse() {
  json::Path::Root root("traceSession");
  JSONTraceSession<JSONTraceIntelPTSettings> session;
  if (!json::fromJSON(m_trace_session_file, session, root))
    return CreateJSONError(root, m_trace_session_file);

  if (Expected<std::vector<ParsedProcess>> parsed_processes =
          ParseCommonSessionFile(session))
    return CreateTraceIntelPTInstance(ParsePTCPU(session.trace.cpuInfo),
                                      *parsed_processes);
  else
    return parsed_processes.takeError();
}
