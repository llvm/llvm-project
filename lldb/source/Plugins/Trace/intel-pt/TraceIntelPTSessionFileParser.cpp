//===-- TraceIntelPTSessionFileParser.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTSessionFileParser.h"

#include "ThreadIntelPT.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

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

void TraceIntelPTSessionFileParser::ParseThread(
    ProcessSP &process_sp, const TraceSessionFileParser::JSONThread &thread) {
  lldb::tid_t tid = static_cast<lldb::tid_t>(thread.tid);

  FileSpec trace_file(thread.trace_file);
  NormalizePath(trace_file);

  ThreadSP thread_sp =
      std::make_shared<ThreadIntelPT>(*process_sp, tid, trace_file);
  process_sp->GetThreadList().AddThread(thread_sp);
}

Error TraceIntelPTSessionFileParser::ParseProcess(
    const TraceSessionFileParser::JSONProcess &process) {
  TargetSP target_sp;
  Status error = m_debugger.GetTargetList().CreateTarget(
      m_debugger, /*user_exe_path*/ StringRef(), process.triple,
      eLoadDependentsNo,
      /*platform_options*/ nullptr, target_sp);

  if (!target_sp)
    return error.ToError();

  m_targets.push_back(target_sp);
  m_debugger.GetTargetList().SetSelectedTarget(target_sp.get());

  ProcessSP process_sp(target_sp->CreateProcess(
      /*listener*/ nullptr, "trace",
      /*crash_file*/ nullptr));
  process_sp->SetID(static_cast<lldb::pid_t>(process.pid));

  for (const TraceSessionFileParser::JSONThread &thread : process.threads)
    ParseThread(process_sp, thread);

  for (const TraceSessionFileParser::JSONModule &module : process.modules) {
    if (Error err = ParseModule(target_sp, module))
      return err;
  }

  if (!process.threads.empty())
    process_sp->GetThreadList().SetSelectedThreadByIndexID(0);

  // We invoke DidAttach to create a correct stopped state for the process and
  // its threads.
  ArchSpec process_arch;
  process_sp->DidAttach(process_arch);

  return llvm::Error::success();
}

void TraceIntelPTSessionFileParser::ParsePTCPU(const JSONPTCPU &pt_cpu) {
  m_pt_cpu = {pt_cpu.vendor.compare("intel") == 0 ? pcv_intel : pcv_unknown,
              static_cast<uint16_t>(pt_cpu.family),
              static_cast<uint8_t>(pt_cpu.model),
              static_cast<uint8_t>(pt_cpu.stepping)};
}

Error TraceIntelPTSessionFileParser::ParseImpl() {
  json::Path::Root root("traceSession");
  TraceSessionFileParser::JSONTraceSession<JSONTraceIntelPTSettings> session;
  if (!json::fromJSON(m_trace_session_file, session, root)) {
    return CreateJSONError(root, m_trace_session_file);
  }

  ParsePTCPU(session.trace.pt_cpu);
  for (const TraceSessionFileParser::JSONProcess &process : session.processes) {
    if (Error err = ParseProcess(process))
      return err;
  }
  return Error::success();
}

Expected<TraceSP> TraceIntelPTSessionFileParser::Parse() {
  if (Error err = ParseImpl()) {
    // Delete all targets that were created
    for (auto target_sp : m_targets)
      m_debugger.GetTargetList().DeleteTarget(target_sp);
    m_targets.clear();
    return std::move(err);
  }

  return TraceIntelPT::CreateInstance(m_pt_cpu, m_targets);
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
