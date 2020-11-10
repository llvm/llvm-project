//===-- TraceIntelPT.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPT.h"

#include "TraceIntelPTSessionFileParser.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadTrace.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

LLDB_PLUGIN_DEFINE(TraceIntelPT)

void TraceIntelPT::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Intel Processor Trace",
                                CreateInstance,
                                TraceIntelPTSessionFileParser::GetSchema());
}

void TraceIntelPT::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ConstString TraceIntelPT::GetPluginNameStatic() {
  static ConstString g_name("intel-pt");
  return g_name;
}

StringRef TraceIntelPT::GetSchema() {
  return TraceIntelPTSessionFileParser::GetSchema();
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------

ConstString TraceIntelPT::GetPluginName() { return GetPluginNameStatic(); }

uint32_t TraceIntelPT::GetPluginVersion() { return 1; }

void TraceIntelPT::Dump(Stream *s) const {}

Expected<TraceSP>
TraceIntelPT::CreateInstance(const json::Value &trace_session_file,
                             StringRef session_file_dir, Debugger &debugger) {
  return TraceIntelPTSessionFileParser(debugger, trace_session_file,
                                       session_file_dir)
      .Parse();
}

TraceIntelPT::TraceIntelPT(
    const pt_cpu &pt_cpu,
    const std::vector<std::shared_ptr<ThreadTrace>> &traced_threads)
    : m_pt_cpu(pt_cpu) {
  for (const std::shared_ptr<ThreadTrace> &thread : traced_threads)
    m_trace_threads.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(thread->GetProcess()->GetID(), thread->GetID()),
        std::forward_as_tuple(thread, pt_cpu));
}

const DecodedThread *TraceIntelPT::Decode(const Thread &thread) {
  auto it = m_trace_threads.find(
      std::make_pair(thread.GetProcess()->GetID(), thread.GetID()));
  if (it == m_trace_threads.end())
    return nullptr;
  return &it->second.Decode();
}

size_t TraceIntelPT::GetCursorPosition(const Thread &thread) {
  const DecodedThread *decoded_thread = Decode(thread);
  if (!decoded_thread)
    return 0;
  return decoded_thread->GetCursorPosition();
}

void TraceIntelPT::TraverseInstructions(
    const Thread &thread, size_t position, TraceDirection direction,
    std::function<bool(size_t index, Expected<lldb::addr_t> load_addr)>
        callback) {
  const DecodedThread *decoded_thread = Decode(thread);
  if (!decoded_thread)
    return;

  ArrayRef<IntelPTInstruction> instructions = decoded_thread->GetInstructions();

  ssize_t delta = direction == TraceDirection::Forwards ? 1 : -1;
  for (ssize_t i = position; i < (ssize_t)instructions.size() && i >= 0;
       i += delta)
    if (!callback(i, instructions[i].GetLoadAddress()))
      break;
}

size_t TraceIntelPT::GetInstructionCount(const Thread &thread) {
  if (const DecodedThread *decoded_thread = Decode(thread))
    return decoded_thread->GetInstructions().size();
  else
    return 0;
}
