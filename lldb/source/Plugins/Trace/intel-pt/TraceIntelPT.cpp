//===-- TraceIntelPT.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPT.h"

#include "CommandObjectTraceStartIntelPT.h"
#include "TraceIntelPTSessionFileParser.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPostMortemTrace.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

LLDB_PLUGIN_DEFINE(TraceIntelPT)

lldb::CommandObjectSP
TraceIntelPT::GetProcessTraceStartCommand(CommandInterpreter &interpreter) {
  return CommandObjectSP(
      new CommandObjectProcessTraceStartIntelPT(*this, interpreter));
}

lldb::CommandObjectSP
TraceIntelPT::GetThreadTraceStartCommand(CommandInterpreter &interpreter) {
  return CommandObjectSP(
      new CommandObjectThreadTraceStartIntelPT(*this, interpreter));
}

void TraceIntelPT::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Intel Processor Trace",
                                CreateInstanceForSessionFile,
                                CreateInstanceForLiveProcess,
                                TraceIntelPTSessionFileParser::GetSchema());
}

void TraceIntelPT::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstanceForSessionFile);
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

Expected<TraceSP> TraceIntelPT::CreateInstanceForSessionFile(
    const json::Value &trace_session_file, StringRef session_file_dir,
    Debugger &debugger) {
  return TraceIntelPTSessionFileParser(debugger, trace_session_file,
                                       session_file_dir)
      .Parse();
}

Expected<TraceSP> TraceIntelPT::CreateInstanceForLiveProcess(Process &process) {
  TraceSP instance(new TraceIntelPT(process));
  process.GetTarget().SetTrace(instance);
  return instance;
}

TraceIntelPT::TraceIntelPT(
    const pt_cpu &cpu_info,
    const std::vector<ThreadPostMortemTraceSP> &traced_threads)
    : m_cpu_info(cpu_info) {
  for (const ThreadPostMortemTraceSP &thread : traced_threads)
    m_thread_decoders.emplace(
        thread.get(), std::make_unique<PostMortemThreadDecoder>(thread, *this));
}

const DecodedThread *TraceIntelPT::Decode(Thread &thread) {
  RefreshLiveProcessState();
  if (m_failed_live_threads_decoder.hasValue())
    return &*m_failed_live_threads_decoder;

  auto it = m_thread_decoders.find(&thread);
  if (it == m_thread_decoders.end())
    return nullptr;
  return &it->second->Decode();
}

size_t TraceIntelPT::GetCursorPosition(Thread &thread) {
  const DecodedThread *decoded_thread = Decode(thread);
  if (!decoded_thread)
    return 0;
  return decoded_thread->GetCursorPosition();
}

void TraceIntelPT::TraverseInstructions(
    Thread &thread, size_t position, TraceDirection direction,
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

Optional<size_t> TraceIntelPT::GetInstructionCount(Thread &thread) {
  if (const DecodedThread *decoded_thread = Decode(thread))
    return decoded_thread->GetInstructions().size();
  else
    return None;
}

Expected<pt_cpu> TraceIntelPT::GetCPUInfoForLiveProcess() {
  Expected<std::vector<uint8_t>> cpu_info = GetLiveProcessBinaryData("cpuInfo");
  if (!cpu_info)
    return cpu_info.takeError();

  int64_t cpu_family = -1;
  int64_t model = -1;
  int64_t stepping = -1;
  std::string vendor_id;

  StringRef rest(reinterpret_cast<const char *>(cpu_info->data()),
                 cpu_info->size());
  while (!rest.empty()) {
    StringRef line;
    std::tie(line, rest) = rest.split('\n');

    SmallVector<StringRef, 2> columns;
    line.split(columns, StringRef(":"), -1, false);

    if (columns.size() < 2)
      continue; // continue searching

    columns[1] = columns[1].trim(" ");
    if (columns[0].contains("cpu family") &&
        columns[1].getAsInteger(10, cpu_family))
      continue;

    else if (columns[0].contains("model") && columns[1].getAsInteger(10, model))
      continue;

    else if (columns[0].contains("stepping") &&
             columns[1].getAsInteger(10, stepping))
      continue;

    else if (columns[0].contains("vendor_id")) {
      vendor_id = columns[1].str();
      if (!vendor_id.empty())
        continue;
    }

    if ((cpu_family != -1) && (model != -1) && (stepping != -1) &&
        (!vendor_id.empty())) {
      return pt_cpu{vendor_id == "GenuineIntel" ? pcv_intel : pcv_unknown,
                    static_cast<uint16_t>(cpu_family),
                    static_cast<uint8_t>(model),
                    static_cast<uint8_t>(stepping)};
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "Failed parsing the target's /proc/cpuinfo file");
}

Expected<pt_cpu> TraceIntelPT::GetCPUInfo() {
  if (!m_cpu_info) {
    if (llvm::Expected<pt_cpu> cpu_info = GetCPUInfoForLiveProcess())
      m_cpu_info = *cpu_info;
    else
      return cpu_info.takeError();
  }
  return *m_cpu_info;
}

void TraceIntelPT::DoRefreshLiveProcessState(
    Expected<TraceGetStateResponse> state) {
  m_thread_decoders.clear();

  if (!state) {
    m_failed_live_threads_decoder = DecodedThread(state.takeError());
    return;
  }

  for (const TraceThreadState &thread_state : state->tracedThreads) {
    Thread &thread =
        *m_live_process->GetThreadList().FindThreadByID(thread_state.tid);
    m_thread_decoders.emplace(
        &thread, std::make_unique<LiveThreadDecoder>(thread, *this));
  }
}

bool TraceIntelPT::IsTraced(const Thread &thread) {
  return m_thread_decoders.count(&thread);
}

Error TraceIntelPT::Start(size_t thread_buffer_size,
                          size_t total_buffer_size_limit) {
  TraceIntelPTStartRequest request;
  request.threadBufferSize = thread_buffer_size;
  request.processBufferSizeLimit = total_buffer_size_limit;
  request.type = GetPluginName().AsCString();
  return Trace::Start(toJSON(request));
}

llvm::Error TraceIntelPT::Start(const std::vector<lldb::tid_t> &tids,
                                size_t thread_buffer_size) {
  TraceIntelPTStartRequest request;
  request.threadBufferSize = thread_buffer_size;
  request.type = GetPluginName().AsCString();
  request.tids.emplace();
  for (lldb::tid_t tid : tids)
    request.tids->push_back(tid);
  return Trace::Start(toJSON(request));
}

Expected<std::vector<uint8_t>>
TraceIntelPT::GetLiveThreadBuffer(lldb::tid_t tid) {
  return Trace::GetLiveThreadBinaryData(tid, "threadTraceBuffer");
}
