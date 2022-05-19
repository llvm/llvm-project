//===-- TraceIntelPT.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPT.h"

#include "../common/ThreadPostMortemTrace.h"
#include "CommandObjectTraceStartIntelPT.h"
#include "DecodedThread.h"
#include "TraceIntelPTConstants.h"
#include "TraceIntelPTSessionFileParser.h"
#include "TraceIntelPTSessionSaver.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "llvm/ADT/None.h"

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

StringRef TraceIntelPT::GetSchema() {
  return TraceIntelPTSessionFileParser::GetSchema();
}

void TraceIntelPT::Dump(Stream *s) const {}

llvm::Error TraceIntelPT::SaveLiveTraceToDisk(FileSpec directory) {
  RefreshLiveProcessState();
  return TraceIntelPTSessionSaver().SaveToDisk(*this, directory);
}

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
  for (const ThreadPostMortemTraceSP &thread : traced_threads) {
    m_thread_decoders.emplace(thread->GetID(),
                              std::make_unique<ThreadDecoder>(thread, *this));
    SetPostMortemThreadDataFile(thread->GetID(), IntelPTDataKinds::kTraceBuffer,
                                thread->GetTraceFile());
  }
}

DecodedThreadSP TraceIntelPT::Decode(Thread &thread) {
  if (const char *error = RefreshLiveProcessState())
    return std::make_shared<DecodedThread>(
        thread.shared_from_this(),
        createStringError(inconvertibleErrorCode(), error));

  auto it = m_thread_decoders.find(thread.GetID());
  if (it == m_thread_decoders.end())
    return std::make_shared<DecodedThread>(
        thread.shared_from_this(),
        createStringError(inconvertibleErrorCode(), "thread not traced"));
  return it->second->Decode();
}

lldb::TraceCursorUP TraceIntelPT::GetCursor(Thread &thread) {
  return Decode(thread)->GetCursor();
}

void TraceIntelPT::DumpTraceInfo(Thread &thread, Stream &s, bool verbose) {
  lldb::tid_t tid = thread.GetID();
  s.Format("\nthread #{0}: tid = {1}", thread.GetIndexID(), thread.GetID());
  if (!IsTraced(tid)) {
    s << ", not traced\n";
    return;
  }
  s << "\n";

  Expected<size_t> raw_size = GetRawTraceSize(thread);
  if (!raw_size) {
    s.Format("  {0}\n", toString(raw_size.takeError()));
    return;
  }

  DecodedThreadSP decoded_trace_sp = Decode(thread);
  size_t insn_len = decoded_trace_sp->GetInstructionsCount();
  size_t mem_used = decoded_trace_sp->CalculateApproximateMemoryUsage();

  s.Format("  Total number of instructions: {0}\n", insn_len);

  s << "\n  Memory usage:\n";
  s.Format("    Raw trace size: {0} KiB\n", *raw_size / 1024);
  s.Format(
      "    Total approximate memory usage (excluding raw trace): {0:2} KiB\n",
      (double)mem_used / 1024);
  if (insn_len != 0)
    s.Format("    Average memory usage per instruction (excluding raw trace): "
             "{0:2} bytes\n",
             (double)mem_used / insn_len);

  s << "\n  Timing:\n";
  GetTimer().ForThread(tid).ForEachTimedTask(
      [&](const std::string &name, std::chrono::milliseconds duration) {
        s.Format("    {0}: {1:2}s\n", name, duration.count() / 1000.0);
      });

  const DecodedThread::EventsStats &events_stats =
      decoded_trace_sp->GetEventsStats();
  s << "\n  Events:\n";
  s.Format("    Number of instructions with events: {0}\n",
           events_stats.total_instructions_with_events);
  s.Format("    Number of individual events: {0}\n", events_stats.total_count);
  for (const auto &event_to_count : events_stats.events_counts) {
    s.Format("      {0}: {1}\n",
             trace_event_utils::EventToDisplayString(event_to_count.first),
             event_to_count.second);
  }

  s << "\n  Errors:\n";
  const DecodedThread::LibiptErrorsStats &tsc_errors_stats =
      decoded_trace_sp->GetTscErrorsStats();
  s.Format("    Number of TSC decoding errors: {0}\n",
           tsc_errors_stats.total_count);
  for (const auto &error_message_to_count :
       tsc_errors_stats.libipt_errors_counts) {
    s.Format("      {0}: {1}\n", error_message_to_count.first,
             error_message_to_count.second);
  }
}

llvm::Expected<size_t> TraceIntelPT::GetRawTraceSize(Thread &thread) {
  size_t size;
  auto callback = [&](llvm::ArrayRef<uint8_t> data) {
    size = data.size();
    return Error::success();
  };
  if (Error err = OnThreadBufferRead(thread.GetID(), callback))
    return std::move(err);

  return size;
}

Expected<pt_cpu> TraceIntelPT::GetCPUInfoForLiveProcess() {
  Expected<std::vector<uint8_t>> cpu_info =
      GetLiveProcessBinaryData(IntelPTDataKinds::kProcFsCpuInfo);
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

Error TraceIntelPT::DoRefreshLiveProcessState(TraceGetStateResponse state,
                                              StringRef json_response) {
  m_thread_decoders.clear();

  for (const TraceThreadState &thread_state : state.traced_threads) {
    ThreadSP thread_sp =
        GetLiveProcess()->GetThreadList().FindThreadByID(thread_state.tid);
    m_thread_decoders.emplace(
        thread_state.tid, std::make_unique<ThreadDecoder>(thread_sp, *this));
  }

  Expected<TraceIntelPTGetStateResponse> intelpt_state =
      json::parse<TraceIntelPTGetStateResponse>(json_response,
                                                "TraceIntelPTGetStateResponse");
  if (!intelpt_state)
    return intelpt_state.takeError();

  m_tsc_conversion = intelpt_state->tsc_perf_zero_conversion;
  if (m_tsc_conversion) {
    Log *log = GetLog(LLDBLog::Target);
    LLDB_LOG(log, "TraceIntelPT found TSC conversion information");
  }
  return Error::success();
}

bool TraceIntelPT::IsTraced(lldb::tid_t tid) {
  RefreshLiveProcessState();
  return m_thread_decoders.count(tid);
}

// The information here should match the description of the intel-pt section
// of the jLLDBTraceStart packet in the lldb/docs/lldb-gdb-remote.txt
// documentation file. Similarly, it should match the CLI help messages of the
// TraceIntelPTOptions.td file.
const char *TraceIntelPT::GetStartConfigurationHelp() {
  static Optional<std::string> message;
  if (!message) {
    message.emplace(formatv(R"(Parameters:

  See the jLLDBTraceStart section in lldb/docs/lldb-gdb-remote.txt for a
  description of each parameter below.

  - int traceBufferSize (defaults to {0} bytes):
    [process and thread tracing]

  - boolean enableTsc (default to {1}):
    [process and thread tracing]

  - int psbPeriod (defaults to {2}):
    [process and thread tracing]

  - boolean perCoreTracing (default to {3}):
    [process tracing only]

  - int processBufferSizeLimit (defaults to {4} MiB):
    [process tracing only])",
                            kDefaultTraceBufferSize, kDefaultEnableTscValue,
                            kDefaultPsbPeriod, kDefaultPerCoreTracing,
                            kDefaultProcessBufferSizeLimit / 1024 / 1024));
  }
  return message->c_str();
}

Error TraceIntelPT::Start(size_t trace_buffer_size,
                          size_t total_buffer_size_limit, bool enable_tsc,
                          Optional<size_t> psb_period, bool per_core_tracing) {
  TraceIntelPTStartRequest request;
  request.trace_buffer_size = trace_buffer_size;
  request.process_buffer_size_limit = total_buffer_size_limit;
  request.enable_tsc = enable_tsc;
  request.psb_period =
      psb_period.map([](size_t val) { return static_cast<uint64_t>(val); });
  request.type = GetPluginName().str();
  request.per_core_tracing = per_core_tracing;
  return Trace::Start(toJSON(request));
}

Error TraceIntelPT::Start(StructuredData::ObjectSP configuration) {
  size_t trace_buffer_size = kDefaultTraceBufferSize;
  size_t process_buffer_size_limit = kDefaultProcessBufferSizeLimit;
  bool enable_tsc = kDefaultEnableTscValue;
  Optional<size_t> psb_period = kDefaultPsbPeriod;
  bool per_core_tracing = kDefaultPerCoreTracing;

  if (configuration) {
    if (StructuredData::Dictionary *dict = configuration->GetAsDictionary()) {
      dict->GetValueForKeyAsInteger("traceBufferSize", trace_buffer_size);
      dict->GetValueForKeyAsInteger("processBufferSizeLimit",
                                    process_buffer_size_limit);
      dict->GetValueForKeyAsBoolean("enableTsc", enable_tsc);
      dict->GetValueForKeyAsInteger("psbPeriod", psb_period);
      dict->GetValueForKeyAsBoolean("perCoreTracing", per_core_tracing);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "configuration object is not a dictionary");
    }
  }

  return Start(trace_buffer_size, process_buffer_size_limit, enable_tsc,
               psb_period, per_core_tracing);
}

llvm::Error TraceIntelPT::Start(llvm::ArrayRef<lldb::tid_t> tids,
                                size_t trace_buffer_size, bool enable_tsc,
                                Optional<size_t> psb_period) {
  TraceIntelPTStartRequest request;
  request.trace_buffer_size = trace_buffer_size;
  request.enable_tsc = enable_tsc;
  request.psb_period =
      psb_period.map([](size_t val) { return static_cast<uint64_t>(val); });
  request.type = GetPluginName().str();
  request.tids.emplace();
  for (lldb::tid_t tid : tids)
    request.tids->push_back(tid);
  return Trace::Start(toJSON(request));
}

Error TraceIntelPT::Start(llvm::ArrayRef<lldb::tid_t> tids,
                          StructuredData::ObjectSP configuration) {
  uint64_t trace_buffer_size = kDefaultTraceBufferSize;
  bool enable_tsc = kDefaultEnableTscValue;
  Optional<uint64_t> psb_period = kDefaultPsbPeriod;

  if (configuration) {
    if (StructuredData::Dictionary *dict = configuration->GetAsDictionary()) {
      dict->GetValueForKeyAsInteger("traceBufferSize", trace_buffer_size);
      dict->GetValueForKeyAsBoolean("enableTsc", enable_tsc);
      dict->GetValueForKeyAsInteger("psbPeriod", psb_period);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "configuration object is not a dictionary");
    }
  }

  return Start(tids, trace_buffer_size, enable_tsc, psb_period);
}

Error TraceIntelPT::OnThreadBufferRead(lldb::tid_t tid,
                                       OnBinaryDataReadCallback callback) {
  return OnThreadBinaryDataRead(tid, IntelPTDataKinds::kTraceBuffer, callback);
}

TaskTimer &TraceIntelPT::GetTimer() { return m_task_timer; }
