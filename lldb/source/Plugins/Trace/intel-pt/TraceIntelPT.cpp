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
  for (const ThreadPostMortemTraceSP &thread : traced_threads)
    m_thread_decoders.emplace(
        thread->GetID(),
        std::make_unique<PostMortemThreadDecoder>(thread, *this));
}

DecodedThreadSP TraceIntelPT::Decode(Thread &thread) {
  RefreshLiveProcessState();
  if (m_live_refresh_error.hasValue())
    return std::make_shared<DecodedThread>(
        thread.shared_from_this(),
        createStringError(inconvertibleErrorCode(), *m_live_refresh_error));

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
  Optional<size_t> raw_size = GetRawTraceSize(thread);
  s.Format("\nthread #{0}: tid = {1}", thread.GetIndexID(), thread.GetID());
  if (!raw_size) {
    s << ", not traced\n";
    return;
  }
  s << "\n";
  DecodedThreadSP decoded_trace_sp = Decode(thread);
  size_t insn_len = decoded_trace_sp->GetInstructionsCount();
  size_t mem_used = decoded_trace_sp->CalculateApproximateMemoryUsage();

  s.Format("  Raw trace size: {0} KiB\n", *raw_size / 1024);
  s.Format("  Total number of instructions: {0}\n", insn_len);
  s.Format("  Total approximate memory usage: {0:2} KiB\n",
           (double)mem_used / 1024);
  if (insn_len != 0)
    s.Format("  Average memory usage per instruction: {0:2} bytes\n",
             (double)mem_used / insn_len);

  const DecodedThread::LibiptErrors &tsc_errors =
      decoded_trace_sp->GetTscErrors();
  s.Format("\n  Number of TSC decoding errors: {0}\n", tsc_errors.total_count);
  for (const auto &error_message_to_count : tsc_errors.libipt_errors) {
    s.Format("    {0}: {1}\n", error_message_to_count.first,
             error_message_to_count.second);
  }
}

Optional<size_t> TraceIntelPT::GetRawTraceSize(Thread &thread) {
  if (IsTraced(thread.GetID()))
    return Decode(thread)->GetRawTraceSize();
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

Process *TraceIntelPT::GetLiveProcess() { return m_live_process; }

void TraceIntelPT::DoRefreshLiveProcessState(
    Expected<TraceGetStateResponse> state) {
  m_thread_decoders.clear();

  if (!state) {
    m_live_refresh_error = toString(state.takeError());
    return;
  }

  for (const TraceThreadState &thread_state : state->tracedThreads) {
    Thread &thread =
        *m_live_process->GetThreadList().FindThreadByID(thread_state.tid);
    m_thread_decoders.emplace(
        thread_state.tid, std::make_unique<LiveThreadDecoder>(thread, *this));
  }
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
  return R"(Parameters:

  Note: If a parameter is not specified, a default value will be used.

  - int threadBufferSize (defaults to 4096 bytes):
    [process and thread tracing]
    Trace size in bytes per thread. It must be a power of 2 greater
    than or equal to 4096 (2^12). The trace is circular keeping the
    the most recent data.

  - boolean enableTsc (default to false):
    [process and thread tracing]
    Whether to use enable TSC timestamps or not. This is supported on
    all devices that support intel-pt.

  - psbPeriod (defaults to null):
    [process and thread tracing]
    This value defines the period in which PSB packets will be generated.
    A PSB packet is a synchronization packet that contains a TSC
    timestamp and the current absolute instruction pointer.

    This parameter can only be used if

        /sys/bus/event_source/devices/intel_pt/caps/psb_cyc

    is 1. Otherwise, the PSB period will be defined by the processor.

    If supported, valid values for this period can be found in

        /sys/bus/event_source/devices/intel_pt/caps/psb_periods

    which contains a hexadecimal number, whose bits represent
    valid values e.g. if bit 2 is set, then value 2 is valid.

    The psb_period value is converted to the approximate number of
    raw trace bytes between PSB packets as:

        2 ^ (value + 11)

    e.g. value 3 means 16KiB between PSB packets. Defaults to 0 if
    supported.

  - int processBufferSizeLimit (defaults to 500 MB):
    [process tracing only]
    Maximum total trace size per process in bytes. This limit applies
    to the sum of the sizes of all thread traces of this process,
    excluding the ones created explicitly with "thread tracing".
    Whenever a thread is attempted to be traced due to this command
    and the limit would be reached, the process is stopped with a
    "processor trace" reason, so that the user can retrace the process
    if needed.)";
}

Error TraceIntelPT::Start(size_t thread_buffer_size,
                          size_t total_buffer_size_limit, bool enable_tsc,
                          Optional<size_t> psb_period) {
  TraceIntelPTStartRequest request;
  request.threadBufferSize = thread_buffer_size;
  request.processBufferSizeLimit = total_buffer_size_limit;
  request.enableTsc = enable_tsc;
  request.psbPeriod = psb_period.map([](size_t val) { return (int64_t)val; });
  request.type = GetPluginName().str();
  return Trace::Start(toJSON(request));
}

Error TraceIntelPT::Start(StructuredData::ObjectSP configuration) {
  size_t thread_buffer_size = kDefaultThreadBufferSize;
  size_t process_buffer_size_limit = kDefaultProcessBufferSizeLimit;
  bool enable_tsc = kDefaultEnableTscValue;
  Optional<size_t> psb_period = kDefaultPsbPeriod;

  if (configuration) {
    if (StructuredData::Dictionary *dict = configuration->GetAsDictionary()) {
      dict->GetValueForKeyAsInteger("threadBufferSize", thread_buffer_size);
      dict->GetValueForKeyAsInteger("processBufferSizeLimit",
                                    process_buffer_size_limit);
      dict->GetValueForKeyAsBoolean("enableTsc", enable_tsc);
      dict->GetValueForKeyAsInteger("psbPeriod", psb_period);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "configuration object is not a dictionary");
    }
  }

  return Start(thread_buffer_size, process_buffer_size_limit, enable_tsc,
               psb_period);
}

llvm::Error TraceIntelPT::Start(llvm::ArrayRef<lldb::tid_t> tids,
                                size_t thread_buffer_size, bool enable_tsc,
                                Optional<size_t> psb_period) {
  TraceIntelPTStartRequest request;
  request.threadBufferSize = thread_buffer_size;
  request.enableTsc = enable_tsc;
  request.psbPeriod = psb_period.map([](size_t val) { return (int64_t)val; });
  request.type = GetPluginName().str();
  request.tids.emplace();
  for (lldb::tid_t tid : tids)
    request.tids->push_back(tid);
  return Trace::Start(toJSON(request));
}

Error TraceIntelPT::Start(llvm::ArrayRef<lldb::tid_t> tids,
                          StructuredData::ObjectSP configuration) {
  size_t thread_buffer_size = kDefaultThreadBufferSize;
  bool enable_tsc = kDefaultEnableTscValue;
  Optional<size_t> psb_period = kDefaultPsbPeriod;

  if (configuration) {
    if (StructuredData::Dictionary *dict = configuration->GetAsDictionary()) {
      dict->GetValueForKeyAsInteger("threadBufferSize", thread_buffer_size);
      dict->GetValueForKeyAsBoolean("enableTsc", enable_tsc);
      dict->GetValueForKeyAsInteger("psbPeriod", psb_period);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "configuration object is not a dictionary");
    }
  }

  return Start(tids, thread_buffer_size, enable_tsc, psb_period);
}

Expected<std::vector<uint8_t>>
TraceIntelPT::GetLiveThreadBuffer(lldb::tid_t tid) {
  return Trace::GetLiveThreadBinaryData(tid, "threadTraceBuffer");
}
