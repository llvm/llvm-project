//===-- IntelPTMultiCoreTrace.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntelPTMultiCoreTrace.h"

#include "Procfs.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"

using namespace lldb;
using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

static bool IsTotalBufferLimitReached(ArrayRef<core_id_t> cores,
                                      const TraceIntelPTStartRequest &request) {
  uint64_t required = cores.size() * request.trace_buffer_size;
  uint64_t limit = request.process_buffer_size_limit.getValueOr(
      std::numeric_limits<uint64_t>::max());
  return required > limit;
}

static Error IncludePerfEventParanoidMessageInError(Error &&error) {
  return createStringError(
      inconvertibleErrorCode(),
      "%s\nYou might need to rerun as sudo or to set "
      "/proc/sys/kernel/perf_event_paranoid to a value of 0 or -1.",
      toString(std::move(error)).c_str());
}

static Expected<PerfEvent> CreateContextSwitchTracePerfEvent(
    bool disabled, lldb::core_id_t core_id,
    IntelPTSingleBufferTrace &intelpt_core_trace) {
  Log *log = GetLog(POSIXLog::Trace);
#ifndef PERF_ATTR_SIZE_VER5
  return createStringError(inconvertibleErrorCode(),
                           "Intel PT Linux perf event not supported");
#else
  perf_event_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.sample_period = 0;
  attr.sample_type = PERF_SAMPLE_TID | PERF_SAMPLE_TIME;
  attr.type = PERF_TYPE_SOFTWARE;
  attr.context_switch = 1;
  attr.exclude_kernel = 1;
  attr.sample_id_all = 1;
  attr.exclude_hv = 1;
  attr.disabled = disabled;

  // The given perf configuration will product context switch records of 32
  // bytes each. Assuming that every context switch will be emitted twice (one
  // for context switch ins and another one for context switch outs), and that a
  // context switch will happen at least every half a millisecond per core, we
  // need 500 * 32 bytes (~16 KB) for a trace of one second, which is much more
  // than what a regular intel pt trace can get. Pessimistically we pick as
  // 32KiB for the size of our context switch trace.

  uint64_t data_buffer_size = 32768;
  uint64_t data_buffer_numpages = data_buffer_size / getpagesize();

  LLDB_LOG(log, "Will create context switch trace buffer of size {0}",
           data_buffer_size);

  if (Expected<PerfEvent> perf_event = PerfEvent::Init(
          attr, /*pid=*/None, core_id,
          intelpt_core_trace.GetPerfEvent().GetFd(), /*flags=*/0)) {
    if (Error mmap_err = perf_event->MmapMetadataAndBuffers(
            data_buffer_numpages, 0, /*data_buffer_write=*/false)) {
      return std::move(mmap_err);
    }
    return perf_event;
  } else {
    return perf_event.takeError();
  }
#endif
}

Expected<IntelPTProcessTraceUP>
IntelPTMultiCoreTrace::StartOnAllCores(const TraceIntelPTStartRequest &request,
                                       NativeProcessProtocol &process) {
  Expected<ArrayRef<core_id_t>> core_ids = GetAvailableLogicalCoreIDs();
  if (!core_ids)
    return core_ids.takeError();

  if (IsTotalBufferLimitReached(*core_ids, request))
    return createStringError(
        inconvertibleErrorCode(),
        "The process can't be traced because the process trace size limit "
        "has been reached. Consider retracing with a higher limit.");

  DenseMap<core_id_t, std::pair<IntelPTSingleBufferTrace, ContextSwitchTrace>>
      traces;

  for (core_id_t core_id : *core_ids) {
    Expected<IntelPTSingleBufferTrace> core_trace =
        IntelPTSingleBufferTrace::Start(request, /*tid=*/None, core_id,
                                        /*disabled=*/true);
    if (!core_trace)
      return IncludePerfEventParanoidMessageInError(core_trace.takeError());

    if (Expected<PerfEvent> context_switch_trace =
            CreateContextSwitchTracePerfEvent(/*disabled=*/true, core_id,
                                              core_trace.get())) {
      traces.try_emplace(core_id,
                         std::make_pair(std::move(*core_trace),
                                        std::move(*context_switch_trace)));
    } else {
      return context_switch_trace.takeError();
    }
  }

  return IntelPTProcessTraceUP(
      new IntelPTMultiCoreTrace(std::move(traces), process));
}

void IntelPTMultiCoreTrace::ForEachCore(
    std::function<void(core_id_t core_id, IntelPTSingleBufferTrace &core_trace)>
        callback) {
  for (auto &it : m_traces_per_core)
    callback(it.first, it.second.first);
}

void IntelPTMultiCoreTrace::ForEachCore(
    std::function<void(core_id_t core_id,
                       IntelPTSingleBufferTrace &intelpt_trace,
                       PerfEvent &context_switch_trace)>
        callback) {
  for (auto &it : m_traces_per_core)
    callback(it.first, it.second.first, it.second.second);
}

void IntelPTMultiCoreTrace::ProcessDidStop() {
  ForEachCore([](core_id_t core_id, IntelPTSingleBufferTrace &core_trace) {
    if (Error err = core_trace.Pause()) {
      LLDB_LOG_ERROR(GetLog(POSIXLog::Trace), std::move(err),
                     "Unable to pause the core trace for core {0}", core_id);
    }
  });
}

void IntelPTMultiCoreTrace::ProcessWillResume() {
  ForEachCore([](core_id_t core_id, IntelPTSingleBufferTrace &core_trace) {
    if (Error err = core_trace.Resume()) {
      LLDB_LOG_ERROR(GetLog(POSIXLog::Trace), std::move(err),
                     "Unable to resume the core trace for core {0}", core_id);
    }
  });
}

TraceIntelPTGetStateResponse IntelPTMultiCoreTrace::GetState() {
  TraceIntelPTGetStateResponse state;

  for (size_t i = 0; m_process.GetThreadAtIndex(i); i++)
    state.traced_threads.push_back(
        TraceThreadState{m_process.GetThreadAtIndex(i)->GetID(), {}});

  state.cores.emplace();
  ForEachCore([&](lldb::core_id_t core_id,
                  const IntelPTSingleBufferTrace &core_trace,
                  const PerfEvent &context_switch_trace) {
    state.cores->push_back(
        {core_id,
         {{IntelPTDataKinds::kTraceBuffer, core_trace.GetTraceBufferSize()},
          {IntelPTDataKinds::kPerfContextSwitchTrace,
           context_switch_trace.GetEffectiveDataBufferSize()}}});
  });

  return state;
}

bool IntelPTMultiCoreTrace::TracesThread(lldb::tid_t tid) const {
  // All the process' threads are being traced automatically.
  return (bool)m_process.GetThreadByID(tid);
}

llvm::Error IntelPTMultiCoreTrace::TraceStart(lldb::tid_t tid) {
  // This instance is already tracing all threads automatically.
  return llvm::Error::success();
}

Error IntelPTMultiCoreTrace::TraceStop(lldb::tid_t tid) {
  return createStringError(inconvertibleErrorCode(),
                           "Can't stop tracing an individual thread when "
                           "per-core process tracing is enabled.");
}

Expected<Optional<std::vector<uint8_t>>>
IntelPTMultiCoreTrace::TryGetBinaryData(
    const TraceGetBinaryDataRequest &request) {
  if (!request.core_id)
    return None;
  auto it = m_traces_per_core.find(*request.core_id);
  if (it == m_traces_per_core.end())
    return createStringError(
        inconvertibleErrorCode(),
        formatv("Core {0} is not being traced", *request.core_id));

  if (request.kind == IntelPTDataKinds::kTraceBuffer)
    return it->second.first.GetTraceBuffer(request.offset, request.size);
  if (request.kind == IntelPTDataKinds::kPerfContextSwitchTrace)
    return it->second.second.ReadFlushedOutDataCyclicBuffer(request.offset,
                                                            request.size);
  return None;
}
