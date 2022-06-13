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

Expected<std::unique_ptr<IntelPTMultiCoreTrace>>
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
            CreateContextSwitchTracePerfEvent(core_id,
                                              &core_trace->GetPerfEvent())) {
      traces.try_emplace(core_id,
                         std::make_pair(std::move(*core_trace),
                                        std::move(*context_switch_trace)));
    } else {
      return context_switch_trace.takeError();
    }
  }

  return std::unique_ptr<IntelPTMultiCoreTrace>(
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
                       ContextSwitchTrace &context_switch_trace)>
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
                  const ContextSwitchTrace &context_switch_trace) {
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
  // All the process' threads are being traced automatically.
  if (!TracesThread(tid))
    return createStringError(
        inconvertibleErrorCode(),
        "Thread %" PRIu64 " is not part of the target process", tid);
  return Error::success();
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
    return it->second.first.GetTraceBuffer();
  if (request.kind == IntelPTDataKinds::kPerfContextSwitchTrace)
    return it->second.second.GetReadOnlyDataBuffer();
  return None;
}
