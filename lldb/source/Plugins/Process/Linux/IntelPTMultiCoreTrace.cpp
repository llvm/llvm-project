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

  llvm::DenseMap<core_id_t, IntelPTSingleBufferTraceUP> buffers;
  for (core_id_t core_id : *core_ids) {
    if (Expected<IntelPTSingleBufferTraceUP> core_trace =
            IntelPTSingleBufferTrace::Start(request, /*tid=*/None, core_id,
                                            TraceCollectionState::Paused))
      buffers.try_emplace(core_id, std::move(*core_trace));
    else
      return IncludePerfEventParanoidMessageInError(core_trace.takeError());
  }

  return IntelPTProcessTraceUP(
      new IntelPTMultiCoreTrace(std::move(buffers), process));
}

void IntelPTMultiCoreTrace::ForEachCore(
    std::function<void(core_id_t core_id, IntelPTSingleBufferTrace &core_trace)>
        callback) {
  for (auto &it : m_traces_per_core)
    callback(it.first, *it.second);
}

void IntelPTMultiCoreTrace::OnProcessStateChanged(lldb::StateType state) {
  if (m_process_state == state)
    return;
  switch (state) {
  case eStateStopped:
  case eStateExited: {
    ForEachCore([](core_id_t core_id, IntelPTSingleBufferTrace &core_trace) {
      if (Error err =
              core_trace.ChangeCollectionState(TraceCollectionState::Paused)) {
        LLDB_LOG_ERROR(GetLog(POSIXLog::Trace), std::move(err),
                       "Unable to pause the core trace for core {0}", core_id);
      }
    });
    break;
  }
  case eStateRunning: {
    ForEachCore([](core_id_t core_id, IntelPTSingleBufferTrace &core_trace) {
      if (Error err =
              core_trace.ChangeCollectionState(TraceCollectionState::Running)) {
        LLDB_LOG_ERROR(GetLog(POSIXLog::Trace), std::move(err),
                       "Unable to resume the core trace for core {0}", core_id);
      }
    });
    break;
  }
  default:
    break;
  }
}

TraceGetStateResponse IntelPTMultiCoreTrace::GetState() {
  TraceGetStateResponse state;

  for (size_t i = 0; m_process.GetThreadAtIndex(i); i++)
    state.traced_threads.push_back(
        TraceThreadState{m_process.GetThreadAtIndex(i)->GetID(), {}});

  state.cores.emplace();
  ForEachCore([&](lldb::core_id_t core_id,
                  const IntelPTSingleBufferTrace &core_trace) {
    state.cores->push_back(
        {core_id,
         {{IntelPTDataKinds::kTraceBuffer, core_trace.GetTraceBufferSize()}}});
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

Expected<std::vector<uint8_t>>
IntelPTMultiCoreTrace::GetBinaryData(const TraceGetBinaryDataRequest &request) {
  return createStringError(inconvertibleErrorCode(), "Unimplemented");
}
