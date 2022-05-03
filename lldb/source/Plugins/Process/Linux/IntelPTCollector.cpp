//===-- IntelPTCollector.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntelPTCollector.h"

#include "Perf.h"
#include "Procfs.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/linux/Support.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <linux/perf_event.h>
#include <sstream>
#include <sys/ioctl.h>
#include <sys/syscall.h>

using namespace lldb;
using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

/// IntelPTThreadTraceCollection

bool IntelPTThreadTraceCollection::TracesThread(lldb::tid_t tid) const {
  return m_thread_traces.count(tid);
}

Error IntelPTThreadTraceCollection::TraceStop(lldb::tid_t tid) {
  auto it = m_thread_traces.find(tid);
  if (it == m_thread_traces.end())
    return createStringError(inconvertibleErrorCode(),
                             "Thread %" PRIu64 " not currently traced", tid);
  m_total_buffer_size -= it->second->GetTraceBufferSize();
  m_thread_traces.erase(tid);
  return Error::success();
}

Error IntelPTThreadTraceCollection::TraceStart(
    lldb::tid_t tid, const TraceIntelPTStartRequest &request) {
  if (TracesThread(tid))
    return createStringError(inconvertibleErrorCode(),
                             "Thread %" PRIu64 " already traced", tid);

  Expected<IntelPTSingleBufferTraceUP> trace_up =
      IntelPTSingleBufferTrace::Start(request, tid);
  if (!trace_up)
    return trace_up.takeError();

  m_total_buffer_size += (*trace_up)->GetTraceBufferSize();
  m_thread_traces.try_emplace(tid, std::move(*trace_up));
  return Error::success();
}

size_t IntelPTThreadTraceCollection::GetTotalBufferSize() const {
  return m_total_buffer_size;
}

std::vector<TraceThreadState>
IntelPTThreadTraceCollection::GetThreadStates() const {
  std::vector<TraceThreadState> states;
  for (const auto &it : m_thread_traces)
    states.push_back({it.first,
                      {TraceBinaryData{IntelPTDataKinds::kTraceBuffer,
                                       it.second->GetTraceBufferSize()}}});
  return states;
}

Expected<const IntelPTSingleBufferTrace &>
IntelPTThreadTraceCollection::GetTracedThread(lldb::tid_t tid) const {
  auto it = m_thread_traces.find(tid);
  if (it == m_thread_traces.end())
    return createStringError(inconvertibleErrorCode(),
                             "Thread %" PRIu64 " not currently traced", tid);
  return *it->second.get();
}

void IntelPTThreadTraceCollection::Clear() {
  m_thread_traces.clear();
  m_total_buffer_size = 0;
}

size_t IntelPTThreadTraceCollection::GetTracedThreadsCount() const {
  return m_thread_traces.size();
}

/// IntelPTPerThreadProcessTrace

bool IntelPTPerThreadProcessTrace::TracesThread(lldb::tid_t tid) const {
  return m_thread_traces.TracesThread(tid);
}

Error IntelPTPerThreadProcessTrace::TraceStop(lldb::tid_t tid) {
  return m_thread_traces.TraceStop(tid);
}

Error IntelPTPerThreadProcessTrace::TraceStart(lldb::tid_t tid) {
  if (m_thread_traces.GetTotalBufferSize() +
          m_tracing_params.trace_buffer_size >
      static_cast<size_t>(*m_tracing_params.process_buffer_size_limit))
    return createStringError(
        inconvertibleErrorCode(),
        "Thread %" PRIu64 " can't be traced as the process trace size limit "
        "has been reached. Consider retracing with a higher "
        "limit.",
        tid);

  return m_thread_traces.TraceStart(tid, m_tracing_params);
}

const IntelPTThreadTraceCollection &
IntelPTPerThreadProcessTrace::GetThreadTraces() const {
  return m_thread_traces;
}

/// IntelPTCollector

IntelPTCollector::IntelPTCollector(NativeProcessProtocol &process)
    : m_process(process) {
  if (Expected<LinuxPerfZeroTscConversion> tsc_conversion =
          LoadPerfTscConversionParameters())
    m_tsc_conversion =
        std::make_unique<LinuxPerfZeroTscConversion>(*tsc_conversion);
  else
    LLDB_LOG_ERROR(GetLog(POSIXLog::Trace), tsc_conversion.takeError(),
                   "unable to load TSC to wall time conversion: {0}");
}

Error IntelPTCollector::TraceStop(lldb::tid_t tid) {
  if (m_per_thread_process_trace_up &&
      m_per_thread_process_trace_up->TracesThread(tid))
    return m_per_thread_process_trace_up->TraceStop(tid);
  else if (m_per_core_process_trace_up)
    return createStringError(inconvertibleErrorCode(),
                             "Can't stop tracing an individual thread when "
                             "per-core process tracing is enabled.");
  return m_thread_traces.TraceStop(tid);
}

Error IntelPTCollector::TraceStop(const TraceStopRequest &request) {
  if (request.IsProcessTracing()) {
    Clear();
    return Error::success();
  } else {
    Error error = Error::success();
    for (int64_t tid : *request.tids)
      error = joinErrors(std::move(error),
                         TraceStop(static_cast<lldb::tid_t>(tid)));
    return error;
  }
}

Expected<IntelPTPerThreadProcessTraceUP>
IntelPTPerThreadProcessTrace::Start(const TraceIntelPTStartRequest &request,
                                    ArrayRef<lldb::tid_t> current_tids) {
  IntelPTPerThreadProcessTraceUP trace(
      new IntelPTPerThreadProcessTrace(request));

  Error error = Error::success();
  for (lldb::tid_t tid : current_tids)
    error = joinErrors(std::move(error), trace->TraceStart(tid));
  if (error)
    return std::move(error);
  return trace;
}

Error IntelPTCollector::TraceStart(const TraceIntelPTStartRequest &request) {
  if (request.IsProcessTracing()) {
    if (IsProcessTracingEnabled()) {
      return createStringError(
          inconvertibleErrorCode(),
          "Process currently traced. Stop process tracing first");
    }
    if (request.IsPerCoreTracing()) {
      if (m_thread_traces.GetTracedThreadsCount() > 0)
        return createStringError(
            inconvertibleErrorCode(),
            "Threads currently traced. Stop tracing them first.");
      if (Expected<IntelPTMultiCoreTraceUP> trace =
              IntelPTMultiCoreTrace::StartOnAllCores(request)) {
        m_per_core_process_trace_up = std::move(*trace);
        return Error::success();
      } else {
        return trace.takeError();
      }
    } else {
      std::vector<lldb::tid_t> process_threads;
      for (size_t i = 0; m_process.GetThreadAtIndex(i); i++)
        process_threads.push_back(m_process.GetThreadAtIndex(i)->GetID());

      // per-thread process tracing
      if (Expected<IntelPTPerThreadProcessTraceUP> trace =
              IntelPTPerThreadProcessTrace::Start(request, process_threads)) {
        m_per_thread_process_trace_up = std::move(trace.get());
        return Error::success();
      } else {
        return trace.takeError();
      }
    }
  } else {
    // individual thread tracing
    if (m_per_core_process_trace_up)
      return createStringError(inconvertibleErrorCode(),
                               "Process currently traced with per-core "
                               "tracing. Stop process tracing first");

    Error error = Error::success();
    for (int64_t tid : *request.tids)
      error = joinErrors(std::move(error),
                         m_thread_traces.TraceStart(tid, request));
    return error;
  }
}

Error IntelPTCollector::OnThreadCreated(lldb::tid_t tid) {
  if (m_per_thread_process_trace_up)
    return m_per_thread_process_trace_up->TraceStart(tid);

  return Error::success();
}

Error IntelPTCollector::OnThreadDestroyed(lldb::tid_t tid) {
  if (m_per_thread_process_trace_up &&
      m_per_thread_process_trace_up->TracesThread(tid))
    return m_per_thread_process_trace_up->TraceStop(tid);
  else if (m_thread_traces.TracesThread(tid))
    return m_thread_traces.TraceStop(tid);
  return Error::success();
}

Expected<json::Value> IntelPTCollector::GetState() const {
  Expected<ArrayRef<uint8_t>> cpu_info = GetProcfsCpuInfo();
  if (!cpu_info)
    return cpu_info.takeError();

  TraceGetStateResponse state;
  state.process_binary_data.push_back(
      {IntelPTDataKinds::kProcFsCpuInfo, cpu_info->size()});

  std::vector<TraceThreadState> thread_states =
      m_thread_traces.GetThreadStates();
  state.traced_threads.insert(state.traced_threads.end(), thread_states.begin(),
                              thread_states.end());

  if (m_per_thread_process_trace_up) {
    thread_states =
        m_per_thread_process_trace_up->GetThreadTraces().GetThreadStates();
    state.traced_threads.insert(state.traced_threads.end(),
                                thread_states.begin(), thread_states.end());
  } else if (m_per_core_process_trace_up) {
    for (size_t i = 0; m_process.GetThreadAtIndex(i); i++)
      state.traced_threads.push_back(
          TraceThreadState{m_process.GetThreadAtIndex(i)->GetID(), {}});

    state.cores.emplace();
    m_per_core_process_trace_up->ForEachCore(
        [&](lldb::core_id_t core_id,
            const IntelPTSingleBufferTrace &core_trace) {
          state.cores->push_back({core_id,
                                  {{IntelPTDataKinds::kTraceBuffer,
                                    core_trace.GetTraceBufferSize()}}});
        });
  }
  return toJSON(state);
}

Expected<const IntelPTSingleBufferTrace &>
IntelPTCollector::GetTracedThread(lldb::tid_t tid) const {
  if (m_per_thread_process_trace_up &&
      m_per_thread_process_trace_up->TracesThread(tid))
    return m_per_thread_process_trace_up->GetThreadTraces().GetTracedThread(
        tid);
  return m_thread_traces.GetTracedThread(tid);
}

Expected<std::vector<uint8_t>>
IntelPTCollector::GetBinaryData(const TraceGetBinaryDataRequest &request) const {
  if (request.kind == IntelPTDataKinds::kTraceBuffer) {
    if (Expected<const IntelPTSingleBufferTrace &> trace =
            GetTracedThread(*request.tid))
      return trace->GetTraceBuffer(request.offset, request.size);
    else
      return trace.takeError();
  } else if (request.kind == IntelPTDataKinds::kProcFsCpuInfo) {
    return GetProcfsCpuInfo();
  }
  return createStringError(inconvertibleErrorCode(),
                           "Unsuported trace binary data kind: %s",
                           request.kind.c_str());
}

void IntelPTCollector::ClearProcessTracing() {
  m_per_thread_process_trace_up.reset();
  m_per_core_process_trace_up.reset();
}

bool IntelPTCollector::IsSupported() {
  if (Expected<uint32_t> intel_pt_type = GetIntelPTOSEventType()) {
    return true;
  } else {
    llvm::consumeError(intel_pt_type.takeError());
    return false;
  }
}

bool IntelPTCollector::IsProcessTracingEnabled() const {
  return (bool)m_per_thread_process_trace_up ||
         (bool)m_per_core_process_trace_up;
}

void IntelPTCollector::Clear() {
  ClearProcessTracing();
  m_thread_traces.Clear();
}
