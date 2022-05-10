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
    states.push_back({static_cast<int64_t>(it.first),
                      {TraceBinaryData{IntelPTDataKinds::kTraceBuffer,
                                       static_cast<int64_t>(
                                           it.second->GetTraceBufferSize())}}});
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

/// IntelPTProcessTrace

bool IntelPTProcessTrace::TracesThread(lldb::tid_t tid) const {
  return m_thread_traces.TracesThread(tid);
}

Error IntelPTProcessTrace::TraceStop(lldb::tid_t tid) {
  return m_thread_traces.TraceStop(tid);
}

Error IntelPTProcessTrace::TraceStart(lldb::tid_t tid) {
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
IntelPTProcessTrace::GetThreadTraces() const {
  return m_thread_traces;
}

/// IntelPTCollector

IntelPTCollector::IntelPTCollector() {
  if (Expected<LinuxPerfZeroTscConversion> tsc_conversion =
          LoadPerfTscConversionParameters())
    m_tsc_conversion =
        std::make_unique<LinuxPerfZeroTscConversion>(*tsc_conversion);
  else
    LLDB_LOG_ERROR(GetLog(POSIXLog::Trace), tsc_conversion.takeError(),
                   "unable to load TSC to wall time conversion: {0}");
}

Error IntelPTCollector::TraceStop(lldb::tid_t tid) {
  if (IsProcessTracingEnabled() && m_process_trace->TracesThread(tid))
    return m_process_trace->TraceStop(tid);
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

Error IntelPTCollector::TraceStart(
    const TraceIntelPTStartRequest &request,
    const std::vector<lldb::tid_t> &process_threads) {
  if (request.IsProcessTracing()) {
    if (IsProcessTracingEnabled()) {
      return createStringError(
          inconvertibleErrorCode(),
          "Process currently traced. Stop process tracing first");
    }
    if (request.per_core_tracing.getValueOr(false)) {
      return createStringError(inconvertibleErrorCode(),
                               "Per-core tracing is not supported.");
    }
    m_process_trace = IntelPTProcessTrace(request);

    Error error = Error::success();
    for (lldb::tid_t tid : process_threads)
      error = joinErrors(std::move(error), m_process_trace->TraceStart(tid));
    return error;
  } else {
    Error error = Error::success();
    for (int64_t tid : *request.tids)
      error = joinErrors(std::move(error),
                         m_thread_traces.TraceStart(tid, request));
    return error;
  }
}

Error IntelPTCollector::OnThreadCreated(lldb::tid_t tid) {
  if (!IsProcessTracingEnabled())
    return Error::success();
  return m_process_trace->TraceStart(tid);
}

Error IntelPTCollector::OnThreadDestroyed(lldb::tid_t tid) {
  if (IsProcessTracingEnabled() && m_process_trace->TracesThread(tid))
    return m_process_trace->TraceStop(tid);
  else if (m_thread_traces.TracesThread(tid))
    return m_thread_traces.TraceStop(tid);
  return Error::success();
}

Expected<json::Value> IntelPTCollector::GetState() const {
  Expected<ArrayRef<uint8_t>> cpu_info = GetProcfsCpuInfo();
  if (!cpu_info)
    return cpu_info.takeError();

  TraceGetStateResponse state;
  state.processBinaryData.push_back({IntelPTDataKinds::kProcFsCpuInfo,
                                     static_cast<int64_t>(cpu_info->size())});

  std::vector<TraceThreadState> thread_states =
      m_thread_traces.GetThreadStates();
  state.tracedThreads.insert(state.tracedThreads.end(), thread_states.begin(),
                             thread_states.end());

  if (IsProcessTracingEnabled()) {
    thread_states = m_process_trace->GetThreadTraces().GetThreadStates();
    state.tracedThreads.insert(state.tracedThreads.end(), thread_states.begin(),
                               thread_states.end());
  }
  return toJSON(state);
}

Expected<const IntelPTSingleBufferTrace &>
IntelPTCollector::GetTracedThread(lldb::tid_t tid) const {
  if (IsProcessTracingEnabled() && m_process_trace->TracesThread(tid))
    return m_process_trace->GetThreadTraces().GetTracedThread(tid);
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

void IntelPTCollector::ClearProcessTracing() { m_process_trace = None; }

bool IntelPTCollector::IsSupported() {
  if (Expected<uint32_t> intel_pt_type = GetIntelPTOSEventType()) {
    return true;
  } else {
    llvm::consumeError(intel_pt_type.takeError());
    return false;
  }
}

bool IntelPTCollector::IsProcessTracingEnabled() const {
  return (bool)m_process_trace;
}

void IntelPTCollector::Clear() {
  ClearProcessTracing();
  m_thread_traces.Clear();
}
