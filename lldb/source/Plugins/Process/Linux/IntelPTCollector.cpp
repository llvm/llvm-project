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

IntelPTCollector::IntelPTCollector(NativeProcessProtocol &process)
    : m_process(process) {}

llvm::Expected<LinuxPerfZeroTscConversion &>
IntelPTCollector::FetchPerfTscConversionParameters() {
  if (Expected<LinuxPerfZeroTscConversion> tsc_conversion =
          LoadPerfTscConversionParameters())
    return *tsc_conversion;
  else
    return createStringError(inconvertibleErrorCode(),
                             "Unable to load TSC to wall time conversion: %s",
                             toString(tsc_conversion.takeError()).c_str());
}

Error IntelPTCollector::TraceStop(lldb::tid_t tid) {
  if (m_process_trace_up && m_process_trace_up->TracesThread(tid))
    return m_process_trace_up->TraceStop(tid);
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

Error IntelPTCollector::TraceStart(const TraceIntelPTStartRequest &request) {
  if (request.IsProcessTracing()) {
    if (m_process_trace_up) {
      return createStringError(
          inconvertibleErrorCode(),
          "Process currently traced. Stop process tracing first");
    }
    if (request.IsPerCpuTracing()) {
      if (m_thread_traces.GetTracedThreadsCount() > 0)
        return createStringError(
            inconvertibleErrorCode(),
            "Threads currently traced. Stop tracing them first.");
      // CPU tracing is useless if we can't convert tsc to nanos.
      Expected<LinuxPerfZeroTscConversion &> tsc_conversion =
          FetchPerfTscConversionParameters();
      if (!tsc_conversion)
        return tsc_conversion.takeError();

      // We force the enabledment of TSCs, which is needed for correlating the
      // cpu traces.
      TraceIntelPTStartRequest effective_request = request;
      effective_request.enable_tsc = true;

      if (Expected<IntelPTProcessTraceUP> trace =
              IntelPTMultiCoreTrace::StartOnAllCores(effective_request,
                                                     m_process)) {
        m_process_trace_up = std::move(*trace);
        return Error::success();
      } else {
        return trace.takeError();
      }
    } else {
      std::vector<lldb::tid_t> process_threads;
      for (size_t i = 0; m_process.GetThreadAtIndex(i); i++)
        process_threads.push_back(m_process.GetThreadAtIndex(i)->GetID());

      // per-thread process tracing
      if (Expected<IntelPTProcessTraceUP> trace =
              IntelPTPerThreadProcessTrace::Start(request, process_threads)) {
        m_process_trace_up = std::move(trace.get());
        return Error::success();
      } else {
        return trace.takeError();
      }
    }
  } else {
    // individual thread tracing
    Error error = Error::success();
    for (int64_t tid : *request.tids) {
      if (m_process_trace_up && m_process_trace_up->TracesThread(tid))
        error = joinErrors(
            std::move(error),
            createStringError(inconvertibleErrorCode(),
                              formatv("Thread with tid {0} is currently "
                                      "traced. Stop tracing it first.",
                                      tid)
                                  .str()
                                  .c_str()));
      else
        error = joinErrors(std::move(error),
                           m_thread_traces.TraceStart(tid, request));
    }
    return error;
  }
}

void IntelPTCollector::ProcessWillResume() {
  if (m_process_trace_up)
    m_process_trace_up->ProcessWillResume();
}

void IntelPTCollector::ProcessDidStop() {
  if (m_process_trace_up)
    m_process_trace_up->ProcessDidStop();
}

Error IntelPTCollector::OnThreadCreated(lldb::tid_t tid) {
  if (m_process_trace_up)
    return m_process_trace_up->TraceStart(tid);

  return Error::success();
}

Error IntelPTCollector::OnThreadDestroyed(lldb::tid_t tid) {
  if (m_process_trace_up && m_process_trace_up->TracesThread(tid))
    return m_process_trace_up->TraceStop(tid);
  else if (m_thread_traces.TracesThread(tid))
    return m_thread_traces.TraceStop(tid);
  return Error::success();
}

Expected<json::Value> IntelPTCollector::GetState() {
  Expected<ArrayRef<uint8_t>> cpu_info = GetProcfsCpuInfo();
  if (!cpu_info)
    return cpu_info.takeError();

  TraceIntelPTGetStateResponse state;
  if (m_process_trace_up)
    state = m_process_trace_up->GetState();

  state.process_binary_data.push_back(
      {IntelPTDataKinds::kProcFsCpuInfo, cpu_info->size()});

  m_thread_traces.ForEachThread(
      [&](lldb::tid_t tid, const IntelPTSingleBufferTrace &thread_trace) {
        state.traced_threads.push_back(
            {tid,
             {{IntelPTDataKinds::kIptTrace, thread_trace.GetIptTraceSize()}}});
      });

  if (Expected<LinuxPerfZeroTscConversion &> tsc_conversion =
          FetchPerfTscConversionParameters())
    state.tsc_perf_zero_conversion = *tsc_conversion;
  else
    state.AddWarning(toString(tsc_conversion.takeError()));
  return toJSON(state);
}

Expected<std::vector<uint8_t>>
IntelPTCollector::GetBinaryData(const TraceGetBinaryDataRequest &request) {
  if (request.kind == IntelPTDataKinds::kProcFsCpuInfo)
    return GetProcfsCpuInfo();

  if (m_process_trace_up) {
    Expected<Optional<std::vector<uint8_t>>> data =
        m_process_trace_up->TryGetBinaryData(request);
    if (!data)
      return data.takeError();
    if (*data)
      return **data;
  }

  {
    Expected<Optional<std::vector<uint8_t>>> data =
        m_thread_traces.TryGetBinaryData(request);
    if (!data)
      return data.takeError();
    if (*data)
      return **data;
  }

  return createStringError(
      inconvertibleErrorCode(),
      formatv("Can't fetch data kind {0} for cpu_id {1}, tid {2} and "
              "\"process tracing\" mode {3}",
              request.kind, request.cpu_id, request.tid,
              m_process_trace_up ? "enabled" : "not enabled"));
}

bool IntelPTCollector::IsSupported() {
  if (Expected<uint32_t> intel_pt_type = GetIntelPTOSEventType()) {
    return true;
  } else {
    llvm::consumeError(intel_pt_type.takeError());
    return false;
  }
}

void IntelPTCollector::Clear() {
  m_process_trace_up.reset();
  m_thread_traces.Clear();
}
