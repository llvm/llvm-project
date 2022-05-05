//===-- IntelPTPerThreadProcessTrace.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntelPTPerThreadProcessTrace.h"

using namespace lldb;
using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

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

IntelPTThreadTraceCollection &IntelPTPerThreadProcessTrace::GetThreadTraces() {
  return m_thread_traces;
}
