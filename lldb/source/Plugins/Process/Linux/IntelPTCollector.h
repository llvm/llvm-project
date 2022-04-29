//===-- IntelPTCollector.h -------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTCollector_H_
#define liblldb_IntelPTCollector_H_

#include "Perf.h"

#include "IntelPTSingleBufferTrace.h"

#include "lldb/Utility/Status.h"
#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"

#include <linux/perf_event.h>
#include <sys/mman.h>
#include <unistd.h>

namespace lldb_private {

namespace process_linux {

/// Manages a list of thread traces.
class IntelPTThreadTraceCollection {
public:
  IntelPTThreadTraceCollection() {}

  /// Dispose of all traces
  void Clear();

  bool TracesThread(lldb::tid_t tid) const;

  size_t GetTotalBufferSize() const;

  std::vector<TraceThreadState> GetThreadStates() const;

  llvm::Expected<const IntelPTSingleBufferTrace &>
  GetTracedThread(lldb::tid_t tid) const;

  llvm::Error TraceStart(lldb::tid_t tid,
                         const TraceIntelPTStartRequest &request);

  llvm::Error TraceStop(lldb::tid_t tid);

private:
  llvm::DenseMap<lldb::tid_t, IntelPTSingleBufferTraceUP> m_thread_traces;
  /// Total actual thread buffer size in bytes
  size_t m_total_buffer_size = 0;
};

/// Manages a "process trace" instance.
class IntelPTProcessTrace {
public:
  IntelPTProcessTrace(const TraceIntelPTStartRequest &request)
      : m_tracing_params(request) {}

  bool TracesThread(lldb::tid_t tid) const;

  const IntelPTThreadTraceCollection &GetThreadTraces() const;

  llvm::Error TraceStart(lldb::tid_t tid);

  llvm::Error TraceStop(lldb::tid_t tid);

private:
  IntelPTThreadTraceCollection m_thread_traces;
  /// Params used to trace threads when the user started "process tracing".
  TraceIntelPTStartRequest m_tracing_params;
};

/// Main class that manages intel-pt process and thread tracing.
class IntelPTCollector {
public:
  IntelPTCollector();

  static bool IsSupported();

  /// If "process tracing" is enabled, then trace the given thread.
  llvm::Error OnThreadCreated(lldb::tid_t tid);

  /// Stops tracing a tracing upon a destroy event.
  llvm::Error OnThreadDestroyed(lldb::tid_t tid);

  /// Implementation of the jLLDBTraceStop packet
  llvm::Error TraceStop(const TraceStopRequest &request);

  /// Implementation of the jLLDBTraceStart packet
  ///
  /// \param[in] process_threads
  ///     A list of all threads owned by the process.
  llvm::Error TraceStart(const TraceIntelPTStartRequest &request,
                         const std::vector<lldb::tid_t> &process_threads);

  /// Implementation of the jLLDBTraceGetState packet
  llvm::Expected<llvm::json::Value> GetState() const;

  /// Implementation of the jLLDBTraceGetBinaryData packet
  llvm::Expected<std::vector<uint8_t>>
  GetBinaryData(const TraceGetBinaryDataRequest &request) const;

  /// Dispose of all traces
  void Clear();

private:
  llvm::Error TraceStop(lldb::tid_t tid);

  /// Start tracing a specific thread.
  llvm::Error TraceStart(lldb::tid_t tid,
                         const TraceIntelPTStartRequest &request);

  llvm::Expected<const IntelPTSingleBufferTrace &>
  GetTracedThread(lldb::tid_t tid) const;

  bool IsProcessTracingEnabled() const;

  void ClearProcessTracing();

  /// Threads traced due to "thread tracing"
  IntelPTThreadTraceCollection m_thread_traces;
  /// Threads traced due to "process tracing". Only one active "process tracing"
  /// instance is assumed for a single process.
  llvm::Optional<IntelPTProcessTrace> m_process_trace;
  /// TSC to wall time conversion.
  TraceTscConversionUP m_tsc_conversion;
};

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTCollector_H_
