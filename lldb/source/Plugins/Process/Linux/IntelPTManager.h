//===-- IntelPTManager.h -------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTManager_H_
#define liblldb_IntelPTManager_H_

#include "lldb/Utility/Status.h"
#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <linux/perf_event.h>
#include <sys/mman.h>
#include <unistd.h>

namespace lldb_private {

namespace process_linux {

/// This class keeps track of one tracing instance of
/// Intel(R) Processor Trace on Linux OS at thread level.
///
/// The kernel interface for us is the perf_event_open.
class IntelPTThreadTrace;
typedef std::unique_ptr<IntelPTThreadTrace> IntelPTThreadTraceUP;

class IntelPTThreadTrace {

  class munmap_delete {
    size_t m_length;

  public:
    munmap_delete(size_t length) : m_length(length) {}
    void operator()(void *ptr) {
      if (m_length)
        munmap(ptr, m_length);
    }
  };

  class file_close {

  public:
    file_close() = default;
    void operator()(int *ptr) {
      if (ptr == nullptr)
        return;
      if (*ptr == -1)
        return;
      close(*ptr);
      std::default_delete<int>()(ptr);
    }
  };

  std::unique_ptr<perf_event_mmap_page, munmap_delete> m_mmap_meta;
  std::unique_ptr<uint8_t, munmap_delete> m_mmap_aux;
  std::unique_ptr<int, file_close> m_fd;
  lldb::tid_t m_tid;

  /// Start tracing a thread
  ///
  /// \param[in] pid
  ///     The pid of the process whose thread will be traced.
  ///
  /// \param[in] buffer_size
  ///     Size of the thread buffer in bytes.
  ///
  /// \param[in] enable_tsc
  ///     Whether to use enable TSC timestamps or not.
  ///     More information in TraceIntelPT::GetStartConfigurationHelp().
  ///
  /// \param[in] psb_period
  ///     This value defines the period in which PSB packets will be generated.
  ///     More information in TraceIntelPT::GetStartConfigurationHelp().
  ///
  /// \return
  ///     \a llvm::Error::success if tracing was successful, or an
  ///     \a llvm::Error otherwise.
  llvm::Error StartTrace(lldb::pid_t pid, lldb::tid_t tid, uint64_t buffer_size,
                         bool enable_tsc, llvm::Optional<size_t> psb_period);

  llvm::MutableArrayRef<uint8_t> GetAuxBuffer() const;
  llvm::MutableArrayRef<uint8_t> GetDataBuffer() const;

  IntelPTThreadTrace()
      : m_mmap_meta(nullptr, munmap_delete(0)),
        m_mmap_aux(nullptr, munmap_delete(0)), m_fd(nullptr, file_close()) {}

public:
  /// Get the content of /proc/cpuinfo that can be later used to decode traces.
  static llvm::Expected<llvm::ArrayRef<uint8_t>> GetCPUInfo();

  /// Start tracing a thread.
  ///
  /// See \a StartTrace.
  ///
  /// \return
  ///   A \a IntelPTThreadTrace instance if tracing was successful, or
  ///   an \a llvm::Error otherwise.
  static llvm::Expected<IntelPTThreadTraceUP>
  Create(lldb::pid_t pid, lldb::tid_t tid, size_t buffer_size, bool enable_tsc,
         llvm::Optional<size_t> psb_period);

  /// Read the trace buffer of the currently traced thread.
  ///
  /// \param[in] offset
  ///     Offset of the data to read.
  ///
  /// \param[in] size
  ///     Number of bytes to read.
  ///
  /// \return
  ///     A vector with the requested binary data. The vector will have the
  ///     size of the requested \a size. Non-available positions will be
  ///     filled with zeroes.
  llvm::Expected<std::vector<uint8_t>> GetIntelPTBuffer(size_t offset,
                                                        size_t size) const;

  Status ReadPerfTraceAux(llvm::MutableArrayRef<uint8_t> &buffer,
                          size_t offset = 0) const;

  Status ReadPerfTraceData(llvm::MutableArrayRef<uint8_t> &buffer,
                           size_t offset = 0) const;

  /// Get the size in bytes of the aux section of the thread or process traced
  /// by this object.
  size_t GetTraceBufferSize() const;

  /// Read data from a cyclic buffer
  ///
  /// \param[in] [out] buf
  ///     Destination buffer, the buffer will be truncated to written size.
  ///
  /// \param[in] src
  ///     Source buffer which must be a cyclic buffer.
  ///
  /// \param[in] src_cyc_index
  ///     The index pointer (start of the valid data in the cyclic
  ///     buffer).
  ///
  /// \param[in] offset
  ///     The offset to begin reading the data in the cyclic buffer.
  static void ReadCyclicBuffer(llvm::MutableArrayRef<uint8_t> &dst,
                               llvm::MutableArrayRef<uint8_t> src,
                               size_t src_cyc_index, size_t offset);

  /// Return the thread-specific part of the jLLDBTraceGetState packet.
  TraceThreadState GetState() const;
};

/// Manages a list of thread traces.
class IntelPTThreadTraceCollection {
public:
  IntelPTThreadTraceCollection(lldb::pid_t pid) : m_pid(pid) {}

  /// Dispose of all traces
  void Clear();

  bool TracesThread(lldb::tid_t tid) const;

  size_t GetTotalBufferSize() const;

  std::vector<TraceThreadState> GetThreadStates() const;

  llvm::Expected<const IntelPTThreadTrace &>
  GetTracedThread(lldb::tid_t tid) const;

  llvm::Error TraceStart(lldb::tid_t tid,
                         const TraceIntelPTStartRequest &request);

  llvm::Error TraceStop(lldb::tid_t tid);

private:
  lldb::pid_t m_pid;
  llvm::DenseMap<lldb::tid_t, IntelPTThreadTraceUP> m_thread_traces;
  /// Total actual thread buffer size in bytes
  size_t m_total_buffer_size = 0;
};

/// Manages a "process trace" instance.
class IntelPTProcessTrace {
public:
  IntelPTProcessTrace(lldb::pid_t pid, const TraceIntelPTStartRequest &request)
      : m_thread_traces(pid), m_tracing_params(request) {}

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
class IntelPTManager {
public:
  IntelPTManager(lldb::pid_t pid) : m_pid(pid), m_thread_traces(pid) {}

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

  llvm::Expected<const IntelPTThreadTrace &>
  GetTracedThread(lldb::tid_t tid) const;

  bool IsProcessTracingEnabled() const;

  void ClearProcessTracing();

  lldb::pid_t m_pid;
  /// Threads traced due to "thread tracing"
  IntelPTThreadTraceCollection m_thread_traces;
  /// Threads traced due to "process tracing". Only one active "process tracing"
  /// instance is assumed for a single process.
  llvm::Optional<IntelPTProcessTrace> m_process_trace;
};

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTManager_H_
