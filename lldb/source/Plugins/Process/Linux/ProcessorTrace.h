//===-- ProcessorTrace.h -------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessorTrace_H_
#define liblldb_ProcessorTrace_H_

#include "lldb/Utility/Status.h"
#include "lldb/Utility/TraceOptions.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <linux/perf_event.h>
#include <sys/mman.h>
#include <unistd.h>

namespace lldb_private {

namespace process_linux {

// This class keeps track of one tracing instance of
// Intel(R) Processor Trace on Linux OS. There is a map keeping track
// of different tracing instances on each thread, which enables trace
// gathering on a per thread level.
//
// The tracing instance is linked with a trace id. The trace id acts like
// a key to the tracing instance and trace manipulations could be
// performed using the trace id.
//
// The trace id could map to trace instances for a group of threads
// (spanning to all the threads in the process) or a single thread.
// The kernel interface for us is the perf_event_open.

class ProcessorTraceMonitor;
typedef std::unique_ptr<ProcessorTraceMonitor> ProcessorTraceMonitorUP;

class ProcessorTraceMonitor {

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

  // perf_event_mmap_page *m_mmap_base;
  lldb::user_id_t m_traceid;
  lldb::tid_t m_thread_id;

  // Counter to track trace instances.
  static lldb::user_id_t m_trace_num;

  void SetTraceID(lldb::user_id_t traceid) { m_traceid = traceid; }

  Status StartTrace(lldb::pid_t pid, lldb::tid_t tid,
                    const TraceOptions &config);

  llvm::MutableArrayRef<uint8_t> GetAuxBuffer();
  llvm::MutableArrayRef<uint8_t> GetDataBuffer();

  ProcessorTraceMonitor()
      : m_mmap_meta(nullptr, munmap_delete(0)),
        m_mmap_aux(nullptr, munmap_delete(0)), m_fd(nullptr, file_close()),
        m_traceid(LLDB_INVALID_UID), m_thread_id(LLDB_INVALID_THREAD_ID){};

  void SetThreadID(lldb::tid_t tid) { m_thread_id = tid; }

public:
  static llvm::Expected<uint32_t> GetOSEventType();

  static bool IsSupported();

  static Status GetCPUType(TraceOptions &config);

  static llvm::Expected<ProcessorTraceMonitorUP>
  Create(lldb::pid_t pid, lldb::tid_t tid, const TraceOptions &config,
         bool useProcessSettings);

  Status ReadPerfTraceAux(llvm::MutableArrayRef<uint8_t> &buffer,
                          size_t offset = 0);

  Status ReadPerfTraceData(llvm::MutableArrayRef<uint8_t> &buffer,
                           size_t offset = 0);

  ~ProcessorTraceMonitor() = default;

  lldb::tid_t GetThreadID() const { return m_thread_id; }

  lldb::user_id_t GetTraceID() const { return m_traceid; }

  Status GetTraceConfig(TraceOptions &config) const;

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
};
} // namespace process_linux
} // namespace lldb_private
#endif
