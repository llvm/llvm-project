//===-- IntelPTSingleBufferTrace.h ---------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTSingleBufferTrace_H_
#define liblldb_IntelPTSingleBufferTrace_H_

#include "Perf.h"

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace lldb_private {
namespace process_linux {

llvm::Expected<uint32_t> GetIntelPTOSEventType();

class IntelPTTrace;
class IntelPTSingleBufferTrace;

using IntelPTThreadTraceUP = std::unique_ptr<IntelPTTrace>;
using IntelPTSingleBufferTraceUP = std::unique_ptr<IntelPTSingleBufferTrace>;

/// This class wraps a single perf event collecting intel pt data in a single
/// buffer.
class IntelPTSingleBufferTrace {
public:
  /// Start tracing using a single Intel PT trace buffer.
  ///
  /// \param[in] request
  ///     Intel PT configuration parameters.
  ///
  /// \param[in] tid
  ///     The tid of the thread to be traced.
  ///
  /// \return
  ///   A \a IntelPTSingleBufferTrace instance if tracing was successful, or
  ///   an \a llvm::Error otherwise.
  static llvm::Expected<IntelPTSingleBufferTraceUP>
  Start(const TraceIntelPTStartRequest &request, lldb::tid_t tid);

  /// \return
  ///    The bytes requested by a jLLDBTraceGetBinaryData packet that was routed
  ///    to this trace instace.
  llvm::Expected<std::vector<uint8_t>>
  GetBinaryData(const TraceGetBinaryDataRequest &request) const;

  /// Read the trace buffer managed by this trace instance.
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
  llvm::Expected<std::vector<uint8_t>> GetTraceBuffer(size_t offset,
                                                      size_t size) const;

  /// \return
  ///   The total the size in bytes used by the trace buffer managed by this
  ///   trace instance.
  size_t GetTraceBufferSize() const;

private:
  /// Construct new \a IntelPTSingleBufferThreadTrace. Users are supposed to
  /// create instances of this class via the \a Start() method and not invoke
  /// this one directly.
  ///
  /// \param[in] perf_event
  ///   perf event configured for IntelPT.
  ///
  /// \param[in] tid
  ///   The thread being traced.
  IntelPTSingleBufferTrace(PerfEvent &&perf_event, lldb::tid_t tid)
      : m_perf_event(std::move(perf_event)) {}

  /// perf event configured for IntelPT.
  PerfEvent m_perf_event;
};

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTSingleBufferTrace_H_
