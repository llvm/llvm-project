//===-- IntelPTProcessTrace.h --------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTProcessTrace_H_
#define liblldb_IntelPTProcessTrace_H_

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"

#include <memory>

namespace lldb_private {
namespace process_linux {

// Abstract class to be inherited by all the process tracing strategies.
class IntelPTProcessTrace {
public:
  virtual ~IntelPTProcessTrace() = default;

  /// This method should be invoked as early as possible whenever the process
  /// resumes or stops so that intel-pt collection is not enabled when
  /// the process is not running. A case in which this is useful in when
  /// tracing is done per-core. In this case we want to prevent polluting the
  /// core traces with executions of unrelated processes, which increases the
  /// data loss of the target process, given that core traces don't filter by
  /// process.
  /// A possible way to avoid this is to use CR3 filtering, which is equivalent
  /// to process filtering, but the perf_event API doesn't support it.
  ///
  /// \param[in] state
  ///     The new state of the target process.
  virtual void OnProcessStateChanged(lldb::StateType state){};

  /// Construct a minimal jLLDBTraceGetState response for this process trace.
  virtual TraceGetStateResponse GetState() = 0;

  virtual bool TracesThread(lldb::tid_t tid) const = 0;

  /// \copydoc IntelPTThreadTraceCollection::TraceStart()
  virtual llvm::Error TraceStart(lldb::tid_t tid) = 0;

  /// \copydoc IntelPTThreadTraceCollection::TraceStop()
  virtual llvm::Error TraceStop(lldb::tid_t tid) = 0;

  /// Get binary data owned by this instance.
  virtual llvm::Expected<std::vector<uint8_t>>
  GetBinaryData(const TraceGetBinaryDataRequest &request) = 0;
};

using IntelPTProcessTraceUP = std::unique_ptr<IntelPTProcessTrace>;

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTProcessTrace_H_
