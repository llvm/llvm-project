//===-- IntelPTMultiCoreTrace.h ------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTMultiCoreTrace_H_
#define liblldb_IntelPTMultiCoreTrace_H_

#include "IntelPTSingleBufferTrace.h"

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace lldb_private {
namespace process_linux {

class IntelPTMultiCoreTrace;
using IntelPTMultiCoreTraceUP = std::unique_ptr<IntelPTMultiCoreTrace>;

class IntelPTMultiCoreTrace {
public:
  /// Start tracing all CPU cores.
  ///
  /// \param[in] request
  ///   Intel PT configuration parameters.
  ///
  /// \return
  ///   An \a IntelPTMultiCoreTrace instance if tracing was successful, or
  ///   an \a llvm::Error otherwise.
  static llvm::Expected<IntelPTMultiCoreTraceUP>
  StartOnAllCores(const TraceIntelPTStartRequest &request);

  /// Execute the provided callback on each core that is being traced.
  ///
  /// \param[in] callback.core_id
  ///   The core id that is being traced.
  ///
  /// \param[in] callback.core_trace
  ///   The single-buffer trace instance for the given core.
  void ForEachCore(std::function<void(lldb::core_id_t core_id,
                                      IntelPTSingleBufferTrace &core_trace)>
                       callback);

  /// This method should be invoked as early as possible whenever the process
  /// resumes or stops so that intel-pt collection is not enabled when
  /// the process is not running. This is done to prevent polluting the core
  /// traces with executions of unrelated processes, which increases the data
  /// loss of the target process, given that core traces don't filter by
  /// process.
  /// A possible way to avoid this is to use CR3 filtering, which is equivalent
  /// to process filtering, but the perf_event API doesn't support it.
  void OnProcessStateChanged(lldb::StateType state);

private:
  IntelPTMultiCoreTrace(
      llvm::DenseMap<lldb::core_id_t, IntelPTSingleBufferTraceUP>
          &&traces_per_core)
      : m_traces_per_core(std::move(traces_per_core)) {}

  llvm::DenseMap<lldb::core_id_t, IntelPTSingleBufferTraceUP> m_traces_per_core;

  /// The initial state is stopped because tracing can only start when the
  /// process is paused.
  lldb::StateType m_process_state = lldb::StateType::eStateStopped;
};

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTMultiCoreTrace_H_
