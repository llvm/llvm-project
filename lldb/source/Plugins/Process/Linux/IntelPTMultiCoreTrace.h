//===-- IntelPTMultiCoreTrace.h ------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IntelPTMultiCoreTrace_H_
#define liblldb_IntelPTMultiCoreTrace_H_

#include "IntelPTProcessTrace.h"
#include "IntelPTSingleBufferTrace.h"

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace lldb_private {
namespace process_linux {

class IntelPTMultiCoreTrace : public IntelPTProcessTrace {
  using ContextSwitchTrace = PerfEvent;

public:
  /// Start tracing all CPU cores.
  ///
  /// \param[in] request
  ///   Intel PT configuration parameters.
  ///
  /// \param[in] process
  ///   The process being debugged.
  ///
  /// \return
  ///   An \a IntelPTMultiCoreTrace instance if tracing was successful, or
  ///   an \a llvm::Error otherwise.
  static llvm::Expected<IntelPTProcessTraceUP>
  StartOnAllCores(const TraceIntelPTStartRequest &request,
                  NativeProcessProtocol &process);

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

  /// Execute the provided callback on each core that is being traced.
  ///
  /// \param[in] callback.core_id
  ///   The core id that is being traced.
  ///
  /// \param[in] callback.intelpt_trace
  ///   The single-buffer intel pt trace instance for the given core.
  ///
  /// \param[in] callback.context_switch_trace
  ///   The perf event collecting context switches for the given core.
  void ForEachCore(std::function<void(lldb::core_id_t core_id,
                                      IntelPTSingleBufferTrace &intelpt_trace,
                                      PerfEvent &context_switch_trace)>
                       callback);

  void ProcessDidStop() override;

  void ProcessWillResume() override;

  TraceIntelPTGetStateResponse GetState() override;

  bool TracesThread(lldb::tid_t tid) const override;

  llvm::Error TraceStart(lldb::tid_t tid) override;

  llvm::Error TraceStop(lldb::tid_t tid) override;

  llvm::Expected<llvm::Optional<std::vector<uint8_t>>>
  TryGetBinaryData(const TraceGetBinaryDataRequest &request) override;

private:
  /// This assumes that all underlying perf_events for each core are part of the
  /// same perf event group.
  IntelPTMultiCoreTrace(
      llvm::DenseMap<lldb::core_id_t,
                     std::pair<IntelPTSingleBufferTrace, ContextSwitchTrace>>
          &&traces_per_core,
      NativeProcessProtocol &process)
      : m_traces_per_core(std::move(traces_per_core)), m_process(process) {}

  llvm::DenseMap<lldb::core_id_t,
                 std::pair<IntelPTSingleBufferTrace, ContextSwitchTrace>>
      m_traces_per_core;

  /// The target process.
  NativeProcessProtocol &m_process;
};

} // namespace process_linux
} // namespace lldb_private

#endif // liblldb_IntelPTMultiCoreTrace_H_
