//===-- TraceIntelPT.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H

#include "IntelPTDecoder.h"
#include "TraceIntelPTSessionFileParser.h"

namespace lldb_private {
namespace trace_intel_pt {

class TraceIntelPT : public Trace {
public:
  void Dump(Stream *s) const override;

  ~TraceIntelPT() override = default;

  /// PluginInterface protocol
  /// \{
  ConstString GetPluginName() override;

  static void Initialize();

  static void Terminate();

  /// Create an instance of this class.
  ///
  /// \param[in] trace_session_file
  ///     The contents of the trace session file. See \a Trace::FindPlugin.
  ///
  /// \param[in] session_file_dir
  ///     The path to the directory that contains the session file. It's used to
  ///     resolved relative paths in the session file.
  ///
  /// \param[in] debugger
  ///     The debugger instance where new Targets will be created as part of the
  ///     JSON data parsing.
  ///
  /// \return
  ///     A trace instance or an error in case of failures.
  static llvm::Expected<lldb::TraceSP>
  CreateInstanceForSessionFile(const llvm::json::Value &trace_session_file,
                               llvm::StringRef session_file_dir,
                               Debugger &debugger);

  static llvm::Expected<lldb::TraceSP>
  CreateInstanceForLiveProcess(Process &process);

  static ConstString GetPluginNameStatic();

  uint32_t GetPluginVersion() override;
  /// \}

  lldb::CommandObjectSP
  GetProcessTraceStartCommand(CommandInterpreter &interpreter) override;

  lldb::CommandObjectSP
  GetThreadTraceStartCommand(CommandInterpreter &interpreter) override;

  llvm::StringRef GetSchema() override;

  void TraverseInstructions(
      Thread &thread, size_t position, TraceDirection direction,
      std::function<bool(size_t index, llvm::Expected<lldb::addr_t> load_addr)>
          callback) override;

  llvm::Optional<size_t> GetInstructionCount(Thread &thread) override;

  size_t GetCursorPosition(Thread &thread) override;

  void DoRefreshLiveProcessState(
      llvm::Expected<TraceGetStateResponse> state) override;

  bool IsTraced(const Thread &thread) override;

  /// Start tracing a live process.
  ///
  /// \param[in] thread_buffer_size
  ///     Trace size per thread in bytes.
  ///
  /// \param[in] total_buffer_size_limit
  ///     Maximum total trace size per process in bytes. This limit applies to
  ///     the sum of the sizes of all thread traces of this process, excluding
  ///     the threads traced explicitly.
  ///
  ///     Whenever a thread is attempted to be traced due to this operation and
  ///     the limit would be reached, the process is stopped with a "tracing"
  ///     reason, so that the user can retrace the process if needed.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error Start(size_t thread_buffer_size, size_t total_buffer_size_limit);

  /// Start tracing a live threads.
  ///
  /// \param[in] tids
  ///     Threads to trace.
  ///
  /// \param[in] thread_buffer_size
  ///     Trace size per thread in bytes.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error Start(const std::vector<lldb::tid_t> &tids,
                    size_t thread_buffer_size);

  /// Get the thread buffer content for a live thread
  llvm::Expected<std::vector<uint8_t>> GetLiveThreadBuffer(lldb::tid_t tid);

  llvm::Expected<pt_cpu> GetCPUInfo();

private:
  friend class TraceIntelPTSessionFileParser;

  llvm::Expected<pt_cpu> GetCPUInfoForLiveProcess();

  /// \param[in] trace_threads
  ///     ThreadTrace instances, which are not live-processes and whose trace
  ///     files are fixed.
  TraceIntelPT(
      const pt_cpu &cpu_info,
      const std::vector<lldb::ThreadPostMortemTraceSP> &traced_threads);

  /// Constructor for live processes
  TraceIntelPT(Process &live_process)
      : Trace(live_process), m_thread_decoders(){};

  /// Decode the trace of the given thread that, i.e. recontruct the traced
  /// instructions. That trace must be managed by this class.
  ///
  /// \param[in] thread
  ///     If \a thread is a \a ThreadTrace, then its internal trace file will be
  ///     decoded. Live threads are not currently supported.
  ///
  /// \return
  ///     A \a DecodedThread instance if decoding was successful, or a \b
  ///     nullptr if the thread's trace is not managed by this class.
  const DecodedThread *Decode(Thread &thread);

  /// It is provided by either a session file or a live process' "cpuInfo"
  /// binary data.
  llvm::Optional<pt_cpu> m_cpu_info;
  std::map<const Thread *, std::unique_ptr<ThreadDecoder>> m_thread_decoders;
  /// Dummy DecodedThread used when decoding threads after there were errors
  /// when refreshing the live process state.
  llvm::Optional<DecodedThread> m_failed_live_threads_decoder;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H
