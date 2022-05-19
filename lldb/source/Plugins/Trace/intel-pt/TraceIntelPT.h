//===-- TraceIntelPT.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H

#include "TaskTimer.h"
#include "ThreadDecoder.h"
#include "TraceIntelPTSessionFileParser.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/raw_ostream.h"

namespace lldb_private {
namespace trace_intel_pt {

class TraceIntelPT : public Trace {
public:
  void Dump(Stream *s) const override;

  llvm::Error SaveLiveTraceToDisk(FileSpec directory) override;

  ~TraceIntelPT() override = default;

  /// PluginInterface protocol
  /// \{
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

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

  static llvm::StringRef GetPluginNameStatic() { return "intel-pt"; }
  /// \}

  lldb::CommandObjectSP
  GetProcessTraceStartCommand(CommandInterpreter &interpreter) override;

  lldb::CommandObjectSP
  GetThreadTraceStartCommand(CommandInterpreter &interpreter) override;

  llvm::StringRef GetSchema() override;

  lldb::TraceCursorUP GetCursor(Thread &thread) override;

  void DumpTraceInfo(Thread &thread, Stream &s, bool verbose) override;

  llvm::Expected<size_t> GetRawTraceSize(Thread &thread);

  llvm::Error DoRefreshLiveProcessState(TraceGetStateResponse state,
                                        llvm::StringRef json_response) override;

  bool IsTraced(lldb::tid_t tid) override;

  const char *GetStartConfigurationHelp() override;

  /// Start tracing a live process.
  ///
  /// More information on the parameters below can be found in the
  /// jLLDBTraceStart section in lldb/docs/lldb-gdb-remote.txt.
  ///
  /// \param[in] trace_buffer_size
  ///     Trace size per thread in bytes.
  ///
  /// \param[in] total_buffer_size_limit
  ///     Maximum total trace size per process in bytes.
  ///
  /// \param[in] enable_tsc
  ///     Whether to use enable TSC timestamps or not.
  ///
  /// \param[in] psb_period
  ///     This value defines the period in which PSB packets will be generated.
  ///
  /// \param[in] per_core_tracing
  ///     This value defines whether to have a trace buffer per thread or per
  ///     cpu core.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error Start(size_t trace_buffer_size, size_t total_buffer_size_limit,
                    bool enable_tsc, llvm::Optional<size_t> psb_period,
                    bool m_per_core_tracing);

  /// \copydoc Trace::Start
  llvm::Error Start(StructuredData::ObjectSP configuration =
                        StructuredData::ObjectSP()) override;

  /// Start tracing live threads.
  ///
  /// More information on the parameters below can be found in the
  /// jLLDBTraceStart section in lldb/docs/lldb-gdb-remote.txt.
  ///
  /// \param[in] tids
  ///     Threads to trace.
  ///
  /// \param[in] trace_buffer_size
  ///     Trace size per thread or per core in bytes.
  ///
  /// \param[in] enable_tsc
  ///     Whether to use enable TSC timestamps or not.
  ///
  /// \param[in] psb_period
  ///     This value defines the period in which PSB packets will be generated.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error Start(llvm::ArrayRef<lldb::tid_t> tids, size_t trace_buffer_size,
                    bool enable_tsc, llvm::Optional<size_t> psb_period);

  /// \copydoc Trace::Start
  llvm::Error Start(llvm::ArrayRef<lldb::tid_t> tids,
                    StructuredData::ObjectSP configuration =
                        StructuredData::ObjectSP()) override;

  /// See \a Trace::OnThreadBinaryDataRead().
  llvm::Error OnThreadBufferRead(lldb::tid_t tid,
                                 OnBinaryDataReadCallback callback);

  /// Get or fetch the cpu information from, for example, /proc/cpuinfo.
  llvm::Expected<pt_cpu> GetCPUInfo();

  /// Get or fetch the values used to convert to and from TSCs and nanos.
  llvm::Optional<LinuxPerfZeroTscConversion> GetPerfZeroTscConversion();

  /// \return
  ///     The timer object for this trace.
  TaskTimer &GetTimer();

private:
  friend class TraceIntelPTSessionFileParser;

  llvm::Expected<pt_cpu> GetCPUInfoForLiveProcess();

  /// Postmortem trace constructor
  ///
  /// \param[in] session
  ///     The definition file for the postmortem session.
  ///
  /// \param[in] traces_proceses
  ///     The processes traced in the live session.
  ///
  /// \param[in] trace_threads
  ///     The threads traced in the live session. They must belong to the
  ///     processes mentioned above.
  TraceIntelPT(JSONTraceSession &session,
               llvm::ArrayRef<lldb::ProcessSP> traced_processes,
               llvm::ArrayRef<lldb::ThreadPostMortemTraceSP> traced_threads);

  /// Constructor for live processes
  TraceIntelPT(Process &live_process)
      : Trace(live_process), m_thread_decoders(){};

  /// Decode the trace of the given thread that, i.e. recontruct the traced
  /// instructions.
  ///
  /// \param[in] thread
  ///     If \a thread is a \a ThreadTrace, then its internal trace file will be
  ///     decoded. Live threads are not currently supported.
  ///
  /// \return
  ///     A \a DecodedThread shared pointer with the decoded instructions. Any
  ///     errors are embedded in the instruction list.
  DecodedThreadSP Decode(Thread &thread);

  /// It is provided by either a session file or a live process' "cpuInfo"
  /// binary data.
  llvm::Optional<pt_cpu> m_cpu_info;
  std::map<lldb::tid_t, std::unique_ptr<ThreadDecoder>> m_thread_decoders;
  /// Helper variable used to track long running operations for telemetry.
  TaskTimer m_task_timer;
  /// It is provided by either a session file or a live process to convert TSC
  /// counters to and from nanos. It might not be available on all hosts.
  llvm::Optional<LinuxPerfZeroTscConversion> m_tsc_conversion;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H
