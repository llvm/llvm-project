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
  CreateInstance(const llvm::json::Value &trace_session_file,
                 llvm::StringRef session_file_dir, Debugger &debugger);

  static ConstString GetPluginNameStatic();

  uint32_t GetPluginVersion() override;
  /// \}

  llvm::StringRef GetSchema() override;

  void TraverseInstructions(
      const Thread &thread, size_t position, TraceDirection direction,
      std::function<bool(size_t index, llvm::Expected<lldb::addr_t> load_addr)>
          callback) override;

  size_t GetInstructionCount(const Thread &thread) override;

  size_t GetCursorPosition(const Thread &thread) override;

private:
  friend class TraceIntelPTSessionFileParser;

  /// \param[in] trace_threads
  ///     ThreadTrace instances, which are not live-processes and whose trace
  ///     files are fixed.
  TraceIntelPT(const pt_cpu &pt_cpu,
               const std::vector<std::shared_ptr<ThreadTrace>> &traced_threads);

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
  const DecodedThread *Decode(const Thread &thread);

  pt_cpu m_pt_cpu;
  std::map<std::pair<lldb::pid_t, lldb::tid_t>, ThreadTraceDecoder>
      m_trace_threads;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPT_H
