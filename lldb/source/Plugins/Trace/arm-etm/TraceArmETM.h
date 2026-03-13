//===-- TraceArmETM.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETM_H
#define LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETM_H

#include "TraceArmETMBundleLoader.h"
#include "forward-declarations.h"
#include "lldb/Target/Trace.h"

namespace lldb_private {
namespace trace_arm_etm {

class TraceArmETM : public Trace {
public:
  void Dump(lldb_private::Stream *s) const override;

  llvm::Expected<FileSpec> SaveToDisk(FileSpec directory,
                                      bool compact) override;

  /// PluginInterface protocol
  /// \{
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  static void Initialize();

  static void Terminate();

  /// Create an instance of this class from a trace bundle.
  ///
  /// \param[in] trace_bundle_description
  ///     The description of the trace bundle. See \a Trace::FindPlugin.
  ///
  /// \param[in] bundle_dir
  ///     The path to the directory that contains the trace bundle.
  ///
  /// \param[in] debugger
  ///     The debugger instance where new Targets will be created as part of the
  ///     JSON data parsing.
  ///
  /// \return
  ///     A trace instance or an error in case of failures.
  static llvm::Expected<lldb::TraceSP> CreateInstanceForTraceBundle(
      const llvm::json::Value &trace_bundle_description,
      llvm::StringRef bundle_dir, Debugger &debugger);

  static llvm::Expected<lldb::TraceSP>
  CreateInstanceForLiveProcess(Process &process);

  static llvm::StringRef GetPluginNameStatic() { return "arm-etm"; }

  static void DebuggerInitialize(Debugger &debugger);
  /// \}

  lldb::CommandObjectSP
  GetProcessTraceStartCommand(CommandInterpreter &interpreter) override;

  lldb::CommandObjectSP
  GetThreadTraceStartCommand(CommandInterpreter &interpreter) override;

  llvm::StringRef GetSchema() override;

  llvm::Expected<lldb::TraceCursorSP> CreateNewCursor(Thread &thread) override;

  void DumpTraceInfo(Thread &thread, Stream &s, bool verbose,
                     bool json) override;

  llvm::Error DoRefreshLiveProcessState(TraceGetStateResponse state,
                                        llvm::StringRef json_response) override;

  bool IsTraced(lldb::tid_t tid) override;

  const char *GetStartConfigurationHelp() override;

  /// \copydoc Trace::Start
  llvm::Error Start(StructuredData::ObjectSP configuration =
                        StructuredData::ObjectSP()) override;

  /// \copydoc Trace::Start
  llvm::Error Start(llvm::ArrayRef<lldb::tid_t> tids,
                    StructuredData::ObjectSP configuration =
                        StructuredData::ObjectSP()) override;

private:
  friend class TraceArmETMBundleLoader;

  /// Postmortem trace constructor
  ///
  /// \param[in] bundle_description
  ///     The definition file for the postmortem bundle.
  ///
  /// \param[in] traced_processes
  ///     The processes traced in the postmortem session.
  ///
  /// \param[in] trace_threads
  ///     The threads traced in the postmortem session. They must belong to the
  ///     processes mentioned above.
  ///
  /// \param[in] trace_mode
  ///     The tracing mode of the postmortem session.
  ///
  /// \return
  ///     A TraceArmETM shared pointer instance.
  /// \{
  static TraceArmETMSP CreateInstanceForPostmortemTrace(
      JSONTraceBundleDescription &bundle_description,
      llvm::ArrayRef<lldb::ProcessSP> traced_processes,
      llvm::ArrayRef<lldb::ThreadPostMortemTraceSP> traced_threads);

  /// This constructor is used by CreateInstanceForPostmortemTrace to get the
  /// instance ready before using shared pointers, which is a limitation of C++.
  TraceArmETM(JSONTraceBundleDescription &bundle_description,
              llvm::ArrayRef<lldb::ProcessSP> traced_processes);
  /// \}

  /// Constructor for live processes
  TraceArmETM(Process &live_process) : Trace(live_process) {};
};

} // namespace trace_arm_etm
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_ARM_ETM_TRACEARMETM_H
