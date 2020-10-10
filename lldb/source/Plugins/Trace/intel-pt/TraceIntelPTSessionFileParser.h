//===-- TraceIntelPTSessionFileParser.h -----------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONFILEPARSER_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONFILEPARSER_H

#include "intel-pt.h"

#include "TraceIntelPT.h"
#include "lldb/Target/TraceSessionFileParser.h"

namespace lldb_private {
namespace trace_intel_pt {

class TraceIntelPT;

class TraceIntelPTSessionFileParser : public TraceSessionFileParser {
public:
  struct JSONPTCPU {
    std::string vendor;
    int64_t family;
    int64_t model;
    int64_t stepping;
  };

  struct JSONTraceIntelPTSettings
      : TraceSessionFileParser::JSONTracePluginSettings {
    JSONPTCPU pt_cpu;
  };

  /// See \a TraceSessionFileParser::TraceSessionFileParser for the description
  /// of these fields.
  TraceIntelPTSessionFileParser(Debugger &debugger,
                                const llvm::json::Value &trace_session_file,
                                llvm::StringRef session_file_dir)
      : TraceSessionFileParser(session_file_dir, GetSchema()),
        m_debugger(debugger), m_trace_session_file(trace_session_file) {}

  /// \return
  ///   The JSON schema for the session data.
  static llvm::StringRef GetSchema();

  /// Parse the structured data trace session and create the corresponding \a
  /// Target objects. In case of an error, no targets are created.
  ///
  /// \return
  ///   A \a lldb::TraceSP instance with the trace session data. In case of
  ///   errors, return a null pointer.
  llvm::Expected<lldb::TraceSP> Parse();

private:
  llvm::Error ParseImpl();

  llvm::Error ParseProcess(const TraceSessionFileParser::JSONProcess &process);

  void ParseThread(lldb::ProcessSP &process_sp,
                   const TraceSessionFileParser::JSONThread &thread);

  void ParsePTCPU(const JSONPTCPU &pt_cpu);

  Debugger &m_debugger;
  const llvm::json::Value &m_trace_session_file;

  /// Objects created as product of the parsing
  /// \{
  pt_cpu m_pt_cpu;
  std::vector<lldb::TargetSP> m_targets;
  /// \}
};

} // namespace trace_intel_pt
} // namespace lldb_private

namespace llvm {
namespace json {

bool fromJSON(
    const Value &value,
    lldb_private::trace_intel_pt::TraceIntelPTSessionFileParser::JSONPTCPU
        &pt_cpu,
    Path path);

bool fromJSON(const Value &value,
              lldb_private::trace_intel_pt::TraceIntelPTSessionFileParser::
                  JSONTraceIntelPTSettings &plugin_settings,
              Path path);

} // namespace json
} // namespace llvm

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONFILEPARSER_H
