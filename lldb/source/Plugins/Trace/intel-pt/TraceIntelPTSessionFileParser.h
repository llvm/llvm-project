//===-- TraceIntelPTSessionFileParser.h -----------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONFILEPARSER_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTSESSIONFILEPARSER_H

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
      : TraceSessionFileParser(debugger, session_file_dir, GetSchema()),
        m_trace_session_file(trace_session_file) {}

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

  lldb::TraceSP
  CreateTraceIntelPTInstance(const pt_cpu &pt_cpu,
                             std::vector<ParsedProcess> &parsed_processes);

private:
  pt_cpu ParsePTCPU(const JSONPTCPU &pt_cpu);

  const llvm::json::Value &m_trace_session_file;
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
