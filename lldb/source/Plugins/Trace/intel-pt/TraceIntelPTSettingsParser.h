//===-- TraceIntelPTSettingsParser.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TraceIntelPTSettingsParser_h_
#define liblldb_TraceIntelPTSettingsParser_h_

#include "intel-pt.h"

#include "TraceIntelPT.h"
#include "lldb/Target/TraceSettingsParser.h"
#include "lldb/Utility/StructuredData.h"

class TraceIntelPT;

class TraceIntelPTSettingsParser : public lldb_private::TraceSettingsParser {
public:
  struct JSONPTCPU {
    std::string vendor;
    int64_t family;
    int64_t model;
    int64_t stepping;
  };

  struct JSONIntelPTSettings {
    JSONPTCPU pt_cpu;
  };

  TraceIntelPTSettingsParser(TraceIntelPT &trace)
      : lldb_private::TraceSettingsParser((lldb_private::Trace &)trace),
        m_trace(trace) {}

protected:
  llvm::StringRef GetPluginSchema() override;

  llvm::Error
  ParsePluginSettings(const llvm::json::Value &plugin_settings) override;

private:
  void ParsePTCPU(const JSONPTCPU &pt_cpu);

  TraceIntelPT &m_trace;
  pt_cpu m_pt_cpu;
};

namespace llvm {
namespace json {

inline bool fromJSON(const llvm::json::Value &value,
                     TraceIntelPTSettingsParser::JSONPTCPU &pt_cpu,
                     llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("vendor", pt_cpu.vendor) &&
         o.map("family", pt_cpu.family) && o.map("model", pt_cpu.model) &&
         o.map("stepping", pt_cpu.stepping);
}

inline bool
fromJSON(const llvm::json::Value &value,
         TraceIntelPTSettingsParser::JSONIntelPTSettings &intel_pt_settings,
         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("pt_cpu", intel_pt_settings.pt_cpu);
}

} // namespace json
} // namespace llvm

#endif // liblldb_TraceIntelPTSettingsParser_h_
