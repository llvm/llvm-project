//===-- TraceIntelPTSettingsParser.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTSettingsParser.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

StringRef TraceIntelPTSettingsParser::GetPluginSchema() {
  return R"({
    "type": "intel-pt",
    "pt_cpu": {
      "vendor": "intel" | "unknown",
      "family": integer,
      "model": integer,
      "stepping": integer
    }
  })";
}

void TraceIntelPTSettingsParser::ParsePTCPU(const JSONPTCPU &pt_cpu) {
  m_pt_cpu = {pt_cpu.vendor.compare("intel") == 0 ? pcv_intel : pcv_unknown,
              static_cast<uint16_t>(pt_cpu.family),
              static_cast<uint8_t>(pt_cpu.model),
              static_cast<uint8_t>(pt_cpu.stepping)};
}

llvm::Error TraceIntelPTSettingsParser::ParsePluginSettings(
    const llvm::json::Value &plugin_settings) {
  json::Path::Root root("settings.trace");
  JSONIntelPTSettings settings;
  if (!json::fromJSON(plugin_settings, settings, root))
    return CreateJSONError(root, plugin_settings);

  ParsePTCPU(settings.pt_cpu);

  m_trace.m_pt_cpu = m_pt_cpu;
  return llvm::Error::success();
}
