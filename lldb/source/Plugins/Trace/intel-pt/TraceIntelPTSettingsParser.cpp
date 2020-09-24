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

llvm::Error TraceIntelPTSettingsParser::ParsePTCPU(const json::Object &trace) {
  llvm::Expected<const json::Object &> pt_cpu =
      json_helpers::GetObjectOrError(trace, "pt_cpu");
  if (!pt_cpu)
    return pt_cpu.takeError();

  llvm::Expected<llvm::StringRef> vendor =
      json_helpers::GetStringOrError(*pt_cpu, "vendor");
  if (!vendor)
    return vendor.takeError();

  llvm::Expected<int64_t> family =
      json_helpers::GetIntegerOrError(*pt_cpu, "family");
  if (!family)
    return family.takeError();

  llvm::Expected<int64_t> model =
      json_helpers::GetIntegerOrError(*pt_cpu, "model");
  if (!model)
    return model.takeError();

  llvm::Expected<int64_t> stepping =
      json_helpers::GetIntegerOrError(*pt_cpu, "stepping");
  if (!stepping)
    return stepping.takeError();

  m_pt_cpu = {vendor->compare("intel") == 0 ? pcv_intel : pcv_unknown,
              static_cast<uint16_t>(*family), static_cast<uint8_t>(*model),
              static_cast<uint8_t>(*stepping)};
  return llvm::Error::success();
}

llvm::Error TraceIntelPTSettingsParser::ParsePluginSettings() {
  llvm::Expected<const json::Object &> trace =
      json_helpers::GetObjectOrError(m_settings, "trace");
  if (!trace)
    return trace.takeError();
  if (llvm::Error err = ParsePTCPU(*trace))
    return err;

  m_trace.m_pt_cpu = m_pt_cpu;
  return llvm::Error::success();
}
