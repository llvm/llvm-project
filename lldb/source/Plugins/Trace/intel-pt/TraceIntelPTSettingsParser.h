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
  TraceIntelPTSettingsParser(TraceIntelPT &trace)
      : lldb_private::TraceSettingsParser((lldb_private::Trace &)trace),
        m_trace(trace) {}

protected:
  llvm::StringRef GetPluginSchema() override;

  llvm::Error ParsePluginSettings() override;

private:
  llvm::Error ParsePTCPU(const llvm::json::Object &trace);

  TraceIntelPT &m_trace;
  pt_cpu m_pt_cpu;
};

#endif // liblldb_TraceIntelPTSettingsParser_h_
