//===-- TraceIntelPT.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPT.h"

#include "TraceIntelPTSessionFileParser.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

LLDB_PLUGIN_DEFINE(TraceIntelPT)

void TraceIntelPT::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Intel Processor Trace",
                                CreateInstance,
                                TraceIntelPTSessionFileParser::GetSchema());
}

void TraceIntelPT::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ConstString TraceIntelPT::GetPluginNameStatic() {
  static ConstString g_name("intel-pt");
  return g_name;
}

StringRef TraceIntelPT::GetSchema() {
  return TraceIntelPTSessionFileParser::GetSchema();
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------

ConstString TraceIntelPT::GetPluginName() { return GetPluginNameStatic(); }

uint32_t TraceIntelPT::GetPluginVersion() { return 1; }

void TraceIntelPT::Dump(lldb_private::Stream *s) const {}

Expected<lldb::TraceSP>
TraceIntelPT::CreateInstance(const json::Value &trace_session_file,
                             StringRef session_file_dir, Debugger &debugger) {
  return TraceIntelPTSessionFileParser(debugger, trace_session_file,
                                       session_file_dir)
      .Parse();
}
