//===-- TraceIntelPT.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPT.h"

#include "TraceIntelPTSettingsParser.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

LLDB_PLUGIN_DEFINE_ADV(TraceIntelPT, TraceIntelPT)

void TraceIntelPT::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Intel Processor Trace",
                                CreateInstance);
}

void TraceIntelPT::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ConstString TraceIntelPT::GetPluginNameStatic() {
  static ConstString g_name("intel-pt");
  return g_name;
}

std::unique_ptr<lldb_private::TraceSettingsParser>
TraceIntelPT::CreateParser() {
  return std::make_unique<TraceIntelPTSettingsParser>(*this);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------

ConstString TraceIntelPT::GetPluginName() { return GetPluginNameStatic(); }

uint32_t TraceIntelPT::GetPluginVersion() { return 1; }

void TraceIntelPT::Dump(lldb_private::Stream *s) const {}

lldb::TraceSP TraceIntelPT::CreateInstance() {
  return lldb::TraceSP(new TraceIntelPT());
}
