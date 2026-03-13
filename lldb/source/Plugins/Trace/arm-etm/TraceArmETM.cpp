//===-- TraceArmETM.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceArmETM.h"

#include "TraceArmETMBundleLoader.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_arm_etm;
using namespace llvm;

LLDB_PLUGIN_DEFINE(TraceArmETM)

lldb::CommandObjectSP
TraceArmETM::GetProcessTraceStartCommand(CommandInterpreter &interpreter) {
  llvm_unreachable("Unimplemented");
}

lldb::CommandObjectSP
TraceArmETM::GetThreadTraceStartCommand(CommandInterpreter &interpreter) {
  llvm_unreachable("Unimplemented");
}

void TraceArmETM::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "Arm ETM", CreateInstanceForTraceBundle,
      CreateInstanceForLiveProcess, TraceArmETMBundleLoader::GetSchema(),
      DebuggerInitialize);
}

void TraceArmETM::DebuggerInitialize(Debugger &debugger) {}

void TraceArmETM::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstanceForTraceBundle);
}

StringRef TraceArmETM::GetSchema() {
  return TraceArmETMBundleLoader::GetSchema();
}

void TraceArmETM::Dump(Stream *s) const {}

Expected<FileSpec> TraceArmETM::SaveToDisk(FileSpec directory, bool compact) {
  llvm_unreachable("Unimplemented");
}

Expected<TraceSP>
TraceArmETM::CreateInstanceForTraceBundle(const json::Value &bundle_description,
                                          StringRef bundle_dir,
                                          Debugger &debugger) {
  return TraceArmETMBundleLoader(debugger, bundle_description, bundle_dir)
      .Load();
}

Expected<TraceSP> TraceArmETM::CreateInstanceForLiveProcess(Process &process) {
  TraceSP instance(new TraceArmETM(process));
  process.GetTarget().SetTrace(instance);
  return instance;
}

TraceArmETMSP TraceArmETM::CreateInstanceForPostmortemTrace(
    JSONTraceBundleDescription &bundle_description,
    ArrayRef<ProcessSP> traced_processes,
    ArrayRef<ThreadPostMortemTraceSP> traced_threads) {
  TraceArmETMSP trace_sp(new TraceArmETM(bundle_description, traced_processes));

  for (const ProcessSP &process_sp : traced_processes)
    process_sp->GetTarget().SetTrace(trace_sp);
  return trace_sp;
}

TraceArmETM::TraceArmETM(JSONTraceBundleDescription &bundle_description,
                         llvm::ArrayRef<lldb::ProcessSP> traced_processes)
    : Trace(traced_processes, std::nullopt) {}

llvm::Expected<lldb::TraceCursorSP>
TraceArmETM::CreateNewCursor(Thread &thread) {
  llvm_unreachable("Unimplemented");
}

void TraceArmETM::DumpTraceInfo(Thread &thread, Stream &s, bool verbose,
                                bool json) {}

Error TraceArmETM::DoRefreshLiveProcessState(TraceGetStateResponse state,
                                             StringRef json_response) {
  llvm_unreachable("Unimplemented");
}

bool TraceArmETM::IsTraced(lldb::tid_t tid) { return false; }

const char *TraceArmETM::GetStartConfigurationHelp() {
  llvm_unreachable("Unimplemented");
}

Error TraceArmETM::Start(StructuredData::ObjectSP configuration) {
  llvm_unreachable("Unimplemented");
}

Error TraceArmETM::Start(llvm::ArrayRef<lldb::tid_t> tids,
                         StructuredData::ObjectSP configuration) {
  llvm_unreachable("Unimplemented");
}
