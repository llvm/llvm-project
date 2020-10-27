//===-- ProcessTrace.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ProcessTrace.h"

#include <memory>

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

ConstString ProcessTrace::GetPluginNameStatic() {
  static ConstString g_name("trace");
  return g_name;
}

const char *ProcessTrace::GetPluginDescriptionStatic() {
  return "Trace process plug-in.";
}

void ProcessTrace::Terminate() {
  PluginManager::UnregisterPlugin(ProcessTrace::CreateInstance);
}

ProcessSP ProcessTrace::CreateInstance(TargetSP target_sp,
                                       ListenerSP listener_sp,
                                       const FileSpec *crash_file) {
  return std::make_shared<ProcessTrace>(target_sp, listener_sp);
}

bool ProcessTrace::CanDebug(TargetSP target_sp, bool plugin_specified_by_name) {
  return plugin_specified_by_name;
}

ProcessTrace::ProcessTrace(TargetSP target_sp, ListenerSP listener_sp)
    : PostMortemProcess(target_sp, listener_sp) {}

ProcessTrace::~ProcessTrace() {
  Clear();
  // We need to call finalize on the process before destroying ourselves to
  // make sure all of the broadcaster cleanup goes as planned. If we destruct
  // this class, then Process::~Process() might have problems trying to fully
  // destroy the broadcaster.
  Finalize();
}

ConstString ProcessTrace::GetPluginName() { return GetPluginNameStatic(); }

uint32_t ProcessTrace::GetPluginVersion() { return 1; }

void ProcessTrace::DidAttach(ArchSpec &process_arch) {
  ListenerSP listener_sp(
      Listener::MakeListener("lldb.process_trace.did_attach_listener"));
  HijackProcessEvents(listener_sp);

  SetCanJIT(false);
  StartPrivateStateThread();
  SetPrivateState(eStateStopped);

  EventSP event_sp;
  WaitForProcessToStop(llvm::None, &event_sp, true, listener_sp);

  RestoreProcessEvents();

  Process::DidAttach(process_arch);
}

bool ProcessTrace::UpdateThreadList(ThreadList &old_thread_list,
                                    ThreadList &new_thread_list) {
  return false;
}

void ProcessTrace::RefreshStateAfterStop() {}

Status ProcessTrace::DoDestroy() { return Status(); }

bool ProcessTrace::IsAlive() { return true; }

size_t ProcessTrace::ReadMemory(addr_t addr, void *buf, size_t size,
                                Status &error) {
  // Don't allow the caching that lldb_private::Process::ReadMemory does since
  // we have it all cached in the trace files.
  return DoReadMemory(addr, buf, size, error);
}

void ProcessTrace::Clear() { m_thread_list.Clear(); }

void ProcessTrace::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

ArchSpec ProcessTrace::GetArchitecture() {
  return GetTarget().GetArchitecture();
}

bool ProcessTrace::GetProcessInfo(ProcessInstanceInfo &info) {
  info.Clear();
  info.SetProcessID(GetID());
  info.SetArchitecture(GetArchitecture());
  ModuleSP module_sp = GetTarget().GetExecutableModule();
  if (module_sp) {
    const bool add_exe_file_as_first_arg = false;
    info.SetExecutableFile(GetTarget().GetExecutableModule()->GetFileSpec(),
                           add_exe_file_as_first_arg);
  }
  return true;
}

size_t ProcessTrace::DoReadMemory(addr_t addr, void *buf, size_t size,
                                  Status &error) {
  Address resolved_address;
  GetTarget().GetSectionLoadList().ResolveLoadAddress(addr, resolved_address);

  return GetTarget().ReadMemoryFromFileCache(resolved_address, buf, size,
                                             error);
}
