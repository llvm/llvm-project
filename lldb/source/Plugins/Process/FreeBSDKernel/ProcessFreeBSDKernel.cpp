//===-- ProcessFreeBSDKernel.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/DynamicLoader.h"

#include "ProcessFreeBSDKernel.h"
#include "ThreadFreeBSDKernel.h"
#include "Plugins/DynamicLoader/Static/DynamicLoaderStatic.h"

#include <fvc.h>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ProcessFreeBSDKernel)

ProcessFreeBSDKernel::ProcessFreeBSDKernel(lldb::TargetSP target_sp,
                                           ListenerSP listener_sp,
                                           const FileSpec &core_file, void *fvc)
    : PostMortemProcess(target_sp, listener_sp), m_fvc(fvc) {}

ProcessFreeBSDKernel::~ProcessFreeBSDKernel() {
  if (m_fvc)
    fvc_close(static_cast<fvc_t *>(m_fvc));
}

lldb::ProcessSP ProcessFreeBSDKernel::CreateInstance(lldb::TargetSP target_sp,
                                                     ListenerSP listener_sp,
                                                     const FileSpec *crash_file,
                                                     bool can_connect) {
  lldb::ProcessSP process_sp;
  ModuleSP executable = target_sp->GetExecutableModule();
  if (crash_file && !can_connect && executable) {
    fvc_t *fvc = fvc_open(
        executable->GetFileSpec().GetPath().c_str(),
        crash_file->GetPath().c_str(), nullptr, nullptr, nullptr);
    if (fvc)
      process_sp = std::make_shared<ProcessFreeBSDKernel>(
          target_sp, listener_sp, *crash_file, fvc);
  }
  return process_sp;
}

void ProcessFreeBSDKernel::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

void ProcessFreeBSDKernel::Terminate() {
  PluginManager::UnregisterPlugin(ProcessFreeBSDKernel::CreateInstance);
}

Status ProcessFreeBSDKernel::DoDestroy() { return Status(); }

bool ProcessFreeBSDKernel::CanDebug(lldb::TargetSP target_sp,
                                    bool plugin_specified_by_name) {
  return true;
}

void ProcessFreeBSDKernel::RefreshStateAfterStop() {}

bool ProcessFreeBSDKernel::DoUpdateThreadList(ThreadList &old_thread_list,
                                              ThreadList &new_thread_list) {
  if (old_thread_list.GetSize(false) == 0) {
    // Make up the thread the first time this is called so we can set our one
    // and only core thread state up.

    // We cannot construct a thread without a register context as that crashes
    // LLDB but we can construct a process without threads to provide minimal
    // memory reading support.
    switch (GetTarget().GetArchitecture().GetMachine()) {
    case llvm::Triple::aarch64:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      break;
    default:
      return false;
    }

    const Symbol *pcb_sym =
        GetTarget().GetExecutableModule()->FindFirstSymbolWithNameAndType(
            ConstString("dumppcb"));
    ThreadSP thread_sp(new ThreadFreeBSDKernel(
        *this, 1, pcb_sym ? pcb_sym->GetFileAddress() : LLDB_INVALID_ADDRESS));
    new_thread_list.AddThread(thread_sp);
  } else {
    const uint32_t num_threads = old_thread_list.GetSize(false);
    for (uint32_t i = 0; i < num_threads; ++i)
      new_thread_list.AddThread(old_thread_list.GetThreadAtIndex(i, false));
  }
  return new_thread_list.GetSize(false) > 0;
}

Status ProcessFreeBSDKernel::DoLoadCore() {
  // The core is already loaded by CreateInstance().
  return Status();
}

size_t ProcessFreeBSDKernel::DoReadMemory(lldb::addr_t addr, void *buf,
                                          size_t size, Status &error) {
  ssize_t rd = fvc_read(static_cast<fvc_t *>(m_fvc), addr, buf, size);
  if (rd < 0 || static_cast<size_t>(rd) != size) {
    error.SetErrorStringWithFormat("Reading memory failed: %s",
                                   fvc_geterr(static_cast<fvc_t *>(m_fvc)));
    return rd > 0 ? rd : 0;
  }
  return rd;
}

DynamicLoader *ProcessFreeBSDKernel::GetDynamicLoader() {
  if (m_dyld_up.get() == nullptr)
    m_dyld_up.reset(DynamicLoader::FindPlugin(
        this, DynamicLoaderStatic::GetPluginNameStatic()));
  return m_dyld_up.get();
}
