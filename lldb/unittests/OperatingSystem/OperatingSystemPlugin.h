//===-- OperatingSystemPlugin.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/OperatingSystem.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadList.h"

/// An operating system plugin that does nothing: simply keeps the thread lists
/// as they are.
class OperatingSystemIdentityMap : public lldb_private::OperatingSystem {
public:
  OperatingSystemIdentityMap(lldb_private::Process *process)
      : OperatingSystem(process) {}

  static OperatingSystem *CreateInstance(lldb_private::Process *process,
                                         bool force) {
    return new OperatingSystemIdentityMap(process);
  }
  static llvm::StringRef GetPluginNameStatic() { return "identity map"; }
  static llvm::StringRef GetPluginDescriptionStatic() { return ""; }

  static void Initialize() {
    lldb_private::PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                                GetPluginDescriptionStatic(),
                                                CreateInstance, nullptr);
  }
  static void Terminate() {
    lldb_private::PluginManager::UnregisterPlugin(CreateInstance);
  }
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // Simply adds the threads from real_thread_list into new_thread_list.
  bool UpdateThreadList(lldb_private::ThreadList &old_thread_list,
                        lldb_private::ThreadList &real_thread_list,
                        lldb_private::ThreadList &new_thread_list) override {
    for (const auto &real_thread : real_thread_list.Threads())
      new_thread_list.AddThread(real_thread);
    return true;
  }

  void ThreadWasSelected(lldb_private::Thread *thread) override {}

  lldb::RegisterContextSP
  CreateRegisterContextForThread(lldb_private::Thread *thread,
                                 lldb::addr_t reg_data_addr) override {
    return thread->GetRegisterContext();
  }

  lldb::StopInfoSP
  CreateThreadStopReason(lldb_private::Thread *thread) override {
    return thread->GetStopInfo();
  }
};
