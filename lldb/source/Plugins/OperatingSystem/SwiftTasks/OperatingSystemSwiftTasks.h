//===-- OperatingSystemSwiftTasks.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OperatingSystemSwiftTasks_h_
#define liblldb_OperatingSystemSwiftTasks_h_

#if LLDB_ENABLE_SWIFT

#include "lldb/Target/OperatingSystem.h"

namespace lldb_private {
class OperatingSystemSwiftTasks : public OperatingSystem {
public:
  OperatingSystemSwiftTasks(Process &process);
  ~OperatingSystemSwiftTasks() override;

  static OperatingSystem *CreateInstance(Process *process, bool force);
  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() { return "swift"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  /// PluginInterface Methods

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  /// OperatingSystem Methods

  bool UpdateThreadList(ThreadList &old_thread_list,
                        ThreadList &real_thread_list,
                        ThreadList &new_thread_list) override;

  void ThreadWasSelected(Thread *thread) override;

  lldb::RegisterContextSP
  CreateRegisterContextForThread(Thread *thread,
                                 lldb::addr_t reg_data_addr) override;

  lldb::StopInfoSP CreateThreadStopReason(Thread *thread) override;

  bool DoesPluginReportAllThreads() override { return false; }

private:
  /// If a thread for task_id had been created in the last stop, return it.
  /// Otherwise, create a new MemoryThread for it.
  lldb::ThreadSP FindOrCreateSwiftThread(ThreadList &old_thread_list,
                                         uint64_t task_id);

  /// Find the Task ID of the task being executed by `thread`, if any.
  std::optional<uint64_t> FindTaskId(Thread &thread);

  /// The offset of the Job ID inside a Task data structure.
  size_t m_job_id_offset;
};
} // namespace lldb_private

#endif // LLDB_ENABLE_SWIFT

#endif // liblldb_OperatingSystemSwiftTasks_h_
