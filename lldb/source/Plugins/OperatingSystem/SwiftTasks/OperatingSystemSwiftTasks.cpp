//===-- OperatingSystemSwiftTasks.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if LLDB_ENABLE_SWIFT

#include "OperatingSystemSwiftTasks.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/Process/Utility/ThreadMemory.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Utility/LLDBLog.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(OperatingSystemSwiftTasks)

void OperatingSystemSwiftTasks::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                nullptr);
}

void OperatingSystemSwiftTasks::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

OperatingSystem *OperatingSystemSwiftTasks::CreateInstance(Process *process,
                                                           bool force) {
  if (!process || !process->GetTarget().GetSwiftUseTasksPlugin())
    return nullptr;

  Log *log = GetLog(LLDBLog::OS);
  std::optional<uint32_t> concurrency_version =
      SwiftLanguageRuntime::FindConcurrencyDebugVersion(*process);
  if (!concurrency_version) {
    LLDB_LOG(log,
             "OperatingSystemSwiftTasks: did not find concurrency module.");
    return nullptr;
  }

  LLDB_LOGF(log,
            "OperatingSystemSwiftTasks: got a concurrency version symbol of %u",
            *concurrency_version);
  if (*concurrency_version > 1) {
    auto warning =
        llvm::formatv("Unexpected Swift concurrency version {0}. Stepping on "
                      "concurrent code may behave incorrectly.",
                      *concurrency_version);
    lldb::user_id_t debugger_id = process->GetTarget().GetDebugger().GetID();
    static std::once_flag concurrency_warning_flag;
    Debugger::ReportWarning(warning, debugger_id, &concurrency_warning_flag);
    return nullptr;
  }
  return new OperatingSystemSwiftTasks(*process);
}

llvm::StringRef OperatingSystemSwiftTasks::GetPluginDescriptionStatic() {
  return "Operating system plug-in converting Swift Tasks into Threads.";
}

OperatingSystemSwiftTasks::~OperatingSystemSwiftTasks() = default;

OperatingSystemSwiftTasks::OperatingSystemSwiftTasks(
    lldb_private::Process &process)
    : OperatingSystem(&process) {
  size_t ptr_size = process.GetAddressByteSize();
  // Offset of a Task ID inside a Task data structure, guaranteed by the ABI.
  // See Job in swift/RemoteInspection/RuntimeInternals.h.
  m_job_id_offset = 4 * ptr_size + 4;
}

ThreadSP
OperatingSystemSwiftTasks::FindOrCreateSwiftThread(ThreadList &old_thread_list,
                                                   uint64_t task_id) {
  // Mask higher bits to avoid conflicts with core thread IDs.
  uint64_t masked_task_id = 0x0000000f00000000 | task_id;

  // If we already had a thread for this Task in the last stop, re-use it.
  if (ThreadSP old_thread = old_thread_list.FindThreadByID(masked_task_id);
      IsOperatingSystemPluginThread(old_thread))
    return old_thread;

  std::string name = llvm::formatv("Swift Task {0}", task_id);
  return std::make_shared<ThreadMemoryProvidingName>(*m_process, masked_task_id,
                                                     /*register_data_addr*/ 0,
                                                     name);
}

bool OperatingSystemSwiftTasks::UpdateThreadList(ThreadList &old_thread_list,
                                                 ThreadList &core_thread_list,
                                                 ThreadList &new_thread_list) {
  Log *log = GetLog(LLDBLog::OS);
  LLDB_LOG(log, "OperatingSystemSwiftTasks: Updating thread list");

  for (const ThreadSP &real_thread : core_thread_list.Threads()) {
    std::optional<uint64_t> task_id = FindTaskId(*real_thread);

    // If this is not a thread running a Task, add it to the list as is.
    if (!task_id) {
      new_thread_list.AddThread(real_thread);
      LLDB_LOGF(log,
                "OperatingSystemSwiftTasks: thread %" PRIx64
                " is not executing a Task",
                real_thread->GetID());
      continue;
    }

    ThreadSP swift_thread = FindOrCreateSwiftThread(old_thread_list, *task_id);
    swift_thread->SetBackingThread(real_thread);
    new_thread_list.AddThread(swift_thread);
    LLDB_LOGF(log,
              "OperatingSystemSwiftTasks: mapping thread IDs: %" PRIx64
              " -> %" PRIx64,
              real_thread->GetID(), swift_thread->GetID());
  }
  return true;
}

void OperatingSystemSwiftTasks::ThreadWasSelected(Thread *thread) {}

RegisterContextSP OperatingSystemSwiftTasks::CreateRegisterContextForThread(
    Thread *thread, addr_t reg_data_addr) {
  if (!thread || !IsOperatingSystemPluginThread(thread->shared_from_this()))
    return nullptr;
  return thread->GetRegisterContext();
}

StopInfoSP OperatingSystemSwiftTasks::CreateThreadStopReason(
    lldb_private::Thread *thread) {
  return thread->GetStopInfo();
}

std::optional<uint64_t> OperatingSystemSwiftTasks::FindTaskId(Thread &thread) {
  llvm::Expected<addr_t> task_addr = GetTaskAddrFromThreadLocalStorage(thread);
  if (!task_addr) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::OS), task_addr.takeError(),
                   "OperatingSystemSwiftTasks: failed to find task address in "
                   "thread local storage: {0}");
    return {};
  }

  Status error;
  // The Task ID is at offset m_job_id_offset from the Task pointer.
  constexpr uint32_t num_bytes_task_id = 4;
  auto task_id = m_process->ReadUnsignedIntegerFromMemory(
      *task_addr + m_job_id_offset, num_bytes_task_id, LLDB_INVALID_ADDRESS,
      error);
  if (error.Fail())
    return {};
  return task_id;
}

#endif // #if LLDB_ENABLE_SWIFT
