//===-- OperatingSystemSwiftTasks.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include <optional>
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

/// Mask applied to task IDs to generate thread IDs that to don't conflicts with
/// core thread IDs.
static constexpr uint64_t TASK_MASK = 0x0000000f00000000ULL;

/// A wrapper around ThreadMemory providing lazy name evaluation, as this is
/// expensive to compute for Swift Tasks.
class SwiftTaskThreadMemory : public ThreadMemory {
public:
  SwiftTaskThreadMemory(lldb_private::Process &process, lldb::tid_t tid,
                        lldb::addr_t register_data_addr)
      : ThreadMemory(process, tid, register_data_addr) {}

  /// Updates the backing thread of this Task, as well as the location where the
  /// task pointer is stored.
  void UpdateBackingThread(const ThreadSP &new_backing_thread,
                           lldb::addr_t task_addr) {
    SetBackingThread(new_backing_thread);
    m_task_addr = task_addr;
  }

  const char *GetName() override {
    if (m_task_name.empty())
      m_task_name = FindTaskName();
    return m_task_name.c_str();
  }

private:
  uint64_t GetTaskID() const {
    auto thread_id = GetID();
    return thread_id & ~TASK_MASK;
  }

  std::string GetDefaultTaskName() const {
    return llvm::formatv("Task {0}", GetTaskID());
  }

  /// If possible, read a user-provided task name from memory, otherwise use a
  /// default name. This never returns an empty string.
  std::string FindTaskName() const {
    llvm::Expected<std::optional<std::string>> task_name =
        GetTaskName(m_task_addr, *GetProcess());
    if (auto err = task_name.takeError()) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::OS), std::move(err),
                     "OperatingSystemSwiftTasks: failed while looking for name "
                     "of task {1:x}: {0}",
                     m_task_addr);
      return GetDefaultTaskName();
    }

    if (!task_name->has_value())
      return GetDefaultTaskName();
    return llvm::formatv("{0} (Task {1})", *task_name, GetTaskID());
  }

  std::string m_task_name = "";
  lldb::addr_t m_task_addr = LLDB_INVALID_ADDRESS;
};

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
    : OperatingSystem(&process) {}

ThreadSP
OperatingSystemSwiftTasks::FindOrCreateSwiftThread(ThreadList &old_thread_list,
                                                   uint64_t task_id) {
  // Mask higher bits to avoid conflicts with core thread IDs.
  uint64_t masked_task_id = TASK_MASK | task_id;

  // If we already had a thread for this Task in the last stop, re-use it.
  if (ThreadSP old_thread = old_thread_list.FindThreadByID(masked_task_id);
      IsOperatingSystemPluginThread(old_thread))
    return old_thread;

  return std::make_shared<SwiftTaskThreadMemory>(*m_process, masked_task_id,
                                                 /*register_data_addr*/ 0);
}

/// For each thread in `threads_it`, computes the task address that is being run
/// by the thread, if any.
static llvm::SmallVector<std::optional<addr_t>>
FindTaskAddresses(TaskInspector &task_inspector,
                  ThreadCollection::ThreadIterable &threads_it) {
  llvm::SmallVector<Thread *> threads;
  for (const ThreadSP &thread : threads_it)
    threads.push_back(thread.get());

  llvm::SmallVector<std::optional<addr_t>> task_addrs;
  task_addrs.reserve(threads.size());

  for (std::optional<addr_t> &task_addr :
       task_inspector.GetTaskAddrFromThreadLocalStorage(threads)) {
    if (!task_addr) {
      LLDB_LOG(GetLog(LLDBLog::OS), "OperatingSystemSwiftTasks: failed to find "
                                    "task address in thread local storage");
      task_addrs.push_back(std::nullopt);
      continue;
    }
    if (*task_addr == 0 || *task_addr == LLDB_INVALID_ADDRESS)
      task_addrs.push_back(std::nullopt);
    else
      task_addrs.push_back(*task_addr);
  }
  return task_addrs;
}

static std::optional<uint64_t> FindTaskId(addr_t task_addr, Process &process) {
  size_t ptr_size = process.GetAddressByteSize();
  // Offset of a Task ID inside a Task data structure, guaranteed by the ABI.
  // See Job in swift/RemoteInspection/RuntimeInternals.h.
  const offset_t job_id_offset = 4 * ptr_size + 4;

  Status error;
  // The Task ID is at offset job_id_offset from the Task pointer.
  constexpr uint32_t num_bytes_task_id = 4;
  auto task_id = process.ReadUnsignedIntegerFromMemory(
      task_addr + job_id_offset, num_bytes_task_id, LLDB_INVALID_ADDRESS,
      error);
  if (error.Fail())
    return {};
  return task_id;
}

bool OperatingSystemSwiftTasks::UpdateThreadList(ThreadList &old_thread_list,
                                                 ThreadList &core_thread_list,
                                                 ThreadList &new_thread_list) {
  Log *log = GetLog(LLDBLog::OS);
  LLDB_LOG(log, "OperatingSystemSwiftTasks: Updating thread list");

  ThreadCollection::ThreadIterable locked_core_threads =
      core_thread_list.Threads();
  llvm::SmallVector<std::optional<addr_t>> task_addrs =
      FindTaskAddresses(m_task_inspector, locked_core_threads);

  for (const auto &[real_thread, task_addr] :
       llvm::zip(locked_core_threads, task_addrs)) {
    // If this is not a thread running a Task, add it to the list as is.
    if (!task_addr) {
      new_thread_list.AddThread(real_thread);
      LLDB_LOGF(log,
                "OperatingSystemSwiftTasks: thread %" PRIx64
                " is not executing a Task",
                real_thread->GetID());
      continue;
    }

    assert(m_process != nullptr);
    std::optional<uint64_t> task_id = FindTaskId(*task_addr, *m_process);
    if (!task_id) {
      LLDB_LOG(log, "OperatingSystemSwiftTasks: could not get ID of Task {0:x}",
               *task_addr);
      continue;
    }

    ThreadSP swift_thread = FindOrCreateSwiftThread(old_thread_list, *task_id);
    static_cast<SwiftTaskThreadMemory &>(*swift_thread)
        .UpdateBackingThread(real_thread, *task_addr);
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

#endif // #if LLDB_ENABLE_SWIFT
