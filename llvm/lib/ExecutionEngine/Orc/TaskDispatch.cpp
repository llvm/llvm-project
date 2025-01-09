//===------------ TaskDispatch.cpp - ORC task dispatch utils --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ENABLE_THREADS
#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

char Task::ID = 0;
char GenericNamedTask::ID = 0;
const char *GenericNamedTask::DefaultDescription = "Generic Task";

void Task::anchor() {}
TaskDispatcher::~TaskDispatcher() = default;

void InPlaceTaskDispatcher::dispatch(std::unique_ptr<Task> T) { T->run(); }

void InPlaceTaskDispatcher::shutdown() {}

#if LLVM_ENABLE_THREADS
void DynamicThreadPoolTaskDispatcher::dispatch(std::unique_ptr<Task> T) {
  bool IsMaterializationTask = isa<MaterializationTask>(*T);

  {
    std::lock_guard<std::mutex> Lock(DispatchMutex);

    // Reject new tasks if they're dispatched after a call to shutdown.
    if (Shutdown)
      return;

    if (IsMaterializationTask) {

      // If this is a materialization task and there are too many running
      // already then queue this one up and return early.
      if (MaxMaterializationThreads &&
          NumMaterializationThreads == *MaxMaterializationThreads) {
        MaterializationTaskQueue.push_back(std::move(T));
        return;
      }

      // Otherwise record that we have a materialization task running.
      ++NumMaterializationThreads;
    }

    ++Outstanding;
  }

  std::thread([this, T = std::move(T), IsMaterializationTask]() mutable {
    while (true) {

      // Run the task.
      T->run();

      // Reset the task to free any resources. We need this to happen *before*
      // we notify anyone (via Outstanding) that this thread is done to ensure
      // that we don't proceed with JIT shutdown while still holding resources.
      // (E.g. this was causing "Dangling SymbolStringPtr" assertions).
      T.reset();

      // Check the work queue state and either proceed with the next task or
      // end this thread.
      std::lock_guard<std::mutex> Lock(DispatchMutex);
      if (!MaterializationTaskQueue.empty()) {
        // If there are any materialization tasks running then steal that work.
        T = std::move(MaterializationTaskQueue.front());
        MaterializationTaskQueue.pop_front();
        if (!IsMaterializationTask) {
          ++NumMaterializationThreads;
          IsMaterializationTask = true;
        }
      } else {
        if (IsMaterializationTask)
          --NumMaterializationThreads;
        --Outstanding;
        if (Outstanding == 0)
          OutstandingCV.notify_all();
        return;
      }
    }
  }).detach();
}

void DynamicThreadPoolTaskDispatcher::shutdown() {
  std::unique_lock<std::mutex> Lock(DispatchMutex);
  Shutdown = true;
  OutstandingCV.wait(Lock, [this]() { return Outstanding == 0; });
}
#endif

} // namespace orc
} // namespace llvm
