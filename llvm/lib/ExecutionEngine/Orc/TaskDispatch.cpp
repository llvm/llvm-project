//===------------ TaskDispatch.cpp - ORC task dispatch utils --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"

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
  {
    std::lock_guard<std::mutex> Lock(DispatchMutex);
    ++Outstanding;
  }

  std::thread([this, T = std::move(T)]() mutable {
    T->run();
    std::lock_guard<std::mutex> Lock(DispatchMutex);
    --Outstanding;
    OutstandingCV.notify_all();
  }).detach();
}

void DynamicThreadPoolTaskDispatcher::shutdown() {
  std::unique_lock<std::mutex> Lock(DispatchMutex);
  Running = false;
  OutstandingCV.wait(Lock, [this]() { return Outstanding == 0; });
}
#endif

} // namespace orc
} // namespace llvm
