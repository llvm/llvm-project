//===- QueueingTaskDispatcher.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/QueueingTaskDispatcher.h
// header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/QueueingTaskDispatcher.h"

#include <cassert>

namespace orc_rt {

void QueueingTaskDispatcher::TaskQueue::addTask(std::unique_ptr<Task> T) {
  {
    std::scoped_lock<std::mutex> Lock(M);
    if (State == Running)
      Tasks.push_back(std::move(T));
  }
  CV.notify_one();
}

void QueueingTaskDispatcher::TaskQueue::shutdown() {
  {
    std::scoped_lock<std::mutex> Lock(M);
    State = Shutdown;
  }
  CV.notify_all();
}

std::unique_ptr<Task> QueueingTaskDispatcher::TaskQueue::takeLastIn() {
  std::unique_lock<std::mutex> Lock(M);
  CV.wait(Lock, [&]() { return !Tasks.empty() || State == Shutdown; });
  if (Tasks.empty())
    return nullptr;
  auto T = std::move(Tasks.back());
  Tasks.pop_back();
  return T;
}

std::unique_ptr<Task> QueueingTaskDispatcher::TaskQueue::takeFirstIn() {
  std::unique_lock<std::mutex> Lock(M);
  CV.wait(Lock, [&]() { return !Tasks.empty() || State == Shutdown; });
  if (Tasks.empty())
    return nullptr;
  auto T = std::move(Tasks.front());
  Tasks.pop_front();
  return T;
}

void QueueingTaskDispatcher::TaskQueue::runLIFOUntilEmpty() {
  while (auto T = takeLastIn())
    T->run();
}

void QueueingTaskDispatcher::TaskQueue::runFIFOUntilEmpty() {
  while (auto T = takeFirstIn())
    T->run();
}

void QueueingTaskDispatcher::dispatch(std::unique_ptr<Task> T) {
  Q.addTask(std::move(T));
}

void QueueingTaskDispatcher::shutdown() { Q.shutdown(); }

} // namespace orc_rt
