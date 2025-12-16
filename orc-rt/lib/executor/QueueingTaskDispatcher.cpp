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

void QueueingTaskDispatcher::dispatch(std::unique_ptr<Task> T) {
  std::scoped_lock<std::mutex> Lock(M);
  if (State == Running)
    Tasks.push_back(std::move(T));
}

void QueueingTaskDispatcher::shutdown() {
  std::deque<std::unique_ptr<Task>> ResidualTasks;
  {
    std::scoped_lock<std::mutex> Lock(M);
    State = Shutdown;
    ResidualTasks = std::move(Tasks);
  }
  // ResidualTask destruction can run Task destructors outside the lock.
}

std::unique_ptr<Task> QueueingTaskDispatcher::pop_back() {
  std::scoped_lock<std::mutex> Lock(M);
  if (Tasks.empty())
    return nullptr;
  auto T = std::move(Tasks.back());
  Tasks.pop_back();
  return T;
}

std::unique_ptr<Task> QueueingTaskDispatcher::pop_front() {
  std::scoped_lock<std::mutex> Lock(M);
  if (Tasks.empty())
    return nullptr;
  auto T = std::move(Tasks.front());
  Tasks.pop_front();
  return T;
}

void QueueingTaskDispatcher::runLIFOUntilEmpty() {
  while (auto T = pop_back())
    T->run();
}

void QueueingTaskDispatcher::runFIFOUntilEmpty() {
  while (auto T = pop_front())
    T->run();
}

} // namespace orc_rt
