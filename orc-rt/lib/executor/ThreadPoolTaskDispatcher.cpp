//===- ThreadPoolTaskDispatch.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/ThreadPoolTaskDispatch.h
// header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/ThreadPoolTaskDispatcher.h"

#include <cassert>

namespace orc_rt {

ThreadPoolTaskDispatcher::~ThreadPoolTaskDispatcher() {
  assert(!AcceptingTasks && "shutdown was not run");
}

ThreadPoolTaskDispatcher::ThreadPoolTaskDispatcher(size_t NumThreads) {
  Threads.reserve(NumThreads);
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([this]() { taskLoop(); });
}

void ThreadPoolTaskDispatcher::dispatch(std::unique_ptr<Task> T) {
  {
    std::scoped_lock<std::mutex> Lock(M);
    if (!AcceptingTasks)
      return;
    PendingTasks.push_back(std::move(T));
  }
  CV.notify_one();
}

void ThreadPoolTaskDispatcher::shutdown() {
  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(AcceptingTasks && "ThreadPoolTaskDispatcher already shut down?");
    AcceptingTasks = false;
  }
  CV.notify_all();
  for (auto &Thread : Threads)
    Thread.join();
}

void ThreadPoolTaskDispatcher::taskLoop() {
  while (true) {
    std::unique_ptr<Task> T;
    {
      std::unique_lock<std::mutex> Lock(M);
      CV.wait(Lock,
              [this]() { return !PendingTasks.empty() || !AcceptingTasks; });

      if (!AcceptingTasks && PendingTasks.empty())
        return;

      T = std::move(PendingTasks.back());
      PendingTasks.pop_back();
    }

    T->run();
  }
}

} // namespace orc_rt
