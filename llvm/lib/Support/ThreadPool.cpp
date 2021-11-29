//==-- llvm/Support/ThreadPool.cpp - A ThreadPool implementation -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a crude C++11 based thread pool.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadPool.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#if LLVM_ENABLE_THREADS

ThreadPool::ThreadPool(ThreadPoolStrategy S)
    : ThreadCount(S.compute_thread_count()) {
  // Create ThreadCount threads that will loop forever, wait on QueueCondition
  // for tasks to be queued or the Pool to be destroyed.
  Threads.reserve(ThreadCount);
  for (unsigned ThreadID = 0; ThreadID < ThreadCount; ++ThreadID) {
    Threads.emplace_back([S, ThreadID, this] {
      S.apply_thread_strategy(ThreadID);
      while (true) {
        std::function<void()> Task;
        {
          std::unique_lock<std::mutex> LockGuard(QueueLock);
          // Wait for tasks to be pushed in the queue
          QueueCondition.wait(LockGuard,
                              [&] { return !EnableFlag || !Tasks.empty(); });
          // Exit condition
          if (!EnableFlag && Tasks.empty())
            return;
          // Yeah, we have a task, grab it and release the lock on the queue

          // We first need to signal that we are active before popping the queue
          // in order for wait() to properly detect that even if the queue is
          // empty, there is still a task in flight.
          ++ActiveThreads;
          Task = std::move(Tasks.front());
          Tasks.pop();
        }
        // Run the task we just grabbed
        Task();

        bool Notify;
        {
          // Adjust `ActiveThreads`, in case someone waits on ThreadPool::wait()
          std::lock_guard<std::mutex> LockGuard(QueueLock);
          --ActiveThreads;
          Notify = workCompletedUnlocked();
        }
        // Notify task completion if this is the last active thread, in case
        // someone waits on ThreadPool::wait().
        if (Notify)
          CompletionCondition.notify_all();
      }
    });
  }
}

void ThreadPool::wait() {
  // Wait for all threads to complete and the queue to be empty
  std::unique_lock<std::mutex> LockGuard(QueueLock);
  CompletionCondition.wait(LockGuard, [&] { return workCompletedUnlocked(); });
}

bool ThreadPool::isWorkerThread() const {
  llvm::thread::id CurrentThreadId = llvm::this_thread::get_id();
  for (const llvm::thread &Thread : Threads)
    if (CurrentThreadId == Thread.get_id())
      return true;
  return false;
}

// The destructor joins all threads, waiting for completion.
ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> LockGuard(QueueLock);
    EnableFlag = false;
  }
  QueueCondition.notify_all();
  for (auto &Worker : Threads)
    Worker.join();
}

#else // LLVM_ENABLE_THREADS Disabled

// No threads are launched, issue a warning if ThreadCount is not 0
ThreadPool::ThreadPool(ThreadPoolStrategy S)
    : ThreadCount(S.compute_thread_count()) {
  if (ThreadCount != 1) {
    errs() << "Warning: request a ThreadPool with " << ThreadCount
           << " threads, but LLVM_ENABLE_THREADS has been turned off\n";
  }
}

void ThreadPool::wait() {
  // Sequential implementation running the tasks
  while (!Tasks.empty()) {
    auto Task = std::move(Tasks.front());
    Tasks.pop();
    Task();
  }
}

ThreadPool::~ThreadPool() { wait(); }

#endif
