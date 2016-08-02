//==-- llvm/Support/ThreadPool.cpp - A ThreadPool implementation -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a crude C++11 based thread pool.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadPool.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#if LLVM_ENABLE_THREADS

// Default to std::thread::hardware_concurrency
ThreadPool::ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

ThreadPool::ThreadPool(unsigned ThreadCount)
    : ActiveThreads(0), EnableFlag(true) {
  // Create ThreadCount threads that will loop forever, wait on QueueCondition
  // for tasks to be queued or the Pool to be destroyed.
  Threads.reserve(ThreadCount);
  for (unsigned ThreadID = 0; ThreadID < ThreadCount; ++ThreadID) {
    Threads.emplace_back([&] {
      while (true) {
        PackagedTaskTy Task;
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
          {
            ++ActiveThreads;
            std::unique_lock<std::mutex> LockGuard(CompletionLock);
          }
          Task = std::move(Tasks.front());
          Tasks.pop();
        }
        // Run the task we just grabbed
#ifndef _MSC_VER
        Task();
#else
        Task(/* unused */ false);
#endif

        {
          // Adjust `ActiveThreads`, in case someone waits on ThreadPool::wait()
          std::unique_lock<std::mutex> LockGuard(CompletionLock);
          --ActiveThreads;
        }

        // Notify task completion, in case someone waits on ThreadPool::wait()
        CompletionCondition.notify_all();
      }
    });
  }
}

void ThreadPool::wait() {
  // Wait for all threads to complete and the queue to be empty
  std::unique_lock<std::mutex> LockGuard(CompletionLock);
  CompletionCondition.wait(LockGuard,
                           [&] { return Tasks.empty() && !ActiveThreads; });
}

std::shared_future<ThreadPool::VoidTy> ThreadPool::asyncImpl(TaskTy Task) {
  /// Wrap the Task in a packaged_task to return a future object.
  PackagedTaskTy PackagedTask(std::move(Task));
  auto Future = PackagedTask.get_future();
  {
    // Lock the queue and push the new task
    std::unique_lock<std::mutex> LockGuard(QueueLock);

    // Don't allow enqueueing after disabling the pool
    assert(EnableFlag && "Queuing a thread during ThreadPool destruction");

    Tasks.push(std::move(PackagedTask));
  }
  QueueCondition.notify_one();
  return Future.share();
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

ThreadPool::ThreadPool() : ThreadPool(0) {}

// No threads are launched, issue a warning if ThreadCount is not 0
ThreadPool::ThreadPool(unsigned ThreadCount)
    : ActiveThreads(0) {
  if (ThreadCount) {
    errs() << "Warning: request a ThreadPool with " << ThreadCount
           << " threads, but LLVM_ENABLE_THREADS has been turned off\n";
  }
}

void ThreadPool::wait() {
  // Sequential implementation running the tasks
  while (!Tasks.empty()) {
    auto Task = std::move(Tasks.front());
    Tasks.pop();
#ifndef _MSC_VER
        Task();
#else
        Task(/* unused */ false);
#endif
  }
}

std::shared_future<ThreadPool::VoidTy> ThreadPool::asyncImpl(TaskTy Task) {
#ifndef _MSC_VER
  // Get a Future with launch::deferred execution using std::async
  auto Future = std::async(std::launch::deferred, std::move(Task)).share();
  // Wrap the future so that both ThreadPool::wait() can operate and the
  // returned future can be sync'ed on.
  PackagedTaskTy PackagedTask([Future]() { Future.get(); });
#else
  auto Future = std::async(std::launch::deferred, std::move(Task), false).share();
  PackagedTaskTy PackagedTask([Future](bool) -> bool { Future.get(); return false; });
#endif
  Tasks.push(std::move(PackagedTask));
  return Future;
}

ThreadPool::~ThreadPool() {
  wait();
}

#endif
