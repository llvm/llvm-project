//===- llvm/Support/Parallel.cpp - Parallel algorithms --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Parallel.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ExponentialBackoff.h"
#include "llvm/Support/Jobserver.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Threading.h"

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace llvm;
using namespace llvm::parallel;

llvm::ThreadPoolStrategy parallel::strategy;

#if LLVM_ENABLE_THREADS

#ifdef _WIN32
static thread_local unsigned threadIndex = UINT_MAX;

unsigned parallel::getThreadIndex() { GET_THREAD_INDEX_IMPL; }
#else
thread_local unsigned parallel::threadIndex = UINT_MAX;
#endif

namespace {

/// Runs closures on a thread pool in filo order.
class ThreadPoolExecutor {
public:
  explicit ThreadPoolExecutor(ThreadPoolStrategy S) {
    if (S.UseJobserver)
      TheJobserver = JobserverClient::getInstance();

    ThreadCount = S.compute_thread_count();
    // Spawn all but one of the threads in another thread as spawning threads
    // can take a while.
    Threads.reserve(ThreadCount);
    Threads.resize(1);
    std::lock_guard<std::mutex> Lock(Mutex);
    // Use operator[] before creating the thread to avoid data race in .size()
    // in 'safe libc++' mode.
    auto &Thread0 = Threads[0];
    Thread0 = std::thread([this, S] {
      for (unsigned I = 1; I < ThreadCount; ++I) {
        Threads.emplace_back([this, S, I] { work(S, I); });
        if (Stop)
          break;
      }
      ThreadsCreated.set_value();
      work(S, 0);
    });
  }

  // To make sure the thread pool executor can only be created with a parallel
  // strategy.
  ThreadPoolExecutor() = delete;

  void stop() {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      if (Stop)
        return;
      Stop = true;
    }
    Cond.notify_all();
    ThreadsCreated.get_future().wait();

    std::thread::id CurrentThreadId = std::this_thread::get_id();
    for (std::thread &T : Threads)
      if (T.get_id() == CurrentThreadId)
        T.detach();
      else
        T.join();
  }

  ~ThreadPoolExecutor() { stop(); }

  struct Creator {
    static void *call() { return new ThreadPoolExecutor(strategy); }
  };
  struct Deleter {
    static void call(void *Ptr) { ((ThreadPoolExecutor *)Ptr)->stop(); }
  };

  struct WorkItem {
    std::function<void()> F;
    std::reference_wrapper<parallel::detail::Latch> L;
    void operator()() {
      F();
      L.get().dec();
    }
  };

  void add(std::function<void()> F, parallel::detail::Latch &L) {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      WorkStack.push_back({std::move(F), std::ref(L)});
    }
    Cond.notify_one();
  }

  size_t getThreadCount() const { return ThreadCount; }

private:
  // Pop one task from the queue and run it. Must be called with Lock held;
  // releases Lock before executing the task.
  void popAndRun(std::unique_lock<std::mutex> &Lock) {
    auto Item = std::move(WorkStack.back());
    WorkStack.pop_back();
    Lock.unlock();
    Item();
  }

  void work(ThreadPoolStrategy S, unsigned ThreadID) {
    threadIndex = ThreadID;
    S.apply_thread_strategy(ThreadID);
    // Note on jobserver deadlock avoidance:
    // GNU Make grants each invoked process one implicit job slot. Our
    // JobserverClient models this by returning an implicit JobSlot on the
    // first successful tryAcquire() in a process. This guarantees forward
    // progress without requiring a dedicated "always-on" thread here.

    while (true) {
      if (TheJobserver) {
        // Jobserver-mode scheduling:
        // - Acquire one job slot (with exponential backoff to avoid busy-wait).
        // - While holding the slot, drain and run tasks from the local queue.
        // - Release the slot when the queue is empty or when shutting down.
        // Rationale: Holding a slot amortizes acquire/release overhead over
        // multiple tasks and avoids requeue/yield churn, while still enforcing
        // the jobserver’s global concurrency limit. With K available slots,
        // up to K workers run tasks in parallel; within each worker tasks run
        // sequentially until the local queue is empty.
        ExponentialBackoff Backoff(std::chrono::hours(24));
        JobSlot Slot;
        do {
          if (Stop)
            return;
          Slot = TheJobserver->tryAcquire();
          if (Slot.isValid())
            break;
        } while (Backoff.waitForNextAttempt());

        llvm::scope_exit SlotReleaser(
            [&] { TheJobserver->release(std::move(Slot)); });

        while (true) {
          std::unique_lock<std::mutex> Lock(Mutex);
          Cond.wait(Lock, [&] { return Stop || !WorkStack.empty(); });
          if (Stop && WorkStack.empty())
            return;
          if (WorkStack.empty())
            break;
          popAndRun(Lock);
        }
      } else {
        std::unique_lock<std::mutex> Lock(Mutex);
        Cond.wait(Lock, [&] { return Stop || !WorkStack.empty(); });
        if (Stop)
          break;
        popAndRun(Lock);
      }
    }
  }

  std::atomic<bool> Stop{false};
  std::vector<WorkItem> WorkStack;
  std::mutex Mutex;
  std::condition_variable Cond;
  std::promise<void> ThreadsCreated;
  std::vector<std::thread> Threads;
  unsigned ThreadCount;

  JobserverClient *TheJobserver = nullptr;
};
} // namespace

static ThreadPoolExecutor *getDefaultExecutor() {
#ifdef _WIN32
  // The ManagedStatic enables the ThreadPoolExecutor to be stopped via
  // llvm_shutdown() on Windows. This is important to avoid various race
  // conditions at process exit that can cause crashes or deadlocks.

  static ManagedStatic<ThreadPoolExecutor, ThreadPoolExecutor::Creator,
                       ThreadPoolExecutor::Deleter>
      ManagedExec;
  static std::unique_ptr<ThreadPoolExecutor> Exec(&(*ManagedExec));
  return Exec.get();
#else
  // ManagedStatic is not desired on other platforms. When `Exec` is destroyed
  // by llvm_shutdown(), worker threads will clean up and invoke TLS
  // destructors. This can lead to race conditions if other threads attempt to
  // access TLS objects that have already been destroyed.
  static ThreadPoolExecutor Exec(strategy);
  return &Exec;
#endif
}

size_t parallel::getThreadCount() {
  return getDefaultExecutor()->getThreadCount();
}
#endif

// Latch::sync() called by the dtor may cause one thread to block. If is a dead
// lock if all threads in the default executor are blocked. To prevent the dead
// lock, only allow the root TaskGroup to run tasks parallelly. In the scenario
// of nested parallel_for_each(), only the outermost one runs parallelly.
TaskGroup::TaskGroup()
    : Parallel(
#if LLVM_ENABLE_THREADS
          strategy.ThreadsRequested != 1 && threadIndex == UINT_MAX
#else
          false
#endif
      ) {
}
TaskGroup::~TaskGroup() {
  // We must ensure that all the workloads have finished before decrementing the
  // instances count.
  L.sync();
}

void TaskGroup::spawn(std::function<void()> F) {
#if LLVM_ENABLE_THREADS
  if (Parallel) {
    L.inc();
    getDefaultExecutor()->add(std::move(F), L);
    return;
  }
#endif
  F();
}

void llvm::parallelFor(size_t Begin, size_t End,
                       function_ref<void(size_t)> Fn) {
#if LLVM_ENABLE_THREADS
  if (strategy.ThreadsRequested != 1) {
    size_t NumItems = End - Begin;
    if (NumItems == 0)
      return;
    // Distribute work via an atomic counter shared by NumWorkers threads,
    // keeping the task count (and thus Linux futex calls) at O(ThreadCount)
    // For lld, per-file work is somewhat uneven, so a multipler > 1 is safer.
    // While 2 vs 4 vs 8 makes no measurable difference, 4 is used as a
    // reasonable default.
    size_t NumWorkers = std::min<size_t>(NumItems, getThreadCount());
    size_t ChunkSize = std::max(size_t(1), NumItems / (NumWorkers * 4));
    std::atomic<size_t> Idx{Begin};
    auto Worker = [&] {
      while (true) {
        size_t I = Idx.fetch_add(ChunkSize, std::memory_order_relaxed);
        if (I >= End)
          break;
        size_t IEnd = std::min(I + ChunkSize, End);
        for (; I < IEnd; ++I)
          Fn(I);
      }
    };

    TaskGroup TG;
    for (size_t I = 0; I != NumWorkers; ++I)
      TG.spawn(Worker);
    return;
  }
#endif

  for (; Begin != End; ++Begin)
    Fn(Begin);
}
