//===- llvm/Support/Parallel.cpp - Parallel algorithms --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Parallel.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Threading.h"

#include <atomic>
#include <deque>
#include <future>
#include <thread>
#include <vector>

llvm::ThreadPoolStrategy llvm::parallel::strategy;

namespace llvm {
namespace parallel {
#if LLVM_ENABLE_THREADS

#ifdef _WIN32
static thread_local unsigned threadIndex = UINT_MAX;

unsigned getThreadIndex() { GET_THREAD_INDEX_IMPL; }
#else
thread_local unsigned threadIndex = UINT_MAX;
#endif

namespace detail {

namespace {

/// An abstract class that takes closures and runs them asynchronously.
class Executor {
public:
  virtual ~Executor() = default;
  virtual void add(std::function<void()> func, bool Sequential = false) = 0;
  virtual size_t getThreadCount() const = 0;

  static Executor *getDefaultExecutor();
};

/// An implementation of an Executor that runs closures on a thread pool
///   in filo order.
class ThreadPoolExecutor : public Executor {
public:
  explicit ThreadPoolExecutor(ThreadPoolStrategy S) {
    ThreadCount = S.compute_thread_count() + 1;
    Threads.reserve(ThreadCount);
    Threads.resize(2);

    {
      std::lock_guard<std::mutex> Lock(MutexSequential);
      auto &Thread0 = Threads[0];
      Thread0 = std::thread([this, S]() {
        work(S, 0, MutexSequential, CondSequential, WorkQueueSequential);
      });
    }

    // Spawn all but one of the threads in another thread as spawning threads
    // can take a while.
    std::lock_guard<std::mutex> Lock(Mutex);
    // Use operator[] before creating the thread to avoid data race in .size()
    // in 'safe libc++' mode.
    auto &Thread1 = Threads[1];
    Thread1 = std::thread([this, S]() {
      for (unsigned I = 2; I < ThreadCount; ++I) {
        Threads.emplace_back([=] { work(S, I, Mutex, Cond, WorkQueue); });
        if (Stop)
          break;
      }
      ThreadsCreated.set_value();
      work(S, 1, Mutex, Cond, WorkQueue);
    });
  }

  void stop() {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      if (Stop)
        return;
      Stop = true;
    }
    Cond.notify_all();
    CondSequential.notify_all();
    ThreadsCreated.get_future().wait();
  }

  ~ThreadPoolExecutor() override {
    stop();
    std::thread::id CurrentThreadId = std::this_thread::get_id();
    for (std::thread &T : Threads)
      if (T.get_id() == CurrentThreadId)
        T.detach();
      else
        T.join();
  }

  struct Creator {
    static void *call() { return new ThreadPoolExecutor(strategy); }
  };
  struct Deleter {
    static void call(void *Ptr) { ((ThreadPoolExecutor *)Ptr)->stop(); }
  };

  void add(std::function<void()> F, bool Sequential = false) override {
    if (Sequential) {
      addImpl<true>(F, MutexSequential, CondSequential, WorkQueueSequential);
      return;
    }

    addImpl<false>(F, Mutex, Cond, WorkQueue);
  }

  size_t getThreadCount() const override { return ThreadCount; }

private:
  template <bool Sequential>
  void addImpl(std::function<void()> F, std::mutex &M,
               std::condition_variable &C,
               std::deque<std::function<void()>> &Q) {
    {
      std::lock_guard<std::mutex> Lock(M);
      if constexpr (Sequential)
        Q.emplace_front(std::move(F));
      else
        Q.emplace_back(std::move(F));
    }
    C.notify_one();
  }

  void work(ThreadPoolStrategy S, unsigned ThreadID, std::mutex &M,
            std::condition_variable &C, std::deque<std::function<void()>> &Q) {
    threadIndex = ThreadID;
    S.apply_thread_strategy(ThreadID);
    while (true) {
      std::unique_lock<std::mutex> Lock(M);
      C.wait(Lock, [&] { return Stop || !Q.empty(); });
      // Stop if requested.
      if (Stop)
        break;

      if (Q.empty()) {
        Lock.unlock();
        continue;
      }

      // Unlock queue and execute task.
      auto Task = std::move(Q.back());
      Q.pop_back();
      Lock.unlock();
      Task();
    }
  }

  std::atomic<bool> Stop{false};
  std::deque<std::function<void()>> WorkQueue;
  std::mutex Mutex;
  std::condition_variable Cond;
  std::deque<std::function<void()>> WorkQueueSequential;
  std::mutex MutexSequential;
  std::condition_variable CondSequential;
  std::promise<void> ThreadsCreated;
  std::vector<std::thread> Threads;
  unsigned ThreadCount;
};

Executor *Executor::getDefaultExecutor() {
#ifdef _WIN32
  // The ManagedStatic enables the ThreadPoolExecutor to be stopped via
  // llvm_shutdown() which allows a "clean" fast exit, e.g. via _exit(). This
  // stops the thread pool and waits for any worker thread creation to complete
  // but does not wait for the threads to finish. The wait for worker thread
  // creation to complete is important as it prevents intermittent crashes on
  // Windows due to a race condition between thread creation and process exit.
  //
  // The ThreadPoolExecutor will only be destroyed when the static unique_ptr to
  // it is destroyed, i.e. in a normal full exit. The ThreadPoolExecutor
  // destructor ensures it has been stopped and waits for worker threads to
  // finish. The wait is important as it prevents intermittent crashes on
  // Windows when the process is doing a full exit.
  //
  // The Windows crashes appear to only occur with the MSVC static runtimes and
  // are more frequent with the debug static runtime.
  //
  // This also prevents intermittent deadlocks on exit with the MinGW runtime.

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
} // namespace
} // namespace detail

size_t getThreadCount() {
  return detail::Executor::getDefaultExecutor()->getThreadCount();
}
#endif

// Latch::sync() called by the dtor may cause one thread to block. If is a dead
// lock if all threads in the default executor are blocked. To prevent the dead
// lock, only allow the root TaskGroup to run tasks parallelly. In the scenario
// of nested parallel_for_each(), only the outermost one runs parallelly.
TaskGroup::TaskGroup()
#if LLVM_ENABLE_THREADS
    : Parallel((parallel::strategy.ThreadsRequested != 1) &&
               (threadIndex == UINT_MAX)) {}
#else
    : Parallel(false) {}
#endif
TaskGroup::~TaskGroup() {
  // We must ensure that all the workloads have finished before decrementing the
  // instances count.
  L.sync();
}

void TaskGroup::spawn(std::function<void()> F, bool Sequential) {
#if LLVM_ENABLE_THREADS
  if (Parallel) {
    L.inc();
    detail::Executor::getDefaultExecutor()->add(
        [&, F = std::move(F)] {
          F();
          L.dec();
        },
        Sequential);
    return;
  }
#endif
  F();
}

} // namespace parallel
} // namespace llvm

void llvm::parallelFor(size_t Begin, size_t End,
                       llvm::function_ref<void(size_t)> Fn) {
#if LLVM_ENABLE_THREADS
  if (parallel::strategy.ThreadsRequested != 1) {
    auto NumItems = End - Begin;
    // Limit the number of tasks to MaxTasksPerGroup to limit job scheduling
    // overhead on large inputs.
    auto TaskSize = NumItems / parallel::detail::MaxTasksPerGroup;
    if (TaskSize == 0)
      TaskSize = 1;

    parallel::TaskGroup TG;
    for (; Begin + TaskSize < End; Begin += TaskSize) {
      TG.spawn([=, &Fn] {
        for (size_t I = Begin, E = Begin + TaskSize; I != E; ++I)
          Fn(I);
      });
    }
    if (Begin != End) {
      TG.spawn([=, &Fn] {
        for (size_t I = Begin; I != End; ++I)
          Fn(I);
      });
    }
    return;
  }
#endif

  for (; Begin != End; ++Begin)
    Fn(Begin);
}
