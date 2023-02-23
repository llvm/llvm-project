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
#include <future>
#include <stack>
#include <thread>
#include <vector>

llvm::ThreadPoolStrategy llvm::parallel::strategy;

namespace llvm {
namespace parallel {
#if LLVM_ENABLE_THREADS

#ifdef _WIN32
static thread_local unsigned threadIndex;

unsigned getThreadIndex() { return threadIndex; }
#else
thread_local unsigned threadIndex;
#endif

namespace detail {

namespace {

/// An abstract class that takes closures and runs them asynchronously.
class Executor {
public:
  virtual ~Executor() = default;
  virtual void add(std::function<void()> func) = 0;

  static Executor *getDefaultExecutor();
};

/// An implementation of an Executor that runs closures on a thread pool
///   in filo order.
class ThreadPoolExecutor : public Executor {
public:
  explicit ThreadPoolExecutor(ThreadPoolStrategy S = hardware_concurrency()) {
    unsigned ThreadCount = S.compute_thread_count();
    // Spawn all but one of the threads in another thread as spawning threads
    // can take a while.
    Threads.reserve(ThreadCount);
    Threads.resize(1);
    std::lock_guard<std::mutex> Lock(Mutex);
    // Use operator[] before creating the thread to avoid data race in .size()
    // in “safe libc++” mode.
    auto& Thread0 = Threads[0];
    Thread0 = std::thread([this, ThreadCount, S] {
      for (unsigned I = 1; I < ThreadCount; ++I) {
        Threads.emplace_back([=] { work(S, I); });
        if (Stop)
          break;
      }
      ThreadsCreated.set_value();
      work(S, 0);
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

  void add(std::function<void()> F) override {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      WorkStack.push(std::move(F));
    }
    Cond.notify_one();
  }

private:
  void work(ThreadPoolStrategy S, unsigned ThreadID) {
    threadIndex = ThreadID;
    S.apply_thread_strategy(ThreadID);
    while (true) {
      std::unique_lock<std::mutex> Lock(Mutex);
      Cond.wait(Lock, [&] { return Stop || !WorkStack.empty(); });
      if (Stop)
        break;
      auto Task = std::move(WorkStack.top());
      WorkStack.pop();
      Lock.unlock();
      Task();
    }
  }

  std::atomic<bool> Stop{false};
  std::stack<std::function<void()>> WorkStack;
  std::mutex Mutex;
  std::condition_variable Cond;
  std::promise<void> ThreadsCreated;
  std::vector<std::thread> Threads;
};

Executor *Executor::getDefaultExecutor() {
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
}
} // namespace
} // namespace detail
#endif

static std::atomic<int> TaskGroupInstances;

// Latch::sync() called by the dtor may cause one thread to block. If is a dead
// lock if all threads in the default executor are blocked. To prevent the dead
// lock, only allow the first TaskGroup to run tasks parallelly. In the scenario
// of nested parallel_for_each(), only the outermost one runs parallelly.
TaskGroup::TaskGroup() : Parallel(TaskGroupInstances++ == 0) {}
TaskGroup::~TaskGroup() {
  // We must ensure that all the workloads have finished before decrementing the
  // instances count.
  L.sync();
  --TaskGroupInstances;
}

void TaskGroup::spawn(std::function<void()> F) {
#if LLVM_ENABLE_THREADS
  if (Parallel) {
    L.inc();
    detail::Executor::getDefaultExecutor()->add([&, F = std::move(F)] {
      F();
      L.dec();
    });
    return;
  }
#endif
  F();
}

void TaskGroup::execute(std::function<void()> F) {
  if (parallel::strategy.ThreadsRequested == 1)
    F();
  else
    spawn(F);
}
} // namespace parallel
} // namespace llvm

void llvm::parallelFor(size_t Begin, size_t End,
                       llvm::function_ref<void(size_t)> Fn) {
  // If we have zero or one items, then do not incur the overhead of spinning up
  // a task group.  They are surprisingly expensive, and because they do not
  // support nested parallelism, a single entry task group can block parallel
  // execution underneath them.
#if LLVM_ENABLE_THREADS
  auto NumItems = End - Begin;
  if (NumItems > 1 && parallel::strategy.ThreadsRequested != 1) {
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
