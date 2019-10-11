//===- llvm/Support/Parallel.cpp - Parallel algorithms --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Parallel.h"
#include "llvm/Config/llvm-config.h"

#if LLVM_ENABLE_THREADS

#include "llvm/Support/Threading.h"

#include <atomic>
#include <stack>
#include <thread>

namespace llvm {
namespace parallel {
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
  explicit ThreadPoolExecutor(unsigned ThreadCount = hardware_concurrency())
      : Done(ThreadCount) {
    // Spawn all but one of the threads in another thread as spawning threads
    // can take a while.
    std::thread([&, ThreadCount] {
      for (size_t i = 1; i < ThreadCount; ++i) {
        std::thread([=] { work(); }).detach();
      }
      work();
    }).detach();
  }

  ~ThreadPoolExecutor() override {
    std::unique_lock<std::mutex> Lock(Mutex);
    Stop = true;
    Lock.unlock();
    Cond.notify_all();
    // Wait for ~Latch.
  }

  void add(std::function<void()> F) override {
    std::unique_lock<std::mutex> Lock(Mutex);
    WorkStack.push(F);
    Lock.unlock();
    Cond.notify_one();
  }

private:
  void work() {
    while (true) {
      std::unique_lock<std::mutex> Lock(Mutex);
      Cond.wait(Lock, [&] { return Stop || !WorkStack.empty(); });
      if (Stop)
        break;
      auto Task = WorkStack.top();
      WorkStack.pop();
      Lock.unlock();
      Task();
    }
    Done.dec();
  }

  std::atomic<bool> Stop{false};
  std::stack<std::function<void()>> WorkStack;
  std::mutex Mutex;
  std::condition_variable Cond;
  parallel::detail::Latch Done;
};

Executor *Executor::getDefaultExecutor() {
  static ThreadPoolExecutor exec;
  return &exec;
}
} // namespace

static std::atomic<int> TaskGroupInstances;

// Latch::sync() called by the dtor may cause one thread to block. If is a dead
// lock if all threads in the default executor are blocked. To prevent the dead
// lock, only allow the first TaskGroup to run tasks parallelly. In the scenario
// of nested parallel_for_each(), only the outermost one runs parallelly.
TaskGroup::TaskGroup() : Parallel(TaskGroupInstances++ == 0) {}
TaskGroup::~TaskGroup() { --TaskGroupInstances; }

void TaskGroup::spawn(std::function<void()> F) {
  if (Parallel) {
    L.inc();
    Executor::getDefaultExecutor()->add([&, F] {
      F();
      L.dec();
    });
  } else {
    F();
  }
}

} // namespace detail
} // namespace parallel
} // namespace llvm
#endif // LLVM_ENABLE_THREADS
