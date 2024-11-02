//===-- llvm/Support/ThreadPool.h - A ThreadPool implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a crude C++11 based thread pool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_THREADPOOL_H
#define LLVM_SUPPORT_THREADPOOL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/thread.h"

#include <future>

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

namespace llvm {

class ThreadPoolTaskGroup;

/// A ThreadPool for asynchronous parallel execution on a defined number of
/// threads.
///
/// The pool keeps a vector of threads alive, waiting on a condition variable
/// for some work to become available.
///
/// It is possible to reuse one thread pool for different groups of tasks
/// by grouping tasks using ThreadPoolTaskGroup. All tasks are processed using
/// the same queue, but it is possible to wait only for a specific group of
/// tasks to finish.
///
/// It is also possible for worker threads to submit new tasks and wait for
/// them. Note that this may result in a deadlock in cases such as when a task
/// (directly or indirectly) tries to wait for its own completion, or when all
/// available threads are used up by tasks waiting for a task that has no thread
/// left to run on (this includes waiting on the returned future). It should be
/// generally safe to wait() for a group as long as groups do not form a cycle.
class ThreadPool {
public:
  /// Construct a pool using the hardware strategy \p S for mapping hardware
  /// execution resources (threads, cores, CPUs)
  /// Defaults to using the maximum execution resources in the system, but
  /// accounting for the affinity mask.
  ThreadPool(ThreadPoolStrategy S = hardware_concurrency());

  /// Blocking destructor: the pool will wait for all the threads to complete.
  ~ThreadPool();

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename Function, typename... Args>
  auto async(Function &&F, Args &&...ArgList) {
    auto Task =
        std::bind(std::forward<Function>(F), std::forward<Args>(ArgList)...);
    return async(std::move(Task));
  }

  /// Overload, task will be in the given task group.
  template <typename Function, typename... Args>
  auto async(ThreadPoolTaskGroup &Group, Function &&F, Args &&...ArgList) {
    auto Task =
        std::bind(std::forward<Function>(F), std::forward<Args>(ArgList)...);
    return async(Group, std::move(Task));
  }

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename Func>
  auto async(Func &&F) -> std::shared_future<decltype(F())> {
    return asyncImpl(std::function<decltype(F())()>(std::forward<Func>(F)),
                     nullptr);
  }

  template <typename Func>
  auto async(ThreadPoolTaskGroup &Group, Func &&F)
      -> std::shared_future<decltype(F())> {
    return asyncImpl(std::function<decltype(F())()>(std::forward<Func>(F)),
                     &Group);
  }

  /// Blocking wait for all the threads to complete and the queue to be empty.
  /// It is an error to try to add new tasks while blocking on this call.
  /// Calling wait() from a task would deadlock waiting for itself.
  void wait();

  /// Blocking wait for only all the threads in the given group to complete.
  /// It is possible to wait even inside a task, but waiting (directly or
  /// indirectly) on itself will deadlock. If called from a task running on a
  /// worker thread, the call may process pending tasks while waiting in order
  /// not to waste the thread.
  void wait(ThreadPoolTaskGroup &Group);

  // TODO: misleading legacy name warning!
  // Returns the maximum number of worker threads in the pool, not the current
  // number of threads!
  unsigned getThreadCount() const { return MaxThreadCount; }

  /// Returns true if the current thread is a worker thread of this thread pool.
  bool isWorkerThread() const;

private:
  /// Helpers to create a promise and a callable wrapper of \p Task that sets
  /// the result of the promise. Returns the callable and a future to access the
  /// result.
  template <typename ResTy>
  static std::pair<std::function<void()>, std::future<ResTy>>
  createTaskAndFuture(std::function<ResTy()> Task) {
    std::shared_ptr<std::promise<ResTy>> Promise =
        std::make_shared<std::promise<ResTy>>();
    auto F = Promise->get_future();
    return {
        [Promise = std::move(Promise), Task]() { Promise->set_value(Task()); },
        std::move(F)};
  }
  static std::pair<std::function<void()>, std::future<void>>
  createTaskAndFuture(std::function<void()> Task) {
    std::shared_ptr<std::promise<void>> Promise =
        std::make_shared<std::promise<void>>();
    auto F = Promise->get_future();
    return {[Promise = std::move(Promise), Task]() {
              Task();
              Promise->set_value();
            },
            std::move(F)};
  }

  /// Returns true if all tasks in the given group have finished (nullptr means
  /// all tasks regardless of their group). QueueLock must be locked.
  bool workCompletedUnlocked(ThreadPoolTaskGroup *Group) const;

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename ResTy>
  std::shared_future<ResTy> asyncImpl(std::function<ResTy()> Task,
                                      ThreadPoolTaskGroup *Group) {

#if LLVM_ENABLE_THREADS
    /// Wrap the Task in a std::function<void()> that sets the result of the
    /// corresponding future.
    auto R = createTaskAndFuture(Task);

    int requestedThreads;
    {
      // Lock the queue and push the new task
      std::unique_lock<std::mutex> LockGuard(QueueLock);

      // Don't allow enqueueing after disabling the pool
      assert(EnableFlag && "Queuing a thread during ThreadPool destruction");
      Tasks.emplace_back(std::make_pair(std::move(R.first), Group));
      requestedThreads = ActiveThreads + Tasks.size();
    }
    QueueCondition.notify_one();
    grow(requestedThreads);
    return R.second.share();

#else // LLVM_ENABLE_THREADS Disabled

    // Get a Future with launch::deferred execution using std::async
    auto Future = std::async(std::launch::deferred, std::move(Task)).share();
    // Wrap the future so that both ThreadPool::wait() can operate and the
    // returned future can be sync'ed on.
    Tasks.emplace_back(std::make_pair([Future]() { Future.get(); }, Group));
    return Future;
#endif
  }

#if LLVM_ENABLE_THREADS
  // Grow to ensure that we have at least `requested` Threads, but do not go
  // over MaxThreadCount.
  void grow(int requested);

  void processTasks(ThreadPoolTaskGroup *WaitingForGroup);
#endif

  /// Threads in flight
  std::vector<llvm::thread> Threads;
  /// Lock protecting access to the Threads vector.
  mutable llvm::sys::RWMutex ThreadsLock;

  /// Tasks waiting for execution in the pool.
  std::deque<std::pair<std::function<void()>, ThreadPoolTaskGroup *>> Tasks;

  /// Locking and signaling for accessing the Tasks queue.
  std::mutex QueueLock;
  std::condition_variable QueueCondition;

  /// Signaling for job completion (all tasks or all tasks in a group).
  std::condition_variable CompletionCondition;

  /// Keep track of the number of thread actually busy
  unsigned ActiveThreads = 0;
  /// Number of threads active for tasks in the given group (only non-zero).
  DenseMap<ThreadPoolTaskGroup *, unsigned> ActiveGroups;

#if LLVM_ENABLE_THREADS // avoids warning for unused variable
  /// Signal for the destruction of the pool, asking thread to exit.
  bool EnableFlag = true;
#endif

  const ThreadPoolStrategy Strategy;

  /// Maximum number of threads to potentially grow this pool to.
  const unsigned MaxThreadCount;
};

/// A group of tasks to be run on a thread pool. Thread pool tasks in different
/// groups can run on the same threadpool but can be waited for separately.
/// It is even possible for tasks of one group to submit and wait for tasks
/// of another group, as long as this does not form a loop.
class ThreadPoolTaskGroup {
public:
  /// The ThreadPool argument is the thread pool to forward calls to.
  ThreadPoolTaskGroup(ThreadPool &Pool) : Pool(Pool) {}

  /// Blocking destructor: will wait for all the tasks in the group to complete
  /// by calling ThreadPool::wait().
  ~ThreadPoolTaskGroup() { wait(); }

  /// Calls ThreadPool::async() for this group.
  template <typename Function, typename... Args>
  inline auto async(Function &&F, Args &&...ArgList) {
    return Pool.async(*this, std::forward<Function>(F),
                      std::forward<Args>(ArgList)...);
  }

  /// Calls ThreadPool::wait() for this group.
  void wait() { Pool.wait(*this); }

private:
  ThreadPool &Pool;
};

} // namespace llvm

#endif // LLVM_SUPPORT_THREADPOOL_H
