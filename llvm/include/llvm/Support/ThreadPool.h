//===-- llvm/Support/ThreadPool.h - A ThreadPool implementation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a crude C++11 based thread pool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_THREAD_POOL_H
#define LLVM_SUPPORT_THREAD_POOL_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/thread.h"

#include <future>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

namespace llvm {

/// A ThreadPool for asynchronous parallel execution on a defined number of
/// threads.
///
/// The pool keeps a vector of threads alive, waiting on a condition variable
/// for some work to become available.
class ThreadPool {
  struct TaskBase {
    virtual ~TaskBase() {}
    virtual void execute() = 0;
  };

  template <typename ReturnType> struct TypedTask : public TaskBase {
    explicit TypedTask(std::packaged_task<ReturnType()> Task)
        : Task(std::move(Task)) {}

    void execute() override { Task(); }

    std::packaged_task<ReturnType()> Task;
  };

public:
  /// Construct a pool with the number of threads found by
  /// hardware_concurrency().
  ThreadPool();

  /// Construct a pool of \p ThreadCount threads
  ThreadPool(unsigned ThreadCount);

  /// Blocking destructor: the pool will wait for all the threads to complete.
  ~ThreadPool();

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename Function, typename... Args>
  inline std::shared_future<typename std::result_of<Function(Args...)>::type>
  async(Function &&F, Args &&... ArgList) {
    auto Task =
        std::bind(std::forward<Function>(F), std::forward<Args>(ArgList)...);
    return asyncImpl(std::move(Task));
  }

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename Function>
  inline std::shared_future<typename std::result_of<Function()>::type>
  async(Function &&F) {
    return asyncImpl(std::forward<Function>(F));
  }

  /// Blocking wait for all the threads to complete and the queue to be empty.
  /// It is an error to try to add new tasks while blocking on this call.
  void wait();

private:
  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename TaskTy>
  std::shared_future<typename std::result_of<TaskTy()>::type>
  asyncImpl(TaskTy &&Task) {
    typedef decltype(Task()) ResultTy;

    /// Wrap the Task in a packaged_task to return a future object.
    std::packaged_task<ResultTy()> PackagedTask(std::move(Task));
    auto Future = PackagedTask.get_future();
    std::unique_ptr<TaskBase> TB =
        llvm::make_unique<TypedTask<ResultTy>>(std::move(PackagedTask));

    {
      // Lock the queue and push the new task
      std::unique_lock<std::mutex> LockGuard(QueueLock);

      // Don't allow enqueueing after disabling the pool
      assert(EnableFlag && "Queuing a thread during ThreadPool destruction");

      Tasks.push(std::move(TB));
    }
    QueueCondition.notify_one();
    return Future.share();
  }

  /// Threads in flight
  std::vector<llvm::thread> Threads;

  /// Tasks waiting for execution in the pool.
  std::queue<std::unique_ptr<TaskBase>> Tasks;

  /// Locking and signaling for accessing the Tasks queue.
  std::mutex QueueLock;
  std::condition_variable QueueCondition;

  /// Locking and signaling for job completion
  std::mutex CompletionLock;
  std::condition_variable CompletionCondition;

  /// Keep track of the number of thread actually busy
  std::atomic<unsigned> ActiveThreads;

#if LLVM_ENABLE_THREADS // avoids warning for unused variable
  /// Signal for the destruction of the pool, asking thread to exit.
  bool EnableFlag;
#endif
};
}

#endif // LLVM_SUPPORT_THREAD_POOL_H
