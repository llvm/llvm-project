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

#include "llvm/Support/thread.h"

#include <future>

#include <atomic>
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
public:
#ifndef _MSC_VER
  using VoidTy = void;
  using TaskTy = std::function<void()>;
  using PackagedTaskTy = std::packaged_task<void()>;
#else
  // MSVC 2013 has a bug and can't use std::packaged_task<void()>;
  // We force it to use bool(bool) instead.
  using VoidTy = bool;
  using TaskTy = std::function<bool(bool)>;
  using PackagedTaskTy = std::packaged_task<bool(bool)>;
#endif

  /// Construct a pool with the number of core available on the system (or
  /// whatever the value returned by std::thread::hardware_concurrency() is).
  ThreadPool();

  /// Construct a pool of \p ThreadCount threads
  ThreadPool(unsigned ThreadCount);

  /// Blocking destructor: the pool will wait for all the threads to complete.
  ~ThreadPool();

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename Function, typename... Args>
  inline std::shared_future<VoidTy> async(Function &&F, Args &&... ArgList) {
    auto Task =
        std::bind(std::forward<Function>(F), std::forward<Args>(ArgList)...);
#ifndef _MSC_VER
    return asyncImpl(std::move(Task));
#else
    // This lambda has to be marked mutable because MSVC 2013's std::bind call
    // operator isn't const qualified.
    return asyncImpl([Task](VoidTy) mutable -> VoidTy {
      Task();
      return VoidTy();
    });
#endif
  }

  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  template <typename Function>
  inline std::shared_future<VoidTy> async(Function &&F) {
#ifndef _MSC_VER
    return asyncImpl(std::forward<Function>(F));
#else
    return asyncImpl([F] (VoidTy) -> VoidTy { F(); return VoidTy(); });
#endif
  }

  /// Blocking wait for all the threads to complete and the queue to be empty.
  /// It is an error to try to add new tasks while blocking on this call.
  void wait();

private:
  /// Asynchronous submission of a task to the pool. The returned future can be
  /// used to wait for the task to finish and is *non-blocking* on destruction.
  std::shared_future<VoidTy> asyncImpl(TaskTy F);

  /// Threads in flight
  std::vector<llvm::thread> Threads;

  /// Tasks waiting for execution in the pool.
  std::queue<PackagedTaskTy> Tasks;

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
