//===- ThreadPoolRunner.h -- Run wrapper calls in a thread pool -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ThreadPoolRunner implementation.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_THREADPOOLRUNNER_H
#define ORC_RT_THREADPOOLRUNNER_H

#include "orc-rt/WrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include "orc-rt-c/CoreTypes.h"
#include "orc-rt-c/WrapperFunction.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace orc_rt {

/// A wrapper-call runner backed by a fixed-size pool of worker threads.
///
/// Each incoming call is enqueued onto an internal work queue and picked up
/// by a worker thread.
///
/// Lifetime: the runner must outlive the Session that uses it. By the time
/// the runner is destroyed the Session must have shut down — operator()
/// after destruction begins is a contract violation (asserted). Any calls
/// already pending in the queue at destruction time will be drained by the
/// workers before they exit.
class ThreadPoolRunner {
public:
  ThreadPoolRunner(size_t NumThreads);
  ~ThreadPoolRunner();

  ThreadPoolRunner(const ThreadPoolRunner &) = delete;
  ThreadPoolRunner &operator=(const ThreadPoolRunner &) = delete;
  ThreadPoolRunner(ThreadPoolRunner &&) = delete;
  ThreadPoolRunner &operator=(ThreadPoolRunner &&) = delete;

  /// Enqueue a wrapper-function call to be run by a worker thread. Must not
  /// be called once destruction has begun.
  void operator()(orc_rt_SessionRef S, uint64_t CallId,
                  orc_rt_WrapperFunctionReturn Return,
                  orc_rt_WrapperFunction Fn, WrapperFunctionBuffer ArgBytes);

private:
  void workerLoop();

  std::vector<std::thread> Workers;

  std::mutex M;
  std::condition_variable CV;
  /// Set by the destructor. Stops new dispatches (asserted in operator())
  /// and tells worker threads to exit once they've drained Pending.
  bool Stop = false;
  std::vector<move_only_function<void()>> Pending;
};

} // namespace orc_rt

#endif // ORC_RT_THREADPOOLRUNNER_H
