//===--- ThreadPoolTaskDispatcher.h - Run tasks in thread pool --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ThreadPoolTaskDispatcher implementation.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_THREADPOOLTASKDISPATCHER_H
#define ORC_RT_THREADPOOLTASKDISPATCHER_H

#include "orc-rt/TaskDispatcher.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace orc_rt {

/// Thread-pool based TaskDispatcher.
///
/// Will spawn NumThreads threads to run dispatched Tasks.
class ThreadPoolTaskDispatcher : public TaskDispatcher {
public:
  ThreadPoolTaskDispatcher(size_t NumThreads);
  ~ThreadPoolTaskDispatcher() override;
  void dispatch(std::unique_ptr<Task> T) override;
  void shutdown() override;

private:
  void taskLoop();

  std::vector<std::thread> Threads;

  std::mutex M;
  bool AcceptingTasks = true;
  std::condition_variable CV;
  std::vector<std::unique_ptr<Task>> PendingTasks;
};

} // End namespace orc_rt

#endif // ORC_RT_THREADPOOLTASKDISPATCHER_H
