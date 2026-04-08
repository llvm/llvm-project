//===------------------ QueueingTaskDispatcher.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// QueueingTaskDispatcher class.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_QUEUEINGTASKDISPATCHER_H
#define ORC_RT_QUEUEINGTASKDISPATCHER_H

#include "orc-rt/TaskDispatcher.h"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>

namespace orc_rt {

/// A TaskDispatcher implementation that puts tasks in a queue to be run.
/// QueueingTaskDispatcher provides direct access to the queue, allowing
/// clients to decide how to run tasks. It is intended for use on systems
/// where threads are not available, and for unit tests.
/// For most uses of the ORC runtime, use of QueueingTaskDispatcher is strongly
/// discouraged, and alternatives like ThreadPoolTaskDispatcher are preferred.
class QueueingTaskDispatcher : public TaskDispatcher {
public:
  class TaskQueue {
  public:
    /// Append a task to the queue.
    void addTask(std::unique_ptr<Task> T);

    /// Shut down the queue. Further calls to addTask will be ignored (the task
    /// arguments will be discarded).
    void shutdown();

    /// Take the task most recently added to the queue. Blocks until a task is
    /// available or the dispatcher shuts down.
    std::unique_ptr<Task> takeLastIn();

    /// Take the earliest task from the queue. Blocks until a task is available
    /// or the dispatcher shuts down.
    std::unique_ptr<Task> takeFirstIn();

    /// Run tasks in last-in-first-out order until the queue is empty.
    void runLIFOUntilEmpty();

    /// Run tasks in first-in-first-out order until the queue is empty.
    void runFIFOUntilEmpty();

  private:
    std::mutex M;
    std::condition_variable CV;
    enum { Running, Shutdown } State = Running;
    std::deque<std::unique_ptr<Task>> Tasks;
  };

  QueueingTaskDispatcher(TaskQueue &Q) : Q(Q) {}
  void dispatch(std::unique_ptr<Task> T) override;
  void shutdown() override;

private:
  TaskQueue &Q;
};

} // namespace orc_rt

#endif // ORC_RT_QUEUEINGTASKDISPATCHER_H
