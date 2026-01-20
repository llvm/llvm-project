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
  void dispatch(std::unique_ptr<Task> T) override;
  void shutdown() override;

  /// Take a task from the back of the queue. If there are no tasks, returns
  /// nullptr.
  std::unique_ptr<Task> pop_back();

  /// Take a task from the front of the queue. If there are no tasks, returns
  /// nullptr.
  std::unique_ptr<Task> pop_front();

  /// Run tasks in last-in-first-out order until the queue is empty.
  void runLIFOUntilEmpty();

  /// Run tasks in first-in-first-out order until the queue is empty.
  void runFIFOUntilEmpty();

private:
  std::mutex M;
  enum { Running, Shutdown } State = Running;
  std::deque<std::unique_ptr<Task>> Tasks;
};

} // namespace orc_rt

#endif // ORC_RT_QUEUEINGTASKDISPATCHER_H
