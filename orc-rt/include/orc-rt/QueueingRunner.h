//===------------------------ QueueingRunner.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// QueueingRunner class template.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_QUEUEINGRUNNER_H
#define ORC_RT_QUEUEINGRUNNER_H

#include "orc-rt/move_only_function.h"

#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace orc_rt {
namespace detail {

template <typename T> class SynchronizedDeque {
public:
  void push_back(T V) {
    std::scoped_lock<std::mutex> Lock(M);
    Q.push_back(std::move(V));
  }

  std::optional<T> pop_back() {
    std::scoped_lock<std::mutex> Lock(M);
    if (Q.empty())
      return std::nullopt;
    auto V = std::move(Q.back());
    Q.pop_back();
    return V;
  }

  std::optional<T> pop_front() {
    std::scoped_lock<std::mutex> Lock(M);
    if (Q.empty())
      return std::nullopt;
    auto V = std::move(Q.front());
    Q.pop_front();
    return V;
  }

private:
  std::mutex M;
  std::deque<T> Q;
};

} // namespace detail

/// A task runner that pushes dispatched tasks onto a caller-owned work queue,
/// leaving the caller free to drain the queue however and whenever they
/// choose.
///
/// QueueingRunner is intended for use on systems where threads are not
/// available, and for unit tests. For most uses of the ORC runtime,
/// alternatives like ThreadPoolRunner are preferred.
///
/// WorkQueue may be any container that stores `void()`-callable values and
/// supports `push_back(T)`, `std::optional<T> pop_back()`, and
/// `std::optional<T> pop_front()`, where the pop operations return
/// `std::nullopt` on an empty queue (e.g. detail::SynchronizedDeque). In
/// multi-threaded setups the WorkQueue type itself is responsible for
/// providing whatever synchronization is needed for concurrent push and
/// drain operations.
template <typename WorkQueueT =
              detail::SynchronizedDeque<move_only_function<void()>>>
class QueueingRunner {
public:
  using WorkQueue = WorkQueueT;

  QueueingRunner(WorkQueueT &Pending) : Pending(Pending) {}

  /// Enqueue a task to be run later.
  void operator()(move_only_function<void()> Task) {
    Pending.push_back(std::move(Task));
  }

  /// Run all currently-queued calls in last-in-first-out order, returning when
  /// the queue is empty. Calls enqueued during draining are run too.
  static void runLIFOUntilEmpty(WorkQueueT &Q) {
    while (auto Call = Q.pop_back())
      (*Call)();
  }

  /// Run all currently-queued calls in first-in-first-out order, returning
  /// when the queue is empty. Calls enqueued during draining are run too.
  static void runFIFOUntilEmpty(WorkQueueT &Q) {
    while (auto Call = Q.pop_front())
      (*Call)();
  }

private:
  WorkQueueT &Pending;
};

template <typename WorkQueueT>
QueueingRunner(WorkQueueT &) -> QueueingRunner<WorkQueueT>;

} // namespace orc_rt

#endif // ORC_RT_QUEUEINGRUNNER_H
