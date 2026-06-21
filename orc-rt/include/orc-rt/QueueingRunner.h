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

#include "orc-rt/WrapperFunction.h"

#include "orc-rt-c/CoreTypes.h"
#include "orc-rt-c/WrapperFunction.h"

#include <cstdint>
#include <utility>

namespace orc_rt {

/// A wrapper-call runner that pushes incoming calls onto a caller-owned work
/// queue, leaving the caller free to drain the queue however and whenever
/// they choose.
///
/// QueueingRunner is intended for use on systems where threads are not
/// available, and for unit tests. For most uses of the ORC runtime,
/// alternatives like ThreadPoolRunner are preferred.
///
/// WorkQueue may be any container that stores `void()`-callable values and
/// supports `push_back`, `pop_back`, `pop_front`, `back()`, `front()`, and
/// `empty()` (e.g. `std::deque<move_only_function<void()>>`). In
/// multi-threaded setups the WorkQueue type itself is responsible for
/// providing whatever synchronization is needed for concurrent push and
/// drain operations.
template <typename WorkQueue> class QueueingRunner {
public:
  QueueingRunner(WorkQueue &Pending) : Pending(Pending) {}

  /// Enqueue a wrapper-function call to be run later.
  void operator()(orc_rt_SessionRef S, uint64_t CallId,
                  orc_rt_WrapperFunctionReturn Return,
                  orc_rt_WrapperFunction Fn, WrapperFunctionBuffer ArgBytes) {
    Pending.push_back([=, ArgBytes = std::move(ArgBytes)]() mutable {
      Fn(S, CallId, Return, ArgBytes.release());
    });
  }

  /// Run all currently-queued calls in last-in-first-out order, returning when
  /// the queue is empty. Calls enqueued during draining are run too.
  static void runLIFOUntilEmpty(WorkQueue &Q) {
    while (!Q.empty()) {
      auto Call = std::move(Q.back());
      Q.pop_back();
      Call();
    }
  }

  /// Run all currently-queued calls in first-in-first-out order, returning
  /// when the queue is empty. Calls enqueued during draining are run too.
  static void runFIFOUntilEmpty(WorkQueue &Q) {
    while (!Q.empty()) {
      auto Call = std::move(Q.front());
      Q.pop_front();
      Call();
    }
  }

private:
  WorkQueue &Pending;
};

template <typename WorkQueue>
QueueingRunner(WorkQueue &) -> QueueingRunner<WorkQueue>;

} // namespace orc_rt

#endif // ORC_RT_QUEUEINGRUNNER_H
