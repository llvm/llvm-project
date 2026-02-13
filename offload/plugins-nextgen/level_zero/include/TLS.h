//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thread Level Storage abstraction.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_TLS_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_TLS_H

#include <bitset>

#include "AsyncQueue.h"
#include "L0Memory.h"
#include "L0Trace.h"
#include "PerThreadTable.h"

namespace llvm::omp::target::plugin {

/// All thread-local data used by the Plugin.
class L0ThreadTLSTy {
  /// Async info tracking.
  static constexpr int32_t PerThreadQueues = 10;
  std::bitset<PerThreadQueues> InUseQueues{0};
  AsyncQueueTy AsyncQueues[PerThreadQueues];

public:
  L0ThreadTLSTy() = default;
  L0ThreadTLSTy(const L0ThreadTLSTy &) = delete;
  L0ThreadTLSTy(L0ThreadTLSTy &&) = delete;
  L0ThreadTLSTy &operator=(const L0ThreadTLSTy &) = delete;
  L0ThreadTLSTy &operator=(const L0ThreadTLSTy &&) = delete;
  ~L0ThreadTLSTy() = default;

  AsyncQueueTy *getAsyncQueue() {
    AsyncQueueTy *Ret = nullptr;
    if (!InUseQueues.all()) {
      // there's a free queue in this thread, find it.
      for (size_t Queue = 0; Queue < PerThreadQueues; Queue++) {
        if (!InUseQueues.test(Queue)) {
          InUseQueues.set(Queue);
          Ret = &AsyncQueues[Queue];
          break;
        }
      }
      assert(Ret && "A queue should have been found!");
    }
    return Ret;
  }

  bool releaseAsyncQueue(AsyncQueueTy *Queue) {
    if (Queue >= &AsyncQueues[0] && Queue < &AsyncQueues[PerThreadQueues]) {
      // it's a local queue
      size_t QueueId = Queue - &AsyncQueues[0];
      InUseQueues.reset(QueueId);
      return true;
    }
    return false;
  }
};

using L0ThreadTblTy = PerThread<L0ThreadTLSTy>;

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_TLS_H
