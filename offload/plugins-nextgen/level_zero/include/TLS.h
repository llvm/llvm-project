//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thread Level Storage abstraction
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_TLS_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_TLS_H

#include "AsyncQueue.h"
#include "L0Memory.h"
#include "L0Trace.h"
#include "PerThreadTable.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// All thread-local data used by the Plugin
class L0ThreadTLSTy {
  /// Async info tracking
  static constexpr int32_t PerThreadQueues = 10;
  AsyncQueueTy AsyncQueues[PerThreadQueues];
  int32_t UsedQueues = 0;

public:
  L0ThreadTLSTy() = default;
  L0ThreadTLSTy(const L0ThreadTLSTy &) = delete;
  L0ThreadTLSTy(L0ThreadTLSTy &&) = delete;
  L0ThreadTLSTy &operator=(const L0ThreadTLSTy &) = delete;
  L0ThreadTLSTy &operator=(const L0ThreadTLSTy &&) = delete;
  ~L0ThreadTLSTy() {}

  AsyncQueueTy *getAsyncQueue() {
    AsyncQueueTy *Ret = nullptr;
    if (UsedQueues < PerThreadQueues) {
      // there's a free queue in this thread, find it
      for (int32_t Queue = 0; Queue < PerThreadQueues; Queue++) {
        if (!AsyncQueues[Queue].InUse) {
          UsedQueues++;
          Ret = &AsyncQueues[Queue];
          break;
        }
      }
      assert(Ret && "A queue should have been found!");
      Ret->InUse = true;
    }
    return Ret;
  }

  bool releaseAsyncQueue(AsyncQueueTy *Queue) {
    if (Queue >= &AsyncQueues[0] && Queue < &AsyncQueues[PerThreadQueues]) {
      // it's a local queue
      Queue->InUse = false;
      UsedQueues--;
      return true;
    }
    return false;
  }
};

using L0ThreadTblTy = PerThread<L0ThreadTLSTy>;

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_TLS_H
