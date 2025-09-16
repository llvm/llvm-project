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

#pragma once

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
  /// Subdevice encoding
  int64_t SubDeviceCode = 0;

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

  void clear() {}

  int64_t getSubDeviceCode() { return SubDeviceCode; }

  void setSubDeviceCode(int64_t Code) { SubDeviceCode = Code; }

  AsyncQueueTy *getAsyncQueue() {
    AsyncQueueTy *ret = nullptr;
    if (UsedQueues < PerThreadQueues) {
      // there's a free queue in this thread, find it
      for (int32_t q = 0; q < PerThreadQueues; q++) {
        if (!AsyncQueues[q].InUse) {
          UsedQueues++;
          ret = &AsyncQueues[q];
          break;
        }
      }
      assert(ret && "A queue should have been found!");
      ret->InUse = true;
    }
    return ret;
  }

  bool releaseAsyncQueue(AsyncQueueTy *queue) {
    if (queue >= &AsyncQueues[0] && queue < &AsyncQueues[PerThreadQueues]) {
      // it's a local queue
      queue->InUse = false;
      UsedQueues--;
      return true;
    }
    return false;
  }
};

struct L0ThreadTblTy : public PerThread<L0ThreadTLSTy> {
  void clear() {
    PerThread::clear([](auto &Entry) { Entry.clear(); });
  }
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
