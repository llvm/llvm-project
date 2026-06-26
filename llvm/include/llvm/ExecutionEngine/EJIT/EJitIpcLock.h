//===-- EJitIpcLock.h - Inter-core lock/barrier wrappers -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Lightweight inter-core synchronization wrappers used by the SRE taskpool.
//  This layer deliberately avoids std::mutex/std::thread facilities.
//
//  Design constraints:
//  - lock scope must stay tiny (state transitions / short metadata updates);
//  - never hold a bucket lock while compiling, allocating large memory,
//    invoking ORC/JITLink, printing diagnostics, or blocking on queue I/O;
//  - real platform lock symbols are only declared when enabled (no weak/local
//    fallback that could shadow strong platform implementations).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITIPCLOCK_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITIPCLOCK_H

#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"
#include <cstdint>

namespace llvm {
namespace ejit {

class EJitSharedBarrier {
public:
  static inline void fenceAcquire() { __atomic_thread_fence(__ATOMIC_ACQUIRE); }
  static inline void fenceRelease() { __atomic_thread_fence(__ATOMIC_RELEASE); }
  static inline void fenceFull() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
};

class EJitIpcBucketLock {
public:
  static constexpr uint32_t kDefaultBucketCount = 32u;

  explicit EJitIpcBucketLock(uint32_t bucketCount = kDefaultBucketCount)
      : bucketCount_(bucketCount) {
    for (uint32_t i = 0; i < kDefaultBucketCount; ++i)
      lockWords_[i].storeRelaxed(0u);
  }

  bool tryLock(uint32_t bucketId);
  void lock(uint32_t bucketId);
  void unlock(uint32_t bucketId);

private:
  uint32_t normalizeBucket(uint32_t bucketId) const {
    return bucketCount_ ? (bucketId % bucketCount_) : 0u;
  }

  uint32_t bucketCount_;
  EJitAtomicU32 lockWords_[kDefaultBucketCount];
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITIPCLOCK_H
