//===-- EJitIpcLock.cpp - Inter-core lock/barrier wrappers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitIpcLock.h"

using namespace llvm;
using namespace llvm::ejit;

#ifdef EJIT_SRE_TASKPOOL_PLATFORM_IPC_LOCK
extern "C" int ejit_sre_bucket_lock_try(uint32_t bucketId)
    __asm__("EJitBucketTryLock");
extern "C" void ejit_sre_bucket_lock(uint32_t bucketId) __asm__("EJitBucketLock");
extern "C" void ejit_sre_bucket_unlock(uint32_t bucketId)
    __asm__("EJitBucketUnlock");
#endif

bool EJitIpcBucketLock::tryLock(uint32_t bucketId) {
  const uint32_t b = normalizeBucket(bucketId);
#ifdef EJIT_SRE_TASKPOOL_PLATFORM_IPC_LOCK
  return ejit_sre_bucket_lock_try(b) != 0;
#else
  uint32_t expected = 0u;
  return lockWords_[b].compareExchange(expected, 1u);
#endif
}

void EJitIpcBucketLock::lock(uint32_t bucketId) {
  const uint32_t b = normalizeBucket(bucketId);
#ifdef EJIT_SRE_TASKPOOL_PLATFORM_IPC_LOCK
  ejit_sre_bucket_lock(b);
#else
  while (!tryLock(b)) {
  }
#endif
  EJitSharedBarrier::fenceAcquire();
}

void EJitIpcBucketLock::unlock(uint32_t bucketId) {
  const uint32_t b = normalizeBucket(bucketId);
  EJitSharedBarrier::fenceRelease();
#ifdef EJIT_SRE_TASKPOOL_PLATFORM_IPC_LOCK
  ejit_sre_bucket_unlock(b);
#else
  lockWords_[b].storeRelease(0u);
#endif
}
