//===-- EJitTaskPoolTest.cpp - Unit tests for the SRE taskpool ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Host-runnable tests for the EmbeddedJIT SRE taskpool. This executable
//  compiles EJitTaskPool.cpp + EJitSreQueue.cpp directly with -DEJIT_SRE_TASKPOOL
//  so it builds and runs regardless of whether LLVMEJIT / EJITTests build, and
//  without any real SRE platform symbols. Everything is single-threaded and
//  uses an injected mock compiler — no JIT/OrcJIT dependency.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"
#include "llvm/ExecutionEngine/EJIT/EJitIpcLock.h"
#include "llvm/ExecutionEngine/EJIT/EJitSreQueue.h"
#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>

using namespace llvm;
using namespace llvm::ejit;

namespace {

/// Build a cacheKey: funcIdx in the high 32 bits, dims in the low 32 bits.
uint64_t key(uint32_t funcIdx, uint32_t low = 0) {
  return (static_cast<uint64_t>(funcIdx) << 32) | low;
}

/// Mock compiler injected into the taskpool via a plain function pointer.
/// Tracks the number of compile calls and can be made to fail once.
struct MockCompiler {
  static int calls;
  static bool failNext;
  static uintptr_t base;

  static void reset() {
    calls = 0;
    failNext = false;
    base = 0x100000;
  }

  static bool compile(void *ctx, const EJitCompileRequest &req, void **outFn) {
    (void)ctx;
    ++calls;
    if (failNext) {
      failNext = false;
      *outFn = nullptr;
      return false;
    }
    // Deterministic, distinct pointer per function index.
    *outFn = reinterpret_cast<void *>(base + req.funcIndex);
    return true;
  }
};
int MockCompiler::calls = 0;
bool MockCompiler::failNext = false;
uintptr_t MockCompiler::base = 0x100000;

void *expectedPtr(uint32_t funcIdx) {
  return reinterpret_cast<void *>(uintptr_t(0x100000) + funcIdx);
}

EJitCompileRequest makeReq(uint32_t funcIdx, uint64_t cacheKey, void *fb) {
  EJitCompileRequest r;
  r.funcIndex = funcIdx;
  r.version = 0;
  r.cacheKey = cacheKey;
  r.fallbackPtr = reinterpret_cast<uintptr_t>(fb);
  r.userData = 0;
  return r;
}

std::string readTextFile(const std::string &path) {
  std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
  if (!in)
    return std::string();
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

} // namespace

//===----------------------------------------------------------------------===//
// 1. Atomic wrapper
//===----------------------------------------------------------------------===//

TEST(EJitAtomic, LoadStoreU32) {
  EJitAtomicU32 a(5u);
  EXPECT_EQ(a.loadAcquire(), 5u);
  a.storeRelease(10u);
  EXPECT_EQ(a.loadRelaxed(), 10u);
  a.storeRelaxed(11u);
  EXPECT_EQ(a.loadAcquire(), 11u);
}

TEST(EJitAtomic, CompareExchangeSuccessAndFailure) {
  EJitAtomicU32 a(10u);

  uint32_t expected = 10u;
  EXPECT_TRUE(a.compareExchange(expected, 20u));
  EXPECT_EQ(a.loadAcquire(), 20u);

  // Failing CAS leaves the value unchanged and updates expected to observed.
  expected = 999u;
  EXPECT_FALSE(a.compareExchange(expected, 30u));
  EXPECT_EQ(expected, 20u);
  EXPECT_EQ(a.loadAcquire(), 20u);
}

TEST(EJitAtomic, U64AndPtrAndFetchAdd) {
  EJitAtomicU64 b(0);
  b.storeRelease(0x1122334455667788ull);
  EXPECT_EQ(b.loadAcquire(), 0x1122334455667788ull);

  EJitAtomicUPtr p(0);
  p.storeRelease(reinterpret_cast<uintptr_t>(&b));
  EXPECT_EQ(p.loadAcquire(), reinterpret_cast<uintptr_t>(&b));

  EJitAtomicU32 c(0);
  EXPECT_EQ(c.fetchAdd(1u), 0u);
  EXPECT_EQ(c.fetchAdd(1u), 1u);
  EXPECT_EQ(c.loadAcquire(), 2u);
}

//===----------------------------------------------------------------------===//
// 1b. IPC lock/barrier wrapper
//===----------------------------------------------------------------------===//

TEST(EJitIpcLock, TryLockAndUnlock) {
  EJitIpcBucketLock locks(32u);
  EXPECT_TRUE(locks.tryLock(3));
  EXPECT_FALSE(locks.tryLock(3));
  locks.unlock(3);
  EXPECT_TRUE(locks.tryLock(3));
  locks.unlock(3);
}

TEST(EJitIpcLock, LockUnlockNormalizedBucket) {
  EJitIpcBucketLock locks(32u);
  locks.lock(35u); // 35 % 32 == 3
  EXPECT_FALSE(locks.tryLock(3u));
  locks.unlock(35u);
  EXPECT_TRUE(locks.tryLock(3u));
  locks.unlock(3u);
}

//===----------------------------------------------------------------------===//
// 2. SwitchController
//===----------------------------------------------------------------------===//

TEST(EJitSwitchController, DefaultsAndSetters) {
  EJitSwitchController sc;
  EXPECT_TRUE(sc.isEnabled());
  EXPECT_EQ(sc.getMode(), EJitCompileMode::Sync);
  EXPECT_EQ(sc.getVersion(), 1u);

  sc.setMode(EJitCompileMode::Async);
  EXPECT_EQ(sc.getMode(), EJitCompileMode::Async);
  sc.setEnabled(false);
  EXPECT_FALSE(sc.isEnabled());
  sc.setMode(EJitCompileMode::Off);
  EXPECT_EQ(sc.getMode(), EJitCompileMode::Off);
}

TEST(EJitSwitchController, BumpVersion) {
  EJitSwitchController sc;
  EXPECT_EQ(sc.getVersion(), 1u);
  EXPECT_EQ(sc.bumpVersion(), 2u);
  EXPECT_EQ(sc.bumpVersion(), 3u);
  EXPECT_EQ(sc.getVersion(), 3u);
}

TEST(EJitSwitchController, OldVersionRequestDroppedByPoll) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xF00D));
  auto r = pool.compileOrGet(3, key(3), fb);
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::EnqueuedPending);

  // Bump the version: the queued request is now stale.
  pool.switchController().bumpVersion();
  EXPECT_TRUE(pool.pollOne()); // item dequeued...
  EXPECT_EQ(MockCompiler::calls, 0); // ...but dropped, never compiled.

  // Nothing was published.
  EXPECT_EQ(pool.cache().lookup(3, key(3), pool.switchController().getVersion()),
            nullptr);
}

TEST(EJitSwitchController, DeactivateAndActivateBumpVersion) {
  EJitSwitchController sc;
  uint32_t v1 = sc.getVersion();
  uint32_t v2 = sc.deactivate();
  EXPECT_FALSE(sc.isEnabled());
  EXPECT_EQ(v2, v1 + 1u);

  uint32_t v3 = sc.activate(EJitCompileMode::Async);
  EXPECT_TRUE(sc.isEnabled());
  EXPECT_EQ(sc.getMode(), EJitCompileMode::Async);
  EXPECT_EQ(v3, v2 + 1u);
}

//===----------------------------------------------------------------------===//
// 3. Queue
//===----------------------------------------------------------------------===//

TEST(EJitQueue, CapacityFullAndFifo) {
  EJitQueue q(4);
  EXPECT_EQ(q.capacity(), 4u);
  EXPECT_EQ(q.approximateSize(), 0u);

  void *fb = nullptr;
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_TRUE(q.push(makeReq(i, key(i), fb)));
  EXPECT_EQ(q.approximateSize(), 4u);

  // Full now.
  EXPECT_FALSE(q.push(makeReq(99, key(99), fb)));

  // FIFO order.
  for (uint32_t i = 0; i < 4; ++i) {
    EJitCompileRequest out;
    EXPECT_TRUE(q.pop(out));
    EXPECT_EQ(out.funcIndex, i);
  }
  EJitCompileRequest out;
  EXPECT_FALSE(q.pop(out)); // empty
}

TEST(EJitQueue, CapacityRoundsUpToPowerOfTwo) {
  EJitQueue q(5);
  EXPECT_EQ(q.capacity(), 8u);
}

//===----------------------------------------------------------------------===//
// 4. Dedup
//===----------------------------------------------------------------------===//

TEST(EJitDedupTable, SameKeyOnlyOnce) {
  EJitDedupTable d;
  EXPECT_EQ(d.tryMarkPending(7, key(7), 1),
            EJitDedupResult::AcquiredPending);
  EXPECT_EQ(d.tryMarkPending(7, key(7), 1),
            EJitDedupResult::AlreadyPending);
}

TEST(EJitDedupTable, DistinctFuncSameBucketNotConfused) {
  EJitDedupTable d;
  const uint32_t a = 1;
  const uint32_t b = 1 + EJitDedupTable::kBuckets; // same bucket, different func
  EXPECT_EQ(a % EJitDedupTable::kBuckets, b % EJitDedupTable::kBuckets);
  EXPECT_EQ(d.tryMarkPending(a, key(a), 1),
            EJitDedupResult::AcquiredPending);
  EXPECT_EQ(d.tryMarkPending(b, key(b), 1),
            EJitDedupResult::AcquiredPending);
  // The first one is still recognized as pending.
  EXPECT_EQ(d.tryMarkPending(a, key(a), 1),
            EJitDedupResult::AlreadyPending);
}

TEST(EJitDedupTable, BucketFullReturnsDedupFull) {
  EJitDedupTable d;
  // Fill all slots of one bucket with distinct funcIndex (same bucket).
  for (uint32_t i = 0; i < EJitDedupTable::kSlots; ++i) {
    uint32_t f = 2 + i * EJitDedupTable::kBuckets;
    EXPECT_EQ(d.tryMarkPending(f, key(f), 1),
              EJitDedupResult::AcquiredPending);
  }
  uint32_t overflow = 2 + EJitDedupTable::kSlots * EJitDedupTable::kBuckets;
  EXPECT_EQ(d.tryMarkPending(overflow, key(overflow), 1),
            EJitDedupResult::DedupFull);
}

TEST(EJitDedupTable, ClearAllowsReacquire) {
  EJitDedupTable d;
  EXPECT_EQ(d.tryMarkPending(5, key(5), 1),
            EJitDedupResult::AcquiredPending);
  d.clear(5, key(5), 1);
  // After clear, it can be reacquired (rollback semantics).
  EXPECT_EQ(d.tryMarkPending(5, key(5), 1),
            EJitDedupResult::AcquiredPending);
}

TEST(EJitDedupTable, CompileTransitions) {
  EJitDedupTable d;
  EXPECT_EQ(d.tryMarkPending(8, key(8), 1),
            EJitDedupResult::AcquiredPending);
  EXPECT_TRUE(d.markCompiling(8, key(8), 1));
  // Cannot mark compiling twice.
  EXPECT_FALSE(d.markCompiling(8, key(8), 1));
  // Commit gate part 1: Compiling -> Publishing, exactly once.
  EXPECT_TRUE(d.beginPublish(8, key(8), 1));
  EXPECT_FALSE(d.beginPublish(8, key(8), 1));
  // Commit gate part 2: Publishing -> Empty, exactly once.
  EXPECT_TRUE(d.finishPublish(8, key(8), 1));
  EXPECT_FALSE(d.finishPublish(8, key(8), 1));
  // Slot is free again.
  EXPECT_EQ(d.pendingCount(), 0u);
}

TEST(EJitDedupTable, FinishPublishBlockedByCancel) {
  EJitDedupTable d;
  EXPECT_EQ(d.tryMarkPending(8, key(8), 1),
            EJitDedupResult::AcquiredPending);
  EXPECT_TRUE(d.markCompiling(8, key(8), 1));
  EXPECT_TRUE(d.beginPublish(8, key(8), 1));
  // freeCode-style cancel forces the Publishing slot back to Empty.
  EXPECT_TRUE(d.cancel(8, key(8)));
  // The worker's finishPublish then loses the gate → must not commit.
  EXPECT_FALSE(d.finishPublish(8, key(8), 1));
}

//===----------------------------------------------------------------------===//
// 5. Sync path
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolSync, MissCompilesThenHits) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Sync);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xBEEF));

  auto r1 = pool.compileOrGet(4, key(4), fb);
  EXPECT_EQ(r1.status, EJitCompileOrGetStatus::SyncCompiled);
  EXPECT_EQ(r1.fnPtr, expectedPtr(4));
  EXPECT_EQ(MockCompiler::calls, 1);

  // Second call hits the cache; no new compile.
  auto r2 = pool.compileOrGet(4, key(4), fb);
  EXPECT_EQ(r2.status, EJitCompileOrGetStatus::CacheHit);
  EXPECT_EQ(r2.fnPtr, expectedPtr(4));
  EXPECT_EQ(MockCompiler::calls, 1);
  EXPECT_EQ(pool.pendingCount(), 0u);
}

TEST(EJitTaskPoolSync, CompileFailureClearsDedupAndRetries) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Sync);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xBEEF));

  MockCompiler::failNext = true;
  auto r1 = pool.compileOrGet(6, key(6), fb);
  EXPECT_EQ(r1.status, EJitCompileOrGetStatus::CompileFailed);
  EXPECT_EQ(r1.fnPtr, fb);
  EXPECT_EQ(pool.pendingCount(), 0u); // dedup cleared

  // Retry succeeds now.
  auto r2 = pool.compileOrGet(6, key(6), fb);
  EXPECT_EQ(r2.status, EJitCompileOrGetStatus::SyncCompiled);
  EXPECT_EQ(r2.fnPtr, expectedPtr(6));
}

TEST(EJitTaskPoolSync, DisabledReturnsFallback) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Off);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xBEEF));
  auto r = pool.compileOrGet(4, key(4), fb);
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::DisabledFallback);
  EXPECT_EQ(r.fnPtr, fb);
  EXPECT_EQ(MockCompiler::calls, 0);
}

//===----------------------------------------------------------------------===//
// 6. Async path
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolAsync, EnqueuePollPublish) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xCAFE));

  auto r1 = pool.compileOrGet(9, key(9), fb);
  EXPECT_EQ(r1.status, EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_EQ(r1.fnPtr, fb);
  EXPECT_EQ(MockCompiler::calls, 0);
  EXPECT_EQ(pool.pendingCount(), 1u);

  // Duplicate submit does not enqueue again.
  auto r2 = pool.compileOrGet(9, key(9), fb);
  EXPECT_EQ(r2.status, EJitCompileOrGetStatus::AlreadyPending);
  EXPECT_EQ(pool.queue().approximateSize(), 1u);

  // Worker compiles and publishes.
  EXPECT_TRUE(pool.pollOne());
  EXPECT_EQ(MockCompiler::calls, 1);
  EXPECT_FALSE(pool.pollOne()); // queue empty now

  // Now it is a cache hit.
  auto r3 = pool.compileOrGet(9, key(9), fb);
  EXPECT_EQ(r3.status, EJitCompileOrGetStatus::CacheHit);
  EXPECT_EQ(r3.fnPtr, expectedPtr(9));
  EXPECT_EQ(pool.pendingCount(), 0u);
}

TEST(EJitTaskPoolAsync, QueueFullRollsBackDedup) {
  MockCompiler::reset();
  EJitTaskPool pool(4); // tiny queue
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xCAFE));

  // Fill the queue with 4 distinct requests.
  for (uint32_t i = 1; i <= 4; ++i) {
    auto r = pool.compileOrGet(i, key(i), fb);
    EXPECT_EQ(r.status, EJitCompileOrGetStatus::EnqueuedPending);
  }
  EXPECT_EQ(pool.pendingCount(), 4u);

  // The 5th request cannot be enqueued; its dedup reservation rolls back.
  auto r5 = pool.compileOrGet(5, key(5), fb);
  EXPECT_EQ(r5.status, EJitCompileOrGetStatus::QueueFullFallback);
  EXPECT_EQ(r5.fnPtr, fb);
  EXPECT_EQ(pool.pendingCount(), 4u); // still 4 — func 5 was rolled back

  // Drain one, then func 5 can be enqueued.
  EXPECT_TRUE(pool.pollOne());
  auto r5b = pool.compileOrGet(5, key(5), fb);
  EXPECT_EQ(r5b.status, EJitCompileOrGetStatus::EnqueuedPending);
}

TEST(EJitTaskPoolAsync, PollBudgetDrainsMultiple) {
  MockCompiler::reset();
  EJitTaskPool pool(32);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);

  void *fb = nullptr;
  for (uint32_t i = 1; i <= 5; ++i)
    pool.compileOrGet(i, key(i), fb);
  EXPECT_EQ(pool.pendingCount(), 5u);

  EXPECT_EQ(pool.pollBudget(3), 3u);
  EXPECT_EQ(pool.pollBudget(10), 2u); // only 2 left
  EXPECT_EQ(pool.pollBudget(10), 0u); // empty
  EXPECT_EQ(MockCompiler::calls, 5);
}

//===----------------------------------------------------------------------===//
// 7. FreeCode
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolFreeCode, ReadyEntryFreedThenMiss) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Sync);

  void *fb = nullptr;
  auto r1 = pool.compileOrGet(4, key(4), fb);
  EXPECT_EQ(r1.status, EJitCompileOrGetStatus::SyncCompiled);

  EXPECT_TRUE(pool.freeCode(4, key(4)));
  uint32_t v = pool.switchController().getVersion();
  EXPECT_EQ(pool.cache().lookup(4, key(4), v), nullptr); // logical miss
}

TEST(EJitTaskPoolFreeCode, PendingFreedWorkerDoesNotPublish) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xCAFE));
  auto r1 = pool.compileOrGet(9, key(9), fb);
  EXPECT_EQ(r1.status, EJitCompileOrGetStatus::EnqueuedPending);

  // Cancel the in-flight request before the worker runs.
  EXPECT_TRUE(pool.freeCode(9, key(9)));

  // Worker dequeues but must not compile/publish a cancelled request.
  EXPECT_TRUE(pool.pollOne());
  EXPECT_EQ(MockCompiler::calls, 0);
  uint32_t v = pool.switchController().getVersion();
  EXPECT_EQ(pool.cache().lookup(9, key(9), v), nullptr);
}

//===----------------------------------------------------------------------===//
// 8. "No threads" structural guards (see also the source grep in the design
//    doc / final report). These compile-time checks assert the request record
//    stays a flat, trivially-copyable POD suitable for a platform queue.
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolNoThreads, RequestIsFlatPod) {
  static_assert(std::is_trivially_copyable<EJitCompileRequest>::value,
                "EJitCompileRequest must be trivially copyable");
  static_assert(std::is_standard_layout<EJitCompileRequest>::value,
                "EJitCompileRequest must be standard layout");
  static_assert(sizeof(EJitCompileRequest) == 16 + 2 * sizeof(uintptr_t),
                "EJitCompileRequest must be tightly packed");
  EXPECT_LE(alignof(EJitCompileRequest), 8u);
}

//===----------------------------------------------------------------------===//
// 9. Publish failure: a full cache bucket must not be reported as success and
//    must not leave a false cache entry. (Task 1)
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolCacheFull, PublishBucketFullReturnsCacheFull) {
  MockCompiler::reset();
  EJitTaskPool pool(64);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Sync);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xBEEF));
  const uint32_t B = EJitTaskPoolCache::kBuckets;
  const uint32_t S = EJitTaskPoolCache::kSlots;

  // Fill every slot of bucket 0 with a distinct Ready entry (funcIndex % B == 0).
  for (uint32_t i = 1; i <= S; ++i) {
    uint32_t f = i * B;
    auto r = pool.compileOrGet(f, key(f), fb);
    EXPECT_EQ(r.status, EJitCompileOrGetStatus::SyncCompiled);
  }
  EXPECT_EQ(pool.cache().readyCount(), S);

  // A new distinct key in the same (full) bucket cannot be published.
  uint32_t overflow = (S + 1) * B;
  auto rf = pool.compileOrGet(overflow, key(overflow), fb);
  EXPECT_EQ(rf.status, EJitCompileOrGetStatus::CacheFullFallback);
  EXPECT_EQ(rf.fnPtr, fb);

  // No false hit, and the dedup reservation was released (not stuck pending).
  uint32_t v = pool.switchController().getVersion();
  EXPECT_EQ(pool.cache().lookup(overflow, key(overflow), v), nullptr);
  EXPECT_EQ(pool.pendingCount(), 0u);

  EJitTaskPoolStatsSnapshot s;
  pool.getStats(s);
  EXPECT_EQ(s.publishFailed, 1u);
}

//===----------------------------------------------------------------------===//
// 10. Dedup Claiming protocol: a slot still in Claiming (identity written but
//     not yet committed) must not be treated as a valid duplicate, and must not
//     stall a concurrent producer. (Task 2)
//===----------------------------------------------------------------------===//

TEST(EJitDedupTableClaiming, ClaimingSlotNotMatchedAsDuplicate) {
  EJitDedupTable d;

  // Leave a slot in Claiming for (7, key7, v1).
  EXPECT_TRUE(d.beginClaimForTest(7, key(7), 1));
  EXPECT_EQ(d.peekStateForTest(7, key(7)),
            static_cast<uint32_t>(EJitDedupClaiming));

  // A producer for the SAME key must still acquire its own Pending slot — the
  // Claiming slot is deliberately invisible to duplicate matching.
  EXPECT_EQ(d.tryMarkPending(7, key(7), 1),
            EJitDedupResult::AcquiredPending);

  // Once the claiming slot commits to Pending, further submissions coalesce.
  d.releaseClaimForTest(7, key(7), 1);
  EXPECT_EQ(d.tryMarkPending(7, key(7), 1),
            EJitDedupResult::AlreadyPending);
}

TEST(EJitDedupTableClaiming, PublishedIdentityIsConsistent) {
  EJitDedupTable d;
  // After a normal tryMarkPending, the slot is committed (Pending) with a fully
  // published identity, so an immediate duplicate is detected.
  EXPECT_EQ(d.tryMarkPending(3, key(3), 5),
            EJitDedupResult::AcquiredPending);
  EXPECT_EQ(d.peekStateForTest(3, key(3)),
            static_cast<uint32_t>(EJitDedupPending));
  EXPECT_EQ(d.tryMarkPending(3, key(3), 5),
            EJitDedupResult::AlreadyPending);
  // A different version of the same func is a different request.
  EXPECT_EQ(d.tryMarkPending(3, key(3), 6),
            EJitDedupResult::AcquiredPending);
}

//===----------------------------------------------------------------------===//
// 11. FreeCode vs worker publish race. The publish gate (Compiling ->
//     Publishing -> Empty) must guarantee a cancelled key never stays cached,
//     whether freeCode strikes during compile or inside the publish window.
//     (Task 3)
//===----------------------------------------------------------------------===//

namespace {
/// Mock compiler that invokes freeCode() from inside the compile callback,
/// modelling a freeCode arriving while the slot is in Compiling.
struct FreeDuringCompile {
  static EJitTaskPool *pool;
  static int calls;
  static bool compile(void *ctx, const EJitCompileRequest &req, void **outFn) {
    (void)ctx;
    ++calls;
    if (pool)
      pool->freeCode(req.funcIndex, req.cacheKey);
    *outFn = reinterpret_cast<void *>(uintptr_t(0x200000) + req.funcIndex);
    return true;
  }
};
EJitTaskPool *FreeDuringCompile::pool = nullptr;
int FreeDuringCompile::calls = 0;

/// Pre-publish hook that invokes freeCode(), modelling a freeCode arriving in
/// the window between the commit gate and the cache write.
struct FreeInPublishWindow {
  static EJitTaskPool *pool;
  static void hook(void *ctx, const EJitCompileRequest &req) {
    (void)ctx;
    if (pool)
      pool->freeCode(req.funcIndex, req.cacheKey);
  }
};
EJitTaskPool *FreeInPublishWindow::pool = nullptr;
} // namespace

TEST(EJitTaskPoolFreeCode, FreeWhileCompilingDropsResult) {
  FreeDuringCompile::calls = 0;
  EJitTaskPool pool(16);
  FreeDuringCompile::pool = &pool;
  pool.setCompiler(&FreeDuringCompile::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Sync);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xBEEF));
  auto r = pool.compileOrGet(11, key(11), fb);

  // freeCode cancelled the Compiling slot, so beginPublish fails and the result
  // is dropped without publishing.
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::CompileFailed);
  EXPECT_EQ(r.fnPtr, fb);
  EXPECT_EQ(FreeDuringCompile::calls, 1);
  uint32_t v = pool.switchController().getVersion();
  EXPECT_EQ(pool.cache().lookup(11, key(11), v), nullptr);
  EXPECT_EQ(pool.cache().readyCount(), 0u);
  EXPECT_EQ(pool.pendingCount(), 0u);
}

TEST(EJitTaskPoolFreeCode, FreeInPublishWindowRollsBack) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  FreeInPublishWindow::pool = &pool;
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.setPrePublishHookForTest(&FreeInPublishWindow::hook, nullptr);
  pool.switchController().setMode(EJitCompileMode::Sync);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xBEEF));
  auto r = pool.compileOrGet(12, key(12), fb);

  // freeCode fired between the commit gate and the cache write; the worker's
  // finishPublish CAS fails and it rolls its own (now-cancelled) entry back.
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::CompileFailed);
  EXPECT_EQ(r.fnPtr, fb);
  uint32_t v = pool.switchController().getVersion();
  EXPECT_EQ(pool.cache().lookup(12, key(12), v), nullptr);
  EXPECT_EQ(pool.cache().readyCount(), 0u);
  EXPECT_EQ(pool.pendingCount(), 0u);
}

TEST(EJitTaskPoolFreeCode, FreeWhileAsyncCompilingViaWorker) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  FreeInPublishWindow::pool = &pool;
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.setPrePublishHookForTest(&FreeInPublishWindow::hook, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xCAFE));
  auto r1 = pool.compileOrGet(13, key(13), fb);
  EXPECT_EQ(r1.status, EJitCompileOrGetStatus::EnqueuedPending);

  // The worker compiles; freeCode strikes in its publish window. Nothing stays
  // cached, and the worker reports work was attempted.
  EXPECT_TRUE(pool.pollOne());
  uint32_t v = pool.switchController().getVersion();
  EXPECT_EQ(pool.cache().lookup(13, key(13), v), nullptr);
  EXPECT_EQ(pool.cache().readyCount(), 0u);
  EXPECT_EQ(pool.pendingCount(), 0u);
}

namespace {
struct DeactivateInPublishWindow {
  static EJitTaskPool *pool;
  static void hook(void *ctx, const EJitCompileRequest &req) {
    (void)ctx;
    (void)req;
    if (pool)
      pool->deactivate();
  }
};
EJitTaskPool *DeactivateInPublishWindow::pool = nullptr;

struct DuplicateInPublishWindow {
  static EJitTaskPool *pool;
  static void *fallback;
  static EJitCompileOrGetStatus observed;
  static void hook(void *ctx, const EJitCompileRequest &req) {
    (void)ctx;
    if (!pool)
      return;
    auto r = pool->compileOrGet(req.funcIndex, req.cacheKey, fallback);
    observed = r.status;
  }
};
EJitTaskPool *DuplicateInPublishWindow::pool = nullptr;
void *DuplicateInPublishWindow::fallback = nullptr;
EJitCompileOrGetStatus DuplicateInPublishWindow::observed =
    EJitCompileOrGetStatus::CompileFailed;
} // namespace

TEST(EJitTaskPoolController, OffBlocksQueueThenActivateAllowsQueue) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);
  pool.deactivate();

  void *fb = reinterpret_cast<void *>(uintptr_t(0xC001));
  auto rOff = pool.compileOrGet(21, key(21), fb);
  EXPECT_EQ(rOff.status, EJitCompileOrGetStatus::DisabledFallback);
  EXPECT_EQ(pool.queue().approximateSize(), 0u);

  pool.activate(EJitCompileMode::Async);
  auto rOn = pool.compileOrGet(21, key(21), fb);
  EXPECT_EQ(rOn.status, EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_EQ(pool.queue().approximateSize(), 1u);
}

TEST(EJitTaskPoolController, DeactivateDuringPublishDropsOldGeneration) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);
  DeactivateInPublishWindow::pool = &pool;
  pool.setPrePublishHookForTest(&DeactivateInPublishWindow::hook, nullptr);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xCAFE));
  auto r1 = pool.compileOrGet(22, key(22), fb);
  EXPECT_EQ(r1.status, EJitCompileOrGetStatus::EnqueuedPending);

  EXPECT_TRUE(pool.pollOne());
  uint32_t v = pool.switchController().getVersion();
  EXPECT_EQ(pool.cache().lookup(22, key(22), v), nullptr);

  pool.setPrePublishHookForTest(nullptr, nullptr);
  DeactivateInPublishWindow::pool = nullptr;
  pool.activate(EJitCompileMode::Async);
  auto r2 = pool.compileOrGet(22, key(22), fb);
  EXPECT_EQ(r2.status, EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(pool.pollOne());
  v = pool.switchController().getVersion();
  EXPECT_EQ(pool.cache().lookup(22, key(22), v), expectedPtr(22));
}

TEST(EJitTaskPoolPublish, DuplicateRequestWhilePublishingCoalesces) {
  MockCompiler::reset();
  EJitTaskPool pool(16);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Sync);

  void *fb = reinterpret_cast<void *>(uintptr_t(0xBEEF));
  DuplicateInPublishWindow::pool = &pool;
  DuplicateInPublishWindow::fallback = fb;
  DuplicateInPublishWindow::observed = EJitCompileOrGetStatus::CompileFailed;
  pool.setPrePublishHookForTest(&DuplicateInPublishWindow::hook, nullptr);

  auto r = pool.compileOrGet(23, key(23), fb);
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::SyncCompiled);
  EXPECT_EQ(DuplicateInPublishWindow::observed,
            EJitCompileOrGetStatus::AlreadyPending);
}

//===----------------------------------------------------------------------===//
// 12. Taskpool stats counters. (Task 4)
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolStats, CountersTrackActivity) {
  MockCompiler::reset();
  EJitTaskPool pool(8);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  void *fb = nullptr;

  // Sync compile then a cache hit.
  pool.switchController().setMode(EJitCompileMode::Sync);
  EXPECT_EQ(pool.compileOrGet(1, key(1), fb).status,
            EJitCompileOrGetStatus::SyncCompiled);
  EXPECT_EQ(pool.compileOrGet(1, key(1), fb).status,
            EJitCompileOrGetStatus::CacheHit);

  // Async enqueue, duplicate (already pending), then worker compile + hit.
  pool.switchController().setMode(EJitCompileMode::Async);
  EXPECT_EQ(pool.compileOrGet(2, key(2), fb).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_EQ(pool.compileOrGet(2, key(2), fb).status,
            EJitCompileOrGetStatus::AlreadyPending);
  EXPECT_TRUE(pool.pollOne());
  EXPECT_EQ(pool.compileOrGet(2, key(2), fb).status,
            EJitCompileOrGetStatus::CacheHit);

  // Logical free of func 1.
  EXPECT_TRUE(pool.freeCode(1, key(1)));

  EJitTaskPoolStatsSnapshot s;
  pool.getStats(s);
  EXPECT_EQ(s.syncCompiles, 1u);
  EXPECT_EQ(s.asyncCompiles, 1u);
  EXPECT_EQ(s.asyncEnqueues, 1u);
  EXPECT_EQ(s.alreadyPending, 1u);
  EXPECT_EQ(s.cacheHits, 2u);
  EXPECT_EQ(s.freeCodeCalls, 1u);
  EXPECT_EQ(s.readyEntries, 1u); // func 2 ready; func 1 freed
  EXPECT_EQ(s.pendingEntries, 0u);
  EXPECT_EQ(s.queueApproxSize, 0u);
}

TEST(EJitTaskPoolStats, QueueFullAndDedupFullCounters) {
  MockCompiler::reset();
  EJitTaskPool pool(4); // tiny queue
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);
  void *fb = nullptr;

  // Fill the queue (capacity 4), then one more enqueue fails → queueFull.
  for (uint32_t i = 1; i <= 4; ++i)
    EXPECT_EQ(pool.compileOrGet(i, key(i), fb).status,
              EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_EQ(pool.compileOrGet(5, key(5), fb).status,
            EJitCompileOrGetStatus::QueueFullFallback);

  EJitTaskPoolStatsSnapshot s;
  pool.getStats(s);
  EXPECT_EQ(s.asyncEnqueues, 4u);
  EXPECT_EQ(s.queueFull, 1u);
}

TEST(EJitTaskPoolStats, DedupFullCounter) {
  MockCompiler::reset();
  EJitTaskPool pool(256);
  pool.setCompiler(&MockCompiler::compile, nullptr);
  pool.switchController().setMode(EJitCompileMode::Async);
  void *fb = nullptr;

  const uint32_t B = EJitDedupTable::kBuckets;
  const uint32_t S = EJitDedupTable::kSlots;
  // Fill every dedup slot of bucket 0 (distinct funcIndex, same bucket).
  for (uint32_t i = 1; i <= S; ++i)
    EXPECT_EQ(pool.compileOrGet(i * B, key(i * B), fb).status,
              EJitCompileOrGetStatus::EnqueuedPending);
  // One more in the same bucket → dedup full.
  auto rf = pool.compileOrGet((S + 1) * B, key((S + 1) * B), fb);
  EXPECT_EQ(rf.status, EJitCompileOrGetStatus::DedupFullFallback);

  EJitTaskPoolStatsSnapshot s;
  pool.getStats(s);
  EXPECT_EQ(s.dedupFull, 1u);
}

//===----------------------------------------------------------------------===//
// 13. Shared-memory layout/representation constraints.
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolLayout, SharedStructAlignmentAndSize) {
  static_assert(alignof(EJitCompileRequest) <= 8,
                "EJitCompileRequest align must stay <= 8");
  static_assert(alignof(EJitDedupSlot) <= 8,
                "EJitDedupSlot align must stay <= 8");
  static_assert(alignof(EJitCacheEntry) <= 8,
                "EJitCacheEntry align must stay <= 8");

  static_assert(sizeof(EJitDedupSlot) == 24,
                "EJitDedupSlot size changed; revisit cross-core layout");
  static_assert(sizeof(EJitCacheEntry) == 32,
                "EJitCacheEntry size changed; revisit cross-core layout");

  static_assert(offsetof(EJitDedupSlot, state) == 0,
                "EJitDedupSlot state offset changed");
  static_assert(offsetof(EJitDedupSlot, funcIndex) == 4,
                "EJitDedupSlot funcIndex offset changed");
  static_assert(offsetof(EJitDedupSlot, version) == 8,
                "EJitDedupSlot version offset changed");
  static_assert(offsetof(EJitDedupSlot, cacheKey) == 16,
                "EJitDedupSlot cacheKey offset changed");
}

//===----------------------------------------------------------------------===//
// 14. Static scan for forbidden C++ thread library usage in taskpool core.
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolNoThreads, CoreFilesContainNoThreadLibraryUsage) {
  const char *files[] = {
      EJIT_TASKPOOL_SOURCE_DIR "/EJitTaskPool.cpp",
      EJIT_TASKPOOL_SOURCE_DIR "/EJitSreQueue.cpp",
      EJIT_TASKPOOL_SOURCE_DIR "/EJitIpcLock.cpp",
      EJIT_TASKPOOL_INCLUDE_DIR "/EJitTaskPool.h",
      EJIT_TASKPOOL_INCLUDE_DIR "/EJitSreQueue.h",
      EJIT_TASKPOOL_INCLUDE_DIR "/EJitIpcLock.h",
  };
  const char *forbidden[] = {
      "<thread>",
      "<future>",
      "<mutex>",
      "<condition_variable>",
      "std::thread(",
      "std::mutex(",
      "std::condition_variable(",
      "std::async(",
  };

  for (const char *path : files) {
    std::string text = readTextFile(path);
    ASSERT_FALSE(text.empty()) << "failed to read: " << path;
    for (const char *token : forbidden)
      EXPECT_EQ(text.find(token), std::string::npos)
          << "forbidden token found in " << path << ": " << token;
  }
}
