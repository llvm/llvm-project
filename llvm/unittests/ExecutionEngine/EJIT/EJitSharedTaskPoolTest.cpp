//===-- EJitSharedTaskPoolTest.cpp - cross-core shared taskpool tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Deterministic, single-thread tests for the cross-core SHARED taskpool. Many
//  "cores" are simulated inside one process by switching EJitCoreId between
//  calls — no real thread is needed to exercise owner election, the shared MPSC
//  queue, cross-core dedup, generation/version invalidation, and the commit
//  gate. One optional test uses the host platform task to run a real worker.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitSharedTaskPool.h"
#include "gtest/gtest.h"
#include <memory>
#include <type_traits>

using namespace llvm::ejit;

namespace {

// A deterministic, non-null "compiled code" address derived from funcIndex. The
// tests never execute it; they only compare/cache it.
void *codeFor(uint32_t funcIndex) {
  return reinterpret_cast<void *>(0x100000ull +
                                  static_cast<uintptr_t>(funcIndex) * 64u);
}

bool mockCompile(void * /*ctx*/, const EJitCompileRequest &req, void **outFn) {
  *outFn = codeFor(req.funcIndex);
  return true;
}

// Compiler that toggles an instance mid-compile to model a deactivate landing
// during compilation (ctx = the pool).
struct ToggleCtx {
  EJitSharedTaskPool *pool;
  uint32_t dimType;
  uint32_t instanceId;
};
bool mockCompileThenToggle(void *ctx, const EJitCompileRequest &req,
                           void **outFn) {
  auto *t = static_cast<ToggleCtx *>(ctx);
  *outFn = codeFor(req.funcIndex);
  t->pool->setInstanceEnabled(t->dimType, t->instanceId, false); // bump version
  return true;
}

// Records every pointer handed to the release callback.
struct ReleaseLog {
  std::vector<void *> freed;
};
void mockRelease(void *ctx, void *oldFn) {
  static_cast<ReleaseLog *>(ctx)->freed.push_back(oldFn);
}

// Compiler that returns a distinct, non-null pointer on every call (models a
// recompile landing at a new code address).
struct SeqCompiler {
  uint32_t n = 0;
};
bool mockCompileSeq(void *ctx, const EJitCompileRequest & /*req*/,
                    void **outFn) {
  auto *s = static_cast<SeqCompiler *>(ctx);
  *outFn = reinterpret_cast<void *>(0x200000ull +
                                    static_cast<uintptr_t>(++s->n) * 64u);
  return true;
}

// Injectable worker hooks. The entry is never run on a real thread here; tests
// drive pollOne() manually, so these only prove "exactly one worker started".
struct WorkerHooks {
  int starts = 0;
  int stops = 0;
  bool failNext = false;
};
bool mockWorkerStart(void *ctx, EJitSharedTaskPool::WorkerEntryFn /*entry*/,
                     void * /*entryCtx*/, uint64_t *outTaskId) {
  auto *w = static_cast<WorkerHooks *>(ctx);
  if (w->failNext)
    return false;
  ++w->starts;
  *outTaskId = 0xABCDull;
  return true;
}
void mockWorkerStop(void *ctx) { ++static_cast<WorkerHooks *>(ctx)->stops; }

// A worker start hook that ignores its ctx (returns success), so a test can
// share a non-WorkerHooks ctx between start and stop hooks.
bool startOkIgnoreCtx(void * /*ctx*/,
                      EJitSharedTaskPool::WorkerEntryFn /*entry*/,
                      void * /*entryCtx*/, uint64_t *outTaskId) {
  if (outTaskId)
    *outTaskId = 1;
  return true;
}

EJitDimPair dim(uint32_t t, uint32_t i) { return EJitDimPair{t, i}; }

class SharedTaskPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    EJitCoreId::resetForTest();
    state_ = std::make_unique<EJitSharedTaskPoolState>();
  }
  void TearDown() override { EJitCoreId::resetForTest(); }

  // Bring up a single owner on core 0 with the mock compiler and (by default)
  // no injected worker (the test drives pollOne()).
  void bringUpOwner(EJitSharedTaskPool &pool, bool codeSharing = false) {
    EJitCoreId::setCurrentForTest(0);
    pool.bind(state_.get());
    pool.setCompiler(&mockCompile, nullptr);
    pool.setMode(EJitCompileMode::Async);
    pool.setCodeSharingEnabled(codeSharing);
    ASSERT_EQ(pool.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  }

  std::unique_ptr<EJitSharedTaskPoolState> state_;
};

//===----------------------------------------------------------------------===//
// 15/ABI: layout + static-assert-backed properties, header stamped on init.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, AbiLayoutAndHeader) {
  EXPECT_TRUE(std::is_standard_layout<EJitSharedTaskPoolState>::value);
  EXPECT_TRUE(std::is_trivially_destructible<EJitSharedTaskPoolState>::value);
  EXPECT_EQ(alignof(EJitSharedTaskPoolState), kEJitSharedCacheLine);
  EXPECT_EQ(offsetof(EJitSharedTaskPoolState, magic), 0u);

  EJitSharedTaskPool pool;
  bringUpOwner(pool);
  EXPECT_EQ(state_->magic, kEJitSharedAbiMagic);
  EXPECT_EQ(state_->abiVersion, kEJitSharedAbiVersion);
  EXPECT_EQ(state_->structSize, sizeof(EJitSharedTaskPoolState));
}

// A process-global instance of the shared blob must require no C++ dynamic
// initialization. Otherwise each image/core can emit and run a
// _GLOBAL__sub_I constructor that clears the shared section after another core
// has already published queue/cache/owner state.
TEST_F(SharedTaskPoolTest, SharedStateRequiresNoDynamicInitialization) {
  EXPECT_TRUE(
      std::is_trivially_default_constructible<EJitSharedTaskPoolState>::value)
      << "shared state must not emit .init_array initialization";
}

//===----------------------------------------------------------------------===//
// 1/ Owner election: exactly one owner across simulated cores.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, ExactlyOneOwnerAcrossCores) {
  EJitSharedTaskPool c0, c1, c2;
  for (auto *p : {&c0, &c1, &c2}) {
    p->bind(state_.get());
    p->setCompiler(&mockCompile, nullptr);
    p->setMode(EJitCompileMode::Async);
  }
  EJitCoreId::setCurrentForTest(0);
  EXPECT_EQ(c0.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  EJitCoreId::setCurrentForTest(1);
  EXPECT_EQ(c1.init(), EJitSharedTaskPool::InitResult::AttachedReady);
  EJitCoreId::setCurrentForTest(2);
  EXPECT_EQ(c2.init(), EJitSharedTaskPool::InitResult::AttachedReady);

  EXPECT_TRUE(c0.isOwner());
  EXPECT_FALSE(c1.isOwner());
  EXPECT_FALSE(c2.isOwner());
  EXPECT_EQ(state_->ownerCoreId.loadAcquire(), 0u);

  // Idempotency: the owner re-observing init() stays Ready, no re-election.
  EJitCoreId::setCurrentForTest(0);
  EXPECT_EQ(c0.init(), EJitSharedTaskPool::InitResult::AttachedReady);
}

//===----------------------------------------------------------------------===//
// 2/ Multiple cores → exactly one worker created (only the owner starts it).
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, OnlyOwnerStartsOneWorker) {
  WorkerHooks hooks;
  EJitSharedTaskPool c0, c1, c2;
  for (auto *p : {&c0, &c1, &c2}) {
    p->bind(state_.get());
    p->setCompiler(&mockCompile, nullptr);
    p->setWorkerHooks(&mockWorkerStart, &mockWorkerStop, &hooks);
    p->setMode(EJitCompileMode::Async);
  }
  EJitCoreId::setCurrentForTest(0);
  EXPECT_EQ(c0.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  EJitCoreId::setCurrentForTest(1);
  EXPECT_EQ(c1.init(), EJitSharedTaskPool::InitResult::AttachedReady);
  EJitCoreId::setCurrentForTest(2);
  EXPECT_EQ(c2.init(), EJitSharedTaskPool::InitResult::AttachedReady);
  EXPECT_EQ(hooks.starts, 1); // single worker
}

//===----------------------------------------------------------------------===//
// 3/ Producers never observe a half-initialized blob; Initializing → pending.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, InitializingExposesNoHalfState) {
  EJitSharedTaskPool pool;
  pool.bind(state_.get());
  pool.setCompiler(&mockCompile, nullptr);
  pool.setMode(EJitCompileMode::Async);
  // Force the "another core is still initializing" state.
  state_->initState.storeRelease(
      static_cast<uint32_t>(EJitSharedInitState::Initializing));

  // A producer before Ready cleanly falls back and touches no shared queue.
  EJitCoreId::setCurrentForTest(5);
  auto r = pool.compileOrGet(7, nullptr, 0, codeFor(7));
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::OffMode);
  EXPECT_EQ(r.fnPtr, codeFor(7));
  EJitSharedDiagnostics d;
  pool.getDiagnostics(d);
  EXPECT_EQ(d.queueDepth, 0u);

  // init() against an Initializing peer returns pending (bounded, no deadlock).
  EXPECT_EQ(pool.init(), EJitSharedTaskPool::InitResult::InitInProgress);
}

//===----------------------------------------------------------------------===//
// 4 & 18/ Owner worker-start failure → Failed, no fake JIT success.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, OwnerFailurePropagatesFailed) {
  WorkerHooks hooks;
  hooks.failNext = true;
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setCompiler(&mockCompile, nullptr);
  owner.setWorkerHooks(&mockWorkerStart, &mockWorkerStop, &hooks);
  owner.setMode(EJitCompileMode::Async);
  EXPECT_EQ(owner.init(), EJitSharedTaskPool::InitResult::OwnerFailed);
  EXPECT_EQ(state_->initState.loadAcquire(),
            static_cast<uint32_t>(EJitSharedInitState::Failed));
  EXPECT_EQ(state_->lastInitError.loadAcquire(),
            static_cast<uint32_t>(EJitSharedInitError::WorkerStartFailed));

  // A peer observing Failed gets a clean fallback, never an infinite wait.
  EJitSharedTaskPool peer;
  EJitCoreId::setCurrentForTest(1);
  peer.bind(state_.get());
  EXPECT_EQ(peer.init(), EJitSharedTaskPool::InitResult::OwnerFailed);

  // No fake JIT success: compileOrGet returns the fallback, never a fnPtr.
  auto r = peer.compileOrGet(3, nullptr, 0, codeFor(3));
  EXPECT_EQ(r.fnPtr, codeFor(3));
  EXPECT_FALSE(r.hasReadToken);
}

//===----------------------------------------------------------------------===//
// 5/ Multiple producer cores enqueue into ONE shared queue; owner drains.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, MultiProducerSharedQueue) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner);
  // Three different cores each submit a distinct funcIndex.
  for (uint32_t core = 0; core < 3; ++core) {
    EJitCoreId::setCurrentForTest(core);
    auto r = owner.compileOrGet(100 + core, nullptr, 0, codeFor(100 + core));
    EXPECT_EQ(r.status, EJitCompileOrGetStatus::EnqueuedPending);
  }
  EJitSharedDiagnostics d;
  owner.getDiagnostics(d);
  EXPECT_EQ(d.queueDepth, 3u);
  EXPECT_EQ(d.asyncEnqueues, 3u);

  // The single owner worker (driven here by pollBudget) compiles all three.
  EJitCoreId::setCurrentForTest(0);
  EXPECT_EQ(owner.pollBudget(8), 3u);
  owner.getDiagnostics(d);
  EXPECT_EQ(d.cacheReadyCount, 3u);
  EXPECT_EQ(d.asyncCompiles, 3u);
}

//===----------------------------------------------------------------------===//
// 6/ Cross-core dedup: same key submitted by two cores compiles once.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, CrossCoreSameKeyDedup) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner);
  EJitDimPair d0[1] = {dim(0, 3)};

  EJitCoreId::setCurrentForTest(1);
  auto a = owner.compileOrGet(42, d0, 1, codeFor(42));
  EXPECT_EQ(a.status, EJitCompileOrGetStatus::EnqueuedPending);

  EJitCoreId::setCurrentForTest(2);
  auto b = owner.compileOrGet(42, d0, 1, codeFor(42));
  EXPECT_EQ(b.status, EJitCompileOrGetStatus::AlreadyPending);

  EJitSharedDiagnostics d;
  owner.getDiagnostics(d);
  EXPECT_EQ(d.pendingCount, 1u);
  EXPECT_EQ(d.queueDepth, 1u);
}

//===----------------------------------------------------------------------===//
// 7/ Queue full → clean fallback AND dedup rolled back.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, QueueFullRollsBackDedup) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner);
  EJitCoreId::setCurrentForTest(0);
  // Fill the ring to capacity with distinct funcIndexes.
  for (uint32_t f = 0; f < kEJitSharedQueueSlots; ++f) {
    auto r = owner.compileOrGet(f, nullptr, 0, codeFor(f));
    ASSERT_EQ(r.status, EJitCompileOrGetStatus::EnqueuedPending);
  }
  // One more distinct funcIndex overflows: clean fallback, dedup rolled back.
  uint32_t overflow = kEJitSharedQueueSlots; // still < max func index
  auto r = owner.compileOrGet(overflow, nullptr, 0, codeFor(overflow));
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::QueueFullFallback);

  EJitSharedDiagnostics d;
  owner.getDiagnostics(d);
  EXPECT_EQ(d.pendingCount, kEJitSharedQueueSlots); // overflow NOT counted
  EXPECT_EQ(d.queueFull, 1u);
  EXPECT_EQ(state_->inFlight[overflow].loadAcquire(), 0u); // rolled back
}

//===----------------------------------------------------------------------===//
// 8/ Generation switch (owner re-init) drops stale queue + cache.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, GenerationSwitchDropsStaleState) {
  EJitSharedTaskPool owner;
  owner.setWorkerHooks(&mockWorkerStart, &mockWorkerStop, new WorkerHooks());
  bringUpOwner(owner);
  EJitCoreId::setCurrentForTest(0);
  // Publish one entry and queue one more (left un-polled).
  ASSERT_EQ(owner.compileOrGet(1, nullptr, 0, codeFor(1)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne());
  ASSERT_EQ(owner.compileOrGet(2, nullptr, 0, codeFor(2)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EJitSharedDiagnostics d;
  owner.getDiagnostics(d);
  EXPECT_EQ(d.cacheReadyCount, 1u);
  EXPECT_EQ(d.queueDepth, 1u);
  uint32_t gen0 = d.generation;

  // Orderly shutdown + re-init bumps generation and resets shared storage.
  owner.ownerShutdown();
  EXPECT_EQ(owner.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  owner.getDiagnostics(d);
  EXPECT_GT(d.generation, gen0);
  EXPECT_EQ(d.cacheReadyCount, 0u); // stale cache dropped
  EXPECT_EQ(d.queueDepth, 0u);      // stale queue dropped
  // The previously published entry no longer hits.
  EJitCoreId::setCurrentForTest(0);
  auto miss = owner.compileOrGet(1, nullptr, 0, codeFor(1));
  EXPECT_NE(miss.status, EJitCompileOrGetStatus::CacheHit);
}

//===----------------------------------------------------------------------===//
// 9/ Deactivate during compile blocks publish (commit gate).
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, DeactivateDuringCompileBlocksPublish) {
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setMode(EJitCompileMode::Async);
  ToggleCtx tctx{&owner, 0, 7};
  owner.setCompiler(&mockCompileThenToggle, &tctx);
  ASSERT_EQ(owner.init(), EJitSharedTaskPool::InitResult::BecameOwner);

  EJitDimPair d0[1] = {dim(0, 7)};
  ASSERT_EQ(owner.compileOrGet(9, d0, 1, codeFor(9)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne()); // compiles, then toggle bumps version → reject
  EJitSharedDiagnostics d;
  owner.getDiagnostics(d);
  EXPECT_EQ(d.cacheReadyCount, 0u); // nothing published
  EXPECT_EQ(d.compileFailed, 1u);
}

//===----------------------------------------------------------------------===//
// 10 & 12/ Cache publish visibility + read-token release.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, PublishLookupAndReadTokenRelease) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner, /*codeSharing=*/true);
  EJitCoreId::setCurrentForTest(0);
  EJitDimPair d0[1] = {dim(1, 4)};
  ASSERT_EQ(owner.compileOrGet(11, d0, 1, codeFor(11)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne());

  auto hit = owner.compileOrGet(11, d0, 1, codeFor(11));
  ASSERT_EQ(hit.status, EJitCompileOrGetStatus::CacheHit);
  EXPECT_EQ(hit.fnPtr, codeFor(11));
  EXPECT_TRUE(hit.hasReadToken);
  // A held read token keeps readers > 0.
  EXPECT_GT(state_->buckets[hit.bucketIndex].readers.loadAcquire(), 0u);
  owner.releaseRead(hit.bucketIndex);
  EXPECT_EQ(state_->buckets[hit.bucketIndex].readers.loadAcquire(), 0u);
}

//===----------------------------------------------------------------------===//
// 11/ Cross-core fnPtr sharing gate.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, CrossCoreFnPtrSharingGate) {
  // codeSharing OFF: a non-owner cleanly rejects the pointer.
  {
    EJitSharedTaskPool owner;
    bringUpOwner(owner, /*codeSharing=*/false);
    EJitCoreId::setCurrentForTest(0);
    ASSERT_EQ(owner.compileOrGet(20, nullptr, 0, codeFor(20)).status,
              EJitCompileOrGetStatus::EnqueuedPending);
    EXPECT_TRUE(owner.pollOne());
    // Owner can read its own pointer.
    auto ownerHit = owner.compileOrGet(20, nullptr, 0, codeFor(20));
    EXPECT_EQ(ownerHit.status, EJitCompileOrGetStatus::CacheHit);
    if (ownerHit.hasReadToken)
      owner.releaseRead(ownerHit.bucketIndex);
    // Non-owner core may NOT: clean reject, no token, no recompile churn.
    EJitSharedTaskPool peer;
    peer.bind(state_.get());
    EJitCoreId::setCurrentForTest(9);
    auto peerHit = peer.compileOrGet(20, nullptr, 0, codeFor(20));
    EXPECT_FALSE(peerHit.hasReadToken);
    EXPECT_TRUE(peerHit.readyButNotShareable);
    EXPECT_EQ(peerHit.fnPtr, codeFor(20)); // fallback
  }
  // codeSharing ON: any core reads the SAME fnPtr.
  {
    state_ = std::make_unique<EJitSharedTaskPoolState>();
    EJitSharedTaskPool owner;
    bringUpOwner(owner, /*codeSharing=*/true);
    EJitCoreId::setCurrentForTest(0);
    ASSERT_EQ(owner.compileOrGet(21, nullptr, 0, codeFor(21)).status,
              EJitCompileOrGetStatus::EnqueuedPending);
    EXPECT_TRUE(owner.pollOne());
    EJitSharedTaskPool peer;
    peer.bind(state_.get());
    EJitCoreId::setCurrentForTest(9);
    auto peerHit = peer.compileOrGet(21, nullptr, 0, codeFor(21));
    ASSERT_EQ(peerHit.status, EJitCompileOrGetStatus::CacheHit);
    EXPECT_EQ(peerHit.fnPtr, codeFor(21)); // same pointer cross-core
    peer.releaseRead(peerHit.bucketIndex);
  }
}

//===----------------------------------------------------------------------===//
// 13/ FreeCode/publish: overwriting an identity releases the old pointer.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, PublishOverwriteReleasesOldCode) {
  ReleaseLog log;
  SeqCompiler seq;
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setCompiler(&mockCompileSeq, &seq);
  owner.setReleaser(&mockRelease, &log);
  owner.setCodeSharingEnabled(true);
  owner.setMode(EJitCompileMode::Async);
  ASSERT_EQ(owner.init(), EJitSharedTaskPool::InitResult::BecameOwner);

  EJitDimPair d0[1] = {dim(0, 1)};
  // First publish for (func=30, (0,1)).
  ASSERT_EQ(owner.compileOrGet(30, d0, 1, codeFor(30)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne());
  auto first = owner.compileOrGet(30, d0, 1, codeFor(30));
  ASSERT_EQ(first.status, EJitCompileOrGetStatus::CacheHit);
  void *firstPtr = first.fnPtr;
  owner.releaseRead(first.bucketIndex);
  // Toggle off then on: version advances by two but identity is unchanged, so a
  // re-compile (new address) overwrites the SAME slot and releases the old.
  owner.setInstanceEnabled(0, 1, false);
  owner.setInstanceEnabled(0, 1, true);
  ASSERT_EQ(owner.compileOrGet(30, d0, 1, codeFor(30)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne());
  ASSERT_EQ(log.freed.size(), 1u);
  EXPECT_EQ(log.freed[0], firstPtr); // old address freed on recompile
}

//===----------------------------------------------------------------------===//
// 14/ Big-endian field semantics: values round-trip by field, never byte-swap.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, BigEndianFieldSemantics) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner, /*codeSharing=*/true);
  EJitCoreId::setCurrentForTest(0);
  const uint32_t func = 0x0ABCu; // distinct bytes, still < max func index
  EJitDimPair d0[2] = {dim(0x03u, 0x0005u), dim(0x07u, 0x00FFu)};
  ASSERT_EQ(owner.compileOrGet(func, d0, 2, codeFor(func)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne());

  // Inspect the published slot directly: each field equals exactly what we put
  // in (correct on aarch64_be precisely because access is by-field, by-value).
  uint64_t key = 0; // recompute the same identity hash the pool uses
  key = static_cast<uint64_t>(func);
  for (uint32_t i = 0; i < 2; ++i) {
    key ^= (static_cast<uint64_t>(d0[i].dimType) << 32) |
           static_cast<uint64_t>(d0[i].instanceId);
    key *= 0x9e3779b97f4a7c15ULL;
  }
  uint32_t bucket = static_cast<uint32_t>(key % kEJitSharedCacheBuckets);
  const EJitSharedCacheBucket &B = state_->buckets[bucket];
  bool found = false;
  for (uint32_t s = 0; s < kEJitSharedCacheSlots; ++s) {
    const EJitSharedCacheSlot &Slot = B.slots[s];
    if (Slot.state.loadAcquire() !=
        static_cast<uint32_t>(EJitSharedSlotState::Ready))
      continue;
    if (Slot.funcIndex != func)
      continue;
    found = true;
    EXPECT_EQ(Slot.numDims, 2u);
    EXPECT_EQ(Slot.dims[0].dimType, 0x03u);
    EXPECT_EQ(Slot.dims[0].instanceId, 0x0005u);
    EXPECT_EQ(Slot.dims[1].dimType, 0x07u);
    EXPECT_EQ(Slot.dims[1].instanceId, 0x00FFu);
    EXPECT_EQ(Slot.identityHash, key);
    EXPECT_EQ(reinterpret_cast<void *>(Slot.fnPtr.loadAcquire()),
              codeFor(func));
    break;
  }
  EXPECT_TRUE(found);
}

//===----------------------------------------------------------------------===//
// ABI mismatch on a Ready blob is refused.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, AbiMismatchRefused) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner);
  state_->magic = kEJitSharedAbiMagic + 1; // corrupt the header
  EJitSharedTaskPool peer;
  peer.bind(state_.get());
  EJitCoreId::setCurrentForTest(1);
  EXPECT_EQ(peer.init(), EJitSharedTaskPool::InitResult::AbiMismatch);
}

//===----------------------------------------------------------------------===//
// Instance-disabled producers fall back and never enqueue.
//===----------------------------------------------------------------------===//
TEST_F(SharedTaskPoolTest, DisabledInstanceFallsBackNoEnqueue) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner);
  EJitCoreId::setCurrentForTest(0);
  EJitDimPair d0[1] = {dim(2, 5)};
  EXPECT_TRUE(owner.setInstanceEnabled(2, 5, false));
  auto r = owner.compileOrGet(50, d0, 1, codeFor(50));
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::InstanceDisabled);
  EJitSharedDiagnostics d;
  owner.getDiagnostics(d);
  EXPECT_EQ(d.queueDepth, 0u);
  EXPECT_EQ(d.instanceDisabled, 1u);
}

//===----------------------------------------------------------------------===//
// Round-2 review fixes (spec §11).
//===----------------------------------------------------------------------===//

// An idle-hook "script" that drives the REAL runWorkerLoop deterministically
// with no thread: it counts the worker's yields, and on a controlled schedule
// publishes Ready + enqueues a request, then (after the request is consumed)
// transitions to Stopping so the loop exits. This proves the SAME worker yields
// while Initializing, survives to Ready, consumes, yields on the empty queue,
// and exits on Stopping — without any spin-budget early exit.
struct IdleScript {
  EJitSharedTaskPool *pool;
  EJitSharedTaskPoolState *st;
  void *fallback;
  int idleCalls = 0;
  int initializingYields = 0;
  bool readyPublished = false;
  bool stopped = false;
};
void scriptedIdle(void *ctx) {
  auto *s = static_cast<IdleScript *>(ctx);
  ++s->idleCalls;
  uint32_t st = s->st->initState.loadAcquire();
  if (st == static_cast<uint32_t>(EJitSharedInitState::Initializing)) {
    ++s->initializingYields;
    // After a few yields proving the worker did NOT exit, the "owner" publishes
    // Ready and enqueues one request.
    if (s->initializingYields == 3) {
      s->st->initState.storeRelease(
          static_cast<uint32_t>(EJitSharedInitState::Ready));
      s->readyPublished = true;
      s->pool->compileOrGet(7, nullptr, 0, s->fallback); // enqueue (now Ready)
    }
    return;
  }
  // Ready + empty queue (the request was already consumed): stop the loop.
  if (s->readyPublished && !s->stopped) {
    s->st->initState.storeRelease(
        static_cast<uint32_t>(EJitSharedInitState::Stopping));
    s->stopped = true;
  }
}

// A compiler that flips the shared state to Stopping once it has compiled a
// target number of requests, so a REAL runWorkerLoop drains then exits.
struct StopAfterCtx {
  EJitSharedTaskPoolState *st;
  int remaining;
};
bool compileThenStopAfter(void *ctx, const EJitCompileRequest &req,
                          void **outFn) {
  auto *c = static_cast<StopAfterCtx *>(ctx);
  *outFn = codeFor(req.funcIndex);
  if (--c->remaining == 0)
    c->st->initState.storeRelease(
        static_cast<uint32_t>(EJitSharedInitState::Stopping));
  return true;
}

// A compiler that bumps the shared generation mid-compile (models an owner
// re-init landing during compilation).
struct GenBumpCtx {
  EJitSharedTaskPoolState *st;
};
bool compileThenBumpGeneration(void *ctx, const EJitCompileRequest &req,
                               void **outFn) {
  auto *c = static_cast<GenBumpCtx *>(ctx);
  *outFn = codeFor(req.funcIndex);
  c->st->generation.fetchAdd(1);
  return true;
}

// A worker stopper that records the init state it observed when called, to
// prove ownerShutdown signals Stopping BEFORE the join.
struct StopObserver {
  EJitSharedTaskPoolState *st;
  uint32_t stateAtStop = 0xFFFFFFFFu;
  int calls = 0;
};
void observingStop(void *ctx) {
  auto *o = static_cast<StopObserver *>(ctx);
  o->stateAtStop = o->st->initState.loadAcquire();
  ++o->calls;
}

// 七.1 — the REAL worker state machine: it WAITS on Initializing (never exits),
// reaches Ready and consumes, and exits on a terminal state. Driven
// step-by-step (workerPollOnce) so it is fully deterministic with no thread.
TEST_F(SharedTaskPoolTest, WorkerStartsWhileInitializingAndWaitsForReady) {
  EJitSharedTaskPool pool;
  bringUpOwner(pool); // state Ready, owner core 0
  EJitCoreId::setCurrentForTest(0);

  // Simulate the SRE task being scheduled before the owner published Ready.
  state_->initState.storeRelease(
      static_cast<uint32_t>(EJitSharedInitState::Initializing));
  EXPECT_EQ(pool.workerPollOnce(), EJitWorkerStep::WaitForReady); // NO exit
  EXPECT_EQ(pool.workerPollOnce(), EJitWorkerStep::WaitForReady);
  EXPECT_TRUE(pool.workerWaitedForReady());

  // Owner publishes Ready; the worker now reaches the consume phase.
  state_->initState.storeRelease(
      static_cast<uint32_t>(EJitSharedInitState::Ready));
  ASSERT_EQ(pool.compileOrGet(1, nullptr, 0, codeFor(1)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_EQ(pool.workerPollOnce(), EJitWorkerStep::Consumed);
  EXPECT_GT(pool.workerConsumeLoops(), 0u);
  EXPECT_EQ(pool.workerPollOnce(), EJitWorkerStep::Idle); // Ready, queue empty

  // Terminal states exit the loop.
  state_->initState.storeRelease(
      static_cast<uint32_t>(EJitSharedInitState::Stopping));
  EXPECT_EQ(pool.workerPollOnce(), EJitWorkerStep::Exit);
  state_->initState.storeRelease(
      static_cast<uint32_t>(EJitSharedInitState::Failed));
  EXPECT_EQ(pool.workerPollOnce(), EJitWorkerStep::Exit);
}

// 七.1 (real entry) — the REAL runWorkerLoop, started while the owner is still
// Initializing, YIELDS (does not busy-spin, does not exit early), and the SAME
// worker survives to Ready, consumes the enqueued request, yields on the empty
// queue, and exits on Stopping. Driven by an injected idle hook (no thread, no
// spin budget, no deadlock).
TEST_F(SharedTaskPoolTest, RealWorkerEntrySurvivesInitializingAndConsumes) {
  EJitSharedTaskPool pool;
  bringUpOwner(pool); // runs initSharedStorage (valid ring), state Ready
  EJitCoreId::setCurrentForTest(0);
  IdleScript script{&pool, state_.get(), codeFor(7)};
  pool.setWorkerIdleHook(&scriptedIdle, &script);
  // Simulate the SRE task being scheduled BEFORE the owner published Ready.
  state_->initState.storeRelease(
      static_cast<uint32_t>(EJitSharedInitState::Initializing));
  pool.runWorkerLoop(); // REAL entry; the idle script drives the transitions.

  EXPECT_GE(script.initializingYields,
            3); // yielded (not exited) on Initializing
  EXPECT_GT(pool.workerIdleYields(), 0u); // worker yielded, never busy-spun
  EXPECT_TRUE(pool.workerWaitedForReady());
  EXPECT_GT(pool.workerConsumeLoops(),
            0u); // SAME worker reached Ready+consumed
  EJitSharedDiagnostics d;
  pool.getDiagnostics(d);
  EXPECT_EQ(d.cacheReadyCount,
            1u); // the enqueued request was actually compiled
}

// 七.1 (real entry) — the actual runWorkerLoop reaches Ready, consumes queued
// work, and exits via a controlled Stopping transition (no thread, no
// deadlock).
TEST_F(SharedTaskPoolTest, RealWorkerEntryConsumesThenStops) {
  EJitSharedTaskPool pool;
  bringUpOwner(pool);
  EJitCoreId::setCurrentForTest(0);
  StopAfterCtx sa{state_.get(), 3};
  pool.setCompiler(&compileThenStopAfter, &sa);
  for (uint32_t f = 1; f <= 3; ++f)
    ASSERT_EQ(pool.compileOrGet(f, nullptr, 0, codeFor(f)).status,
              EJitCompileOrGetStatus::EnqueuedPending);
  pool.runWorkerLoop(); // REAL entry: consumes 3 then sees Stopping → exits.
  EXPECT_GE(pool.workerConsumeLoops(), 3u);
  EJitSharedDiagnostics d;
  pool.getDiagnostics(d);
  EXPECT_EQ(d.cacheReadyCount, 3u);
}

// 七.2 — worker start failure publishes Failed and records the reason.
TEST_F(SharedTaskPoolTest, WorkerStartFailurePublishesFailed) {
  WorkerHooks hooks;
  hooks.failNext = true;
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setCompiler(&mockCompile, nullptr);
  owner.setWorkerHooks(&mockWorkerStart, &mockWorkerStop, &hooks);
  owner.setMode(EJitCompileMode::Async);
  EXPECT_EQ(owner.init(), EJitSharedTaskPool::InitResult::OwnerFailed);
  EXPECT_EQ(state_->initState.loadAcquire(),
            static_cast<uint32_t>(EJitSharedInitState::Failed));
  EXPECT_EQ(state_->lastInitError.loadAcquire(),
            static_cast<uint32_t>(EJitSharedInitError::WorkerStartFailed));
  EXPECT_EQ(hooks.starts, 0);
}

// 七.3 — the host test build must NOT select the platform core-id path, and the
// (settable) core id actually participates in owner election (not defaulted 0).
TEST_F(SharedTaskPoolTest, PlatformCoreIdBuildSelection) {
#ifdef EJIT_SRE_SHARED_TASKPOOL_PLATFORM
  FAIL() << "host unit-test build must not define "
            "EJIT_SRE_SHARED_TASKPOOL_PLATFORM";
#else
  EJitSharedTaskPool pool;
  pool.bind(state_.get());
  pool.setCompiler(&mockCompile, nullptr);
  pool.setMode(EJitCompileMode::Async);
  EJitCoreId::setCurrentForTest(7); // a non-zero core wins election
  ASSERT_EQ(pool.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  EXPECT_EQ(state_->ownerCoreId.loadAcquire(), 7u); // core id participated
#endif
}

// 七.4 — the configured worker stack-size macro is present and valid (the value
// the freestanding SRE adapter passes to SRE_TaskCreate). The authoritative
// "reaches TaskCreate" check is the freestanding compile in the build phase.
TEST_F(SharedTaskPoolTest, ConfiguredWorkerStackReachesTaskCreate) {
#ifdef EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE
  EXPECT_GT(
      static_cast<unsigned long long>(EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE),
      0ull);
  EXPECT_EQ(EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE % 16u, 0u);
  EXPECT_LE(
      static_cast<unsigned long long>(EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE),
      0xFFFFFFFFull);
#else
  FAIL() << "EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE must be defined by the build";
#endif
}

// 七.5 — code sharing OFF: a non-owner core hits a Ready entry but gets NO
// fnPtr and does NOT re-enqueue (no recompile churn).
TEST_F(SharedTaskPoolTest, CodeSharingOffRejectsPeerWithoutReenqueue) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner, /*codeSharing=*/false);
  EJitCoreId::setCurrentForTest(0);
  ASSERT_EQ(owner.compileOrGet(60, nullptr, 0, codeFor(60)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne());

  EJitSharedTaskPool peer;
  peer.bind(state_.get());
  EJitCoreId::setCurrentForTest(9);
  EJitSharedDiagnostics before;
  peer.getDiagnostics(before);
  auto r = peer.compileOrGet(60, nullptr, 0, codeFor(60));
  EXPECT_FALSE(r.hasReadToken);
  EXPECT_TRUE(r.readyButNotShareable);
  EXPECT_EQ(r.fnPtr, codeFor(60)); // fallback
  EJitSharedDiagnostics after;
  peer.getDiagnostics(after);
  EXPECT_EQ(after.queueDepth, before.queueDepth); // NOT re-enqueued
  EXPECT_EQ(after.pendingCount, before.pendingCount);
}

// 七.6 — code sharing ON: a non-owner core gets the SAME fnPtr + read token,
// and the owner can read its own pointer too.
TEST_F(SharedTaskPoolTest, CodeSharingOnReturnsSamePointerToPeer) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner, /*codeSharing=*/true);
  EJitCoreId::setCurrentForTest(0);
  ASSERT_EQ(owner.compileOrGet(61, nullptr, 0, codeFor(61)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne());

  auto ownerHit = owner.compileOrGet(61, nullptr, 0, codeFor(61));
  EXPECT_EQ(ownerHit.status, EJitCompileOrGetStatus::CacheHit);
  EXPECT_EQ(ownerHit.fnPtr, codeFor(61));
  if (ownerHit.hasReadToken)
    owner.releaseRead(ownerHit.bucketIndex);

  EJitSharedTaskPool peer;
  peer.bind(state_.get());
  EJitCoreId::setCurrentForTest(9);
  auto peerHit = peer.compileOrGet(61, nullptr, 0, codeFor(61));
  ASSERT_EQ(peerHit.status, EJitCompileOrGetStatus::CacheHit);
  EXPECT_EQ(peerHit.fnPtr, codeFor(61)); // same pointer cross-core
  EXPECT_TRUE(peerHit.hasReadToken);
  peer.releaseRead(peerHit.bucketIndex);
}

// 七.7 — a request whose generation has been superseded is dropped at the
// worker (no compile, no publish) and its OWN-generation dedup slot is
// released.
TEST_F(SharedTaskPoolTest, StaleQueuedGenerationDropped) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner);
  EJitCoreId::setCurrentForTest(0);
  ASSERT_EQ(owner.compileOrGet(5, nullptr, 0, codeFor(5)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  uint32_t g = state_->generation.loadAcquire();
  EXPECT_EQ(state_->inFlight[5].loadAcquire(), g); // dedup slot holds gen g
  // Bump the generation as an owner re-init would (without resetting the
  // queue).
  state_->generation.storeRelease(g + 1);
  EXPECT_TRUE(owner.pollOne()); // worker pops the stale request → drops it
  EJitSharedDiagnostics d;
  owner.getDiagnostics(d);
  EXPECT_EQ(d.cacheReadyCount, 0u); // nothing published
  EXPECT_EQ(d.compileFailed, 1u);
  EXPECT_EQ(state_->inFlight[5].loadAcquire(), 0u); // gen-g slot released
}

// 七.8 — a generation change DURING compilation drops the result (released, not
// published).
TEST_F(SharedTaskPoolTest, GenerationChangesDuringCompileDropsResult) {
  ReleaseLog log;
  GenBumpCtx gctx{state_.get()};
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setCompiler(&compileThenBumpGeneration, &gctx);
  owner.setReleaser(&mockRelease, &log);
  owner.setMode(EJitCompileMode::Async);
  ASSERT_EQ(owner.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  ASSERT_EQ(owner.compileOrGet(6, nullptr, 0, codeFor(6)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(owner.pollOne()); // compiles, bumps gen → checkpoint 2 rejects
  EJitSharedDiagnostics d;
  owner.getDiagnostics(d);
  EXPECT_EQ(d.cacheReadyCount, 0u);
  EXPECT_EQ(d.compileFailed, 1u);
  ASSERT_EQ(log.freed.size(), 1u);
  EXPECT_EQ(log.freed[0], codeFor(6)); // stale result released
}

// 七.9 — a stale (older-generation) worker clearing its dedup slot must NOT
// clear a newer generation's in-flight slot for the same funcIndex.
TEST_F(SharedTaskPoolTest, StaleWorkerCannotClearNewGenerationDedup) {
  EJitSharedTaskPool owner;
  bringUpOwner(owner);
  EJitCoreId::setCurrentForTest(0);
  // Gen g1 producer enqueues funcIndex 7 (dedup slot := g1).
  ASSERT_EQ(owner.compileOrGet(7, nullptr, 0, codeFor(7)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  uint32_t g1 = state_->generation.loadAcquire();
  // Simulate an owner re-init: dedup slot reset + generation bumped to g2.
  state_->inFlight[7].storeRelease(0);
  state_->generation.storeRelease(g1 + 1);
  // New gen-g2 producer re-claims funcIndex 7 (dedup slot := g2).
  ASSERT_EQ(owner.compileOrGet(7, nullptr, 0, codeFor(7)).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_EQ(state_->inFlight[7].loadAcquire(), g1 + 1);
  // The stale gen-g1 request is first in the queue; the worker pops + drops it,
  // calling dedupClear(7, g1) = CAS(g1->0) which FAILS against the g2 value.
  EXPECT_TRUE(owner.pollOne());
  EXPECT_EQ(state_->inFlight[7].loadAcquire(),
            g1 + 1); // new-gen slot preserved
}

// 七.10 — destroying a non-owner peer must NOT stop the owner's worker.
TEST_F(SharedTaskPoolTest, PeerDestructionDoesNotStopOwnerWorker) {
  WorkerHooks hooks;
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setCompiler(&mockCompile, nullptr);
  owner.setWorkerHooks(&mockWorkerStart, &mockWorkerStop, &hooks);
  owner.setMode(EJitCompileMode::Async);
  ASSERT_EQ(owner.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  EXPECT_EQ(hooks.starts, 1);
  {
    EJitSharedTaskPool peer;
    peer.bind(state_.get());
    EJitCoreId::setCurrentForTest(1);
    ASSERT_EQ(peer.init(), EJitSharedTaskPool::InitResult::AttachedReady);
    EXPECT_FALSE(peer.isOwner());
    peer.ownerShutdown(); // non-owner: must be a no-op
  }
  EXPECT_EQ(hooks.stops, 0); // owner worker NOT stopped by the peer
  EXPECT_EQ(state_->initState.loadAcquire(),
            static_cast<uint32_t>(EJitSharedInitState::Ready));
  EXPECT_TRUE(owner.isOwner());
  owner.ownerShutdown();
  EXPECT_EQ(hooks.stops, 1);
}

// 七.11 — owner shutdown signals Stopping and JOINS the worker BEFORE returning
// the shared state to Uninitialized (so private ORC/driver teardown is safe).
TEST_F(SharedTaskPoolTest,
       OwnerShutdownStopsWorkerBeforePrivateContextDestruction) {
  StopObserver obs{state_.get()};
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setCompiler(&mockCompile, nullptr);
  owner.setWorkerHooks(&startOkIgnoreCtx, &observingStop, &obs);
  owner.setMode(EJitCompileMode::Async);
  ASSERT_EQ(owner.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  owner.ownerShutdown();
  EXPECT_EQ(obs.calls, 1);
  // The worker stop (join) observed Stopping: the worker was told to stop
  // BEFORE the join, never after the state was already torn down.
  EXPECT_EQ(obs.stateAtStop,
            static_cast<uint32_t>(EJitSharedInitState::Stopping));
  EXPECT_EQ(state_->initState.loadAcquire(),
            static_cast<uint32_t>(EJitSharedInitState::Uninitialized));
  EXPECT_FALSE(owner.isOwner());
}

//===----------------------------------------------------------------------===//
// Round-3 review fixes (spec §11): registration fingerprint consistency.
//===----------------------------------------------------------------------===//

// 三 — a peer whose registration fingerprint matches the owner's attaches.
TEST_F(SharedTaskPoolTest, RegistrationFingerprintMatchAttaches) {
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setCompiler(&mockCompile, nullptr);
  owner.setMode(EJitCompileMode::Async);
  owner.setRegistrationFingerprint(0xA1B2C3D4E5F60718ull);
  ASSERT_EQ(owner.init(), EJitSharedTaskPool::InitResult::BecameOwner);
  EXPECT_EQ(state_->registrationFingerprint.loadAcquire(),
            0xA1B2C3D4E5F60718ull);

  EJitSharedTaskPool peer;
  peer.bind(state_.get());
  peer.setRegistrationFingerprint(0xA1B2C3D4E5F60718ull); // same mapping
  EJitCoreId::setCurrentForTest(1);
  EXPECT_EQ(peer.init(), EJitSharedTaskPool::InitResult::AttachedReady);
}

// 三 — a peer whose registration fingerprint differs is cleanly rejected and
// must NOT submit requests against a mismatched mapping.
TEST_F(SharedTaskPoolTest, RegistrationFingerprintMismatchRejected) {
  EJitSharedTaskPool owner;
  EJitCoreId::setCurrentForTest(0);
  owner.bind(state_.get());
  owner.setCompiler(&mockCompile, nullptr);
  owner.setMode(EJitCompileMode::Async);
  owner.setRegistrationFingerprint(0x1111111111111111ull);
  ASSERT_EQ(owner.init(), EJitSharedTaskPool::InitResult::BecameOwner);

  EJitSharedTaskPool peer;
  peer.bind(state_.get());
  peer.setRegistrationFingerprint(0x2222222222222222ull); // divergent mapping
  EJitCoreId::setCurrentForTest(2);
  EXPECT_EQ(peer.init(), EJitSharedTaskPool::InitResult::FingerprintMismatch);
  EXPECT_FALSE(peer.isOwner());
}

} // namespace
