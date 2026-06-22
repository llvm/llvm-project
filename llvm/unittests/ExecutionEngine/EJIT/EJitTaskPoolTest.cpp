//===-- EJitTaskPoolTest.cpp - Unit tests for SRE taskpool ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Host-runnable tests for the EmbeddedJIT SRE taskpool. The taskpool sources
//  are compiled directly into this executable with -DEJIT_SRE_TASKPOOL, so the
//  tests run without OrcJIT/EJIT or any real SRE platform symbols.
//
//  These tests assert SPEC semantics (jit_design_doc/EJIT_SRE_TASKPOOL.md):
//   - §3.5 dedup: inFlight_[funcIndex] direct index, funcIndex-only, no fold;
//   - §5.1 SwitchController: enabled_[8][256]/version_[8][256] direct index;
//   - §5.2 compileOrGet order: instance -> cache -> Off -> enqueue;
//   - §5.3/§5.4 publish commit gate closing the checkpoint2->publish window.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"
#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include "llvm/ExecutionEngine/EJIT/EJitFuncRegistry.h"
#include "llvm/ExecutionEngine/EJIT/EJitLifecycleRegistry.h"
#include "llvm/ExecutionEngine/EJIT/EJitRwLock.h"
#include "llvm/ExecutionEngine/EJIT/EJitSreQueue.h"
#include "gtest/gtest.h"
#include <string>
#include <thread>
#include <type_traits>

using namespace llvm;
using namespace llvm::ejit;

namespace {

static int DummyFn0() { return 7; }
static int DummyFn1() { return 11; }
static int DummyFn2() { return 19; }

EJitCompileRequest makeReq(uint32_t FuncIndex, const EJitDimPair *Dims,
                           uint32_t NumDims, uint32_t Version = 0) {
  EJitCompileRequest R{};
  R.funcIndex = FuncIndex;
  R.numDims = NumDims;
  for (uint32_t i = 0; i < NumDims; ++i) {
    R.dims[i] = Dims[i];
    R.versions[i] = Version;
  }
  return R;
}

struct MockCompiler {
  EJitAtomicU32 calls{0};
  bool fail = false;
  bool nullOut = false;
  uintptr_t base = 0x100000;

  static bool compile(void *ctx, const EJitCompileRequest &req, void **outFn) {
    auto *self = static_cast<MockCompiler *>(ctx);
    self->calls.fetchAdd(1);
    if (self->fail) {
      *outFn = nullptr;
      return false;
    }
    if (self->nullOut) {
      *outFn = nullptr;
      return true;
    }
    *outFn = reinterpret_cast<void *>(self->base + req.funcIndex);
    return true;
  }
};

/// Bounded busy-wait (no sleep) used by the real-worker tests.
template <typename Pred> bool spinUntil(Pred P, uint64_t MaxIters = 200000000) {
  for (uint64_t i = 0; i < MaxIters; ++i) {
    if (P())
      return true;
    std::this_thread::yield();
  }
  return P();
}

} // namespace

//===----------------------------------------------------------------------===//
// Request layout
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolLayout, RequestIsFlatPod) {
  static_assert(std::is_trivially_copyable<EJitCompileRequest>::value,
                "EJitCompileRequest must be trivially copyable");
  static_assert(std::is_standard_layout<EJitCompileRequest>::value,
                "EJitCompileRequest must be standard layout");
  EXPECT_LE(alignof(EJitCompileRequest), 8u);
  // funcIndex + numDims + dims[4] + versions[4] + fallbackPtr + generation,
  // with tail padding to alignof on a 64-bit target (72) and none on 32-bit
  // (64). See the static_assert in EJitSreQueue.h.
  EXPECT_EQ(sizeof(EJitCompileRequest), sizeof(uintptr_t) == 8 ? 72u : 64u);
}

//===----------------------------------------------------------------------===//
// EJitAtomic
//===----------------------------------------------------------------------===//

TEST(EJitAtomicTest, U8CompareExchange) {
  EJitAtomicU8 A(0);
  uint8_t expected = 0;
  EXPECT_TRUE(A.compareExchange(expected, 1));
  EXPECT_EQ(A.loadAcquire(), 1u);
  expected = 0; // wrong expectation
  EXPECT_FALSE(A.compareExchange(expected, 2));
  EXPECT_EQ(expected, 1u); // updated to observed
  EXPECT_EQ(A.loadAcquire(), 1u);
}

TEST(EJitAtomicTest, U32LoadStoreAndArith) {
  EJitAtomicU32 A(5);
  EXPECT_EQ(A.loadAcquire(), 5u);
  A.storeRelease(8);
  EXPECT_EQ(A.fetchAdd(2), 8u);
  EXPECT_EQ(A.fetchSub(1), 10u);
  EXPECT_EQ(A.loadRelaxed(), 9u);
}

TEST(EJitAtomicTest, PointerRoundtrip) {
  EJitAtomicUPtr A(0);
  uintptr_t P = reinterpret_cast<uintptr_t>(&A);
  A.storeRelease(P);
  EXPECT_EQ(A.loadAcquire(), P);
}

//===----------------------------------------------------------------------===//
// EJitRwLock
//===----------------------------------------------------------------------===//

TEST(EJitRwLockTest, MultipleReadersThenWrite) {
  EJitRwLock L;
  EXPECT_TRUE(L.tryRead());
  EXPECT_TRUE(L.tryRead()); // multiple concurrent readers allowed
  EXPECT_FALSE(L.tryWrite());
  L.readRelease();
  EXPECT_FALSE(L.tryWrite()); // one reader still active
  L.readRelease();
  EXPECT_TRUE(L.tryWrite()); // all readers gone
  L.writeRelease();
}

TEST(EJitRwLockTest, WriterBlocksReaders) {
  EJitRwLock L;
  EXPECT_TRUE(L.tryWrite());
  EXPECT_FALSE(L.tryRead());
  L.writeRelease();
  EXPECT_TRUE(L.tryRead());
  L.readRelease();
}

TEST(EJitRwLockTest, RepeatedReadReleaseIsUnderflowSafe) {
  EJitRwLock L;
  // Releasing when no readers are held must not underflow the counter.
  L.readRelease();
  L.readRelease();
  EXPECT_TRUE(L.tryWrite());
  L.writeRelease();
}

TEST(EJitRwLockTest, DoubleCheckAfterReaderIncrement) {
  EJitRwLock L;
  EXPECT_TRUE(L.tryRead());
  // While a reader is active, a writer cannot acquire; the reader's double
  // check guarantees the writer never proceeds with readers present.
  EXPECT_FALSE(L.tryWrite());
  L.readRelease();
}

/// Legacy-ABI token lifetime: a held read token must keep a writer out until
/// the caller releases it (the token covers the function-execution window —
/// releasing early before use is the use-after-free the driver fix removes).
/// ejit_compile_or_get therefore stays on the LRU path, never the taskpool.
TEST(EJitRwLockTest, LegacyTokenLifetimeReadBlocksWrite) {
  EJitRwLock L;
  ASSERT_TRUE(L.tryRead());   // token acquired (stands in for a cache hit)
  EXPECT_FALSE(L.tryWrite()); // writer (publish/free) blocked during use
  L.readRelease();            // caller done using fnPtr
  EXPECT_TRUE(L.tryWrite());  // now safe to overwrite/release
  L.writeRelease();
}

//===----------------------------------------------------------------------===//
// EJitSwitchController (§5.1 strict [8][256])
//===----------------------------------------------------------------------===//

TEST(EJitSwitchControllerTest, DefaultsEnabledVersionZero) {
  EJitSwitchController S;
  EXPECT_TRUE(S.isInstanceEnabled(0, 0));
  EXPECT_TRUE(S.isInstanceEnabled(7, 255));
  EXPECT_EQ(S.getInstanceVersion(0, 0), 0u);
  EXPECT_EQ(S.getInstanceVersion(7, 255), 0u);
}

TEST(EJitSwitchControllerTest, DimType0And7Valid) {
  EJitSwitchController S;
  EXPECT_TRUE(S.setEnabled(0, 1, false));
  EXPECT_EQ(S.getInstanceVersion(0, 1), 1u);
  EXPECT_TRUE(S.setEnabled(7, 1, false));
  EXPECT_EQ(S.getInstanceVersion(7, 1), 1u);
}

TEST(EJitSwitchControllerTest, DimType8Rejected) {
  EJitSwitchController S;
  EXPECT_FALSE(S.setEnabled(8, 0, false));   // dimType == MAX_DIM_TYPES
  EXPECT_FALSE(S.isInstanceEnabled(8, 0));   // out of range → false
  EXPECT_EQ(S.getInstanceVersion(8, 0), 0u); // out of range → 0
}

TEST(EJitSwitchControllerTest, Instance255ValidId256Rejected) {
  EJitSwitchController S;
  EXPECT_TRUE(S.setEnabled(0, 255, false)); // instanceId 255 valid
  EXPECT_EQ(S.getInstanceVersion(0, 255), 1u);
  EXPECT_FALSE(S.setEnabled(0, 256, false)); // instanceId 256 rejected
  EXPECT_FALSE(S.isInstanceEnabled(0, 256));
  EXPECT_EQ(S.getInstanceVersion(0, 256), 0u);
}

TEST(EJitSwitchControllerTest, ToggleBumpsVersionOnlyOnChange) {
  EJitSwitchController S;
  uint32_t v0 = S.getInstanceVersion(3, 1);
  EXPECT_FALSE(S.setEnabled(3, 1, true)); // already enabled → no change
  EXPECT_EQ(S.getInstanceVersion(3, 1), v0);
  EXPECT_TRUE(S.setEnabled(3, 1, false)); // enabled → disabled
  EXPECT_EQ(S.getInstanceVersion(3, 1), v0 + 1);
  EXPECT_FALSE(S.setEnabled(3, 1, false)); // already disabled → no change
  EXPECT_EQ(S.getInstanceVersion(3, 1), v0 + 1);
  EXPECT_TRUE(S.setEnabled(3, 1, true)); // re-enable bumps again
  EXPECT_EQ(S.getInstanceVersion(3, 1), v0 + 2);
}

TEST(EJitSwitchControllerTest, DistinctDimTypesIndependent) {
  EJitSwitchController S;
  EXPECT_TRUE(S.setEnabled(2, 5, false));
  EXPECT_EQ(S.getInstanceVersion(2, 5), 1u);
  // A different dimType/instance is completely independent.
  EXPECT_EQ(S.getInstanceVersion(3, 5), 0u);
  EXPECT_TRUE(S.isInstanceEnabled(3, 5));
  EXPECT_TRUE(S.isInstanceEnabled(2, 6));
}

// Repeated concurrent toggles: version is monotonic and equals the number of
// real flips (no version is ever lost or reordered below a prior value).
TEST(EJitSwitchControllerTest, ConcurrentToggleVersionMonotonic) {
  EJitSwitchController S;
  constexpr int kThreads = 4;
  constexpr int kFlipsPerThread = 2000;
  std::thread ts[kThreads];
  for (int t = 0; t < kThreads; ++t)
    ts[t] = std::thread([&S] {
      for (int i = 0; i < kFlipsPerThread; ++i) {
        S.setEnabled(1, 9, false);
        S.setEnabled(1, 9, true);
      }
    });
  uint32_t last = 0;
  for (int i = 0; i < 100000; ++i) {
    uint32_t v = S.getInstanceVersion(1, 9);
    EXPECT_GE(v, last); // never goes backward
    last = v;
  }
  for (auto &th : ts)
    th.join();
  // Every successful flip bumped version once; both endpoints were exercised.
  EXPECT_GT(S.getInstanceVersion(1, 9), 0u);
}

TEST(EJitSwitchControllerTest, ModeRoundtrip) {
  EJitSwitchController S;
  S.setMode(EJitCompileMode::Off);
  EXPECT_EQ(S.getMode(), EJitCompileMode::Off);
  S.setMode(EJitCompileMode::Async);
  EXPECT_EQ(S.getMode(), EJitCompileMode::Async);
}

//===----------------------------------------------------------------------===//
// EJitQueue
//===----------------------------------------------------------------------===//

TEST(EJitQueueTest, CapacityRoundsUpPowerOfTwo) {
  EJitQueue Q(3);
  EXPECT_EQ(Q.capacity(), 4u);
}

TEST(EJitQueueTest, FifoAndCapacity) {
  EJitQueue Q(4);
  EJitDimPair D[1] = {{0, 1}};
  EXPECT_TRUE(Q.push(makeReq(10, D, 1)));
  EXPECT_TRUE(Q.push(makeReq(11, D, 1)));
  EXPECT_TRUE(Q.push(makeReq(12, D, 1)));
  EXPECT_TRUE(Q.push(makeReq(13, D, 1)));
  EXPECT_FALSE(Q.push(makeReq(99, D, 1))); // full
  EJitCompileRequest O{};
  EXPECT_TRUE(Q.pop(O));
  EXPECT_EQ(O.funcIndex, 10u);
  EXPECT_TRUE(Q.pop(O));
  EXPECT_EQ(O.funcIndex, 11u);
  EXPECT_TRUE(Q.pop(O));
  EXPECT_EQ(O.funcIndex, 12u);
  EXPECT_TRUE(Q.pop(O));
  EXPECT_EQ(O.funcIndex, 13u);
  EXPECT_FALSE(Q.pop(O)); // empty
}

TEST(EJitQueueTest, ApproximateSizeTracksPushPop) {
  EJitQueue Q(4);
  EJitDimPair D[1] = {{0, 1}};
  EXPECT_EQ(Q.approximateSize(), 0u);
  EXPECT_TRUE(Q.push(makeReq(1, D, 1)));
  EXPECT_TRUE(Q.push(makeReq(2, D, 1)));
  EXPECT_EQ(Q.approximateSize(), 2u);
  EJitCompileRequest O{};
  EXPECT_TRUE(Q.pop(O));
  EXPECT_EQ(Q.approximateSize(), 1u);
}

//===----------------------------------------------------------------------===//
// EJitDedupTable (§3.5 flat, funcIndex-only, direct index, NO fold)
//===----------------------------------------------------------------------===//

TEST(EJitDedupTest, SameFuncIndexOnlyOnce) {
  EJitDedupTable T;
  EXPECT_EQ(T.tryMarkPending(42), EJitDedupResult::Claimed);
  EXPECT_EQ(T.tryMarkPending(42), EJitDedupResult::AlreadyPending);
  T.clear(42);
  EXPECT_EQ(T.tryMarkPending(42), EJitDedupResult::Claimed);
}

TEST(EJitDedupTest, DistinctFuncIndexesNeverAlias) {
  EJitDedupTable T;
  // Two distinct funcIndexes that the old (kSlots-1) mask would have aliased
  // must remain independent: claiming one never blocks the other.
  uint32_t a = 1;
  uint32_t b = 1 + EJitDedupTable::kCapacity; // would alias under modulo
  EXPECT_EQ(T.tryMarkPending(a), EJitDedupResult::Claimed);
  // b is out of range (>= capacity): rejected, NOT folded onto a's slot.
  EXPECT_EQ(T.tryMarkPending(b), EJitDedupResult::InvalidFuncIndex);
  // A different in-range funcIndex is fully independent of a.
  EXPECT_EQ(T.tryMarkPending(2), EJitDedupResult::Claimed);
  EXPECT_EQ(T.pendingCount(), 2u);
}

TEST(EJitDedupTest, OutOfRangeFuncIndexRejected) {
  EJitDedupTable T;
  EXPECT_EQ(T.tryMarkPending(EJitDedupTable::kCapacity),
            EJitDedupResult::InvalidFuncIndex);
  EXPECT_EQ(T.tryMarkPending(0xFFFFFFFFu), EJitDedupResult::InvalidFuncIndex);
  EXPECT_EQ(T.pendingCount(), 0u); // nothing was marked
}

//===----------------------------------------------------------------------===//
// EJitTaskPoolCache
//===----------------------------------------------------------------------===//

TEST(EJitCacheTest, PublishLookupHit) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  uint32_t versions[1] = {0};
  EXPECT_EQ(C.publish(10, D, 1, versions, reinterpret_cast<void *>(&DummyFn0)),
            EJitPublishStatus::Published);
  EJitCacheLookupResult R = C.lookup(10, D, 1);
  EXPECT_EQ(R.fnPtr, reinterpret_cast<void *>(&DummyFn0));
  EXPECT_TRUE(R.hasReadToken);
  C.releaseRead(R.bucketIndex);
}

TEST(EJitCacheTest, LookupMissUnknownKey) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  EXPECT_EQ(C.lookup(11, D, 1).fnPtr, nullptr);
}

TEST(EJitCacheTest, PublishRejectsNullAndTooManyDims) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D1[1] = {{0, 1}};
  uint32_t v1[1] = {0};
  EXPECT_EQ(C.publish(10, D1, 1, v1, nullptr), EJitPublishStatus::InvalidParam);
  EJitDimPair D5[5] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}};
  uint32_t v5[5] = {0, 0, 0, 0, 0};
  EXPECT_EQ(C.publish(10, D5, 5, v5, reinterpret_cast<void *>(&DummyFn0)),
            EJitPublishStatus::InvalidParam);
}

TEST(EJitCacheTest, VersionChangeInvalidatesLookup) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  uint32_t versions[1] = {0};
  EXPECT_EQ(C.publish(10, D, 1, versions, reinterpret_cast<void *>(&DummyFn1)),
            EJitPublishStatus::Published);
  EXPECT_TRUE(S.setEnabled(0, 1, false)); // bump version 0 -> 1
  EXPECT_EQ(C.lookup(10, D, 1).fnPtr, nullptr);
}

// A 64-bit hash CAN collide: funcIndex=10 dims{0,0} and funcIndex=8 dims{0,2}
// share a bucket key. Both identities must coexist and resolve to their own
// pointers — never a false cross hit. (The collision is a correctness hazard
// handled by full-identity chaining, NOT described as inherently "safe".)
TEST(EJitCacheTest, HashCollisionCoexists) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair A[1] = {{0, 0}};
  EJitDimPair B[1] = {{0, 2}};
  uint32_t v[1] = {0};
  void *pA = reinterpret_cast<void *>(&DummyFn0);
  void *pB = reinterpret_cast<void *>(&DummyFn1);
  EXPECT_EQ(C.publish(10, A, 1, v, pA), EJitPublishStatus::Published);
  EXPECT_EQ(C.publish(8, B, 1, v, pB), EJitPublishStatus::Published);
  EJitCacheLookupResult rA = C.lookup(10, A, 1);
  EJitCacheLookupResult rB = C.lookup(8, B, 1);
  EXPECT_EQ(rA.fnPtr, pA);
  EXPECT_EQ(rB.fnPtr, pB);
  C.releaseRead(rA.bucketIndex);
  C.releaseRead(rB.bucketIndex);
  EXPECT_EQ(C.readyCount(), 2u);
}

TEST(EJitCacheTest, PublishOverwriteInvokesReleaser) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  static void *releasedOld = nullptr;
  static int releaseCalls = 0;
  releasedOld = nullptr;
  releaseCalls = 0;
  C.setReleaser(
      [](void *, void *oldFn) {
        releasedOld = oldFn;
        ++releaseCalls;
      },
      nullptr);
  EJitDimPair D[1] = {{0, 1}};
  uint32_t v[1] = {0};
  void *p1 = reinterpret_cast<void *>(&DummyFn0);
  void *p2 = reinterpret_cast<void *>(&DummyFn1);
  EXPECT_EQ(C.publish(10, D, 1, v, p1), EJitPublishStatus::Published);
  EXPECT_EQ(C.publish(10, D, 1, v, p2), EJitPublishStatus::Published);
  EXPECT_EQ(releaseCalls, 1);
  EXPECT_EQ(releasedOld, p1);
  EJitCacheLookupResult R = C.lookup(10, D, 1);
  EXPECT_EQ(R.fnPtr, p2);
  C.releaseRead(R.bucketIndex);
}

TEST(EJitCacheTest, NoReleaserNoFabricatedFree) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  uint32_t v[1] = {0};
  // Without a releaser the overwrite is purely logical (no physical free).
  EXPECT_EQ(C.publish(10, D, 1, v, reinterpret_cast<void *>(&DummyFn0)),
            EJitPublishStatus::Published);
  EXPECT_EQ(C.publish(10, D, 1, v, reinterpret_cast<void *>(&DummyFn1)),
            EJitPublishStatus::Published);
  EJitCacheLookupResult R = C.lookup(10, D, 1);
  EXPECT_EQ(R.fnPtr, reinterpret_cast<void *>(&DummyFn1));
  C.releaseRead(R.bucketIndex);
}

// Commit gate (§5.3/§5.4): if the request's version no longer matches the
// current version at publish time, the entry is NOT written and any existing
// entry is NOT overwritten.
TEST(EJitCacheTest, VersionMismatchDoesNotOverwriteExistingEntry) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  void *good = reinterpret_cast<void *>(&DummyFn0);
  void *stale = reinterpret_cast<void *>(&DummyFn1);

  uint32_t v0[1] = {0};
  EXPECT_EQ(C.publish(10, D, 1, v0, good), EJitPublishStatus::Published);

  // The instance is toggled (version 0 -> 1). A worker that snapshotted v0 now
  // tries to publish: the under-lock recheck rejects it and the good entry is
  // untouched.
  EXPECT_TRUE(S.setEnabled(0, 1, false));
  EXPECT_EQ(C.publish(10, D, 1, v0, stale), EJitPublishStatus::VersionMismatch);

  // Re-enable so the instance check passes; the good entry was stored at v0,
  // which after two flips no longer matches — so lookup misses rather than
  // returning the stale pointer. The key point: 'stale' was never stored.
  EXPECT_TRUE(S.setEnabled(0, 1, true)); // version now 2
  EJitCacheLookupResult R = C.lookup(10, D, 1);
  EXPECT_NE(R.fnPtr, stale);
  if (R.hasReadToken)
    C.releaseRead(R.bucketIndex);
}

// Deterministic window test: a toggle injected via the test hook AFTER the
// publish write lock is taken but BEFORE the under-lock recheck must cause the
// publish to reject the freshly-compiled result.
TEST(EJitCacheTest, ToggleInsidePublishWindowRejectsResult) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  uint32_t v0[1] = {0};
  C.setPrePublishHookForTest(
      [](void *ctx) {
        // Toggle the instance while the publisher holds the write lock.
        static_cast<EJitSwitchController *>(ctx)->setEnabled(0, 1, false);
      },
      &S);
  EXPECT_EQ(C.publish(10, D, 1, v0, reinterpret_cast<void *>(&DummyFn0)),
            EJitPublishStatus::VersionMismatch);
  C.setPrePublishHookForTest(nullptr, nullptr);
  // Nothing was published.
  EXPECT_EQ(C.readyCount(), 0u);
}

TEST(EJitCacheTest, ReadyCountTracksEntries) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair A[1] = {{0, 1}};
  EJitDimPair B[1] = {{0, 2}};
  uint32_t v[1] = {0};
  EXPECT_EQ(C.readyCount(), 0u);
  EXPECT_EQ(C.publish(10, A, 1, v, reinterpret_cast<void *>(&DummyFn0)),
            EJitPublishStatus::Published);
  EXPECT_EQ(C.publish(11, B, 1, v, reinterpret_cast<void *>(&DummyFn1)),
            EJitPublishStatus::Published);
  EXPECT_EQ(C.readyCount(), 2u);
}

// A held read token must block a concurrent publisher (write drains readers to
// 0) so no reader is mid-use when the entry is overwritten — no use-after-free.
TEST(EJitCacheTest, PublishWaitsForReadersBeforeOverwrite) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  uint32_t v[1] = {0};
  EXPECT_EQ(C.publish(10, D, 1, v, reinterpret_cast<void *>(&DummyFn0)),
            EJitPublishStatus::Published);
  EJitCacheLookupResult R = C.lookup(10, D, 1); // hold a read token
  ASSERT_TRUE(R.hasReadToken);

  EJitAtomicU32 done{0};
  std::thread writer([&] {
    C.publish(10, D, 1, v, reinterpret_cast<void *>(&DummyFn1));
    done.storeRelease(1);
  });
  for (int i = 0; i < 1000000; ++i)
    std::this_thread::yield();
  EXPECT_EQ(done.loadAcquire(), 0u); // still blocked while token held
  C.releaseRead(R.bucketIndex);
  EXPECT_TRUE(spinUntil([&] { return done.loadAcquire() == 1u; }));
  writer.join();
  EJitCacheLookupResult R2 = C.lookup(10, D, 1);
  EXPECT_EQ(R2.fnPtr, reinterpret_cast<void *>(&DummyFn1));
  C.releaseRead(R2.bucketIndex);
}

// shutdown() releases each live entry's code through the real callback exactly
// once.
TEST(EJitCacheTest, ShutdownInvokesReleaserForEntries) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  static int releaseCalls = 0;
  releaseCalls = 0;
  C.setReleaser([](void *, void *) { ++releaseCalls; }, nullptr);
  EJitDimPair A[1] = {{0, 1}};
  EJitDimPair B[1] = {{0, 2}};
  uint32_t v[1] = {0};
  C.publish(10, A, 1, v, reinterpret_cast<void *>(&DummyFn0));
  C.publish(11, B, 1, v, reinterpret_cast<void *>(&DummyFn1));
  C.shutdown();
  EXPECT_EQ(releaseCalls, 2); // one per distinct live entry
  EXPECT_EQ(C.readyCount(), 0u);
}

TEST(EJitCacheTest, ShutdownWithoutReleaserDoesNotFabricateFree) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  uint32_t v[1] = {0};
  C.publish(10, D, 1, v, reinterpret_cast<void *>(&DummyFn0));
  // No releaser installed: shutdown logically drops entries, never frees.
  C.shutdown();
  EXPECT_EQ(C.readyCount(), 0u);
}

//===----------------------------------------------------------------------===//
// Finding (四): release callbacks must run OUTSIDE the bucket write lock.
//===----------------------------------------------------------------------===//

namespace {
struct ReentrantReleaseCtx {
  EJitTaskPoolCache *cache = nullptr;
  void *expectedNew = nullptr;
  bool reentrantSawNew = false;
  int calls = 0;
};
} // namespace

// On overwrite, the old code is released after the bucket write lock is
// dropped. We prove the lock is free during the callback by performing a
// re-entrant lookup of the SAME key from inside it: lookup uses a non-blocking
// tryRead, so it observes the just-published pointer ONLY if the writer already
// released the lock. If the release still ran under the write lock, tryRead
// would fail and the callback would see a miss (this test would fail, never
// hang).
TEST(EJitCacheTest, PublishReleaseRunsOutsideBucketLock) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  EJitDimPair D[1] = {{0, 1}};
  uint32_t v[1] = {0};
  void *p1 = reinterpret_cast<void *>(&DummyFn0);
  void *p2 = reinterpret_cast<void *>(&DummyFn1);
  ReentrantReleaseCtx ctx;
  ctx.cache = &C;
  ctx.expectedNew = p2;
  C.setReleaser(
      [](void *c, void *) {
        auto *rc = static_cast<ReentrantReleaseCtx *>(c);
        ++rc->calls;
        EJitDimPair D2[1] = {{0, 1}};
        EJitCacheLookupResult R = rc->cache->lookup(10, D2, 1);
        rc->reentrantSawNew = (R.fnPtr == rc->expectedNew);
        if (R.hasReadToken)
          rc->cache->releaseRead(R.bucketIndex);
      },
      &ctx);
  EXPECT_EQ(C.publish(10, D, 1, v, p1), EJitPublishStatus::Published);
  EXPECT_EQ(C.publish(10, D, 1, v, p2), EJitPublishStatus::Published);
  EXPECT_EQ(ctx.calls, 1);
  EXPECT_TRUE(ctx.reentrantSawNew); // bucket lock was free during the release
  EJitCacheLookupResult R = C.lookup(10, D, 1);
  EXPECT_EQ(R.fnPtr, p2);
  if (R.hasReadToken)
    C.releaseRead(R.bucketIndex);
}

// shutdown() releases each DISTINCT live pointer exactly once even when the
// same pointer was published under several identities (dedup prevents a double
// free).
TEST(EJitCacheTest, ShutdownReleasesDuplicatePointerOnce) {
  EJitSwitchController S;
  EJitTaskPoolCache C(S);
  static int releaseCalls = 0;
  releaseCalls = 0;
  C.setReleaser([](void *, void *) { ++releaseCalls; }, nullptr);
  // Same pointer under two distinct identities (different funcIndex + dims, so
  // they may land in different buckets) plus a second distinct pointer.
  void *shared = reinterpret_cast<void *>(&DummyFn0);
  void *other = reinterpret_cast<void *>(&DummyFn1);
  EJitDimPair A[1] = {{0, 1}};
  EJitDimPair B[1] = {{1, 7}};
  EJitDimPair Cc[1] = {{2, 3}};
  uint32_t v[1] = {0};
  C.publish(10, A, 1, v, shared);
  C.publish(11, B, 1, v, shared); // same pointer, different identity
  C.publish(12, Cc, 1, v, other);
  C.shutdown();
  // shared released once (deduped), other released once → 2 total, not 3.
  EXPECT_EQ(releaseCalls, 2);
  EXPECT_EQ(C.readyCount(), 0u);
}

//===----------------------------------------------------------------------===//
// Findings (一)/(二)/(三): cross-module identity must be order-independent and
// must not alias. funcIndex is a dense, registration-time index handed out by
// EJitFuncRegistry (NOT a modulo name hash); dimType is a process-global
// name-keyed registry slot (NOT a per-module sorted position and NOT a name
// hash — "cell" and "tenant" collide under fnv%8).
//===----------------------------------------------------------------------===//

TEST(EJitFuncRegistry, DenseDistinctIdempotentOrderIndependent) {
  auto &R = EJitFuncRegistry::instance();
  R.reset();
  // Dense, distinct, idempotent.
  uint32_t a = R.resolveAssign("a_func");
  uint32_t b = R.resolveAssign("b_func");
  EXPECT_NE(a, b);
  EXPECT_EQ(R.resolveAssign("a_func"), a); // idempotent
  EXPECT_EQ(R.lookup("b_func"), b);
  EXPECT_EQ(R.lookup("never"), kEJitInvalidFuncIndex);
  EXPECT_LT(a, kEJitMaxFuncIndex);
  EXPECT_LT(b, kEJitMaxFuncIndex);
  // Registering a NEW name never shifts an existing index.
  uint32_t c = R.resolveAssign("c_func");
  EXPECT_EQ(R.lookup("a_func"), a);
  EXPECT_EQ(R.lookup("b_func"), b);
  EXPECT_NE(c, a);
  EXPECT_NE(c, b);
  R.reset();
}

TEST(EJitFuncRegistry, CrossModuleSameNameSameIndexEitherOrder) {
  // Two registration sites (modules) in one process, opposite orders: each name
  // keeps a single stable dense index — the cross-module funcIndex agreement.
  auto &R = EJitFuncRegistry::instance();
  R.reset();
  uint32_t fAlpha = R.resolveAssign("alpha"); // module A first
  uint32_t fOmega = R.resolveAssign("omega");
  EXPECT_EQ(R.resolveAssign("omega"), fOmega); // module B, omega first
  EXPECT_EQ(R.resolveAssign("alpha"), fAlpha);
  EXPECT_NE(fAlpha, fOmega);
  R.reset();
}

TEST(EJitFuncRegistry, CapacityExhaustionRejectsCleanly) {
  auto &R = EJitFuncRegistry::instance();
  R.reset();
  for (uint32_t i = 0; i < kEJitMaxFuncIndex; ++i)
    ASSERT_EQ(R.resolveAssign("f" + std::to_string(i)), i);
  EXPECT_EQ(R.count(), kEJitMaxFuncIndex);
  // The next NEW name is rejected (invalid), never aliased onto an existing
  // one.
  EXPECT_EQ(R.resolveAssign("overflow"), kEJitInvalidFuncIndex);
  EXPECT_EQ(R.lookup("overflow"), kEJitInvalidFuncIndex);
  // An already-assigned name still resolves (idempotent) even when full.
  EXPECT_EQ(R.resolveAssign("f0"), 0u);
  R.reset();
}

TEST(EJitLifecycleRegistry, DenseAssignmentIdempotentAndLookup) {
  auto &R = EJitLifecycleRegistry::instance();
  R.reset();
  EXPECT_EQ(R.resolveAssign("cell"), 0u);
  EXPECT_EQ(R.resolveAssign("trp"), 1u);
  EXPECT_EQ(R.resolveAssign("cell"), 0u); // idempotent, no new slot
  EXPECT_EQ(R.count(), 2u);
  EXPECT_EQ(R.lookup("trp"), 1u);
  EXPECT_EQ(R.lookup("never-registered"), kEJitInvalidDimType);
  R.reset();
}

// The exact cross-module hazard: "cell" and "tenant" hash to the SAME 8-slot
// bucket (fnv1a%8 == 5 for both), which is why dimType cannot be a name hash.
// The registry hands them DISTINCT slots.
TEST(EJitLifecycleRegistry, HashCollidingNamesGetDistinctSlots) {
  auto &R = EJitLifecycleRegistry::instance();
  R.reset();
  uint32_t cell = R.resolveAssign("cell");
  uint32_t tenant = R.resolveAssign("tenant");
  EXPECT_NE(cell, tenant);
  EXPECT_NE(cell, kEJitInvalidDimType);
  EXPECT_NE(tenant, kEJitInvalidDimType);
  R.reset();
}

// Two modules (different registration sites in one process) that name the SAME
// lifecycle observe the SAME slot, in either registration order — the
// cross-module dimType agreement the fix guarantees.
TEST(EJitLifecycleRegistry, CrossModuleSameNameSameSlot) {
  auto &R = EJitLifecycleRegistry::instance();
  R.reset();
  uint32_t aCell = R.resolveAssign("cell"); // module A
  uint32_t aTrp = R.resolveAssign("trp");   // module A
  uint32_t bTrp = R.resolveAssign("trp");   // module B, registered later
  uint32_t bCell = R.resolveAssign("cell"); // module B
  EXPECT_EQ(aTrp, bTrp);
  EXPECT_EQ(aCell, bCell);
  EXPECT_NE(aCell, aTrp);
  R.reset();
}

// The 9th distinct lifecycle is rejected (kEJitInvalidDimType), never aliased
// onto an existing slot.
TEST(EJitLifecycleRegistry, NinthLifecycleRejected) {
  auto &R = EJitLifecycleRegistry::instance();
  R.reset();
  for (uint32_t i = 0; i < kEJitMaxDimTypes; ++i)
    EXPECT_EQ(R.resolveAssign("lc" + std::to_string(i)), i);
  EXPECT_EQ(R.count(), kEJitMaxDimTypes);
  EXPECT_EQ(R.resolveAssign("ninth"), kEJitInvalidDimType);
  EXPECT_EQ(R.lookup("ninth"), kEJitInvalidDimType);
  EXPECT_EQ(R.count(), kEJitMaxDimTypes); // unchanged
  R.reset();
}

// End-to-end: distinct lifecycles occupy distinct SwitchController rows, so
// disabling an instance under one lifecycle never disables the same instanceId
// under another lifecycle (module A "cell" vs module B "tenant").
TEST(EJitLifecycleRegistry, DistinctLifecyclesHaveIndependentInstances) {
  auto &R = EJitLifecycleRegistry::instance();
  R.reset();
  uint32_t cell = R.resolveAssign("cell");
  uint32_t tenant = R.resolveAssign("tenant");
  ASSERT_NE(cell, tenant);
  EJitSwitchController S;
  EXPECT_TRUE(S.isInstanceEnabled(cell, 5));
  EXPECT_TRUE(S.isInstanceEnabled(tenant, 5));
  EXPECT_TRUE(S.setEnabled(cell, 5, false));
  EXPECT_FALSE(S.isInstanceEnabled(cell, 5));
  EXPECT_TRUE(S.isInstanceEnabled(tenant, 5)); // unaffected by the cell toggle
  R.reset();
}

//===----------------------------------------------------------------------===//
// Finding (五): name-based control plane. EJit::activate/deactivate resolve the
// lifecycle NAME to a dimType via EJitLifecycleRegistry, then drive the
// taskpool SwitchController. These tests exercise that exact mechanism (the
// EJit wiring is a thin #ifdef EJIT_SRE_TASKPOOL layer over it).
//===----------------------------------------------------------------------===//

TEST(EJitControlPlane, NameResolvesToDimTypeAndSyncsSwitchController) {
  auto &R = EJitLifecycleRegistry::instance();
  R.reset();
  uint32_t cell = R.resolveAssign("cell");
  uint32_t tenant = R.resolveAssign("tenant");
  ASSERT_NE(cell, kEJitInvalidDimType);
  ASSERT_NE(tenant, kEJitInvalidDimType);
  // An unknown lifecycle resolves to INVALID so EJit::activate returns an error
  // and changes no state.
  EXPECT_EQ(R.lookup("unknown-life"), kEJitInvalidDimType);

  EJitSwitchController S;
  // deactivate("cell", 3): name -> dimType -> disable, version bumps once.
  uint32_t dt = R.lookup("cell");
  ASSERT_NE(dt, kEJitInvalidDimType);
  uint32_t v0 = S.getInstanceVersion(dt, 3);
  EXPECT_TRUE(S.setEnabled(dt, 3, /*wantOn=*/false));
  EXPECT_FALSE(S.isInstanceEnabled(dt, 3));
  EXPECT_EQ(S.getInstanceVersion(dt, 3), v0 + 1);
  // A disable under "cell" must NOT affect "tenant".
  EXPECT_TRUE(S.isInstanceEnabled(tenant, 3));
  // activate("cell", 3) restores it.
  EXPECT_TRUE(S.setEnabled(dt, 3, /*wantOn=*/true));
  EXPECT_TRUE(S.isInstanceEnabled(dt, 3));
  EXPECT_EQ(S.getInstanceVersion(dt, 3), v0 + 2);
  R.reset();
}

TEST(EJitControlPlane, VersionMonotonicOnlyOnRealChange) {
  EJitSwitchController S;
  uint32_t v0 = S.getInstanceVersion(0, 1);
  EXPECT_TRUE(S.isInstanceEnabled(0, 1));    // starts enabled
  EXPECT_FALSE(S.setEnabled(0, 1, true));    // already enabled: no change
  EXPECT_EQ(S.getInstanceVersion(0, 1), v0); // version unchanged
  EXPECT_TRUE(S.setEnabled(0, 1, false));    // real change
  EXPECT_EQ(S.getInstanceVersion(0, 1), v0 + 1);
  EXPECT_FALSE(S.setEnabled(0, 1, false)); // already disabled: no change
  EXPECT_EQ(S.getInstanceVersion(0, 1), v0 + 1);
}

TEST(EJitControlPlane, ActivateAllSweepsEveryInstanceOfOneLifecycleOnly) {
  // activateAll/deactivateAll sweep all instances of ONE lifecycle's dimType
  // (the per-instance sweep EJit::deactivateAll performs) without touching any
  // other lifecycle's row.
  auto &R = EJitLifecycleRegistry::instance();
  R.reset();
  uint32_t cell = R.resolveAssign("cell");
  uint32_t tenant = R.resolveAssign("tenant");
  EJitSwitchController S;
  for (uint32_t i = 0; i < EJitSwitchController::MAX_INSTANCES; ++i)
    S.setEnabled(cell, i, /*wantOn=*/false); // deactivateAll("cell")
  for (uint32_t i = 0; i < EJitSwitchController::MAX_INSTANCES; ++i) {
    EXPECT_FALSE(S.isInstanceEnabled(cell, i));
    EXPECT_TRUE(S.isInstanceEnabled(tenant, i)); // tenant row untouched
  }
  R.reset();
}

//===----------------------------------------------------------------------===//
// EJitTaskPool::compileOrGet ordering + async path
//===----------------------------------------------------------------------===//

TEST(EJitTaskPoolTest, InvalidParam) {
  EJitTaskPool P(8, false);
  EXPECT_EQ(P.compileOrGet(1, nullptr, 1, nullptr).status,
            EJitCompileOrGetStatus::InvalidParam);
  EJitDimPair D5[5] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}};
  EXPECT_EQ(P.compileOrGet(1, D5, 5, nullptr).status,
            EJitCompileOrGetStatus::InvalidParam);
}

TEST(EJitTaskPoolTest, OutOfRangeFuncIndexRejected) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  // funcIndex == flat dedup capacity is out of range: reject, never alias.
  auto r = P.compileOrGet(EJitDedupTable::kCapacity, D, 1, nullptr);
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::InvalidParam);
  EXPECT_EQ(P.pendingCount(), 0u);
}

TEST(EJitTaskPoolTest, OrderingInstanceCheckedBeforeCache) {
  // A disabled instance falls back before the cache is consulted, so it is
  // never served the cached JIT pointer (§5.2 step 0 precedes lookup).
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(5, D, 1, nullptr);
  EXPECT_TRUE(P.pollOne()); // populate cache
  void *fb = reinterpret_cast<void *>(&DummyFn2);
  P.switchController().setEnabled(0, 1, false); // disable the instance
  auto r = P.compileOrGet(5, D, 1, fb);
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::InstanceDisabled);
  EXPECT_EQ(r.fnPtr, fb);
  EXPECT_FALSE(r.hasReadToken);
}

// §5.2: cache lookup precedes the Off check, so an existing cached entry is
// still served while the pool is globally Off (Off only suppresses NEW
// compiles).
TEST(EJitTaskPoolTest, CacheHitStillWorksWhenModeOff) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(21, D, 1, nullptr);
  EXPECT_TRUE(P.pollOne()); // cache now holds func 21
  P.switchController().setMode(EJitCompileMode::Off);
  auto r = P.compileOrGet(21, D, 1, nullptr);
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::CacheHit);
  EXPECT_TRUE(r.hasReadToken);
  P.releaseRead(r.bucketIndex);
}

// Off suppresses a NEW compile (no existing cache entry): fall back, no
// enqueue.
TEST(EJitTaskPoolTest, OffSuppressesNewCompile) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Off);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  void *fb = reinterpret_cast<void *>(&DummyFn2);
  auto r = P.compileOrGet(22, D, 1, fb);
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::OffMode);
  EXPECT_EQ(r.fnPtr, fb);
  EXPECT_EQ(P.pendingCount(), 0u);
}

TEST(EJitTaskPoolTest, DisabledInstanceWithExistingCacheFallsBack) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 7}};
  P.compileOrGet(34, D, 1, nullptr);
  EXPECT_TRUE(P.pollOne());
  auto hit = P.compileOrGet(34, D, 1, nullptr);
  EXPECT_EQ(hit.status, EJitCompileOrGetStatus::CacheHit);
  P.releaseRead(hit.bucketIndex);
  void *fb = reinterpret_cast<void *>(&DummyFn1);
  P.switchController().setEnabled(0, 7, false);
  auto r = P.compileOrGet(34, D, 1, fb);
  EXPECT_EQ(r.status, EJitCompileOrGetStatus::InstanceDisabled);
  EXPECT_EQ(r.fnPtr, fb);
}

TEST(EJitTaskPoolTest, SameFuncDifferentDimsAlreadyPending) {
  EJitTaskPool P(16, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair A[2] = {{0, 3}, {1, 5}};
  EJitDimPair B[1] = {{0, 9}}; // same funcIndex, different dims
  EXPECT_EQ(P.compileOrGet(100, A, 2, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  // Dedup is funcIndex-only: a different-dims request for the same funcIndex
  // is coalesced as AlreadyPending.
  EXPECT_EQ(P.compileOrGet(100, B, 1, nullptr).status,
            EJitCompileOrGetStatus::AlreadyPending);
}

TEST(EJitTaskPoolTest, SameFuncDifferentVersionsAlreadyPending) {
  EJitTaskPool P(16, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 3}};
  EXPECT_EQ(P.compileOrGet(101, D, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  // Bump the version (different version snapshot), same funcIndex: still
  // merged.
  P.switchController().setEnabled(0, 4, false); // unrelated instance bump
  EXPECT_EQ(P.compileOrGet(101, D, 1, nullptr).status,
            EJitCompileOrGetStatus::AlreadyPending);
}

TEST(EJitTaskPoolTest, DistinctFuncIndexesNotMerged) {
  EJitTaskPool P(16, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  EXPECT_EQ(P.compileOrGet(1, D, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  // A different funcIndex is independent: it enqueues, not AlreadyPending.
  EXPECT_EQ(P.compileOrGet(2, D, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
}

TEST(EJitTaskPoolTest, QueueFullRollback) {
  EJitTaskPool P(2, false); // capacity rounds to 2
  P.switchController().setMode(EJitCompileMode::Async);
  EJitDimPair A[1] = {{0, 1}};
  EJitDimPair B[1] = {{0, 2}};
  EJitDimPair Cc[1] = {{0, 3}};
  EXPECT_EQ(P.compileOrGet(10, A, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_EQ(P.compileOrGet(11, B, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  // Queue full: dedup for func 12 must roll back so a later retry works.
  EXPECT_EQ(P.compileOrGet(12, Cc, 1, nullptr).status,
            EJitCompileOrGetStatus::QueueFullFallback);
  EXPECT_TRUE(P.pollOne()); // drain one
  EXPECT_EQ(P.compileOrGet(12, Cc, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending); // dedup was rolled back
}

TEST(EJitTaskPoolTest, FailurePathReleasesSlot) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  C.fail = true;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  EXPECT_EQ(P.compileOrGet(9, D, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(P.pollOne());        // compile fails
  EXPECT_EQ(P.pendingCount(), 0u); // dedup slot released on failure
  // The same funcIndex can be enqueued again (slot was freed).
  EXPECT_EQ(P.compileOrGet(9, D, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
}

TEST(EJitTaskPoolTest, PollCompilesAndCaches) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  EXPECT_EQ(P.compileOrGet(8, D, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(P.pollOne());
  EXPECT_EQ(C.calls.loadAcquire(), 1u);
  auto hit = P.compileOrGet(8, D, 1, nullptr);
  EXPECT_EQ(hit.status, EJitCompileOrGetStatus::CacheHit);
  EXPECT_TRUE(hit.hasReadToken);
  P.releaseRead(hit.bucketIndex);
}

TEST(EJitTaskPoolTest, PollEmptyAndBudget) {
  EJitTaskPool P(8, false);
  EXPECT_FALSE(P.pollOne());
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair A[1] = {{0, 1}};
  EJitDimPair B[1] = {{0, 2}};
  EJitDimPair Cc[1] = {{0, 3}};
  P.compileOrGet(1, A, 1, nullptr);
  P.compileOrGet(2, B, 1, nullptr);
  P.compileOrGet(3, Cc, 1, nullptr);
  EXPECT_EQ(P.pollBudget(2), 2u);
}

TEST(EJitTaskPoolTest, NullCompiledPointerTreatedFailure) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  C.nullOut = true;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(9, D, 1, nullptr);
  EXPECT_TRUE(P.pollOne());
  EXPECT_NE(P.compileOrGet(9, D, 1, nullptr).status,
            EJitCompileOrGetStatus::CacheHit);
}

TEST(EJitTaskPoolTest, VersionMismatchDropsQueuedAtCheckpoint1) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(3, D, 1, nullptr); // snapshots version 0
  EXPECT_TRUE(P.switchController().setEnabled(0, 1, false)); // bump to 1
  EXPECT_TRUE(P.pollOne()); // dequeued but dropped (stale version)
  EXPECT_EQ(C.calls.loadAcquire(), 0u);
}

// End-to-end commit gate via the worker: a toggle injected inside the publish
// window (post-checkpoint-2, under the write lock) rejects the result, releases
// the dedup slot, and leaves the cache empty.
TEST(EJitTaskPoolTest, ToggleBetweenCheckpointAndPublishRejectsResult) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  P.cache().setPrePublishHookForTest(
      [](void *ctx) {
        static_cast<EJitTaskPool *>(ctx)->switchController().setEnabled(0, 1,
                                                                        false);
      },
      &P);
  EJitDimPair D[1] = {{0, 1}};
  EXPECT_EQ(P.compileOrGet(40, D, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(P.pollOne()); // compiles, then publish is rejected at the gate
  P.cache().setPrePublishHookForTest(nullptr, nullptr);
  EXPECT_EQ(C.calls.loadAcquire(), 1u); // it did compile
  EXPECT_EQ(P.pendingCount(), 0u);      // dedup released
  // Re-enable so the instance check passes; lookup must still miss (nothing was
  // published for the stale result).
  P.switchController().setEnabled(0, 1, true);
  auto r = P.compileOrGet(40, D, 1, nullptr);
  EXPECT_NE(r.status, EJitCompileOrGetStatus::CacheHit);
}

// A toggle strictly AFTER a successful publish invalidates the entry on the
// next lookup (the stored snapshot no longer matches the current version).
TEST(EJitTaskPoolTest, ToggleAfterPublishInvalidatesEntry) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(50, D, 1, nullptr);
  EXPECT_TRUE(P.pollOne()); // publish at version 0
  auto hit = P.compileOrGet(50, D, 1, nullptr);
  EXPECT_EQ(hit.status, EJitCompileOrGetStatus::CacheHit);
  P.releaseRead(hit.bucketIndex);
  P.switchController().setEnabled(0, 1, false); // version 1
  P.switchController().setEnabled(0, 1, true);  // version 2
  auto r = P.compileOrGet(50, D, 1, nullptr);
  EXPECT_NE(r.status, EJitCompileOrGetStatus::CacheHit);
}

TEST(EJitTaskPoolTest, PendingCountDropsAfterPoll) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(5, D, 1, nullptr);
  EXPECT_GT(P.pendingCount(), 0u);
  P.pollOne();
  EXPECT_EQ(P.pendingCount(), 0u);
}

TEST(EJitTaskPoolTest, StatsSnapshotPopulated) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(41, D, 1, nullptr);
  P.pollOne();
  auto h = P.compileOrGet(41, D, 1, nullptr);
  P.releaseRead(h.bucketIndex);
  EJitTaskPoolStatsSnapshot S{};
  P.getStats(S);
  EXPECT_GE(S.asyncEnqueues, 1u);
  EXPECT_GE(S.asyncCompiles, 1u);
  EXPECT_GE(S.cacheHits, 1u);
  EXPECT_GE(S.readyEntries, 1u);
}

//===----------------------------------------------------------------------===//
// EJitWorker lifecycle (real std::thread via EJitSreTask_host)
//===----------------------------------------------------------------------===//

TEST(EJitWorkerTest, StartStopRestartIdempotent) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);

  EXPECT_TRUE(P.startWorker());
  EXPECT_TRUE(P.startWorker()); // idempotent: second start is a no-op success
  EXPECT_TRUE(spinUntil([&] { return P.isWorkerRunning(); }));

  EJitDimPair D[1] = {{0, 1}};
  EXPECT_EQ(P.compileOrGet(7, D, 1, nullptr).status,
            EJitCompileOrGetStatus::EnqueuedPending);
  EXPECT_TRUE(spinUntil([&] {
    auto r = P.compileOrGet(7, D, 1, nullptr);
    if (r.status == EJitCompileOrGetStatus::CacheHit) {
      P.releaseRead(r.bucketIndex);
      return true;
    }
    return false;
  }));
  EXPECT_GE(P.workerProcessedCount(), 1u);

  P.stopWorker();
  EXPECT_FALSE(P.isWorkerRunning());
  P.stopWorker(); // idempotent

  EXPECT_TRUE(P.startWorker()); // restart
  EXPECT_TRUE(spinUntil([&] { return P.isWorkerRunning(); }));
  P.stopWorker();
  EXPECT_FALSE(P.isWorkerRunning());
}

TEST(EJitWorkerTest, SpinsWhenIdle) {
  EJitTaskPool P(8, false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EXPECT_TRUE(P.startWorker());
  // With no work queued, the worker loops via the idle yield and accrues spins.
  EXPECT_TRUE(spinUntil([&] { return P.workerSpinCount() > 0; }));
  P.stopWorker();
}

// The taskpool destructor must stop the worker BEFORE tearing down the cache,
// so no worker thread can publish into a half-torn cache.
TEST(EJitWorkerTest, WorkerStopsBeforeCacheShutdown) {
  static int releaseCalls = 0;
  releaseCalls = 0;
  {
    EJitTaskPool P(8, /*autoStartWorker=*/true);
    P.setReleaser([](void *, void *) { ++releaseCalls; }, nullptr);
    P.switchController().setMode(EJitCompileMode::Async);
    MockCompiler C;
    P.setCompiler(&MockCompiler::compile, &C);
    EXPECT_TRUE(spinUntil([&] { return P.isWorkerRunning(); }));
    EJitDimPair D[1] = {{0, 1}};
    P.compileOrGet(60, D, 1, nullptr);
    // Let the real worker compile + publish one entry.
    EXPECT_TRUE(spinUntil([&] {
      auto r = P.compileOrGet(60, D, 1, nullptr);
      if (r.status == EJitCompileOrGetStatus::CacheHit) {
        P.releaseRead(r.bucketIndex);
        return true;
      }
      return false;
    }));
    // Destructor here: stops the worker, then shuts the cache down, releasing
    // the live entry exactly once.
  }
  EXPECT_EQ(releaseCalls, 1);
}

TEST(EJitWorkerTest, AutoStartConstructorRunsWorker) {
  EJitTaskPool P(8, /*autoStartWorker=*/true);
  EXPECT_TRUE(spinUntil([&] { return P.isWorkerRunning(); }));
  P.stopWorker();
  EXPECT_FALSE(P.isWorkerRunning());
}

//===----------------------------------------------------------------------===//
// Finding (三): deferred worker start. The driver builds the taskpool with the
// worker stopped; EJit starts it only after registration + engine are ready.
//===----------------------------------------------------------------------===//

TEST(EJitWorkerTest, ConstructWithoutAutoStartDoesNotRun) {
  EJitTaskPool P(8, /*autoStartWorker=*/false);
  EXPECT_FALSE(P.isWorkerRunning());
  // A request can be enqueued, but with no worker nothing is ever compiled.
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(70, D, 1, nullptr); // enqueued, pending
  // Spin a while: the compiler callback must never fire while the worker is
  // stopped (init failure / not-yet-ready must not compile).
  for (int i = 0; i < 200000; ++i)
    std::this_thread::yield();
  EXPECT_EQ(C.calls.loadAcquire(), 0u);
  EXPECT_FALSE(P.isWorkerRunning());
}

TEST(EJitWorkerTest, StartAfterReadyCompiles) {
  EJitTaskPool P(8, /*autoStartWorker=*/false);
  P.switchController().setMode(EJitCompileMode::Async);
  MockCompiler C;
  P.setCompiler(&MockCompiler::compile, &C);
  EXPECT_FALSE(P.isWorkerRunning());
  // Worker starts only once everything is wired up (mirrors EJit starting it
  // after registration freeze + engine install).
  EXPECT_TRUE(P.startWorker());
  EXPECT_TRUE(spinUntil([&] { return P.isWorkerRunning(); }));
  EJitDimPair D[1] = {{0, 1}};
  P.compileOrGet(71, D, 1, nullptr);
  EXPECT_TRUE(spinUntil([&] {
    auto r = P.compileOrGet(71, D, 1, nullptr);
    if (r.status == EJitCompileOrGetStatus::CacheHit) {
      P.releaseRead(r.bucketIndex);
      return true;
    }
    return false;
  }));
  P.stopWorker();
  EXPECT_FALSE(P.isWorkerRunning());
}

TEST(EJitWorkerTest, StopUnstartedWorkerIsSafe) {
  // Mirrors an init that failed before startTaskPoolWorker(): the destructor /
  // explicit stop must be idempotent on a never-started worker.
  EJitTaskPool P(8, /*autoStartWorker=*/false);
  EXPECT_FALSE(P.isWorkerRunning());
  P.stopWorker(); // no-op, must not crash
  EXPECT_FALSE(P.isWorkerRunning());
  // Destructor (here) on a never-started worker must also be safe.
}

//===----------------------------------------------------------------------===//
// Array-level control-plane wiring (EJit::activateArray/deactivateArray, the
// single-vs-multi-array clean-reject rule, and the version semantics) is tested
// end-to-end through the public EJit / C ABI in EJitRuntimeTest.cpp
// (EJitTaskpoolArray.*), since proving the wiring requires a real EJit
// instance. A bare SwitchController test here could not distinguish array-level
// from period-level behavior and was removed.
//===----------------------------------------------------------------------===//
