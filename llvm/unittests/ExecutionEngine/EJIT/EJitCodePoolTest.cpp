//===-- EJitCodePoolTest.cpp - Unit tests for the SRE code pool -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Host-runnable tests for EJitCodePoolManager. These never touch real SRE:
//  the raw allocator and the seal (enable_ex) primitive are injected mocks, so
//  the tests pass on any host regardless of whether EJIT_SRE_CODE_POOL is set.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCodePool.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace llvm;
using namespace llvm::ejit;

namespace {

/// Mock SRE backend: tracks raw allocations (freed at teardown), records seal
/// calls and the bases that were sealed, and can be configured to fail.
struct MockSre {
  std::vector<void *> Raws;
  size_t AllocCalls = 0;
  bool FailNextAlloc = false;

  std::vector<void *> SealedBases;
  size_t SealCalls = 0;
  unsigned SealRc = 0; // return code handed back by the mock enable_ex

  ~MockSre() {
    for (void *P : Raws)
      std::free(P);
  }

  void *rawAlloc(size_t Bytes) {
    ++AllocCalls;
    if (FailNextAlloc) {
      FailNextAlloc = false;
      return nullptr;
    }
    void *P = std::malloc(Bytes);
    if (P)
      Raws.push_back(P);
    return P;
  }

  unsigned seal(void *Base) {
    ++SealCalls;
    if (SealRc == 0)
      SealedBases.push_back(Base);
    return SealRc;
  }
};

EJitCodePoolManager makeManager(MockSre &M,
                                EJitCodePoolManager::Options Opts) {
  return EJitCodePoolManager(
      Opts, [&M](size_t N) { return M.rawAlloc(N); },
      [&M](void *B) { return M.seal(B); });
}

EJitCodePoolManager::Options smallOpts(size_t PoolSize = 256) {
  EJitCodePoolManager::Options O;
  O.poolSize = PoolSize;
  O.poolAlign = PoolSize; // keep raw allocations tiny for logic tests
  O.minCodeAlign = 64;
  return O;
}

uintptr_t A(const void *P) { return reinterpret_cast<uintptr_t>(P); }

} // namespace

// 1. The usable base of a pool is 2MiB aligned.
TEST(EJitCodePool, BaseIs2MiBAligned) {
  constexpr size_t k2M = static_cast<size_t>(2) * 1024 * 1024;
  EJitCodePoolManager::Options O;
  O.poolSize = k2M;
  O.poolAlign = k2M;
  O.minCodeAlign = 64;
  MockSre M;
  auto Mgr = makeManager(M, O);

  void *P = cantFail(Mgr.allocateCode(128, 16));
  // First allocation sits at offset 0, so it equals the pool base.
  EXPECT_EQ(A(P) % k2M, 0u);
  EXPECT_TRUE(Mgr.contains(P));
}

// 2. Multiple small allocations bump contiguously inside one RW pool.
TEST(EJitCodePool, BumpAllocatesWithinOneRWPool) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts(/*PoolSize=*/4096));

  void *a = cantFail(Mgr.allocateCode(64, 64));
  void *b = cantFail(Mgr.allocateCode(64, 64));
  void *c = cantFail(Mgr.allocateCode(64, 64));

  EXPECT_LT(A(a), A(b));
  EXPECT_LT(A(b), A(c));
  EXPECT_EQ(A(b) - A(a), 64u); // 64-aligned, 64-byte blocks → exactly adjacent
  EXPECT_EQ(A(c) - A(b), 64u);

  auto S = Mgr.getStats();
  EXPECT_EQ(S.poolCount, 1u);
  EXPECT_EQ(S.sealedCount, 0u);
  EXPECT_EQ(S.activeCount, 1u);
  EXPECT_EQ(M.SealCalls, 0u);
}

// 3. Sealing a pool marks it executable and invokes enable_ex exactly once.
TEST(EJitCodePool, SealMarksPoolExecutable) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts());

  void *a = cantFail(Mgr.allocateCode(64, 64));
  cantFail(Mgr.sealPoolContaining(a));

  auto S = Mgr.getStats();
  EXPECT_EQ(S.sealedCount, 1u);
  EXPECT_EQ(S.activeCount, 0u);
  EXPECT_EQ(S.sealInvocations, 1u);
  EXPECT_EQ(M.SealCalls, 1u);
  ASSERT_EQ(M.SealedBases.size(), 1u);
}

// 4. A sealed pool is never reused; the next allocation creates a new pool.
TEST(EJitCodePool, SealedPoolNotReusedNewPoolCreated) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts());

  void *a = cantFail(Mgr.allocateCode(64, 64));
  cantFail(Mgr.sealPoolContaining(a));
  void *b = cantFail(Mgr.allocateCode(64, 64));

  auto S = Mgr.getStats();
  EXPECT_EQ(S.poolCount, 2u);
  EXPECT_EQ(S.sealedCount, 1u);
  EXPECT_EQ(S.activeCount, 1u);
  // b must not fall inside the first (sealed) pool's range.
  EXPECT_GE(A(b) > A(a) ? A(b) - A(a) : A(a) - A(b), 64u);
}

// 5. Repeated seal of the same pool does not re-invoke enable_ex (idempotent).
TEST(EJitCodePool, RepeatedSealNoDuplicateEnableEx) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts());

  void *a = cantFail(Mgr.allocateCode(64, 64));
  void *b = cantFail(Mgr.allocateCode(64, 64)); // same pool as a

  cantFail(Mgr.sealPoolContaining(a));
  EXPECT_EQ(M.SealCalls, 1u);
  cantFail(Mgr.sealPoolContaining(b)); // already sealed → no-op success
  EXPECT_EQ(M.SealCalls, 1u);
  EXPECT_EQ(Mgr.getStats().sealInvocations, 1u);
}

// 6. enable_ex failure surfaces as an Error and the pool stays writable.
TEST(EJitCodePool, EnableExFailureReturnsError) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts());

  void *a = cantFail(Mgr.allocateCode(64, 64));
  M.SealRc = 7; // make enable_ex fail

  Error Err = Mgr.sealPoolContaining(a);
  EXPECT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  auto S = Mgr.getStats();
  EXPECT_EQ(S.sealedCount, 0u); // still RW, not sealed
  EXPECT_EQ(S.sealInvocations, 0u);
}

// 6b. A seal failure during full-pool rollover propagates out of allocateCode.
TEST(EJitCodePool, RolloverSealFailurePropagates) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts(/*PoolSize=*/256));

  // Fill the 256-byte pool with four 64-byte blocks (offsets 0/64/128/192).
  for (int i = 0; i < 4; ++i)
    (void)cantFail(Mgr.allocateCode(64, 64));

  M.SealRc = 9; // the rollover seal of the full pool will fail
  auto E = Mgr.allocateCode(64, 64);
  EXPECT_FALSE(static_cast<bool>(E));
  consumeError(E.takeError());
}

// 7. A request larger than the pool size is rejected cleanly (no allocation).
TEST(EJitCodePool, OversizeRequestCleanReject) {
  MockSre M;
  auto Opts = smallOpts(/*PoolSize=*/4096);
  auto Mgr = makeManager(M, Opts);

  auto E = Mgr.allocateCode(Opts.poolSize + 1, 16);
  EXPECT_FALSE(static_cast<bool>(E));
  consumeError(E.takeError());

  EXPECT_EQ(Mgr.getStats().poolCount, 0u);
  EXPECT_EQ(M.AllocCalls, 0u); // never even tried to allocate a pool
}

// 8. Statistics (pool count, sealed count, used / wasted bytes) are correct.
TEST(EJitCodePool, StatsAreAccurate) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts(/*PoolSize=*/4096));

  (void)cantFail(Mgr.allocateCode(100, 64)); // off 0,   used 100
  void *b = cantFail(Mgr.allocateCode(200, 64)); // off 128, used 328

  auto S = Mgr.getStats();
  EXPECT_EQ(S.poolCount, 1u);
  EXPECT_EQ(S.reservedBytes, 4096u);
  EXPECT_EQ(S.usedBytes, 328u);
  EXPECT_EQ(S.sealedCount, 0u);
  EXPECT_EQ(S.wastedBytes, 0u); // active pool tail is not counted as wasted

  cantFail(Mgr.sealPoolContaining(b));
  S = Mgr.getStats();
  EXPECT_EQ(S.sealedCount, 1u);
  EXPECT_EQ(S.wastedBytes, 4096u - 328u); // sealed tail is wasted
}

// Strategy case 3: a full active pool is sealed automatically before a new
// pool is created on the next allocation.
TEST(EJitCodePool, FullPoolSealedOnRollover) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts(/*PoolSize=*/256));

  for (int i = 0; i < 4; ++i)
    (void)cantFail(Mgr.allocateCode(64, 64)); // fills the pool exactly

  EXPECT_EQ(M.SealCalls, 0u);
  void *e = cantFail(Mgr.allocateCode(64, 64)); // triggers seal + new pool
  EXPECT_TRUE(Mgr.contains(e));

  auto S = Mgr.getStats();
  EXPECT_EQ(S.poolCount, 2u);
  EXPECT_EQ(S.sealedCount, 1u);
  EXPECT_EQ(M.SealCalls, 1u);
}

// Larger alignment requests are honored.
TEST(EJitCodePool, RespectsLargerAlignment) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts(/*PoolSize=*/4096));

  (void)cantFail(Mgr.allocateCode(8, 64));   // off 0
  void *b = cantFail(Mgr.allocateCode(8, 256)); // must be 256-aligned
  EXPECT_EQ(A(b) % 256u, 0u);
}

// sealAllWritablePools seals every still-writable pool.
TEST(EJitCodePool, SealAllWritablePools) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts());

  void *a = cantFail(Mgr.allocateCode(64, 64));
  cantFail(Mgr.sealPoolContaining(a)); // pool 1 sealed
  (void)cantFail(Mgr.allocateCode(64, 64)); // pool 2 active

  cantFail(Mgr.sealAllWritablePools());
  auto S = Mgr.getStats();
  EXPECT_EQ(S.poolCount, 2u);
  EXPECT_EQ(S.sealedCount, 2u);
  EXPECT_EQ(S.activeCount, 0u);
  EXPECT_EQ(M.SealCalls, 2u); // pool 1 not re-sealed
}

// An address not owned by any pool cannot be sealed.
TEST(EJitCodePool, SealUnknownAddressFails) {
  MockSre M;
  auto Mgr = makeManager(M, smallOpts());
  int OnStack = 0;
  Error Err = Mgr.sealPoolContaining(&OnStack);
  EXPECT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));
}

//===----------------------------------------------------------------------===//
// 4K page-seal mode tests
//
// These exercise the SRE-platform 4K execute-permission interface: the pool is
// still 2MiB-aligned (split into 4K mappings via split_2m_to_4k at creation),
// but only the 4KiB pages a finalized allocation covers are sealed (enable_ex
// per page). All injected mocks; no real platform symbols.
//===----------------------------------------------------------------------===//
namespace {

constexpr size_t kTwoMiB = static_cast<size_t>(2) * 1024 * 1024;
constexpr size_t kFourKiB = static_cast<size_t>(4) * 1024;

/// Mock SRE backend for 4K mode: deliberately returns a non-2MiB-aligned raw
/// base, records split_2m_to_4k(base,size) calls and per-page enable_ex calls,
/// and can be made to fail split or a chosen seal call.
struct MockSre4K {
  std::vector<void *> Origs; // posix_memalign bases (freed at teardown)
  uintptr_t LastRawReturned = 0;
  size_t LastBytesRequested = 0;
  size_t AllocCalls = 0;

  std::vector<std::pair<uintptr_t, size_t>> Splits;
  unsigned SplitRc = 0;

  std::vector<uintptr_t> SealedPages;
  size_t SealCalls = 0;
  int FailSealOnCall = -1; // 1-based seal call index to fail; -1 = never
  unsigned SealFailRc = 7;

  ~MockSre4K() {
    for (void *P : Origs)
      std::free(P);
  }

  void *rawAlloc(size_t Bytes) {
    ++AllocCalls;
    LastBytesRequested = Bytes;
    void *Base = nullptr;
    // Over-allocate, 2MiB-aligned, then hand back a deliberately misaligned
    // pointer (offset 4KiB) so the manager must round the base up to 2MiB.
    if (posix_memalign(&Base, kTwoMiB, Bytes + kTwoMiB) != 0)
      return nullptr;
    Origs.push_back(Base);
    void *Raw = static_cast<char *>(Base) + kFourKiB;
    LastRawReturned = reinterpret_cast<uintptr_t>(Raw);
    return Raw;
  }

  unsigned split(void *Base, size_t Size) {
    Splits.push_back({reinterpret_cast<uintptr_t>(Base), Size});
    return SplitRc;
  }

  unsigned seal(void *PageVA) {
    ++SealCalls;
    if (FailSealOnCall > 0 && static_cast<int>(SealCalls) == FailSealOnCall)
      return SealFailRc;
    SealedPages.push_back(reinterpret_cast<uintptr_t>(PageVA));
    return 0;
  }
};

EJitCodePoolManager::Options fourKOpts() {
  EJitCodePoolManager::Options O;
  O.poolSize = kTwoMiB;
  O.poolAlign = kTwoMiB;
  O.minCodeAlign = 64;
  O.fourKSeal = true;
  O.sealPageSize = kFourKiB;
  return O;
}

EJitCodePoolManager makeManager4K(MockSre4K &M,
                                  EJitCodePoolManager::Options Opts) {
  return EJitCodePoolManager(
      Opts, [&M](size_t N) { return M.rawAlloc(N); },
      [&M](void *V) { return M.seal(V); },
      [&M](void *B, size_t S) { return M.split(B, S); });
}

} // namespace

// A deliberately non-2MiB-aligned raw base is rounded up to a 2MiB-aligned pool
// base, and the usable window stays inside the raw allocation.
TEST(EJitCodePool4K, AlignsMisalignedRawBaseTo2MiB) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  void *P = cantFail(Mgr.allocateCode(128, 64));
  // First allocation in 4K mode starts at offset 0, i.e. the pool base.
  EXPECT_EQ(A(P) % kTwoMiB, 0u);          // aligned base is 2MiB aligned
  EXPECT_TRUE(Mgr.contains(P));
  EXPECT_NE(A(P), M.LastRawReturned);     // raw base was misaligned, base != raw
  // Usable window [base, base+poolSize) must fit within [raw, raw+requested).
  EXPECT_LE(A(P) + kTwoMiB, M.LastRawReturned + M.LastBytesRequested);
  // The manager requests poolSize + 2MiB of alignment slack.
  EXPECT_EQ(M.LastBytesRequested, kTwoMiB + kTwoMiB);
}

// split_2m_to_4k is called exactly once per pool, with the aligned base and the
// usable pool size; a second allocation in the same pool does not re-split.
TEST(EJitCodePool4K, SplitCalledOncePerPool) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  void *P = cantFail(Mgr.allocateCode(128, 64));
  ASSERT_EQ(M.Splits.size(), 1u);
  EXPECT_EQ(M.Splits[0].first, A(P));      // split base == aligned pool base
  EXPECT_EQ(M.Splits[0].second, kTwoMiB);  // split size == usable pool size
  EXPECT_EQ(M.Splits[0].first % kTwoMiB, 0u);
  EXPECT_EQ(Mgr.getStats().splitInvocations, 1u);

  (void)cantFail(Mgr.allocateCode(128, 64)); // same pool, no new split
  EXPECT_EQ(M.Splits.size(), 1u);
  EXPECT_EQ(Mgr.getStats().splitInvocations, 1u);
}

// A small function seals only the single 4K page it covers, not the whole pool.
TEST(EJitCodePool4K, SmallCodeSealsOnlyCoveredPage) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  void *P = cantFail(Mgr.allocateCode(100, 64));
  cantFail(Mgr.sealCodeRange(P, 100));

  EXPECT_EQ(M.SealCalls, 1u); // one page, NOT 512 (the whole 2MiB pool)
  ASSERT_EQ(M.SealedPages.size(), 1u);
  EXPECT_EQ(M.SealedPages[0], A(P)); // page base == 4K-aligned code start
  EXPECT_EQ(Mgr.getStats().sealInvocations, 1u);
}

// A code range spanning multiple 4K pages loops enable_ex over each page.
TEST(EJitCodePool4K, MultiPageCodeSealsEachPage) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  size_t Sz = 2 * kFourKiB + 200; // covers 3 pages
  void *P = cantFail(Mgr.allocateCode(Sz, 64));
  cantFail(Mgr.sealCodeRange(P, Sz));

  EXPECT_EQ(M.SealCalls, 3u);
  ASSERT_EQ(M.SealedPages.size(), 3u);
  EXPECT_EQ(M.SealedPages[0], A(P));
  EXPECT_EQ(M.SealedPages[1], A(P) + kFourKiB);
  EXPECT_EQ(M.SealedPages[2], A(P) + 2 * kFourKiB);
}

// If any page's enable_ex fails, sealCodeRange returns an Error.
TEST(EJitCodePool4K, EnableExFailureOnAPageReturnsError) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  size_t Sz = 2 * kFourKiB + 200; // 3 pages
  void *P = cantFail(Mgr.allocateCode(Sz, 64));
  M.FailSealOnCall = 2; // fail the 2nd page

  Error Err = Mgr.sealCodeRange(P, Sz);
  EXPECT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));
}

// A subsequent allocation lands on a fresh 4K page, never inside a sealed page,
// and stays in the same (still partially-writable) pool.
TEST(EJitCodePool4K, NextAllocationSkipsSealedPage) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  void *A1 = cantFail(Mgr.allocateCode(100, 64)); // page 0
  cantFail(Mgr.sealCodeRange(A1, 100));           // seal page 0
  void *A2 = cantFail(Mgr.allocateCode(100, 64)); // must be a later page

  EXPECT_TRUE(Mgr.contains(A2));
  EXPECT_GE(A(A2), A(A1) + kFourKiB);
  EXPECT_FALSE(A(A2) >= A(A1) && A(A2) < A(A1) + kFourKiB); // not in sealed page
  EXPECT_EQ(A(A2) % kFourKiB, 0u);                          // fresh page start
  auto S = Mgr.getStats();
  EXPECT_EQ(S.poolCount, 1u);          // same pool reused
  EXPECT_EQ(S.splitInvocations, 1u);
}

// split_2m_to_4k failure makes pool creation (hence allocateCode) fail cleanly,
// registering no pool.
TEST(EJitCodePool4K, SplitFailureFailsPoolCreation) {
  MockSre4K M;
  M.SplitRc = 5; // split_2m_to_4k fails
  auto Mgr = makeManager4K(M, fourKOpts());

  auto E = Mgr.allocateCode(128, 64);
  EXPECT_FALSE(static_cast<bool>(E));
  consumeError(E.takeError());
  EXPECT_EQ(Mgr.getStats().poolCount, 0u);
  EXPECT_EQ(Mgr.getStats().splitInvocations, 0u);
}

// Rolling over to a new pool splits the new pool exactly once, and no whole-pool
// seal happens (sealing is per-page at finalize, not on rollover).
TEST(EJitCodePool4K, RolloverCreatesNewPoolAndSplitsAgain) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  (void)cantFail(Mgr.allocateCode(kTwoMiB, 64)); // fills pool 1 exactly
  void *P2 = cantFail(Mgr.allocateCode(64, 64)); // forces pool 2

  EXPECT_TRUE(Mgr.contains(P2));
  auto S = Mgr.getStats();
  EXPECT_EQ(S.poolCount, 2u);
  EXPECT_EQ(S.splitInvocations, 2u); // one split per pool
  EXPECT_EQ(M.SealCalls, 0u);        // no whole-pool seal on rollover
  EXPECT_EQ(S.sealedCount, 0u);
}

// In 4K mode the whole-pool seal entry point sealPoolContaining is unsupported
// (a bare pointer has no size); it must return an Error and never enable_ex.
TEST(EJitCodePool4K, SealPoolContainingReturnsErrorIn4KMode) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  void *P = cantFail(Mgr.allocateCode(100, 64));
  Error Err = Mgr.sealPoolContaining(P);
  EXPECT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  // No enable_ex was invoked and the pool was not marked sealed.
  EXPECT_EQ(M.SealCalls, 0u);
  EXPECT_EQ(Mgr.getStats().sealInvocations, 0u);
  EXPECT_EQ(Mgr.getStats().sealedCount, 0u);
}

// In 4K mode sealAllWritablePools (whole-pool sealing) is unsupported and must
// return an Error rather than silently sealing or no-op succeeding.
TEST(EJitCodePool4K, SealAllWritablePoolsReturnsErrorIn4KMode) {
  MockSre4K M;
  auto Mgr = makeManager4K(M, fourKOpts());

  (void)cantFail(Mgr.allocateCode(100, 64));
  Error Err = Mgr.sealAllWritablePools();
  EXPECT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  EXPECT_EQ(M.SealCalls, 0u);
  EXPECT_EQ(Mgr.getStats().sealedCount, 0u);
}


