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
