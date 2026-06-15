//===-- EJitCodePoolMemoryManagerTest.cpp - mem mgr over code pool --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Host-runnable tests that drive EJitCodePoolMemoryManager with a synthetic
//  JITLink LinkGraph and a mock SRE backend. These exercise the exact
//  allocate -> finalize -> seal path the engine uses at lookup, without needing
//  a native backend to actually execute code (the host has no matching JIT
//  target, so end-to-end execution is validated on the target instead).
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCodePool.h"
#include "llvm/ExecutionEngine/EJIT/EJitCodePoolMemoryManager.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <vector>

using namespace llvm;
using namespace llvm::ejit;
using namespace llvm::jitlink;

namespace {

struct MockSre {
  std::vector<void *> Raws;
  size_t SealCalls = 0;
  unsigned SealRc = 0;

  ~MockSre() {
    for (void *P : Raws)
      std::free(P);
  }
  void *rawAlloc(size_t Bytes) {
    void *P = std::malloc(Bytes);
    if (P)
      Raws.push_back(P);
    return P;
  }
  unsigned seal(void *) {
    ++SealCalls;
    return SealRc;
  }
};

EJitCodePoolManager::Options poolOpts(size_t PoolSize) {
  EJitCodePoolManager::Options O;
  O.poolSize = PoolSize;
  O.poolAlign = PoolSize;
  O.minCodeAlign = 64;
  return O;
}

// 64 bytes of filler "code" referenced by the content block (must outlive G).
const char CodeBytes[64] = {0};

std::unique_ptr<LinkGraph> makeCodeGraph(size_t Size, uint64_t VAddr) {
  auto G = std::make_unique<LinkGraph>(
      "g", std::make_shared<orc::SymbolStringPool>(),
      Triple("x86_64-unknown-linux-gnu"), SubtargetFeatures(),
      getGenericEdgeKindName);
  auto &Sec =
      G->createSection("__text", orc::MemProt::Read | orc::MemProt::Exec);
  G->createContentBlock(Sec, ArrayRef<char>(CodeBytes, Size),
                        orc::ExecutorAddr(VAddr), 16, 0);
  return G;
}

void *firstBlockAddr(LinkGraph &G) {
  Block *B = *G.blocks().begin();
  return B->getAddress().toPtr<void *>();
}

} // namespace

// JIT code memory is allocated out of the pool (not mmap), and the resolved
// block address lands inside an owned pool.
TEST(EJitCodePoolMemMgr, CodeMemoryComesFromPool) {
  MockSre M;
  EJitCodePoolManager Pool(
      poolOpts(/*PoolSize=*/256 * 1024),
      [&M](size_t N) { return M.rawAlloc(N); },
      [&M](void *B) { return M.seal(B); });
  EJitCodePoolMemoryManager MM(Pool, /*PageSize=*/4096);

  auto G = makeCodeGraph(64, 0x1000);
  auto IFA = cantFail(MM.allocate(nullptr, *G));
  void *CodeAddr = firstBlockAddr(*G);

  EXPECT_TRUE(Pool.contains(CodeAddr));
  auto S = Pool.getStats();
  EXPECT_EQ(S.poolCount, 1u);
  EXPECT_GT(S.usedBytes, 0u);

  auto FA = cantFail(IFA->finalize());
  cantFail(MM.deallocate(std::move(FA)));
}

// finalize() does NOT seal: the pool stays RW so JITLink can keep writing.
// Sealing happens out-of-band (the engine does it at lookup).
TEST(EJitCodePoolMemMgr, FinalizeKeepsPoolWritable) {
  MockSre M;
  EJitCodePoolManager Pool(
      poolOpts(256 * 1024), [&M](size_t N) { return M.rawAlloc(N); },
      [&M](void *B) { return M.seal(B); });
  EJitCodePoolMemoryManager MM(Pool, 4096);

  auto G = makeCodeGraph(64, 0x1000);
  auto IFA = cantFail(MM.allocate(nullptr, *G));
  void *CodeAddr = firstBlockAddr(*G);
  auto FA = cantFail(IFA->finalize());

  // The memory manager must not have flipped permissions.
  EXPECT_EQ(M.SealCalls, 0u);
  EXPECT_EQ(Pool.getStats().sealedCount, 0u);

  // The engine's lookup step seals the containing pool.
  cantFail(Pool.sealPoolContaining(CodeAddr));
  EXPECT_EQ(M.SealCalls, 1u);
  EXPECT_EQ(Pool.getStats().sealedCount, 1u);

  cantFail(MM.deallocate(std::move(FA)));
}

// Sealing the pool that backs a finalized function is idempotent: a repeated
// seal (e.g. a second lookup of the same function) does not re-invoke enable_ex.
TEST(EJitCodePoolMemMgr, RepeatedSealNoDuplicateEnableEx) {
  MockSre M;
  EJitCodePoolManager Pool(
      poolOpts(256 * 1024), [&M](size_t N) { return M.rawAlloc(N); },
      [&M](void *B) { return M.seal(B); });
  EJitCodePoolMemoryManager MM(Pool, 4096);

  auto G = makeCodeGraph(64, 0x1000);
  auto IFA = cantFail(MM.allocate(nullptr, *G));
  void *CodeAddr = firstBlockAddr(*G);
  auto FA = cantFail(IFA->finalize());

  cantFail(Pool.sealPoolContaining(CodeAddr)); // first lookup
  cantFail(Pool.sealPoolContaining(CodeAddr)); // second lookup, same pool
  EXPECT_EQ(M.SealCalls, 1u);
  EXPECT_EQ(Pool.getStats().sealInvocations, 1u);

  cantFail(MM.deallocate(std::move(FA)));
}

// Runtime scenario: two functions are compiled and "looked up" in turn. The
// first seals its pool; because a sealed pool is never reused, the second
// function is allocated from a brand-new pool. Mirrors the engine flow of
// compile -> lookup(seal) -> compile -> lookup(seal).
TEST(EJitCodePoolMemMgr, SecondFunctionUsesNewPoolAfterSeal) {
  constexpr size_t kPoolSize = 64 * 1024;
  MockSre M;
  EJitCodePoolManager Pool(
      poolOpts(kPoolSize), [&M](size_t N) { return M.rawAlloc(N); },
      [&M](void *B) { return M.seal(B); });
  EJitCodePoolMemoryManager MM(Pool, 4096);

  // Function 1: allocate, finalize, seal (first lookup).
  auto G1 = makeCodeGraph(64, 0x1000);
  auto IFA1 = cantFail(MM.allocate(nullptr, *G1));
  void *Addr1 = firstBlockAddr(*G1);
  auto FA1 = cantFail(IFA1->finalize());
  cantFail(Pool.sealPoolContaining(Addr1));
  EXPECT_EQ(M.SealCalls, 1u);
  EXPECT_EQ(Pool.getStats().poolCount, 1u);

  // Function 2: the active pool is now sealed, so this must land in a new pool
  // even though the first pool still had free space.
  auto G2 = makeCodeGraph(64, 0x2000);
  auto IFA2 = cantFail(MM.allocate(nullptr, *G2));
  void *Addr2 = firstBlockAddr(*G2);
  auto FA2 = cantFail(IFA2->finalize());
  cantFail(Pool.sealPoolContaining(Addr2));

  auto S = Pool.getStats();
  EXPECT_EQ(S.poolCount, 2u);
  EXPECT_EQ(S.sealedCount, 2u);
  EXPECT_EQ(M.SealCalls, 2u);

  // The two functions live in different 64KiB pools.
  auto poolOf = [](void *P) {
    return reinterpret_cast<uintptr_t>(P) & ~static_cast<uintptr_t>(kPoolSize - 1);
  };
  EXPECT_NE(poolOf(Addr1), poolOf(Addr2));

  cantFail(MM.deallocate(std::move(FA1)));
  cantFail(MM.deallocate(std::move(FA2)));
}

