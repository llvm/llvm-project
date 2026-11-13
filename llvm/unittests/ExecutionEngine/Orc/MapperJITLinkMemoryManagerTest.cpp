//===---------------- MapperJITLinkMemoryManagerTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"

#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"
#include "llvm/Testing/Support/Error.h"

#include <new>

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

class CounterMapper final : public MemoryMapper {
public:
  CounterMapper(std::unique_ptr<MemoryMapper> Mapper)
      : Mapper(std::move(Mapper)) {}

  unsigned int getPageSize() override { return Mapper->getPageSize(); }

  void reserve(size_t NumBytes, OnReservedFunction OnReserved) override {
    ++ReserveCount;
    return Mapper->reserve(NumBytes, std::move(OnReserved));
  }

  void initialize(AllocInfo &AI, OnInitializedFunction OnInitialized) override {
    ++InitCount;
    return Mapper->initialize(AI, std::move(OnInitialized));
  }

  char *prepare(LinkGraph &G, ExecutorAddr Addr, size_t ContentSize) override {
    return Mapper->prepare(G, Addr, ContentSize);
  }

  void deinitialize(ArrayRef<ExecutorAddr> Allocations,
                    OnDeinitializedFunction OnDeInitialized) override {
    ++DeinitCount;
    return Mapper->deinitialize(Allocations, std::move(OnDeInitialized));
  }

  void release(ArrayRef<ExecutorAddr> Reservations,
               OnReleasedFunction OnRelease) override {
    ++ReleaseCount;

    return Mapper->release(Reservations, std::move(OnRelease));
  }

  int ReserveCount = 0, InitCount = 0, DeinitCount = 0, ReleaseCount = 0;

private:
  std::unique_ptr<MemoryMapper> Mapper;
};

TEST(MapperJITLinkMemoryManagerTest, InProcess) {
  auto Mapper = std::make_unique<CounterMapper>(
      cantFail(InProcessMemoryMapper::Create()));

  auto *Counter = static_cast<CounterMapper *>(Mapper.get());

  auto MemMgr = std::make_unique<MapperJITLinkMemoryManager>(16 * 1024 * 1024,
                                                             std::move(Mapper));

  EXPECT_EQ(Counter->ReserveCount, 0);
  EXPECT_EQ(Counter->InitCount, 0);

  StringRef Hello = "hello";
  auto SSA1 = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, std::make_shared<SymbolStringPool>(),
      Triple("x86_64-apple-darwin"), nullptr,
      {{MemProt::Read, {Hello.size(), Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA1, Succeeded());

  EXPECT_EQ(Counter->ReserveCount, 1);
  EXPECT_EQ(Counter->InitCount, 0);

  auto SegInfo1 = SSA1->getSegInfo(MemProt::Read);
  memcpy(SegInfo1.WorkingMem.data(), Hello.data(), Hello.size());

  auto FA1 = SSA1->finalize();
  EXPECT_THAT_EXPECTED(FA1, Succeeded());

  EXPECT_EQ(Counter->ReserveCount, 1);
  EXPECT_EQ(Counter->InitCount, 1);

  auto SSA2 = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, std::make_shared<SymbolStringPool>(),
      Triple("x86_64-apple-darwin"), nullptr,
      {{MemProt::Read, {Hello.size(), Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA2, Succeeded());

  // last reservation should be reused
  EXPECT_EQ(Counter->ReserveCount, 1);
  EXPECT_EQ(Counter->InitCount, 1);

  auto SegInfo2 = SSA2->getSegInfo(MemProt::Read);
  memcpy(SegInfo2.WorkingMem.data(), Hello.data(), Hello.size());
  auto FA2 = SSA2->finalize();
  EXPECT_THAT_EXPECTED(FA2, Succeeded());

  EXPECT_EQ(Counter->ReserveCount, 1);
  EXPECT_EQ(Counter->InitCount, 2);

  ExecutorAddr TargetAddr1(SegInfo1.Addr);
  ExecutorAddr TargetAddr2(SegInfo2.Addr);

  const char *TargetMem1 = TargetAddr1.toPtr<const char *>();
  StringRef TargetHello1(TargetMem1, Hello.size());
  EXPECT_EQ(Hello, TargetHello1);

  const char *TargetMem2 = TargetAddr2.toPtr<const char *>();
  StringRef TargetHello2(TargetMem2, Hello.size());
  EXPECT_EQ(Hello, TargetHello2);

  EXPECT_EQ(Counter->DeinitCount, 0);

  auto Err2 = MemMgr->deallocate(std::move(*FA1));
  EXPECT_THAT_ERROR(std::move(Err2), Succeeded());

  EXPECT_EQ(Counter->DeinitCount, 1);

  auto Err3 = MemMgr->deallocate(std::move(*FA2));
  EXPECT_THAT_ERROR(std::move(Err3), Succeeded());

  EXPECT_EQ(Counter->DeinitCount, 2);
}

TEST(MapperJITLinkMemoryManagerTest, Coalescing) {
  auto Mapper = cantFail(InProcessMemoryMapper::Create());
  auto MemMgr = std::make_unique<MapperJITLinkMemoryManager>(16 * 1024 * 1024,
                                                             std::move(Mapper));
  auto SSP = std::make_shared<SymbolStringPool>();

  auto SSA1 = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, SSP, Triple("x86_64-apple-darwin"), nullptr,
      {{MemProt::Read, {1024, Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA1, Succeeded());
  auto SegInfo1 = SSA1->getSegInfo(MemProt::Read);
  ExecutorAddr TargetAddr1(SegInfo1.Addr);
  auto FA1 = SSA1->finalize();
  EXPECT_THAT_EXPECTED(FA1, Succeeded());

  auto SSA2 = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, SSP, Triple("x86_64-apple-darwin"), nullptr,
      {{MemProt::Read, {1024, Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA2, Succeeded());
  auto FA2 = SSA2->finalize();
  EXPECT_THAT_EXPECTED(FA2, Succeeded());

  auto Err2 = MemMgr->deallocate(std::move(*FA1));
  EXPECT_THAT_ERROR(std::move(Err2), Succeeded());

  auto Err3 = MemMgr->deallocate(std::move(*FA2));
  EXPECT_THAT_ERROR(std::move(Err3), Succeeded());

  auto SSA3 = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, SSP, Triple("x86_64-apple-darwin"), nullptr,
      {{MemProt::Read, {2048, Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA3, Succeeded());

  auto SegInfo3 = SSA3->getSegInfo(MemProt::Read);
  ExecutorAddr TargetAddr3(SegInfo3.Addr);

  auto FA3 = SSA3->finalize();
  EXPECT_THAT_EXPECTED(FA3, Succeeded());

  // previous two freed 1024 blocks should be fused to form a 2048 block
  EXPECT_EQ(TargetAddr1, TargetAddr3);

  auto Err4 = MemMgr->deallocate(std::move(*FA3));
  EXPECT_THAT_ERROR(std::move(Err4), Succeeded());
}

// A colocating (slab) allocator places all objects inside a single reservation,
// so any two of them are close enough for a 32-bit PC-relative cross-object
// reference: an x86-64 REL32 is a signed 32-bit displacement, i.e. the target
// must be within +/-2GB of the reference. The default InProcessMemoryManager
// does NOT guarantee this -- it allocates each object independently, so two
// objects can land more than 2GB apart and a direct cross-object call/branch
// would fail to relocate ("out of range"). This test verifies that the
// Mapper-based slab allocator keeps separately-allocated objects within REL32
// range.
TEST(MapperJITLinkMemoryManagerTest, Colocation) {
  auto Mapper = cantFail(InProcessMemoryMapper::Create());
  auto MemMgr = std::make_unique<MapperJITLinkMemoryManager>(16 * 1024 * 1024,
                                                             std::move(Mapper));
  auto SSP = std::make_shared<SymbolStringPool>();

  // Allocate several "objects", the way separate input files would be added.
  constexpr unsigned NumObjects = 8;
  SmallVector<JITLinkMemoryManager::FinalizedAlloc> Allocs;
  uint64_t MinAddr = ~uint64_t(0), MaxAddr = 0;

  for (unsigned I = 0; I != NumObjects; ++I) {
    auto SSA = jitlink::SimpleSegmentAlloc::Create(
        *MemMgr, SSP, Triple("x86_64-apple-darwin"), nullptr,
        {{MemProt::Read, {4096, Align(16)}}});
    EXPECT_THAT_EXPECTED(SSA, Succeeded());

    uint64_t Addr =
        ExecutorAddr(SSA->getSegInfo(MemProt::Read).Addr).getValue();
    if (Addr < MinAddr)
      MinAddr = Addr;
    if (Addr > MaxAddr)
      MaxAddr = Addr;

    auto FA = SSA->finalize();
    EXPECT_THAT_EXPECTED(FA, Succeeded());
    Allocs.push_back(std::move(*FA));
  }

  // All objects sit inside the one ~16MB reservation, so the span between the
  // lowest and highest is far below the 2GB REL32 limit.
  constexpr uint64_t REL32Limit = uint64_t(1) << 31; // 2GB
  EXPECT_LT(MaxAddr - MinAddr, REL32Limit);

  for (auto &FA : Allocs)
    EXPECT_THAT_ERROR(MemMgr->deallocate(std::move(FA)), Succeeded());
}

// With per-JITDylib colocation enabled, each JITDylib draws from its own pool
// of reservations: a second JITDylib's allocation does NOT reuse the first
// JITDylib's leftover space, so it triggers a fresh reservation. With the
// default (single shared pool) the second JITDylib reuses the first's leftover,
// so only one reservation is made. Counting reservations is a deterministic way
// to observe that objects are being grouped per-JITDylib.
TEST(MapperJITLinkMemoryManagerTest, ColocationPerJITDylib) {
  auto SSP = std::make_shared<SymbolStringPool>();

  // Allocate a small object in the given JITDylib and finalize it.
  auto AllocSmall = [&](MapperJITLinkMemoryManager &MemMgr,
                        const JITLinkDylib *JD) {
    auto SSA = jitlink::SimpleSegmentAlloc::Create(
        MemMgr, SSP, Triple("x86_64-apple-darwin"), JD,
        {{MemProt::Read, {4096, Align(16)}}});
    EXPECT_THAT_EXPECTED(SSA, Succeeded());
    auto FA = SSA->finalize();
    EXPECT_THAT_EXPECTED(FA, Succeeded());
    return cantFail(std::move(FA));
  };

  // Per-JITDylib colocation ON: JD B cannot reuse JD A's reservation, so a
  // second reservation is made.
  {
    auto Mapper = std::make_unique<CounterMapper>(
        cantFail(InProcessMemoryMapper::Create()));
    auto *Counter = static_cast<CounterMapper *>(Mapper.get());
    MapperJITLinkMemoryManager MemMgr(16 * 1024 * 1024, std::move(Mapper),
                                      /*ColocatePerJITDylib=*/true);
    // Declared after MemMgr so they're destroyed first: MemMgr must outlive
    // any JITLinkDylib it's used with.
    JITLinkDylib JDA("A"), JDB("B");

    auto FA_A = AllocSmall(MemMgr, &JDA);
    EXPECT_EQ(Counter->ReserveCount, 1);
    auto FA_B = AllocSmall(MemMgr, &JDB);
    EXPECT_EQ(Counter->ReserveCount, 2); // separate reservation for JD B

    // The two JITDylibs draw from separate reservations, so their objects
    // should be at least one reservation unit apart.
    ExecutorAddr AddrA = FA_A.getAddress(), AddrB = FA_B.getAddress();
    EXPECT_GE(AddrA > AddrB ? AddrA - AddrB : AddrB - AddrA,
              MemMgr.reservationUnits());

    EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(FA_A)), Succeeded());
    EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(FA_B)), Succeeded());
  }

  // Default (single shared pool): JD B reuses JD A's leftover, so only one
  // reservation is made.
  {
    auto Mapper = std::make_unique<CounterMapper>(
        cantFail(InProcessMemoryMapper::Create()));
    auto *Counter = static_cast<CounterMapper *>(Mapper.get());
    MapperJITLinkMemoryManager MemMgr(16 * 1024 * 1024, std::move(Mapper));
    // Same ordering requirement as above: destroyed before MemMgr.
    JITLinkDylib JDA("A"), JDB("B");

    auto FA_A = AllocSmall(MemMgr, &JDA);
    EXPECT_EQ(Counter->ReserveCount, 1);
    auto FA_B = AllocSmall(MemMgr, &JDB);
    EXPECT_EQ(Counter->ReserveCount, 1); // reused JD A's reservation

    // The two JITDylibs share the same reservation, so their objects should
    // be within one reservation unit of one another.
    ExecutorAddr AddrA = FA_A.getAddress(), AddrB = FA_B.getAddress();
    EXPECT_LT(AddrA > AddrB ? AddrA - AddrB : AddrB - AddrA,
              MemMgr.reservationUnits());

    EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(FA_A)), Succeeded());
    EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(FA_B)), Succeeded());
  }
}

// A destroyed JITDylib's pool should be freed, so a new JITDylib reusing the
// same address gets its own reservation instead of inheriting the old one's
// leftover space. Uses placement new to force that address reuse.
TEST(MapperJITLinkMemoryManagerTest, PoolFreedOnJITDylibDestruction) {
  constexpr size_t SlabSize = 64 * 1024;
  auto SSP = std::make_shared<SymbolStringPool>();

  auto Mapper = std::make_unique<CounterMapper>(
      cantFail(InProcessMemoryMapper::Create()));
  auto *Counter = static_cast<CounterMapper *>(Mapper.get());
  MapperJITLinkMemoryManager MemMgr(SlabSize, std::move(Mapper),
                                    /*ColocatePerJITDylib=*/true);

  auto AllocSlabSized = [&](const JITLinkDylib *JD) {
    return jitlink::SimpleSegmentAlloc::Create(
        MemMgr, SSP, Triple("x86_64-apple-darwin"), JD,
        {{MemProt::Read, {SlabSize, Align(1)}}});
  };

  alignas(JITLinkDylib) unsigned char Storage[sizeof(JITLinkDylib)];

  auto *JD1 = new (Storage) JITLinkDylib("first");
  auto SSA1 = AllocSlabSized(JD1);
  EXPECT_THAT_EXPECTED(SSA1, Succeeded());
  auto FA1 = SSA1->finalize();
  EXPECT_THAT_EXPECTED(FA1, Succeeded());
  EXPECT_EQ(Counter->ReserveCount, 1);
  // Freeing it leaves the whole slab as free space in JD1's pool.
  EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(*FA1)), Succeeded());

  // JD1's destruction should drop that free space too.
  JD1->~JITLinkDylib();

  // JD2 occupies the exact address JD1 used to.
  auto *JD2 = new (Storage) JITLinkDylib("second");
  ASSERT_EQ(static_cast<void *>(JD1), static_cast<void *>(JD2));

  auto SSA2 = AllocSlabSized(JD2);
  EXPECT_THAT_EXPECTED(SSA2, Succeeded());
  auto FA2 = SSA2->finalize();
  EXPECT_THAT_EXPECTED(FA2, Succeeded());
  // If JD1's pool wasn't freed, JD2 would reuse its leftover slab instead of
  // reserving its own.
  EXPECT_EQ(Counter->ReserveCount, 2);

  EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(*FA2)), Succeeded());
  JD2->~JITLinkDylib();
}

// denySlabGrowth() restricts each pool to a single reservation: once the first
// slab is full, an allocation that needs fresh memory fails instead of spilling
// into another (possibly out-of-range) slab. The default allows the growth.
TEST(MapperJITLinkMemoryManagerTest, SingleSlab) {
  constexpr size_t SlabSize = 64 * 1024;
  auto SSP = std::make_shared<SymbolStringPool>();

  // Allocate an object that fills a whole slab (so the next allocation cannot
  // reuse leftover space and must reserve again).
  auto AllocSlabSized = [&](MapperJITLinkMemoryManager &MemMgr) {
    return jitlink::SimpleSegmentAlloc::Create(
        MemMgr, SSP, Triple("x86_64-apple-darwin"), nullptr,
        {{MemProt::Read, {SlabSize, Align(1)}}});
  };

  // Single-slab: the second allocation is rejected.
  {
    MapperJITLinkMemoryManager MemMgr(
        SlabSize, cantFail(InProcessMemoryMapper::Create()));
    MemMgr.denySlabGrowth();

    auto SSA1 = AllocSlabSized(MemMgr);
    EXPECT_THAT_EXPECTED(SSA1, Succeeded());
    auto FA1 = SSA1->finalize();
    EXPECT_THAT_EXPECTED(FA1, Succeeded());

    EXPECT_THAT_EXPECTED(AllocSlabSized(MemMgr), Failed());

    EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(*FA1)), Succeeded());
  }

  // Default (multiple slabs allowed): the same second allocation succeeds by
  // reserving another slab.
  {
    MapperJITLinkMemoryManager MemMgr(
        SlabSize, cantFail(InProcessMemoryMapper::Create()));

    auto SSA1 = AllocSlabSized(MemMgr);
    EXPECT_THAT_EXPECTED(SSA1, Succeeded());
    auto FA1 = SSA1->finalize();
    EXPECT_THAT_EXPECTED(FA1, Succeeded());

    auto SSA2 = AllocSlabSized(MemMgr);
    EXPECT_THAT_EXPECTED(SSA2, Succeeded());
    auto FA2 = SSA2->finalize();
    EXPECT_THAT_EXPECTED(FA2, Succeeded());

    EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(*FA1)), Succeeded());
    EXPECT_THAT_ERROR(MemMgr.deallocate(std::move(*FA2)), Succeeded());
  }
}

// createColocatingInProcessMemoryManager() builds a working per-JITDylib-
// colocating in-process allocator, and defaultSlabSize() reflects the host
// pointer width.
TEST(MapperJITLinkMemoryManagerTest, CreateColocatingInProcessMemoryManager) {
  // Use a small explicit slab so the test doesn't reserve the 1GB default.
  constexpr size_t ReservationGranularity = 64 * 1024;
  auto MemMgr = createColocatingInProcessMemoryManager(ReservationGranularity);
  ASSERT_THAT_EXPECTED(MemMgr, Succeeded());

  auto SSP = std::make_shared<SymbolStringPool>();
  auto SSA = jitlink::SimpleSegmentAlloc::Create(
      **MemMgr, SSP, Triple("x86_64-apple-darwin"), nullptr,
      {{MemProt::Read, {4096, Align(16)}}});
  ASSERT_THAT_EXPECTED(SSA, Succeeded());
  auto FA = SSA->finalize();
  ASSERT_THAT_EXPECTED(FA, Succeeded());
  EXPECT_THAT_ERROR((*MemMgr)->deallocate(std::move(*FA)), Succeeded());

  if constexpr (sizeof(void *) >= 8)
    EXPECT_EQ(MapperJITLinkMemoryManager::defaultSlabSize(), size_t(1) << 30);
  else
    EXPECT_EQ(MapperJITLinkMemoryManager::defaultSlabSize(), size_t(10) << 20);
}

} // namespace
