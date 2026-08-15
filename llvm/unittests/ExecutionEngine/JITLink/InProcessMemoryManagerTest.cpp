//===------ InProcessMemoryManagerTest.cpp - Test the default mem mgr -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <random>

using namespace llvm;
using namespace llvm::jitlink;

namespace {

struct SegSpec {
  orc::MemProt Prot;
  orc::MemLifetime Lifetime = orc::MemLifetime::Standard;
  size_t ContentSize = 1;
  size_t ZeroFillSize = 0;
};

/// Create a LinkGraph with one section (holding one block per non-zero size in
/// the spec) per SegSpec.
std::unique_ptr<LinkGraph> makeGraph(const Twine &Name,
                                     ArrayRef<SegSpec> Specs) {
  auto G = std::make_unique<LinkGraph>(
      Name.str(), std::make_shared<orc::SymbolStringPool>(),
      Triple("x86_64-unknown-linux-gnu"), SubtargetFeatures(),
      getGenericEdgeKindName);

  orc::ExecutorAddr NextAddr(0x1000);
  unsigned SecIdx = 0;
  for (auto &Spec : Specs) {
    auto &Sec = G->createSection(("sec." + Twine(SecIdx++)).str(), Spec.Prot);
    Sec.setMemLifetime(Spec.Lifetime);
    if (Spec.ContentSize) {
      G->createMutableContentBlock(Sec, Spec.ContentSize, NextAddr, 1, 0);
      NextAddr += Spec.ContentSize;
    }
    if (Spec.ZeroFillSize) {
      G->createZeroFillBlock(Sec, Spec.ZeroFillSize, NextAddr, 1, 0);
      NextAddr += Spec.ZeroFillSize;
    }
  }

  return G;
}

/// The address range that a section was assigned by the memory manager, and
/// the AllocGroup that it belongs to.
struct SegPlacement {
  orc::AllocGroup AG;
  orc::ExecutorAddr Addr;
  uint64_t Size = 0;
};

std::vector<SegPlacement> getPlacements(LinkGraph &G) {
  std::vector<SegPlacement> Placements;
  for (auto &Sec : G.sections()) {
    orc::ExecutorAddr Start(~static_cast<uint64_t>(0)), End(0);
    for (const auto *B : Sec.blocks()) {
      Start = std::min(Start, B->getAddress());
      End = std::max(End, B->getRange().End);
    }
    Placements.push_back(
        {{Sec.getMemProt(), Sec.getMemLifetime()}, Start, End - Start});
  }
  return Placements;
}

std::string toString(orc::AllocGroup AG) {
  std::string S;
  raw_string_ostream(S) << AG;
  return S;
}

/// Returns the lowest address assigned to any of G's blocks. For graphs with a
/// single segment this is the base of that segment.
orc::ExecutorAddr getLowestBlockAddr(LinkGraph &G) {
  orc::ExecutorAddr Lowest(~static_cast<uint64_t>(0));
  for (const auto *B : G.blocks())
    Lowest = std::min(Lowest, B->getAddress());
  return Lowest;
}

/// Returns the address range spanned by all of G's segments.
orc::ExecutorAddrRange getSpan(LinkGraph &G) {
  orc::ExecutorAddr Lowest(~static_cast<uint64_t>(0)), Highest(0);
  for (auto &P : getPlacements(G)) {
    Lowest = std::min(Lowest, P.Addr);
    Highest = std::max(Highest, P.Addr + P.Size);
  }
  return {Lowest, Highest};
}

class InProcessMemoryManagerTest : public testing::Test {
protected:
  void SetUp() override {
    auto PS = sys::Process::getPageSize();
    ASSERT_THAT_EXPECTED(PS, Succeeded());
    PageSize = *PS;
    ChunkSize = 4 * PageSize;
    SlabSize = 64 * ChunkSize;
    ReservationSize = 128 * SlabSize;

    auto MM =
        InProcessMemoryManager::Create({SlabSize, ChunkSize, ReservationSize});
    ASSERT_THAT_EXPECTED(MM, Succeeded());
    MemMgr = std::move(*MM);
  }

  void TearDown() override {
    if (!FinalizedAllocs.empty())
      EXPECT_THAT_ERROR(MemMgr->deallocate(std::move(FinalizedAllocs)),
                        Succeeded());
  }

  /// Allocate and finalize G, keeping the resulting allocation alive until the
  /// end of the test.
  void allocateAndFinalize(LinkGraph &G) {
    auto Alloc = MemMgr->allocate(nullptr, G);
    ASSERT_THAT_EXPECTED(Alloc, Succeeded());
    auto FA = (*Alloc)->finalize();
    ASSERT_THAT_EXPECTED(FA, Succeeded());
    FinalizedAllocs.push_back(std::move(*FA));
  }

  /// Link NumGraphs graphs, each with an R-X, an R-- and an RW- segment of one
  /// page each, and return the placements of all their segments.
  std::vector<SegPlacement> linkRXROAndRWGraphs(unsigned NumGraphs) {
    std::vector<SegPlacement> Placements;
    for (unsigned I = 0; I != NumGraphs; ++I) {
      auto G = makeGraph("g" + Twine(I),
                         {{orc::MemProt::Read | orc::MemProt::Exec},
                          {orc::MemProt::Read},
                          {orc::MemProt::Read | orc::MemProt::Write}});
      allocateAndFinalize(*G);
      append_range(Placements, getPlacements(*G));
      Graphs.push_back(std::move(G));
    }
    return Placements;
  }

  uint64_t PageSize = 0;
  uint64_t ChunkSize = 0;
  uint64_t SlabSize = 0;
  uint64_t ReservationSize = 0;
  std::unique_ptr<InProcessMemoryManager> MemMgr;
  std::vector<std::unique_ptr<LinkGraph>> Graphs;
  std::vector<JITLinkMemoryManager::FinalizedAlloc> FinalizedAllocs;
};

TEST_F(InProcessMemoryManagerTest, ChunksAreNotSharedBetweenAllocGroups) {
  // Link a batch of graphs whose segments would interleave permissions if they
  // were laid out one graph at a time, then check that each chunk of the slab
  // only ever holds segments from a single AllocGroup.
  constexpr unsigned NumGraphs = 32;
  auto Placements = linkRXROAndRWGraphs(NumGraphs);
  ASSERT_EQ(Placements.size(), 3 * NumGraphs);

  // Each graph asked for three single-page segments, so everything should have
  // fit in one slab. Since executable memory is allocated from the bottom of
  // the slab, the lowest address handed out is the slab's base.
  orc::ExecutorAddr SlabBase(~static_cast<uint64_t>(0));
  for (auto &P : Placements)
    SlabBase = std::min(SlabBase, P.Addr);

  DenseMap<uint64_t, orc::AllocGroup> ChunkGroups;
  for (auto &P : Placements) {
    ASSERT_GE(P.Addr, SlabBase);
    ASSERT_LE(P.Addr + P.Size, SlabBase + SlabSize)
        << "Segments spread over more than one slab";

    const uint64_t FirstChunk = (P.Addr - SlabBase) / ChunkSize;
    const uint64_t LastChunk = (P.Addr + P.Size - 1 - SlabBase) / ChunkSize;
    for (uint64_t C = FirstChunk; C <= LastChunk; ++C) {
      auto [I, Inserted] = ChunkGroups.try_emplace(C, P.AG);
      EXPECT_TRUE(Inserted || I->second == P.AG)
          << "Chunk " << C << " shared by " << toString(I->second) << " and "
          << toString(P.AG);
    }
  }
}

TEST_F(InProcessMemoryManagerTest, ExecutableMemoryIsPackedAtTheBottom) {
  // Executable segments are allocated from the bottom of the slab and
  // everything else from the top, so no non-executable segment should ever end
  // up below an executable one.
  constexpr unsigned NumGraphs = 32;
  const auto Placements = linkRXROAndRWGraphs(NumGraphs);

  orc::ExecutorAddr LowestExec(~static_cast<uint64_t>(0)), HighestExec(0);
  orc::ExecutorAddr LowestNonExec(~static_cast<uint64_t>(0));
  for (auto &P : Placements) {
    if ((P.AG.getMemProt() & orc::MemProt::Exec) != orc::MemProt::None) {
      LowestExec = std::min(LowestExec, P.Addr);
      HighestExec = std::max(HighestExec, P.Addr + P.Size);
    } else
      LowestNonExec = std::min(LowestNonExec, P.Addr);
  }

  EXPECT_LE(HighestExec, LowestNonExec)
      << "Executable and non-executable memory are interleaved";

  // The executable segments were one page each, so they should be packed into
  // the smallest number of chunks that can hold them.
  EXPECT_LE(HighestExec - LowestExec,
            divideCeil(NumGraphs * PageSize, ChunkSize) * ChunkSize)
      << "Executable memory is not densely packed";
}

TEST_F(InProcessMemoryManagerTest, SegmentsForOneGraphShareASlab) {
  // Force many slabs to be created by linking graphs that are large relative
  // to the slab size, and check that the segments of any one graph always stay
  // within a slab's worth of address space of each other: intra-graph
  // references have to stay in range of one another.
  constexpr unsigned NumGraphs = 8;
  const size_t SegSize = SlabSize / 8;

  for (unsigned I = 0; I != NumGraphs; ++I) {
    auto G =
        makeGraph("g" + Twine(I),
                  {{orc::MemProt::Read | orc::MemProt::Exec,
                    orc::MemLifetime::Standard, SegSize},
                   {orc::MemProt::Read, orc::MemLifetime::Standard, SegSize},
                   {orc::MemProt::Read | orc::MemProt::Write,
                    orc::MemLifetime::Standard, SegSize}});
    allocateAndFinalize(*G);

    auto Span = getSpan(*G);
    EXPECT_LE(Span.size(), SlabSize)
        << "Segments of graph " << I << " span more than one slab";

    Graphs.push_back(std::move(G));
  }
}

TEST_F(InProcessMemoryManagerTest, FreedMemoryIsReusedAndZeroed) {
  // Check that memory returned by deallocate is handed out again, and that it
  // is writable and zeroed when it is.
  auto G1 = makeGraph("g1", {{orc::MemProt::Read | orc::MemProt::Exec,
                              orc::MemLifetime::Standard, PageSize}});
  auto Alloc1 = MemMgr->allocate(nullptr, *G1);
  ASSERT_THAT_EXPECTED(Alloc1, Succeeded());

  orc::ExecutorAddr Addr1 = getLowestBlockAddr(*G1);

  // Dirty the whole page before finalizing (memory is still writable at this
  // point) so that we can tell whether it is zeroed on re-use.
  memset(Addr1.toPtr<char *>(), 0xAA, PageSize);

  auto FA1 = (*Alloc1)->finalize();
  ASSERT_THAT_EXPECTED(FA1, Succeeded());
  EXPECT_THAT_ERROR(MemMgr->deallocate(std::move(*FA1)), Succeeded());

  // The same AllocGroup should get the same memory back.
  auto G2 = makeGraph("g2", {{orc::MemProt::Read | orc::MemProt::Exec,
                              orc::MemLifetime::Standard, 1, PageSize - 1}});
  auto Alloc2 = MemMgr->allocate(nullptr, *G2);
  ASSERT_THAT_EXPECTED(Alloc2, Succeeded());

  ASSERT_EQ(getLowestBlockAddr(*G2), Addr1) << "Freed memory was not reused";

  // Re-used memory must be writable again (protections are reset on free) and
  // zeroed, so that zero-fill blocks really do read as zero.
  auto *Mem = Addr1.toPtr<char *>();
  Mem[0] = 42;
  for (uint64_t I = 1; I != PageSize; ++I)
    ASSERT_EQ(Mem[I], 0) << "Re-used memory was not zeroed at offset " << I;

  auto FA2 = (*Alloc2)->finalize();
  ASSERT_THAT_EXPECTED(FA2, Succeeded());
  EXPECT_THAT_ERROR(MemMgr->deallocate(std::move(*FA2)), Succeeded());
}

TEST_F(InProcessMemoryManagerTest, FinalizeLifetimeSegmentsAreFreedOnFinalize) {
  // Finalize-lifetime segments should be returned to their slab as soon as the
  // allocation is finalized, so that the memory can be handed straight back
  // out to the next allocation.
  auto G1 = makeGraph("g1", {{orc::MemProt::Read | orc::MemProt::Write,
                              orc::MemLifetime::Finalize}});
  auto Alloc1 = MemMgr->allocate(nullptr, *G1);
  ASSERT_THAT_EXPECTED(Alloc1, Succeeded());
  orc::ExecutorAddr FinalizeSegAddr = getLowestBlockAddr(*G1);
  auto FA1 = (*Alloc1)->finalize();
  ASSERT_THAT_EXPECTED(FA1, Succeeded());

  auto G2 = makeGraph("g2", {{orc::MemProt::Read | orc::MemProt::Write,
                              orc::MemLifetime::Finalize}});
  auto Alloc2 = MemMgr->allocate(nullptr, *G2);
  ASSERT_THAT_EXPECTED(Alloc2, Succeeded());
  EXPECT_EQ(getLowestBlockAddr(*G2), FinalizeSegAddr)
      << "Finalize-lifetime memory was not released on finalize";

  auto FA2 = (*Alloc2)->finalize();
  ASSERT_THAT_EXPECTED(FA2, Succeeded());

  std::vector<JITLinkMemoryManager::FinalizedAlloc> Allocs;
  Allocs.push_back(std::move(*FA1));
  Allocs.push_back(std::move(*FA2));
  EXPECT_THAT_ERROR(MemMgr->deallocate(std::move(Allocs)), Succeeded());
}

TEST_F(InProcessMemoryManagerTest, AbandonReturnsMemoryToSlab) {
  const auto G1 = makeGraph("g1", {{orc::MemProt::Read | orc::MemProt::Write}});
  auto Alloc1 = MemMgr->allocate(nullptr, *G1);
  ASSERT_THAT_EXPECTED(Alloc1, Succeeded());
  const orc::ExecutorAddr Addr1 = getLowestBlockAddr(*G1);

  std::promise<MSVCPError> P;
  auto F = P.get_future();
  (*Alloc1)->abandon([&](Error Err) { P.set_value(std::move(Err)); });
  EXPECT_THAT_ERROR(F.get(), Succeeded());

  const auto G2 = makeGraph("g2", {{orc::MemProt::Read | orc::MemProt::Write}});
  auto Alloc2 = MemMgr->allocate(nullptr, *G2);
  ASSERT_THAT_EXPECTED(Alloc2, Succeeded());
  EXPECT_EQ(getLowestBlockAddr(*G2), Addr1)
      << "Abandoned memory was not returned to its slab";

  auto FA2 = (*Alloc2)->finalize();
  ASSERT_THAT_EXPECTED(FA2, Succeeded());
  EXPECT_THAT_ERROR(MemMgr->deallocate(std::move(*FA2)), Succeeded());
}

TEST_F(InProcessMemoryManagerTest, OversizedGraphGetsItsOwnSlab) {
  // Graphs that are too big for a default-sized slab should still be linkable.
  const auto G = makeGraph("g", {{orc::MemProt::Read | orc::MemProt::Exec,
                                  orc::MemLifetime::Standard, SlabSize},
                                 {orc::MemProt::Read | orc::MemProt::Write,
                                  orc::MemLifetime::Standard, SlabSize}});
  allocateAndFinalize(*G);
  EXPECT_GE(getSpan(*G).size(), 2 * SlabSize);
}

TEST_F(InProcessMemoryManagerTest, AllAllocationsStayWithinTheReservation) {
  // Force lots of slab churn -- allocate an oversized graph (so it needs a
  // fresh slab of its own) and free it immediately, many times over -- and
  // check that every address ever handed out stays within ReservationSize of
  // the very first one. All slabs are carved out of one fixed reservation
  // (see the class comment on InProcessMemoryManager), so this is a
  // deterministic guarantee of the allocator itself: unlike relying on the OS
  // to honor a placement hint for separate mappings, it can't drift.
  auto G0 = makeGraph("g0", {{orc::MemProt::Read | orc::MemProt::Write}});
  allocateAndFinalize(*G0);
  const orc::ExecutorAddr FirstAddr = getLowestBlockAddr(*G0);

  orc::ExecutorAddr Lowest = FirstAddr, Highest = FirstAddr;
  constexpr unsigned NumCycles = 64;
  for (unsigned I = 0; I != NumCycles; ++I) {
    auto G = makeGraph("churn" + Twine(I),
                       {{orc::MemProt::Read | orc::MemProt::Write,
                         orc::MemLifetime::Standard, SlabSize}});
    auto Alloc = MemMgr->allocate(nullptr, *G);
    ASSERT_THAT_EXPECTED(Alloc, Succeeded());
    auto Span = getSpan(*G);
    Lowest = std::min(Lowest, Span.Start);
    Highest = std::max(Highest, Span.End);
    auto FA = (*Alloc)->finalize();
    ASSERT_THAT_EXPECTED(FA, Succeeded());
    EXPECT_THAT_ERROR(MemMgr->deallocate(std::move(*FA)), Succeeded());
  }

  EXPECT_LE((Highest - Lowest), ReservationSize)
      << "Allocations spread further than the reservation is wide";
}

TEST_F(InProcessMemoryManagerTest, ReservationExhaustionFails) {
  // A single graph that needs more room than the whole reservation should
  // fail cleanly rather than falling back to a separate, unbounded mapping
  // (which would defeat the point of the reservation).
  auto G = makeGraph(
      "g", {{orc::MemProt::Read | orc::MemProt::Write,
             orc::MemLifetime::Standard, ReservationSize + SlabSize}});
  auto Alloc = MemMgr->allocate(nullptr, *G);
  ASSERT_THAT_EXPECTED(Alloc, Failed());
}

TEST_F(InProcessMemoryManagerTest, RandomAllocateAndDeallocate) {
  // Exercise the slab free lists with a randomized allocate / deallocate
  // workload, checking that live segments never overlap one another and that
  // memory handed out is always writable and zeroed.
  constexpr unsigned NumOps = 512;
  std::minstd_rand Rng(42);

  struct LiveAlloc {
    JITLinkMemoryManager::FinalizedAlloc FA;
    std::vector<SegPlacement> Segs;
  };
  std::vector<LiveAlloc> Live;

  auto Overlaps = [](const SegPlacement &A, const SegPlacement &B) {
    return A.Addr < B.Addr + B.Size && B.Addr < A.Addr + A.Size;
  };

  for (unsigned Op = 0; Op != NumOps; ++Op) {
    // Deallocate roughly a third of the time, once there's something to free.
    if (!Live.empty() && Rng() % 3 == 0) {
      unsigned Idx = Rng() % Live.size();
      std::swap(Live[Idx], Live.back());
      ASSERT_THAT_ERROR(MemMgr->deallocate(std::move(Live.back().FA)),
                        Succeeded());
      Live.pop_back();
      continue;
    }

    // Build a graph with a random subset of segments at random sizes.
    SmallVector<SegSpec, 4> Specs;
    for (auto Prot :
         {orc::MemProt::Read | orc::MemProt::Exec, orc::MemProt::Read,
          orc::MemProt::Read | orc::MemProt::Write})
      if (Rng() % 2)
        Specs.push_back(
            {Prot, orc::MemLifetime::Standard, 1 + Rng() % (4 * PageSize)});
    if (Specs.empty())
      Specs.push_back({orc::MemProt::Read | orc::MemProt::Write});
    if (Rng() % 4 == 0)
      Specs.push_back({orc::MemProt::Read | orc::MemProt::Write,
                       orc::MemLifetime::Finalize, 1 + Rng() % PageSize});

    auto G = makeGraph("g" + Twine(Op), Specs);
    auto Alloc = MemMgr->allocate(nullptr, *G);
    ASSERT_THAT_EXPECTED(Alloc, Succeeded());

    // Every byte handed out should be zero, and writable.
    for (auto &P : getPlacements(*G)) {
      auto *Mem = P.Addr.toPtr<char *>();
      for (uint64_t I = 0; I != P.Size; ++I)
        ASSERT_EQ(Mem[I], 0) << "Memory was not zeroed on allocation";
      memset(Mem, 0xAA, P.Size);
    }

    auto FA = (*Alloc)->finalize();
    ASSERT_THAT_EXPECTED(FA, Succeeded());

    // Only standard-lifetime segments are still live after finalization.
    std::vector<SegPlacement> Segs;
    for (auto &P : getPlacements(*G))
      if (P.AG.getMemLifetime() == orc::MemLifetime::Standard)
        Segs.push_back(P);

    for (auto &Other : Live)
      for (auto &A : Other.Segs)
        for (auto &B : Segs)
          ASSERT_FALSE(Overlaps(A, B))
              << "Live segments overlap at " << A.Addr.getValue() << " and "
              << B.Addr.getValue();

    Live.push_back({std::move(*FA), std::move(Segs)});
    Graphs.push_back(std::move(G));
  }

  std::vector<JITLinkMemoryManager::FinalizedAlloc> Remaining;
  for (auto &L : Live)
    Remaining.push_back(std::move(L.FA));
  EXPECT_THAT_ERROR(MemMgr->deallocate(std::move(Remaining)), Succeeded());
}

TEST_F(InProcessMemoryManagerTest, DefaultSlabOptionsAreConsistent) {
  auto PS = sys::Process::getPageSize();
  ASSERT_THAT_EXPECTED(PS, Succeeded());

  auto SO = InProcessMemoryManager::SlabOptions::defaults(*PS);
  EXPECT_GT(SO.ChunkSize, 0U);
  EXPECT_EQ(SO.ChunkSize % *PS, 0U);
  EXPECT_EQ(SO.SlabSize % SO.ChunkSize, 0U);
  EXPECT_GE(SO.SlabSize, SO.ChunkSize);
  EXPECT_EQ(SO.ReservationSize % SO.SlabSize, 0U);
  EXPECT_GE(SO.ReservationSize, SO.SlabSize);
  // Comfortably under the 2^32 budget some platforms need every piece of
  // code a session ever JITs to stay within (see the class comment on
  // InProcessMemoryManager).
  EXPECT_LT(SO.ReservationSize, 1ULL << 32);
}

} // namespace
