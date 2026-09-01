//===--- JITLinkMemoryManager.cpp - JITLinkMemoryManager implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Process.h"

#include <map>
#include <optional>

#define DEBUG_TYPE "jitlink"

using namespace llvm;

namespace llvm {
namespace jitlink {

JITLinkMemoryManager::~JITLinkMemoryManager() = default;
JITLinkMemoryManager::InFlightAlloc::~InFlightAlloc() = default;

BasicLayout::BasicLayout(LinkGraph &G) : G(G) {

  for (auto &Sec : G.sections()) {
    // Skip empty sections, and sections with NoAlloc lifetime policies.
    if (Sec.blocks().empty() ||
        Sec.getMemLifetime() == orc::MemLifetime::NoAlloc)
      continue;

    auto &Seg = Segments[{Sec.getMemProt(), Sec.getMemLifetime()}];
    for (auto *B : Sec.blocks())
      if (LLVM_LIKELY(!B->isZeroFill()))
        Seg.ContentBlocks.push_back(B);
      else
        Seg.ZeroFillBlocks.push_back(B);
  }

  // Build Segments map.
  auto CompareBlocks = [](const Block *LHS, const Block *RHS) {
    // Sort by section, address and size
    if (LHS->getSection().getOrdinal() != RHS->getSection().getOrdinal())
      return LHS->getSection().getOrdinal() < RHS->getSection().getOrdinal();
    if (LHS->getAddress() != RHS->getAddress())
      return LHS->getAddress() < RHS->getAddress();
    return LHS->getSize() < RHS->getSize();
  };

  LLVM_DEBUG(dbgs() << "Generated BasicLayout for " << G.getName() << ":\n");
  for (auto &KV : Segments) {
    auto &Seg = KV.second;

    llvm::sort(Seg.ContentBlocks, CompareBlocks);
    llvm::sort(Seg.ZeroFillBlocks, CompareBlocks);

    for (auto *B : Seg.ContentBlocks) {
      Seg.ContentSize = alignToBlock(Seg.ContentSize, *B);
      Seg.ContentSize += B->getSize();
      Seg.Alignment = std::max(Seg.Alignment, Align(B->getAlignment()));
    }

    uint64_t SegEndOffset = Seg.ContentSize;
    for (auto *B : Seg.ZeroFillBlocks) {
      SegEndOffset = alignToBlock(SegEndOffset, *B);
      SegEndOffset += B->getSize();
      Seg.Alignment = std::max(Seg.Alignment, Align(B->getAlignment()));
    }
    Seg.ZeroFillSize = SegEndOffset - Seg.ContentSize;

    LLVM_DEBUG({
      dbgs() << "  Seg " << KV.first
             << ": content-size=" << formatv("{0:x}", Seg.ContentSize)
             << ", zero-fill-size=" << formatv("{0:x}", Seg.ZeroFillSize)
             << ", align=" << formatv("{0:x}", Seg.Alignment.value()) << "\n";
    });
  }
}

Expected<BasicLayout::ContiguousPageBasedLayoutSizes>
BasicLayout::getContiguousPageBasedLayoutSizes(uint64_t PageSize) {
  ContiguousPageBasedLayoutSizes SegsSizes;

  for (auto &KV : segments()) {
    auto &AG = KV.first;
    auto &Seg = KV.second;

    if (Seg.Alignment > PageSize)
      return make_error<StringError>("Segment alignment greater than page size",
                                     inconvertibleErrorCode());

    uint64_t SegSize = alignTo(Seg.ContentSize + Seg.ZeroFillSize, PageSize);
    if (AG.getMemLifetime() == orc::MemLifetime::Standard)
      SegsSizes.StandardSegs += SegSize;
    else
      SegsSizes.FinalizeSegs += SegSize;
  }

  return SegsSizes;
}

Error BasicLayout::apply() {
  for (auto &KV : Segments) {
    auto &Seg = KV.second;

    assert(!(Seg.ContentBlocks.empty() && Seg.ZeroFillBlocks.empty()) &&
           "Empty section recorded?");

    for (auto *B : Seg.ContentBlocks) {
      // Align addr and working-mem-offset.
      Seg.Addr = alignToBlock(Seg.Addr, *B);
      Seg.NextWorkingMemOffset = alignToBlock(Seg.NextWorkingMemOffset, *B);

      // Update block addr.
      B->setAddress(Seg.Addr);
      Seg.Addr += B->getSize();

      // Copy content to working memory, then update content to point at working
      // memory.
      memcpy(Seg.WorkingMem + Seg.NextWorkingMemOffset, B->getContent().data(),
             B->getSize());
      B->setMutableContent(
          {Seg.WorkingMem + Seg.NextWorkingMemOffset, B->getSize()});
      Seg.NextWorkingMemOffset += B->getSize();
    }

    for (auto *B : Seg.ZeroFillBlocks) {
      // Align addr.
      Seg.Addr = alignToBlock(Seg.Addr, *B);
      // Update block addr.
      B->setAddress(Seg.Addr);
      Seg.Addr += B->getSize();
    }

    Seg.ContentBlocks.clear();
    Seg.ZeroFillBlocks.clear();
  }

  return Error::success();
}

orc::shared::AllocActions &BasicLayout::graphAllocActions() {
  return G.allocActions();
}

void SimpleSegmentAlloc::Create(JITLinkMemoryManager &MemMgr,
                                std::shared_ptr<orc::SymbolStringPool> SSP,
                                Triple TT, const JITLinkDylib *JD,
                                SegmentMap Segments,
                                OnCreatedFunction OnCreated) {

  static_assert(orc::AllocGroup::NumGroups == 32,
                "AllocGroup has changed. Section names below must be updated");
  StringRef AGSectionNames[] = {
      "__---.standard", "__R--.standard", "__-W-.standard", "__RW-.standard",
      "__--X.standard", "__R-X.standard", "__-WX.standard", "__RWX.standard",
      "__---.finalize", "__R--.finalize", "__-W-.finalize", "__RW-.finalize",
      "__--X.finalize", "__R-X.finalize", "__-WX.finalize", "__RWX.finalize"};

  auto G =
      std::make_unique<LinkGraph>("", std::move(SSP), std::move(TT),
                                  SubtargetFeatures(), getGenericEdgeKindName);
  orc::AllocGroupSmallMap<Block *> ContentBlocks;

  orc::ExecutorAddr NextAddr(0x100000);
  for (auto &KV : Segments) {
    auto &AG = KV.first;
    auto &Seg = KV.second;

    assert(AG.getMemLifetime() != orc::MemLifetime::NoAlloc &&
           "NoAlloc segments are not supported by SimpleSegmentAlloc");

    auto AGSectionName =
        AGSectionNames[static_cast<unsigned>(AG.getMemProt()) |
                       static_cast<bool>(AG.getMemLifetime()) << 3];

    auto &Sec = G->createSection(AGSectionName, AG.getMemProt());
    Sec.setMemLifetime(AG.getMemLifetime());

    if (Seg.ContentSize != 0) {
      NextAddr =
          orc::ExecutorAddr(alignTo(NextAddr.getValue(), Seg.ContentAlign));
      auto &B =
          G->createMutableContentBlock(Sec, G->allocateBuffer(Seg.ContentSize),
                                       NextAddr, Seg.ContentAlign.value(), 0);
      ContentBlocks[AG] = &B;
      NextAddr += Seg.ContentSize;
    }
  }

  // GRef declared separately since order-of-argument-eval isn't specified.
  auto &GRef = *G;
  MemMgr.allocate(JD, GRef,
                  [G = std::move(G), ContentBlocks = std::move(ContentBlocks),
                   OnCreated = std::move(OnCreated)](
                      JITLinkMemoryManager::AllocResult Alloc) mutable {
                    if (!Alloc)
                      OnCreated(Alloc.takeError());
                    else
                      OnCreated(SimpleSegmentAlloc(std::move(G),
                                                   std::move(ContentBlocks),
                                                   std::move(*Alloc)));
                  });
}

Expected<SimpleSegmentAlloc> SimpleSegmentAlloc::Create(
    JITLinkMemoryManager &MemMgr, std::shared_ptr<orc::SymbolStringPool> SSP,
    Triple TT, const JITLinkDylib *JD, SegmentMap Segments) {
  std::promise<MSVCPExpected<SimpleSegmentAlloc>> AllocP;
  auto AllocF = AllocP.get_future();
  Create(MemMgr, std::move(SSP), std::move(TT), JD, std::move(Segments),
         [&](Expected<SimpleSegmentAlloc> Result) {
           AllocP.set_value(std::move(Result));
         });
  return AllocF.get();
}

SimpleSegmentAlloc::SimpleSegmentAlloc(SimpleSegmentAlloc &&) = default;
SimpleSegmentAlloc &
SimpleSegmentAlloc::operator=(SimpleSegmentAlloc &&) = default;
SimpleSegmentAlloc::~SimpleSegmentAlloc() = default;

SimpleSegmentAlloc::SegmentInfo
SimpleSegmentAlloc::getSegInfo(orc::AllocGroup AG) {
  auto I = ContentBlocks.find(AG);
  if (I != ContentBlocks.end()) {
    auto &B = *I->second;
    return {B.getAddress(), B.getAlreadyMutableContent()};
  }
  return {};
}

SimpleSegmentAlloc::SimpleSegmentAlloc(
    std::unique_ptr<LinkGraph> G,
    orc::AllocGroupSmallMap<Block *> ContentBlocks,
    std::unique_ptr<JITLinkMemoryManager::InFlightAlloc> Alloc)
    : G(std::move(G)), ContentBlocks(std::move(ContentBlocks)),
      Alloc(std::move(Alloc)) {}

/// A single, fixed reservation of address space that every slab a memory
/// manager ever creates is carved out of -- see the class comment on
/// InProcessMemoryManager for why that matters. Backed by one OS mapping,
/// made lazily the first time a slab is needed.
struct InProcessMemoryManager::Reservation {
  explicit Reservation(const sys::MemoryBlock &Mem) : Mem(Mem) {
    FreeRanges[0] = Mem.allocatedSize();
  }

  char *base() const { return static_cast<char *>(Mem.base()); }
  uint64_t size() const { return Mem.allocatedSize(); }

  /// Carve Size bytes out of the reservation, first-fit. Returns the offset
  /// of the allocation within the reservation, or std::nullopt if there's no
  /// room left.
  std::optional<uint64_t> allocate(uint64_t Size);

  /// Return a previously allocated range to the reservation.
  void free(uint64_t Offset, uint64_t Size);

  sys::MemoryBlock Mem;

  /// Free ranges of the reservation, keyed by offset. Ranges are disjoint,
  /// and never adjacent to one another (adjacent ranges are always
  /// coalesced).
  std::map<uint64_t, uint64_t> FreeRanges;
};

std::optional<uint64_t>
InProcessMemoryManager::Reservation::allocate(uint64_t Size) {
  assert(Size && "Cannot allocate zero bytes");

  for (auto I = FreeRanges.begin(), E = FreeRanges.end(); I != E; ++I) {
    if (I->second < Size)
      continue;
    const uint64_t Offset = I->first, RangeSize = I->second;
    FreeRanges.erase(I);
    if (RangeSize > Size)
      FreeRanges[Offset + Size] = RangeSize - Size;
    return Offset;
  }
  return std::nullopt;
}

void InProcessMemoryManager::Reservation::free(uint64_t Offset, uint64_t Size) {
  assert(Size && "Cannot free zero bytes");
  assert(Offset + Size <= size() && "Freed range out of reservation bounds");

  // Add the range to the free list, coalescing with its neighbors.
  uint64_t Start = Offset, End = Offset + Size;
  const auto Next = FreeRanges.upper_bound(Offset);
  if (Next != FreeRanges.begin()) {
    const auto Prev = std::prev(Next);
    assert(Prev->first + Prev->second <= Offset &&
           "Freed range overlaps existing free range");
    if (Prev->first + Prev->second == Offset) {
      Start = Prev->first;
      FreeRanges.erase(Prev);
    }
  }
  if (Next != FreeRanges.end() && Next->first == End) {
    End = Next->first + Next->second;
    FreeRanges.erase(Next);
  }
  FreeRanges[Start] = End - Start;
}

/// A logical range of the reservation that is striped between AllocGroups at
/// chunk granularity.
///
/// Each chunk of the slab is owned by at most one AllocGroup at a time: a
/// range may only be allocated for a group if every chunk that it touches is
/// either unowned, or already owned by that group. Chunks are returned to the
/// unowned state once all memory within them has been freed.
struct InProcessMemoryManager::Slab {
  /// Chunk owners are AllocGroup indexes (see allocGroupIndex), with NoOwner
  /// used for chunks that no group has claimed yet.
  static constexpr uint8_t NoOwner = ~static_cast<uint8_t>(0);

  Slab(char *Base, uint64_t Size, uint64_t ChunkSize)
      : Base(Base), Size(Size), ChunkSize(ChunkSize),
        ChunkOwners(divideCeil(Size, ChunkSize), NoOwner) {
    FreeRanges[0] = Size;
  }

  char *base() const { return Base; }
  uint64_t size() const { return Size; }

  /// Allocate Size bytes for the group with index Owner.
  ///
  /// If FromTop is true then the allocation is taken from the highest suitable
  /// address in the slab, otherwise from the lowest. Returns the offset of the
  /// allocation within the slab, or std::nullopt if the request can't be
  /// satisfied.
  std::optional<uint64_t> allocate(uint64_t Size, uint8_t Owner, bool FromTop);

  /// Return a previously allocated range to the slab.
  void free(uint64_t Offset, uint64_t Size);

  char *Base;
  uint64_t Size;
  uint64_t ChunkSize;
  SmallVector<uint8_t, 64> ChunkOwners;

  /// Free ranges of the slab, keyed by offset. Ranges are disjoint, and never
  /// adjacent to one another (adjacent ranges are always coalesced).
  ///
  /// FIXME: Allocation is first-fit over this map, which is O(#free-ranges) in
  /// the worst case. Revisit if this shows up in profiles.
  std::map<uint64_t, uint64_t> FreeRanges;

  uint64_t NumLiveRanges = 0;
};

std::optional<uint64_t> InProcessMemoryManager::Slab::allocate(uint64_t Size,
                                                               uint8_t Owner,
                                                               bool FromTop) {
  assert(Size && "Cannot allocate zero bytes");

  auto ChunkOf = [&](uint64_t Offset) { return Offset / ChunkSize; };
  auto EndOfChunk = [&](uint64_t Offset) {
    return (ChunkOf(Offset) + 1) * ChunkSize;
  };
  auto AvailableAt = [&](uint64_t Offset) {
    const uint8_t O = ChunkOwners[ChunkOf(Offset)];
    return O == NoOwner || O == Owner;
  };

  // Find a Size-byte position within the free range [Start, End) that only
  // covers chunks available to Owner, preferring the highest such position if
  // FromTop is set and the lowest otherwise.
  auto FindInRange = [&](uint64_t Start,
                         uint64_t End) -> std::optional<uint64_t> {
    std::optional<uint64_t> Result;
    uint64_t Cur = Start;
    while (Cur < End) {
      if (!AvailableAt(Cur)) {
        Cur = EndOfChunk(Cur);
        continue;
      }
      // Extend the run of available chunks as far as we can.
      uint64_t RunStart = Cur;
      while (Cur < End && AvailableAt(Cur))
        Cur = std::min(EndOfChunk(Cur), End);
      if (Cur - RunStart >= Size) {
        Result = FromTop ? Cur - Size : RunStart;
        if (!FromTop)
          break;
      }
    }
    return Result;
  };

  std::optional<uint64_t> Offset;
  if (FromTop) {
    for (auto I = FreeRanges.rbegin(), E = FreeRanges.rend(); I != E; ++I)
      if ((Offset = FindInRange(I->first, I->first + I->second)))
        break;
  } else {
    for (auto &KV : FreeRanges)
      if ((Offset = FindInRange(KV.first, KV.first + KV.second)))
        break;
  }

  if (!Offset)
    return std::nullopt;

  // Remove the allocated range from the free list, keeping any leading or
  // trailing remainder.
  const auto I = std::prev(FreeRanges.upper_bound(*Offset));
  const uint64_t RangeStart = I->first, RangeEnd = I->first + I->second;
  assert(RangeStart <= *Offset && *Offset + Size <= RangeEnd &&
         "Allocation not contained in the free range it came from");
  FreeRanges.erase(I);
  if (RangeStart < *Offset)
    FreeRanges[RangeStart] = *Offset - RangeStart;
  if (*Offset + Size < RangeEnd)
    FreeRanges[*Offset + Size] = RangeEnd - (*Offset + Size);

  // Claim the chunks that the allocation touches for Owner.
  for (uint64_t C = ChunkOf(*Offset), CE = ChunkOf(*Offset + Size - 1); C <= CE;
       ++C)
    ChunkOwners[C] = Owner;

  ++NumLiveRanges;
  return Offset;
}

void InProcessMemoryManager::Slab::free(uint64_t Offset, uint64_t Size) {
  assert(Size && "Cannot free zero bytes");
  assert(Offset + Size <= size() && "Freed range out of slab bounds");
  assert(NumLiveRanges && "Freeing range from empty slab");

  // Add the range to the free list, coalescing with its neighbors.
  uint64_t Start = Offset, End = Offset + Size;
  const auto Next = FreeRanges.upper_bound(Offset);
  if (Next != FreeRanges.begin()) {
    const auto Prev = std::prev(Next);
    assert(Prev->first + Prev->second <= Offset &&
           "Freed range overlaps existing free range");
    if (Prev->first + Prev->second == Offset) {
      Start = Prev->first;
      FreeRanges.erase(Prev);
    }
  }
  if (Next != FreeRanges.end() && Next->first == End) {
    End = Next->first + Next->second;
    FreeRanges.erase(Next);
  }
  FreeRanges[Start] = End - Start;

  // Release any chunks that are now entirely free back to the slab so that
  // they can be re-used by other AllocGroups.
  for (uint64_t C = Offset / ChunkSize, CE = (Offset + Size - 1) / ChunkSize;
       C <= CE; ++C) {
    const uint64_t ChunkStart = C * ChunkSize;
    const uint64_t ChunkEnd = std::min(ChunkStart + ChunkSize, size());
    if (Start <= ChunkStart && ChunkEnd <= End)
      ChunkOwners[C] = NoOwner;
  }

  --NumLiveRanges;
}

char *InProcessMemoryManager::SlabRange::base() const {
  return S->base() + Offset;
}

/// Returns a unique index in the range [0, AllocGroup::NumGroups) for AG.
static uint8_t allocGroupIndex(orc::AllocGroup AG) {
  static_assert(orc::AllocGroup::NumGroups <=
                    std::numeric_limits<uint8_t>::max(),
                "AllocGroup indexes must fit in a uint8_t");
  return static_cast<uint8_t>(static_cast<unsigned>(AG.getMemProt()) |
                              static_cast<unsigned>(AG.getMemLifetime()) << 3);
}

static constexpr auto ReadWrite = static_cast<sys::Memory::ProtectionFlags>(
    sys::Memory::MF_READ | sys::Memory::MF_WRITE);

class InProcessMemoryManager::IPInFlightAlloc : public InFlightAlloc {
public:
  IPInFlightAlloc(InProcessMemoryManager &MemMgr, LinkGraph &G,
                  SlabRangeList Segments)
      : MemMgr(MemMgr), G(&G), Segments(std::move(Segments)) {}

  ~IPInFlightAlloc() override {
    assert(!G && "InFlight alloc neither abandoned nor finalized");
  }

  void finalize(OnFinalizedFunction OnFinalized) override {

    // Apply memory protections to all segments.
    if (auto Err = applyProtections()) {
      OnFinalized(std::move(Err));
      return;
    }

    // Run finalization actions.
    auto DeallocActions = runFinalizeActions(G->allocActions());
    if (!DeallocActions) {
      OnFinalized(DeallocActions.takeError());
      return;
    }

    // Split the segments by lifetime: the finalize-lifetime ones are done with
    // now, the standard-lifetime ones live on in the finalized allocation.
    SlabRangeList StandardSegs, FinalizeSegs;
    for (auto &R : Segments) {
      if (R.AG.getMemLifetime() == orc::MemLifetime::Standard)
        StandardSegs.push_back(R);
      else
        FinalizeSegs.push_back(R);
    }
    Segments.clear();

    if (auto Err = MemMgr.freeSegments(FinalizeSegs,
                                       /*ResetProtections=*/true)) {
      OnFinalized(std::move(Err));
      return;
    }

#ifndef NDEBUG
    // Set 'G' to null to flag that we've been successfully finalized.
    // This allows us to assert at destruction time that a call has been made
    // to either finalize or abandon.
    G = nullptr;
#endif

    // Continue with finalized allocation.
    OnFinalized(MemMgr.createFinalizedAlloc(std::move(StandardSegs),
                                            std::move(*DeallocActions)));
  }

  void abandon(OnAbandonedFunction OnAbandoned) override {
    // Protections have not been applied yet, so the memory is still writable
    // and can be returned to its slab as-is.
    Error Err = MemMgr.freeSegments(Segments, /*ResetProtections=*/false);
    Segments.clear();

#ifndef NDEBUG
    // Set 'G' to null to flag that we've been successfully finalized.
    // This allows us to assert at destruction time that a call has been made
    // to either finalize or abandon.
    G = nullptr;
#endif

    OnAbandoned(std::move(Err));
  }

private:
  Error applyProtections() {
    for (auto &R : Segments) {
      auto Prot = toSysMemoryProtectionFlags(R.AG.getMemProt());

      sys::MemoryBlock MB(R.base(), R.Size);
      if (auto EC = sys::Memory::protectMappedMemory(MB, Prot))
        return errorCodeToError(EC);
      if (Prot & sys::Memory::MF_EXEC)
        sys::Memory::InvalidateInstructionCache(MB.base(), MB.allocatedSize());
    }
    return Error::success();
  }

  InProcessMemoryManager &MemMgr;
  LinkGraph *G;
  SlabRangeList Segments;
};

InProcessMemoryManager::SlabOptions
InProcessMemoryManager::SlabOptions::defaults(uint64_t PageSize) {
  SlabOptions SO;

  // On 64-bit hosts we can afford to reserve address space generously. On
  // 32-bit hosts we have to be much more frugal.
  if constexpr (sizeof(uintptr_t) >= 8) {
    SO.SlabSize = 16 * 1024 * 1024;
    SO.ChunkSize = 256 * 1024;
    // Comfortably under the 2^32 budget that some platforms need every piece
    // of code a session ever JITs to stay within, of a fixed base address
    // (see the class comment).
    SO.ReservationSize = 2ULL * 1024 * 1024 * 1024;
  } else {
    SO.SlabSize = 4 * 1024 * 1024;
    SO.ChunkSize = 64 * 1024;
    SO.ReservationSize = 64 * 1024 * 1024;
  }

  // Chunks must be a whole number of pages, slabs a whole number of chunks,
  // and the reservation a whole number of slabs.
  SO.ChunkSize = alignTo(std::max(SO.ChunkSize, PageSize), PageSize);
  SO.SlabSize = alignTo(std::max(SO.SlabSize, SO.ChunkSize), SO.ChunkSize);
  SO.ReservationSize =
      alignTo(std::max(SO.ReservationSize, SO.SlabSize), SO.SlabSize);

  return SO;
}

InProcessMemoryManager::InProcessMemoryManager(uint64_t PageSize,
                                               SlabOptions SO)
    : PageSize(PageSize), SlabOpts(SO) {
  assert(isPowerOf2_64(PageSize) && "PageSize must be a power of 2");
  assert(SlabOpts.ChunkSize && SlabOpts.ChunkSize % PageSize == 0 &&
         "ChunkSize must be a non-zero multiple of PageSize");
  assert(SlabOpts.SlabSize && SlabOpts.SlabSize % SlabOpts.ChunkSize == 0 &&
         "SlabSize must be a non-zero multiple of ChunkSize");
  assert(SlabOpts.ReservationSize &&
         SlabOpts.ReservationSize % SlabOpts.SlabSize == 0 &&
         "ReservationSize must be a non-zero multiple of SlabSize");
}

InProcessMemoryManager::~InProcessMemoryManager() {
  if (TheReservation)
    (void)sys::Memory::releaseMappedMemory(TheReservation->Mem);
}

Expected<std::unique_ptr<InProcessMemoryManager>>
InProcessMemoryManager::Create() {
  if (auto PageSize = sys::Process::getPageSize())
    return Create(SlabOptions::defaults(*PageSize));
  else
    return PageSize.takeError();
}

Expected<std::unique_ptr<InProcessMemoryManager>>
InProcessMemoryManager::Create(SlabOptions SO) {
  if (auto PageSize = sys::Process::getPageSize()) {
    // FIXME: Just check this once on startup.
    if (!isPowerOf2_64(*PageSize))
      return make_error<StringError>(
          "Could not create InProcessMemoryManager: Page size " +
              Twine(*PageSize) + " is not a power of 2",
          inconvertibleErrorCode());

    if (!SO.ChunkSize || SO.ChunkSize % *PageSize)
      return make_error<StringError>(
          "Could not create InProcessMemoryManager: Chunk size " +
              Twine(SO.ChunkSize) +
              " is not a non-zero multiple of page size " + Twine(*PageSize),
          inconvertibleErrorCode());

    if (!SO.SlabSize || SO.SlabSize % SO.ChunkSize)
      return make_error<StringError>(
          "Could not create InProcessMemoryManager: Slab size " +
              Twine(SO.SlabSize) +
              " is not a non-zero multiple of chunk size " +
              Twine(SO.ChunkSize),
          inconvertibleErrorCode());

    if (!SO.ReservationSize || SO.ReservationSize % SO.SlabSize)
      return make_error<StringError>(
          "Could not create InProcessMemoryManager: Reservation size " +
              Twine(SO.ReservationSize) +
              " is not a non-zero multiple of slab size " + Twine(SO.SlabSize),
          inconvertibleErrorCode());

    return std::make_unique<InProcessMemoryManager>(*PageSize, SO);
  } else
    return PageSize.takeError();
}

Expected<InProcessMemoryManager::Slab *>
InProcessMemoryManager::createSlab(uint64_t MinSize) {
  if (!TheReservation) {
    // Placement doesn't matter here -- unlike a hint passed to an individual
    // mapping, the address the OS puts this reservation at doesn't affect
    // whether later slabs stay within range of one another, since they're
    // all carved out of this one mapping (see the class comment).
    std::error_code EC;
    auto MB = sys::Memory::allocateMappedMemory(SlabOpts.ReservationSize,
                                                nullptr, ReadWrite, EC);
    if (EC)
      return errorCodeToError(EC);

    LLVM_DEBUG({
      dbgs() << "InProcessMemoryManager reserved "
             << formatv("[ {0:x16} -- {1:x16} ]",
                        orc::ExecutorAddr::fromPtr(MB.base()),
                        orc::ExecutorAddr::fromPtr(MB.base()) +
                            MB.allocatedSize())
             << "\n";
    });

    TheReservation = std::make_unique<Reservation>(MB);
  }

  const uint64_t ThisSlabSize =
      alignTo(std::max(MinSize, SlabOpts.SlabSize), SlabOpts.ChunkSize);

  auto Offset = TheReservation->allocate(ThisSlabSize);
  if (!Offset)
    return make_error<StringError>(
        "InProcessMemoryManager's " + Twine(SlabOpts.ReservationSize) +
            "-byte address space reservation is exhausted; consider "
            "increasing SlabOptions::ReservationSize",
        inconvertibleErrorCode());

  char *Base = TheReservation->base() + *Offset;
  LLVM_DEBUG({
    dbgs() << "InProcessMemoryManager created slab "
           << formatv("[ {0:x16} -- {1:x16} ]",
                      orc::ExecutorAddr::fromPtr(Base),
                      orc::ExecutorAddr::fromPtr(Base) + ThisSlabSize)
           << "\n";
  });

  Slabs.push_back(
      std::make_unique<Slab>(Base, ThisSlabSize, SlabOpts.ChunkSize));
  return Slabs.back().get();
}

Expected<InProcessMemoryManager::SlabRangeList>
InProcessMemoryManager::allocateSegments(BasicLayout &BL) {
  // Work out the page-aligned size of each segment, along with the amount of
  // slab space needed to hold all of them in the worst case, i.e. when no
  // chunk can be shared with an existing allocation. BasicLayout guarantees at
  // most one segment per AllocGroup, so this is just the sum of the segment
  // sizes rounded up to chunk size.
  SmallVector<std::pair<orc::AllocGroup, uint64_t>, 8> SegSizes;
  uint64_t WorstCaseSize = 0;
  for (auto &KV : BL.segments()) {
    auto &Seg = KV.second;
    // Zero-sized segments would still need somewhere to point, so round them
    // up to a page like everything else.
    uint64_t SegSize = std::max(
        alignTo(Seg.ContentSize + Seg.ZeroFillSize, PageSize), PageSize);
    SegSizes.push_back({KV.first, SegSize});
    WorstCaseSize += alignTo(SegSize, SlabOpts.ChunkSize);
  }

  SlabRangeList Ranges;

  {
    std::lock_guard<std::mutex> Lock(SlabsMutex);

    // Try to allocate every segment from S. All segments for a graph have to
    // come from the same slab so that intra-graph references stay in range, so
    // this either succeeds completely or leaves S untouched.
    auto TryAllocFrom = [&](Slab &S) {
      for (auto &[AG, SegSize] : SegSizes) {
        // Grow executable memory up from the bottom of the slab and everything
        // else down from the top, so that code stays contiguous.
        bool FromTop =
            (AG.getMemProt() & orc::MemProt::Exec) == orc::MemProt::None;
        auto Offset = S.allocate(SegSize, allocGroupIndex(AG), FromTop);
        if (!Offset) {
          for (auto &R : Ranges)
            S.free(R.Offset, R.Size);
          Ranges.clear();
          return false;
        }
        Ranges.push_back({&S, *Offset, SegSize, AG});
      }
      return true;
    };

    bool Allocated = false;
    for (auto &S : Slabs)
      if ((Allocated = TryAllocFrom(*S)))
        break;

    if (!Allocated) {
      auto S = createSlab(WorstCaseSize);
      if (!S)
        return S.takeError();
      // A fresh slab is at least WorstCaseSize bytes and has no chunks claimed
      // yet, so this can't fail.
      Allocated = TryAllocFrom(**S);
      assert(Allocated && "Failed to allocate segments from fresh slab");
      (void)Allocated;
    }
  }

  // Zero the memory: slab memory may have been used by an earlier allocation,
  // and both zero-fill blocks and inter-block padding are required to be zero.
  for (auto &R : Ranges)
    memset(R.base(), 0, R.Size);

  LLVM_DEBUG({
    dbgs() << "InProcessMemoryManager allocated:\n";
    for (auto &R : Ranges)
      dbgs() << formatv("  [ {0:x16} -- {1:x16} ]",
                        orc::ExecutorAddr::fromPtr(R.base()),
                        orc::ExecutorAddr::fromPtr(R.base()) + R.Size)
             << " to " << R.AG << " segment\n";
  });

  return Ranges;
}

Error InProcessMemoryManager::freeSegments(MutableArrayRef<SlabRange> Ranges,
                                           bool ResetProtections) {
  Error Err = Error::success();

  SmallVector<SlabRange *, 4> ToFree;
  for (auto &R : Ranges) {
    // Restore read/write permissions so that the memory can be written to by
    // whichever allocation picks it up next. If that fails we can't safely
    // hand this range out again, so leave it allocated (which also stops its
    // slab from being released).
    if (ResetProtections &&
        R.AG.getMemProt() != (orc::MemProt::Read | orc::MemProt::Write)) {
      sys::MemoryBlock MB(R.base(), R.Size);
      if (const auto EC = sys::Memory::protectMappedMemory(MB, ReadWrite)) {
        Err = joinErrors(std::move(Err), errorCodeToError(EC));
        continue;
      }
    }
    ToFree.push_back(&R);
  }

  std::lock_guard<std::mutex> Lock(SlabsMutex);

  for (const auto *R : ToFree)
    R->S->free(R->Offset, R->Size);

  // Return fully-freed slabs to the reservation so their space can be reused
  // by future slabs. This never touches the OS -- the reservation itself is
  // held for the lifetime of this memory manager (see the class comment) --
  // so unlike the old per-slab OS mappings this replaced, it can't fail.
  for (auto I = Slabs.begin(); I != Slabs.end();) {
    if ((*I)->NumLiveRanges == 0) {
      TheReservation->free((*I)->base() - TheReservation->base(), (*I)->size());
      I = Slabs.erase(I);
    } else {
      ++I;
    }
  }

  return Err;
}

void InProcessMemoryManager::allocate(const JITLinkDylib *JD, LinkGraph &G,
                                      OnAllocatedFunction OnAllocated) {
  BasicLayout BL(G);

  /// Scan the request and calculate the group and total sizes.
  /// Check that segment size is no larger than a page.
  auto SegsSizes = BL.getContiguousPageBasedLayoutSizes(PageSize);
  if (!SegsSizes) {
    OnAllocated(SegsSizes.takeError());
    return;
  }

  /// Check that the total size requested (including zero fill) is not larger
  /// than a size_t.
  if (SegsSizes->total() > std::numeric_limits<size_t>::max()) {
    OnAllocated(make_error<JITLinkError>(
        "Total requested size " + formatv("{0:x}", SegsSizes->total()) +
        " for graph " + G.getName() + " exceeds address space"));
    return;
  }

  auto Segments = allocateSegments(BL);
  if (!Segments) {
    OnAllocated(Segments.takeError());
    return;
  }

  // Assign addresses and working memory to the segments. allocateSegments
  // produces one range per segment, in segment iteration order.
  unsigned Idx = 0;
  for (auto &KV : BL.segments()) {
    auto &R = (*Segments)[Idx++];
    assert(KV.first == R.AG && "Segment / range mismatch");
    KV.second.WorkingMem = R.base();
    KV.second.Addr = orc::ExecutorAddr::fromPtr(R.base());
  }

  if (auto Err = BL.apply()) {
    // Roll the allocation back: protections haven't been applied yet, so the
    // segments can go straight back to their slab.
    Err = joinErrors(std::move(Err),
                     freeSegments(*Segments, /*ResetProtections=*/false));
    OnAllocated(std::move(Err));
    return;
  }

  OnAllocated(
      std::make_unique<IPInFlightAlloc>(*this, G, std::move(*Segments)));
}

void InProcessMemoryManager::deallocate(std::vector<FinalizedAlloc> Allocs,
                                        OnDeallocatedFunction OnDeallocated) {
  std::vector<SlabRangeList> StandardSegmentsList;
  std::vector<std::vector<orc::shared::WrapperFunctionCall>> DeallocActionsList;

  {
    std::lock_guard<std::mutex> Lock(FinalizedAllocsMutex);
    for (auto &Alloc : Allocs) {
      auto *FA = Alloc.release().toPtr<FinalizedAllocInfo *>();
      StandardSegmentsList.push_back(std::move(FA->StandardSegments));
      DeallocActionsList.push_back(std::move(FA->DeallocActions));
      FA->~FinalizedAllocInfo();
      FinalizedAllocInfos.Deallocate(FA);
    }
  }

  Error DeallocErr = Error::success();

  while (!DeallocActionsList.empty()) {
    auto &DeallocActions = DeallocActionsList.back();
    auto &StandardSegments = StandardSegmentsList.back();

    /// Run any deallocate calls.
    while (!DeallocActions.empty()) {
      if (auto Err = DeallocActions.back().runWithSPSRetErrorMerged())
        DeallocErr = joinErrors(std::move(DeallocErr), std::move(Err));
      DeallocActions.pop_back();
    }

    /// Return the standard segments to their slab.
    if (auto Err = freeSegments(StandardSegments, /*ResetProtections=*/true))
      DeallocErr = joinErrors(std::move(DeallocErr), std::move(Err));

    DeallocActionsList.pop_back();
    StandardSegmentsList.pop_back();
  }

  OnDeallocated(std::move(DeallocErr));
}

JITLinkMemoryManager::FinalizedAlloc
InProcessMemoryManager::createFinalizedAlloc(
    SlabRangeList StandardSegments,
    std::vector<orc::shared::WrapperFunctionCall> DeallocActions) {
  std::lock_guard<std::mutex> Lock(FinalizedAllocsMutex);
  auto *FA = FinalizedAllocInfos.Allocate<FinalizedAllocInfo>();
  new (FA) FinalizedAllocInfo(
      {std::move(StandardSegments), std::move(DeallocActions)});
  return FinalizedAlloc(orc::ExecutorAddr::fromPtr(FA));
}

} // end namespace jitlink
} // end namespace llvm
