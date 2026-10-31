//=== MapperJITLinkMemoryManager.cpp - Memory management with MemoryMapper ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Process.h"

using namespace llvm::jitlink;

namespace llvm {
namespace orc {

class MapperJITLinkMemoryManager::InFlightAlloc
    : public JITLinkMemoryManager::InFlightAlloc {
public:
  InFlightAlloc(MapperJITLinkMemoryManager &Parent, LinkGraph &G,
                ExecutorAddr AllocAddr,
                std::vector<MemoryMapper::AllocInfo::SegInfo> Segs)
      : Parent(Parent), G(G), AllocAddr(AllocAddr), Segs(std::move(Segs)) {}

  void finalize(OnFinalizedFunction OnFinalize) override {
    MemoryMapper::AllocInfo AI;
    AI.MappingBase = AllocAddr;

    std::swap(AI.Segments, Segs);
    std::swap(AI.Actions, G.allocActions());

    Parent.Mapper->initialize(AI, [OnFinalize = std::move(OnFinalize)](
                                      Expected<ExecutorAddr> Result) mutable {
      if (!Result) {
        OnFinalize(Result.takeError());
        return;
      }

      OnFinalize(FinalizedAlloc(*Result));
    });
  }

  void abandon(OnAbandonedFunction OnFinalize) override {
    Parent.Mapper->deinitialize({AllocAddr}, std::move(OnFinalize));
  }

private:
  MapperJITLinkMemoryManager &Parent;
  LinkGraph &G;
  ExecutorAddr AllocAddr;
  std::vector<MemoryMapper::AllocInfo::SegInfo> Segs;
};

MapperJITLinkMemoryManager::MapperJITLinkMemoryManager(
    size_t ReservationGranularity, std::unique_ptr<MemoryMapper> Mapper,
    bool ColocatePerJITDylib)
    : ReservationUnits(ReservationGranularity),
      ColocatePerJITDylib(ColocatePerJITDylib), Mapper(std::move(Mapper)) {}

// Lazily creates and stores Key's pool on first access. Pools are
// heap-allocated behind unique_ptr so the returned reference survives map
// rehashing, and AMAllocator (declared before the pools) outlives them. Also
// registers for a destruction notification the first time a JITDylib is
// seen, so its pool can be freed in notifyDestroying().
MapperJITLinkMemoryManager::AvailableMemoryMap &
MapperJITLinkMemoryManager::getAvailableMemory(const JITLinkDylib *Key) {
  auto [It, Inserted] = Pools.try_emplace(Key);
  if (Inserted && Key)
    const_cast<JITLinkDylib *>(Key)->notifyOnDestruction(*this);
  auto &Pool = It->second.AvailPool;
  if (!Pool)
    Pool = std::make_unique<AvailableMemoryMap>(AMAllocator);
  return *Pool;
}

void MapperJITLinkMemoryManager::notifyDestroying(JITLinkDylib &JD) {
  std::lock_guard<std::mutex> Lock(Mutex);
  Pools.erase(&JD);
}

void MapperJITLinkMemoryManager::denySlabGrowth() {
  std::lock_guard<std::mutex> Lock(Mutex);
  OnSlabGrow = [](const JITLinkDylib *JD, size_t RequestedSize) -> Error {
    return createStringError(
        "JITDylib '%s' slab exhausted (single-slab mode); enable slab "
        "growth to allow further reservations (%zu bytes requested)",
        JD ? JD->getName().c_str() : "<unnamed>", RequestedSize);
  };
}

void MapperJITLinkMemoryManager::allowSlabGrowth() {
  std::lock_guard<std::mutex> Lock(Mutex);
  OnSlabGrow = [](const JITLinkDylib *, size_t) -> Error {
    return Error::success();
  };
}

void MapperJITLinkMemoryManager::setSlabPolicy(SlabGrowthPolicy Policy) {
  std::lock_guard<std::mutex> Lock(Mutex);
  OnSlabGrow = std::move(Policy);
}

Expected<std::unique_ptr<MapperJITLinkMemoryManager>>
createColocatingInProcessMemoryManager(
    std::optional<size_t> ReservationGranularity) {
  auto Mapper = InProcessMemoryMapper::Create();
  if (!Mapper)
    return Mapper.takeError();
  return std::make_unique<MapperJITLinkMemoryManager>(
      ReservationGranularity.value_or(
          MapperJITLinkMemoryManager::defaultSlabSize()),
      std::move(*Mapper),
      /*ColocatePerJITDylib=*/true);
}

void MapperJITLinkMemoryManager::allocate(const JITLinkDylib *JD, LinkGraph &G,
                                          OnAllocatedFunction OnAllocated) {
  BasicLayout BL(G);

  // find required address space
  auto SegsSizes = BL.getContiguousPageBasedLayoutSizes(Mapper->getPageSize());
  if (!SegsSizes) {
    OnAllocated(SegsSizes.takeError());
    return;
  }

  auto TotalSize = SegsSizes->total();

  // Which pool to draw from: the JITDylib's own pool when colocating
  // per-JITDylib, otherwise a single shared (nullptr) pool.
  const JITLinkDylib *PoolKey = ColocatePerJITDylib ? JD : nullptr;

  auto CompleteAllocation = [this, PoolKey, &G, BL = std::move(BL),
                             OnAllocated = std::move(OnAllocated)](
                                Expected<ExecutorAddrRange> Result) mutable {
    if (!Result) {
      Mutex.unlock();
      return OnAllocated(Result.takeError());
    }

    auto NextSegAddr = Result->Start;

    std::vector<MemoryMapper::AllocInfo::SegInfo> SegInfos;

    for (auto &KV : BL.segments()) {
      auto &AG = KV.first;
      auto &Seg = KV.second;

      auto TotalSize = Seg.ContentSize + Seg.ZeroFillSize;

      Seg.Addr = NextSegAddr;
      Seg.WorkingMem = Mapper->prepare(G, NextSegAddr, TotalSize);

      NextSegAddr += alignTo(TotalSize, Mapper->getPageSize());

      MemoryMapper::AllocInfo::SegInfo SI;
      SI.Offset = Seg.Addr - Result->Start;
      SI.ContentSize = Seg.ContentSize;
      SI.ZeroFillSize = Seg.ZeroFillSize;
      SI.AG = AG;
      SI.WorkingMem = Seg.WorkingMem;

      SegInfos.push_back(SI);
    }

    UsedMemory.insert({Result->Start, NextSegAddr - Result->Start});
    AllocPoolKey[Result->Start] = PoolKey;

    if (NextSegAddr < Result->End) {
      // Save the remaining memory for reuse by later allocations from the same
      // pool.
      getAvailableMemory(PoolKey).insert(NextSegAddr, Result->End - 1, true);
    }
    Mutex.unlock();

    if (auto Err = BL.apply()) {
      OnAllocated(std::move(Err));
      return;
    }

    OnAllocated(std::make_unique<InFlightAlloc>(*this, G, Result->Start,
                                                std::move(SegInfos)));
  };

  Mutex.lock();

  // find an already reserved range in this pool that is large enough
  ExecutorAddrRange SelectedRange{};

  AvailableMemoryMap &Avail = getAvailableMemory(PoolKey);
  for (AvailableMemoryMap::iterator It = Avail.begin(); It != Avail.end();
       It++) {
    if (It.stop() - It.start() + 1 >= TotalSize) {
      SelectedRange = ExecutorAddrRange(It.start(), It.stop() + 1);
      It.erase();
      break;
    }
  }

  if (SelectedRange.empty()) { // no already reserved range was found
    // A reservation is needed. If this pool already owns one, reserving
    // another (possibly out-of-range) slab may place its objects out of range,
    // so consult the growth policy first.
    bool IsFirstReservation = !Pools[PoolKey].Reserved;
    Pools[PoolKey].Reserved = true;
    auto TotalAllocation = alignTo(TotalSize, ReservationUnits);
    if (!IsFirstReservation) {
      if (auto Err = OnSlabGrow(JD, TotalAllocation))
        return CompleteAllocation(std::move(Err));
    }
    Mapper->reserve(TotalAllocation,
                    [this, PoolKey, IsFirstReservation,
                     CompleteAllocation = std::move(CompleteAllocation)](
                        Expected<ExecutorAddrRange> Result) mutable {
                      // If this was the pool's first reservation attempt and it
                      // failed, undo the Reserved flag so a later attempt is
                      // treated as the first one again, rather than
                      // incorrectly triggering the growth policy for a pool
                      // that never got a slab.
                      if (!Result && IsFirstReservation)
                        Pools[PoolKey].Reserved = false;
                      CompleteAllocation(std::move(Result));
                    });
  } else {
    CompleteAllocation(SelectedRange);
  }
}

void MapperJITLinkMemoryManager::deallocate(
    std::vector<FinalizedAlloc> Allocs, OnDeallocatedFunction OnDeallocated) {
  std::vector<ExecutorAddr> Bases;
  Bases.reserve(Allocs.size());
  for (auto &FA : Allocs) {
    ExecutorAddr Addr = FA.getAddress();
    Bases.push_back(Addr);
  }

  Mapper->deinitialize(Bases, [this, Allocs = std::move(Allocs),
                               OnDeallocated = std::move(OnDeallocated)](
                                  llvm::Error Err) mutable {
    // TODO: How should we treat memory that we fail to deinitialize?
    // We're currently bailing out and treating it as "burned" -- should we
    // require that a failure to deinitialize still reset the memory so that
    // we can reclaim it?
    if (Err) {
      for (auto &FA : Allocs)
        FA.release();
      OnDeallocated(std::move(Err));
      return;
    }

    {
      std::lock_guard<std::mutex> Lock(Mutex);

      for (auto &FA : Allocs) {
        ExecutorAddr Addr = FA.getAddress();
        ExecutorAddrDiff Size = UsedMemory[Addr];

        UsedMemory.erase(Addr);

        // Return the range to the pool it was allocated from.
        const JITLinkDylib *PoolKey = AllocPoolKey.lookup(Addr);
        AllocPoolKey.erase(Addr);
        getAvailableMemory(PoolKey).insert(Addr, Addr + Size - 1, true);

        FA.release();
      }
    }

    OnDeallocated(Error::success());
  });
}

} // end namespace orc
} // end namespace llvm
