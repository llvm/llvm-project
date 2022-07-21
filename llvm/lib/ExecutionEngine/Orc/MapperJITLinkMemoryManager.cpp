//=== MapperJITLinkMemoryManager.cpp - Memory management with MemoryMapper ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Process.h"

#include <limits>

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

    Parent.Mapper->initialize(AI, [&](Expected<ExecutorAddr> Result) {
      if (!Result) {
        OnFinalize(Result.takeError());
        return;
      }

      OnFinalize(FinalizedAlloc(*Result));
    });
  }

  void abandon(OnAbandonedFunction OnFinalize) override {
    Parent.Mapper->release({AllocAddr}, std::move(OnFinalize));
  }

private:
  MapperJITLinkMemoryManager &Parent;
  LinkGraph &G;
  ExecutorAddr AllocAddr;
  std::vector<MemoryMapper::AllocInfo::SegInfo> Segs;
};

MapperJITLinkMemoryManager::MapperJITLinkMemoryManager(
    std::unique_ptr<MemoryMapper> Mapper)
    : Mapper(std::move(Mapper)) {}

void MapperJITLinkMemoryManager::allocate(const JITLinkDylib *JD, LinkGraph &G,
                                          OnAllocatedFunction OnAllocated) {
  BasicLayout BL(G);

  // find required address space
  auto SegsSizes = BL.getContiguousPageBasedLayoutSizes(Mapper->getPageSize());
  if (!SegsSizes) {
    OnAllocated(SegsSizes.takeError());
    return;
  }

  // Check if total size fits in address space
  if (SegsSizes->total() > std::numeric_limits<size_t>::max()) {
    OnAllocated(make_error<JITLinkError>(
        formatv("Total requested size {:x} for graph {} exceeds address space",
                SegsSizes->total(), G.getName())));
    return;
  }

  Mapper->reserve(
      SegsSizes->total(),
      [this, &G, BL = std::move(BL), OnAllocated = std::move(OnAllocated)](
          Expected<ExecutorAddrRange> Result) mutable {
        if (!Result) {
          return OnAllocated(Result.takeError());
        }

        auto NextSegAddr = Result->Start;

        std::vector<MemoryMapper::AllocInfo::SegInfo> SegInfos;

        for (auto &KV : BL.segments()) {
          auto &AG = KV.first;
          auto &Seg = KV.second;

          auto TotalSize = Seg.ContentSize + Seg.ZeroFillSize;

          Seg.Addr = NextSegAddr;
          Seg.WorkingMem = Mapper->prepare(NextSegAddr, TotalSize);

          NextSegAddr += alignTo(TotalSize, Mapper->getPageSize());

          MemoryMapper::AllocInfo::SegInfo SI;
          SI.Offset = Seg.Addr - Result->Start;
          SI.ContentSize = Seg.ContentSize;
          SI.ZeroFillSize = Seg.ZeroFillSize;
          SI.Prot = (toSysMemoryProtectionFlags(AG.getMemProt()));
          SI.WorkingMem = Seg.WorkingMem;

          SegInfos.push_back(SI);
        }

        if (auto Err = BL.apply()) {
          OnAllocated(std::move(Err));
          return;
        }

        OnAllocated(std::make_unique<InFlightAlloc>(*this, G, Result->Start,
                                                    std::move(SegInfos)));
      });
}

void MapperJITLinkMemoryManager::deallocate(
    std::vector<FinalizedAlloc> Allocs, OnDeallocatedFunction OnDeallocated) {
  std::vector<ExecutorAddr> Bases;
  Bases.reserve(Allocs.size());
  for (auto &FA : Allocs) {
    Bases.push_back(FA.getAddress());
    FA.release();
  }
  Mapper->release(Bases, std::move(OnDeallocated));
}

} // end namespace orc
} // end namespace llvm
