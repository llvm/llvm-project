//===- SimpleExecuorMemoryManagare.cpp - Simple executor-side memory mgmt -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleExecutorMemoryManager.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {
namespace rt_bootstrap {

SimpleExecutorMemoryManager::~SimpleExecutorMemoryManager() {
  assert(Slabs.empty() && "shutdown not called?");
}

Expected<ExecutorAddr> SimpleExecutorMemoryManager::reserve(uint64_t Size) {
  std::error_code EC;
  auto MB = sys::Memory::allocateMappedMemory(
      Size, nullptr, sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC);
  if (EC)
    return errorCodeToError(EC);
  std::lock_guard<std::mutex> Lock(M);
  assert(!Slabs.count(MB.base()) && "Duplicate allocation addr");
  Slabs[MB.base()].Size = Size;
  return ExecutorAddr::fromPtr(MB.base());
}

Expected<ExecutorAddr>
SimpleExecutorMemoryManager::initialize(tpctypes::FinalizeRequest &FR) {
  std::vector<shared::WrapperFunctionCall> DeallocationActions;

  if (FR.Segments.empty()) {
    if (FR.Actions.empty())
      return make_error<StringError>("Finalization request is empty",
                                     inconvertibleErrorCode());
    else
      return make_error<StringError>("Finalization actions attached to empty "
                                     "finalization request",
                                     inconvertibleErrorCode());
  }

  ExecutorAddrRange RR(FR.Segments.front().Addr, FR.Segments.front().Addr);

  std::vector<sys::MemoryBlock> MBsToReset;
  auto ResetMBs = make_scope_exit([&]() {
    for (auto &MB : MBsToReset)
      sys::Memory::protectMappedMemory(MB, sys::Memory::MF_READ |
                                               sys::Memory::MF_WRITE);
    sys::Memory::InvalidateInstructionCache(RR.Start.toPtr<void *>(),
                                            RR.size());
  });

  // Copy content and apply permissions.
  for (auto &Seg : FR.Segments) {
    RR.Start = std::min(RR.Start, Seg.Addr);
    RR.End = std::max(RR.End, Seg.Addr + Seg.Size);

    // Check segment ranges.
    if (LLVM_UNLIKELY(Seg.Size < Seg.Content.size()))
      return make_error<StringError>(
          formatv("Segment {0:x} content size ({1:x} bytes) "
                  "exceeds segment size ({2:x} bytes)",
                  Seg.Addr.getValue(), Seg.Content.size(), Seg.Size),
          inconvertibleErrorCode());
    ExecutorAddr SegEnd = Seg.Addr + ExecutorAddrDiff(Seg.Size);
    if (LLVM_UNLIKELY(Seg.Addr < RR.Start || SegEnd > RR.End))
      return make_error<StringError>(
          formatv("Segment {0:x} -- {1:x} crosses boundary of "
                  "allocation {2:x} -- {3:x}",
                  Seg.Addr, SegEnd, RR.Start, RR.End),
          inconvertibleErrorCode());

    char *Mem = Seg.Addr.toPtr<char *>();
    if (!Seg.Content.empty())
      memcpy(Mem, Seg.Content.data(), Seg.Content.size());
    memset(Mem + Seg.Content.size(), 0, Seg.Size - Seg.Content.size());
    assert(Seg.Size <= std::numeric_limits<size_t>::max());

    sys::MemoryBlock MB(Mem, Seg.Size);
    if (auto EC = sys::Memory::protectMappedMemory(
            MB, toSysMemoryProtectionFlags(Seg.RAG.Prot)))
      return errorCodeToError(EC);

    MBsToReset.push_back(MB);

    if ((Seg.RAG.Prot & MemProt::Exec) == MemProt::Exec)
      sys::Memory::InvalidateInstructionCache(Mem, Seg.Size);
  }

  auto DeallocActions = runFinalizeActions(FR.Actions);
  if (!DeallocActions)
    return DeallocActions.takeError();

  {
    std::lock_guard<std::mutex> Lock(M);
    auto Region = createRegionInfo(RR, "In initialize");
    if (!Region)
      return Region.takeError();
    Region->DeallocActions = std::move(*DeallocActions);
  }

  // Successful initialization.
  ResetMBs.release();

  return RR.Start;
}

Error SimpleExecutorMemoryManager::deinitialize(
    const std::vector<ExecutorAddr> &InitKeys) {
  Error Err = Error::success();

  for (auto &KeyAddr : llvm::reverse(InitKeys)) {
    std::vector<shared::WrapperFunctionCall> DeallocActions;
    {
      std::scoped_lock<std::mutex> Lock(M);
      auto Slab = getSlabInfo(KeyAddr, "In deinitialize");
      if (!Slab) {
        Err = joinErrors(std::move(Err), Slab.takeError());
        continue;
      }

      auto RI = getRegionInfo(*Slab, KeyAddr, "In deinitialize");
      if (!RI) {
        Err = joinErrors(std::move(Err), RI.takeError());
        continue;
      }

      DeallocActions = std::move(RI->DeallocActions);
    }

    Err = joinErrors(std::move(Err),
                     runDeallocActions(std::move(DeallocActions)));
  }

  return Err;
}

Error SimpleExecutorMemoryManager::release(
    const std::vector<ExecutorAddr> &Bases) {
  Error Err = Error::success();

  // TODO: Prohibit new initializations within the slabs being removed?
  for (auto &Base : llvm::reverse(Bases)) {
    std::vector<shared::WrapperFunctionCall> DeallocActions;
    sys::MemoryBlock MB;

    {
      std::scoped_lock<std::mutex> Lock(M);

      auto SlabI = Slabs.find(Base.toPtr<void *>());
      if (SlabI == Slabs.end()) {
        Err = joinErrors(
            std::move(Err),
            make_error<StringError>("In release, " + formatv("{0:x}", Base) +
                                        " is not part of any reserved "
                                        "address range",
                                    inconvertibleErrorCode()));
        continue;
      }

      auto &Slab = SlabI->second;

      for (auto &[Addr, Region] : Slab.Regions)
        llvm::copy(Region.DeallocActions, back_inserter(DeallocActions));

      MB = {Base.toPtr<void *>(), Slab.Size};

      Slabs.erase(SlabI);
    }

    Err = joinErrors(std::move(Err), runDeallocActions(DeallocActions));
    if (auto EC = sys::Memory::releaseMappedMemory(MB))
      Err = joinErrors(std::move(Err), errorCodeToError(EC));
  }

  return Err;
}

Error SimpleExecutorMemoryManager::shutdown() {

  // TODO: Prevent new allocations during shutdown.
  std::vector<ExecutorAddr> Bases;
  {
    std::scoped_lock<std::mutex> Lock(M);
    for (auto &[Base, Slab] : Slabs)
      Bases.push_back(ExecutorAddr::fromPtr(Base));
  }

  return release(Bases);
}

void SimpleExecutorMemoryManager::addBootstrapSymbols(
    StringMap<ExecutorAddr> &M) {
  M[rt::SimpleExecutorMemoryManagerInstanceName] = ExecutorAddr::fromPtr(this);
  M[rt::SimpleExecutorMemoryManagerReserveWrapperName] =
      ExecutorAddr::fromPtr(&reserveWrapper);
  M[rt::SimpleExecutorMemoryManagerInitializeWrapperName] =
      ExecutorAddr::fromPtr(&initializeWrapper);
  M[rt::SimpleExecutorMemoryManagerDeinitializeWrapperName] =
      ExecutorAddr::fromPtr(&deinitializeWrapper);
  M[rt::SimpleExecutorMemoryManagerReleaseWrapperName] =
      ExecutorAddr::fromPtr(&releaseWrapper);
}

Expected<SimpleExecutorMemoryManager::SlabInfo &>
SimpleExecutorMemoryManager::getSlabInfo(ExecutorAddr A, StringRef Context) {
  auto MakeBadSlabError = [&]() {
    return make_error<StringError>(
        Context + ", address " + formatv("{0:x}", A) +
            " is not part of any reserved address range",
        inconvertibleErrorCode());
  };

  auto I = Slabs.upper_bound(A.toPtr<void *>());
  if (I == Slabs.begin())
    return MakeBadSlabError();
  --I;
  if (!ExecutorAddrRange(ExecutorAddr::fromPtr(I->first), I->second.Size)
           .contains(A))
    return MakeBadSlabError();

  return I->second;
}

Expected<SimpleExecutorMemoryManager::SlabInfo &>
SimpleExecutorMemoryManager::getSlabInfo(ExecutorAddrRange R,
                                         StringRef Context) {
  auto MakeBadSlabError = [&]() {
    return make_error<StringError>(
        Context + ", range " + formatv("{0:x}", R) +
            " is not part of any reserved address range",
        inconvertibleErrorCode());
  };

  auto I = Slabs.upper_bound(R.Start.toPtr<void *>());
  if (I == Slabs.begin())
    return MakeBadSlabError();
  --I;
  if (!ExecutorAddrRange(ExecutorAddr::fromPtr(I->first), I->second.Size)
           .contains(R))
    return MakeBadSlabError();

  return I->second;
}

Expected<SimpleExecutorMemoryManager::RegionInfo &>
SimpleExecutorMemoryManager::createRegionInfo(ExecutorAddrRange R,
                                              StringRef Context) {

  auto Slab = getSlabInfo(R, Context);
  if (!Slab)
    return Slab.takeError();

  auto MakeBadRegionError = [&](ExecutorAddrRange Other, bool Prev) {
    return make_error<StringError>(Context + ", region " + formatv("{0:x}", R) +
                                       " overlaps " +
                                       (Prev ? "previous" : "following") +
                                       " region " + formatv("{0:x}", Other),
                                   inconvertibleErrorCode());
  };

  auto I = Slab->Regions.upper_bound(R.Start);
  if (I != Slab->Regions.begin()) {
    auto J = std::prev(I);
    ExecutorAddrRange PrevRange(J->first, J->second.Size);
    if (PrevRange.overlaps(R))
      return MakeBadRegionError(PrevRange, true);
  }
  if (I != Slab->Regions.end()) {
    ExecutorAddrRange NextRange(I->first, I->second.Size);
    if (NextRange.overlaps(R))
      return MakeBadRegionError(NextRange, false);
  }

  auto &RInfo = Slab->Regions[R.Start];
  RInfo.Size = R.size();
  return RInfo;
}

Expected<SimpleExecutorMemoryManager::RegionInfo &>
SimpleExecutorMemoryManager::getRegionInfo(SlabInfo &Slab, ExecutorAddr A,
                                           StringRef Context) {
  auto I = Slab.Regions.find(A);
  if (I == Slab.Regions.end())
    return make_error<StringError>(
        Context + ", address " + formatv("{0:x}", A) +
            " does not correspond to the start of any initialized region",
        inconvertibleErrorCode());

  return I->second;
}

Expected<SimpleExecutorMemoryManager::RegionInfo &>
SimpleExecutorMemoryManager::getRegionInfo(ExecutorAddr A, StringRef Context) {
  auto Slab = getSlabInfo(A, Context);
  if (!Slab)
    return Slab.takeError();

  return getRegionInfo(*Slab, A, Context);
}

llvm::orc::shared::CWrapperFunctionResult
SimpleExecutorMemoryManager::reserveWrapper(const char *ArgData,
                                            size_t ArgSize) {
  return shared::WrapperFunction<rt::SPSSimpleRemoteMemoryMapReserveSignature>::
      handle(ArgData, ArgSize,
             shared::makeMethodWrapperHandler(
                 &SimpleExecutorMemoryManager::reserve))
          .release();
}

llvm::orc::shared::CWrapperFunctionResult
SimpleExecutorMemoryManager::initializeWrapper(const char *ArgData,
                                               size_t ArgSize) {
  return shared::
      WrapperFunction<rt::SPSSimpleRemoteMemoryMapInitializeSignature>::handle(
             ArgData, ArgSize,
             shared::makeMethodWrapperHandler(
                 &SimpleExecutorMemoryManager::initialize))
          .release();
}

llvm::orc::shared::CWrapperFunctionResult
SimpleExecutorMemoryManager::deinitializeWrapper(const char *ArgData,
                                                 size_t ArgSize) {
  return shared::WrapperFunction<
             rt::SPSSimpleRemoteMemoryMapDeinitializeSignature>::
      handle(ArgData, ArgSize,
             shared::makeMethodWrapperHandler(
                 &SimpleExecutorMemoryManager::deinitialize))
          .release();
}

llvm::orc::shared::CWrapperFunctionResult
SimpleExecutorMemoryManager::releaseWrapper(const char *ArgData,
                                            size_t ArgSize) {
  return shared::WrapperFunction<rt::SPSSimpleRemoteMemoryMapReleaseSignature>::
      handle(ArgData, ArgSize,
             shared::makeMethodWrapperHandler(
                 &SimpleExecutorMemoryManager::release))
          .release();
}

} // namespace rt_bootstrap
} // end namespace orc
} // end namespace llvm
