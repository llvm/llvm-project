//===- SimpleNativeMemoryMap.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SimpleNativeMemoryMap and related APIs.
//
// TODO: We don't reset / uncommit pages on deallocate, or on failure during
//       finalize. We should do that to reduce memory pressure.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SimpleNativeMemoryMap.h"
#include "orc-rt/SPSAllocAction.h"
#include "orc-rt/SPSMemoryFlags.h"
#include <sstream>

#if defined(__APPLE__) || defined(__linux__)
#include "Unix/NativeMemoryAPIs.inc"
#else
#error "Target OS memory APIs unsupported"
#endif

namespace orc_rt {

struct SPSSimpleNativeMemoryMapSegment;

template <>
class SPSSerializationTraits<SPSSimpleNativeMemoryMapSegment,
                             SimpleNativeMemoryMap::FinalizeRequest::Segment> {
  using SPSType = SPSTuple<SPSExecutorAddr, uint64_t, SPSAllocGroup, uint8_t>;

public:
  static bool deserialize(SPSInputBuffer &IB,
                          SimpleNativeMemoryMap::FinalizeRequest::Segment &S) {
    using ContentType =
        SimpleNativeMemoryMap::FinalizeRequest::Segment::ContentType;

    ExecutorAddr Address;
    uint64_t Size;
    AllocGroup G;
    uint8_t C;
    if (!SPSType::AsArgList::deserialize(IB, Address, Size, G, C))
      return false;
    if (Size >= std::numeric_limits<size_t>::max())
      return false;
    S.Address = Address.toPtr<void *>();
    S.Size = Size;
    S.G = G;
    S.C = static_cast<ContentType>(C);
    switch (S.C) {
    case ContentType::Uninitialized:
      return true;
    case ContentType::ZeroFill:
      memset(reinterpret_cast<char *>(S.Address), 0, S.Size);
      return true;
    case ContentType::Regular:
      // Read content directly into target address.
      return IB.read(reinterpret_cast<char *>(S.Address), S.Size);
    }
  }
};

struct SPSSimpleNativeMemoryMapFinalizeRequest;

template <>
class SPSSerializationTraits<SPSSimpleNativeMemoryMapFinalizeRequest,
                             SimpleNativeMemoryMap::FinalizeRequest> {
  using SPSType = SPSTuple<SPSSequence<SPSSimpleNativeMemoryMapSegment>,
                           SPSSequence<SPSAllocActionPair>>;

public:
  static bool deserialize(SPSInputBuffer &IB,
                          SimpleNativeMemoryMap::FinalizeRequest &FR) {
    return SPSType::AsArgList::deserialize(IB, FR.Segments, FR.AAPs);
  }
};

void SimpleNativeMemoryMap::reserve(OnReserveCompleteFn &&OnComplete,
                                    size_t Size) {
  // FIXME: Get page size from session object.
  if (Size % (64 * 1024)) {
    return OnComplete(make_error<StringError>(
        (std::ostringstream()
         << "SimpleNativeMemoryMap error: reserved size " << std::hex << Size
         << " is not a page-size multiple")
            .str()));
  }

  auto Addr = hostOSMemoryReserve(Size);
  if (!Addr)
    return OnComplete(Addr.takeError());

  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(!Slabs.count(*Addr) &&
           "hostOSMemoryReserve returned duplicate addresses");
    Slabs.emplace(std::make_pair(*Addr, SlabInfo(Size)));
  }

  OnComplete(*Addr);
}

void SimpleNativeMemoryMap::release(OnReleaseCompleteFn &&OnComplete,
                                    void *Addr) {
  std::optional<SlabInfo> SI;
  {
    std::scoped_lock<std::mutex> Lock(M);
    auto I = Slabs.find(Addr);
    if (I != Slabs.end()) {
      SI = std::move(I->second);
      Slabs.erase(I);
    }
  }

  if (!SI) {
    std::ostringstream ErrMsg;
    ErrMsg << "SimpleNativeMemoryMap error: release called on unrecognized "
              "address "
           << Addr;
    return OnComplete(make_error<StringError>(ErrMsg.str()));
  }

  for (auto &[Addr, DAAs] : SI->DeallocActions)
    runDeallocActions(std::move(DAAs));

  OnComplete(hostOSMemoryRelease(Addr, SI->Size));
}

void SimpleNativeMemoryMap::finalize(OnFinalizeCompleteFn &&OnComplete,
                                     FinalizeRequest FR) {

  void *Base = nullptr;

  // TODO: Record finalize segments for release.
  // std::vector<std::pair<void*, size_t>> FinalizeSegments;

  for (auto &S : FR.Segments) {
    if (auto Err = hostOSMemoryProtect(S.Address, S.Size, S.G.getMemProt()))
      return OnComplete(std::move(Err));
    switch (S.G.getMemLifetime()) {
    case MemLifetime::Standard:
      if (!Base || S.Address < Base)
        Base = S.Address;
      break;
    case MemLifetime::Finalize:
      // TODO: Record finalize segment for release.
      // FinalizeSegments.push_back({S.Address, S.Size});
      break;
    }
  }

  if (!Base)
    return OnComplete(make_error<StringError>(
        "SimpleNativeMemoryMap finalize error: finalization requires at least "
        "one standard-lifetime segment"));

  auto DeallocActions = runFinalizeActions(std::move(FR.AAPs));
  if (!DeallocActions)
    return OnComplete(DeallocActions.takeError());

  if (auto Err = recordDeallocActions(Base, std::move(*DeallocActions))) {
    runDeallocActions(std::move(*DeallocActions));
    return OnComplete(std::move(Err));
  }

  OnComplete(Base);
}

void SimpleNativeMemoryMap::deallocate(OnDeallocateCompleteFn &&OnComplete,
                                       void *Base) {
  std::vector<AllocAction> DAAs;

  {
    std::unique_lock<std::mutex> Lock(M);
    auto *SI = findSlabInfoFor(Base);
    if (!SI) {
      Lock.unlock();
      return OnComplete(makeBadSlabError(Base, "finalize"));
    }

    auto I = SI->DeallocActions.find(Base);
    if (I == SI->DeallocActions.end()) {
      Lock.unlock();
      std::ostringstream ErrMsg;
      ErrMsg << "SimpleNativeMemoryMap deallocate error: no deallocate actions "
                "registered for segment base address "
             << Base;
      return OnComplete(make_error<StringError>(ErrMsg.str()));
    }

    DAAs = std::move(I->second);
    SI->DeallocActions.erase(I);
  }

  runDeallocActions(std::move(DAAs));
  OnComplete(Error::success());
}

void SimpleNativeMemoryMap::detach(ResourceManager::OnCompleteFn OnComplete) {
  // Detach is a noop for now: we just retain all actions to run at shutdown
  // time.
  OnComplete(Error::success());
}

void SimpleNativeMemoryMap::shutdown(ResourceManager::OnCompleteFn OnComplete) {
  // TODO: Establish a clear order to run dealloca actions across slabs,
  // object boundaries.

  // Collect slab base addresses for removal.
  std::vector<void *> Bases;
  {
    std::scoped_lock<std::mutex> Lock(M);
    for (auto &[Base, _] : Slabs)
      Bases.push_back(Base);
  }

  shutdownNext(std::move(OnComplete), std::move(Bases));
}

void SimpleNativeMemoryMap::shutdownNext(
    ResourceManager::OnCompleteFn OnComplete, std::vector<void *> Bases) {
  if (Bases.empty())
    return OnComplete(Error::success());

  auto *Base = Bases.back();
  Bases.pop_back();

  release(
      [this, Bases = std::move(Bases),
       OnComplete = std::move(OnComplete)](Error Err) mutable {
        if (Err) {
          // TODO: Log release error?
          consumeError(std::move(Err));
        }
        shutdownNext(std::move(OnComplete), std::move(Bases));
      },
      Base);
}

Error SimpleNativeMemoryMap::makeBadSlabError(void *Base, const char *Op) {
  std::ostringstream ErrMsg;
  ErrMsg << "SimpleNativeMemoryMap " << Op << " error: segment base address "
         << Base << " does not fall within an allocated slab";
  return make_error<StringError>(ErrMsg.str());
}

SimpleNativeMemoryMap::SlabInfo *
SimpleNativeMemoryMap::findSlabInfoFor(void *Base) {
  // NOTE: We assume that the caller is holding a lock for M.
  auto I = Slabs.upper_bound(Base);
  if (I == Slabs.begin())
    return nullptr;

  --I;
  if (reinterpret_cast<char *>(I->first) + I->second.Size <=
      reinterpret_cast<char *>(Base))
    return nullptr;

  return &I->second;
}

Error SimpleNativeMemoryMap::recordDeallocActions(
    void *Base, std::vector<AllocAction> DeallocActions) {

  std::unique_lock<std::mutex> Lock(M);
  auto *SI = findSlabInfoFor(Base);
  if (!SI) {
    Lock.unlock();
    return makeBadSlabError(Base, "deallocate");
  }

  auto I = SI->DeallocActions.find(Base);
  if (I != SI->DeallocActions.end()) {
    Lock.unlock();
    std::ostringstream ErrMsg;
    ErrMsg << "SimpleNativeMemoryMap finalize error: segment base address "
              "reused in subsequent finalize call";
    return make_error<StringError>(ErrMsg.str());
  }

  SI->DeallocActions[Base] = std::move(DeallocActions);
  return Error::success();
}

ORC_RT_SPS_INTERFACE void orc_rt_SimpleNativeMemoryMap_reserve_sps_wrapper(
    orc_rt_SessionRef Session, void *CallCtx,
    orc_rt_WrapperFunctionReturn Return,
    orc_rt_WrapperFunctionBuffer ArgBytes) {
  using Sig = SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSSize);
  SPSWrapperFunction<Sig>::handle(
      Session, CallCtx, Return, ArgBytes,
      WrapperFunction::handleWithAsyncMethod(&SimpleNativeMemoryMap::reserve));
}

ORC_RT_SPS_INTERFACE void orc_rt_SimpleNativeMemoryMap_release_sps_wrapper(
    orc_rt_SessionRef Session, void *CallCtx,
    orc_rt_WrapperFunctionReturn Return,
    orc_rt_WrapperFunctionBuffer ArgBytes) {
  using Sig = SPSError(SPSExecutorAddr, SPSExecutorAddr);
  SPSWrapperFunction<Sig>::handle(
      Session, CallCtx, Return, ArgBytes,
      WrapperFunction::handleWithAsyncMethod(&SimpleNativeMemoryMap::release));
}

ORC_RT_SPS_INTERFACE void orc_rt_SimpleNativeMemoryMap_finalize_sps_wrapper(
    orc_rt_SessionRef Session, void *CallCtx,
    orc_rt_WrapperFunctionReturn Return,
    orc_rt_WrapperFunctionBuffer ArgBytes) {
  using Sig = SPSExpected<SPSExecutorAddr>(
      SPSExecutorAddr, SPSSimpleNativeMemoryMapFinalizeRequest);
  SPSWrapperFunction<Sig>::handle(
      Session, CallCtx, Return, ArgBytes,
      WrapperFunction::handleWithAsyncMethod(&SimpleNativeMemoryMap::finalize));
}

ORC_RT_SPS_INTERFACE void orc_rt_SimpleNativeMemoryMap_deallocate_sps_wrapper(
    orc_rt_SessionRef Session, void *CallCtx,
    orc_rt_WrapperFunctionReturn Return,
    orc_rt_WrapperFunctionBuffer ArgBytes) {
  using Sig = SPSError(SPSExecutorAddr, SPSExecutorAddr);
  SPSWrapperFunction<Sig>::handle(Session, CallCtx, Return, ArgBytes,
                                  WrapperFunction::handleWithAsyncMethod(
                                      &SimpleNativeMemoryMap::deallocate));
}

} // namespace orc_rt
