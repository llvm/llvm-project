//===--------------------- ResourceManager.cpp ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// The classes here represent processor resource units and their management
/// strategy.  These classes are managed by the Scheduler.
///
//===----------------------------------------------------------------------===//

#include "HardwareUnits/ResourceManager.h"
#include "Support.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mca {

using namespace llvm;

#define DEBUG_TYPE "llvm-mca"
ResourceStrategy::~ResourceStrategy() = default;

void DefaultResourceStrategy::skipMask(uint64_t Mask) {
  NextInSequenceMask &= (~Mask);
  if (!NextInSequenceMask) {
    NextInSequenceMask = ResourceUnitMask ^ RemovedFromNextInSequence;
    RemovedFromNextInSequence = 0;
  }
}

uint64_t DefaultResourceStrategy::select(uint64_t ReadyMask) {
  // This method assumes that ReadyMask cannot be zero.
  uint64_t CandidateMask = llvm::PowerOf2Floor(NextInSequenceMask);
  while (!(ReadyMask & CandidateMask)) {
    skipMask(CandidateMask);
    CandidateMask = llvm::PowerOf2Floor(NextInSequenceMask);
  }
  return CandidateMask;
}

void DefaultResourceStrategy::used(uint64_t Mask) {
  if (Mask > NextInSequenceMask) {
    RemovedFromNextInSequence |= Mask;
    return;
  }
  skipMask(Mask);
}

ResourceState::ResourceState(const MCProcResourceDesc &Desc, unsigned Index,
                             uint64_t Mask)
    : ProcResourceDescIndex(Index), ResourceMask(Mask),
      BufferSize(Desc.BufferSize) {
  if (llvm::countPopulation(ResourceMask) > 1)
    ResourceSizeMask = ResourceMask ^ llvm::PowerOf2Floor(ResourceMask);
  else
    ResourceSizeMask = (1ULL << Desc.NumUnits) - 1;
  ReadyMask = ResourceSizeMask;
  AvailableSlots = BufferSize == -1 ? 0U : static_cast<unsigned>(BufferSize);
  Unavailable = false;
}

bool ResourceState::isReady(unsigned NumUnits) const {
  return (!isReserved() || isADispatchHazard()) &&
         llvm::countPopulation(ReadyMask) >= NumUnits;
}

ResourceStateEvent ResourceState::isBufferAvailable() const {
  if (isADispatchHazard() && isReserved())
    return RS_RESERVED;
  if (!isBuffered() || AvailableSlots)
    return RS_BUFFER_AVAILABLE;
  return RS_BUFFER_UNAVAILABLE;
}

#ifndef NDEBUG
void ResourceState::dump() const {
  dbgs() << "MASK: " << ResourceMask << ", SIZE_MASK: " << ResourceSizeMask
         << ", RDYMASK: " << ReadyMask << ", BufferSize=" << BufferSize
         << ", AvailableSlots=" << AvailableSlots
         << ", Reserved=" << Unavailable << '\n';
}
#endif

static unsigned getResourceStateIndex(uint64_t Mask) {
  return std::numeric_limits<uint64_t>::digits - llvm::countLeadingZeros(Mask);
}

static std::unique_ptr<ResourceStrategy>
getStrategyFor(const ResourceState &RS) {
  if (RS.isAResourceGroup() || RS.getNumUnits() > 1)
    return llvm::make_unique<DefaultResourceStrategy>(RS.getReadyMask());
  return std::unique_ptr<ResourceStrategy>(nullptr);
}

ResourceManager::ResourceManager(const MCSchedModel &SM)
    : ProcResID2Mask(SM.getNumProcResourceKinds()) {
  computeProcResourceMasks(SM, ProcResID2Mask);
  Resources.resize(SM.getNumProcResourceKinds());
  Strategies.resize(SM.getNumProcResourceKinds());

  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    uint64_t Mask = ProcResID2Mask[I];
    unsigned Index = getResourceStateIndex(Mask);
    Resources[Index] =
        llvm::make_unique<ResourceState>(*SM.getProcResource(I), I, Mask);
    Strategies[Index] = getStrategyFor(*Resources[Index]);
  }
}

void ResourceManager::setCustomStrategyImpl(std::unique_ptr<ResourceStrategy> S,
                                            uint64_t ResourceMask) {
  unsigned Index = getResourceStateIndex(ResourceMask);
  assert(Index < Resources.size() && "Invalid processor resource index!");
  assert(S && "Unexpected null strategy in input!");
  Strategies[Index] = std::move(S);
}

unsigned ResourceManager::resolveResourceMask(uint64_t Mask) const {
  return Resources[getResourceStateIndex(Mask)]->getProcResourceID();
}

unsigned ResourceManager::getNumUnits(uint64_t ResourceID) const {
  return Resources[getResourceStateIndex(ResourceID)]->getNumUnits();
}

// Returns the actual resource consumed by this Use.
// First, is the primary resource ID.
// Second, is the specific sub-resource ID.
ResourceRef ResourceManager::selectPipe(uint64_t ResourceID) {
  unsigned Index = getResourceStateIndex(ResourceID);
  ResourceState &RS = *Resources[Index];
  assert(RS.isReady() && "No available units to select!");

  // Special case where RS is not a group, and it only declares a single
  // resource unit.
  if (!RS.isAResourceGroup() && RS.getNumUnits() == 1)
    return std::make_pair(ResourceID, RS.getReadyMask());

  uint64_t SubResourceID = Strategies[Index]->select(RS.getReadyMask());
  if (RS.isAResourceGroup())
    return selectPipe(SubResourceID);
  return std::make_pair(ResourceID, SubResourceID);
}

void ResourceManager::use(const ResourceRef &RR) {
  // Mark the sub-resource referenced by RR as used.
  ResourceState &RS = *Resources[getResourceStateIndex(RR.first)];
  RS.markSubResourceAsUsed(RR.second);
  // If there are still available units in RR.first,
  // then we are done.
  if (RS.isReady())
    return;

  // Notify to other resources that RR.first is no longer available.
  for (std::unique_ptr<ResourceState> &Res : Resources) {
    ResourceState &Current = *Res;
    if (!Current.isAResourceGroup() || Current.getResourceMask() == RR.first)
      continue;

    if (Current.containsResource(RR.first)) {
      unsigned Index = getResourceStateIndex(Current.getResourceMask());
      Current.markSubResourceAsUsed(RR.first);
      Strategies[Index]->used(RR.first);
    }
  }
}

void ResourceManager::release(const ResourceRef &RR) {
  ResourceState &RS = *Resources[getResourceStateIndex(RR.first)];
  bool WasFullyUsed = !RS.isReady();
  RS.releaseSubResource(RR.second);
  if (!WasFullyUsed)
    return;

  for (std::unique_ptr<ResourceState> &Res : Resources) {
    ResourceState &Current = *Res;
    if (!Current.isAResourceGroup() || Current.getResourceMask() == RR.first)
      continue;

    if (Current.containsResource(RR.first))
      Current.releaseSubResource(RR.first);
  }
}

ResourceStateEvent
ResourceManager::canBeDispatched(ArrayRef<uint64_t> Buffers) const {
  ResourceStateEvent Result = ResourceStateEvent::RS_BUFFER_AVAILABLE;
  for (uint64_t Buffer : Buffers) {
    ResourceState &RS = *Resources[getResourceStateIndex(Buffer)];
    Result = RS.isBufferAvailable();
    if (Result != ResourceStateEvent::RS_BUFFER_AVAILABLE)
      break;
  }
  return Result;
}

void ResourceManager::reserveBuffers(ArrayRef<uint64_t> Buffers) {
  for (const uint64_t Buffer : Buffers) {
    ResourceState &RS = *Resources[getResourceStateIndex(Buffer)];
    assert(RS.isBufferAvailable() == ResourceStateEvent::RS_BUFFER_AVAILABLE);
    RS.reserveBuffer();

    if (RS.isADispatchHazard()) {
      assert(!RS.isReserved());
      RS.setReserved();
    }
  }
}

void ResourceManager::releaseBuffers(ArrayRef<uint64_t> Buffers) {
  for (const uint64_t R : Buffers)
    Resources[getResourceStateIndex(R)]->releaseBuffer();
}

bool ResourceManager::canBeIssued(const InstrDesc &Desc) const {
  return std::all_of(Desc.Resources.begin(), Desc.Resources.end(),
                     [&](const std::pair<uint64_t, const ResourceUsage> &E) {
                       unsigned NumUnits =
                           E.second.isReserved() ? 0U : E.second.NumUnits;
                       unsigned Index = getResourceStateIndex(E.first);
                       return Resources[Index]->isReady(NumUnits);
                     });
}

// Returns true if all resources are in-order, and there is at least one
// resource which is a dispatch hazard (BufferSize = 0).
bool ResourceManager::mustIssueImmediately(const InstrDesc &Desc) const {
  if (!canBeIssued(Desc))
    return false;
  bool AllInOrderResources = all_of(Desc.Buffers, [&](uint64_t BufferMask) {
    unsigned Index = getResourceStateIndex(BufferMask);
    const ResourceState &Resource = *Resources[Index];
    return Resource.isInOrder() || Resource.isADispatchHazard();
  });
  if (!AllInOrderResources)
    return false;

  return any_of(Desc.Buffers, [&](uint64_t BufferMask) {
    return Resources[getResourceStateIndex(BufferMask)]->isADispatchHazard();
  });
}

void ResourceManager::issueInstruction(
    const InstrDesc &Desc,
    SmallVectorImpl<std::pair<ResourceRef, double>> &Pipes) {
  for (const std::pair<uint64_t, ResourceUsage> &R : Desc.Resources) {
    const CycleSegment &CS = R.second.CS;
    if (!CS.size()) {
      releaseResource(R.first);
      continue;
    }

    assert(CS.begin() == 0 && "Invalid {Start, End} cycles!");
    if (!R.second.isReserved()) {
      ResourceRef Pipe = selectPipe(R.first);
      use(Pipe);
      BusyResources[Pipe] += CS.size();
      // Replace the resource mask with a valid processor resource index.
      const ResourceState &RS = *Resources[getResourceStateIndex(Pipe.first)];
      Pipe.first = RS.getProcResourceID();
      Pipes.emplace_back(
          std::pair<ResourceRef, double>(Pipe, static_cast<double>(CS.size())));
    } else {
      assert((countPopulation(R.first) > 1) && "Expected a group!");
      // Mark this group as reserved.
      assert(R.second.isReserved());
      reserveResource(R.first);
      BusyResources[ResourceRef(R.first, R.first)] += CS.size();
    }
  }
}

void ResourceManager::cycleEvent(SmallVectorImpl<ResourceRef> &ResourcesFreed) {
  for (std::pair<ResourceRef, unsigned> &BR : BusyResources) {
    if (BR.second)
      BR.second--;
    if (!BR.second) {
      // Release this resource.
      const ResourceRef &RR = BR.first;

      if (countPopulation(RR.first) == 1)
        release(RR);

      releaseResource(RR.first);
      ResourcesFreed.push_back(RR);
    }
  }

  for (const ResourceRef &RF : ResourcesFreed)
    BusyResources.erase(RF);
}

void ResourceManager::reserveResource(uint64_t ResourceID) {
  ResourceState &Resource = *Resources[getResourceStateIndex(ResourceID)];
  assert(!Resource.isReserved());
  Resource.setReserved();
}

void ResourceManager::releaseResource(uint64_t ResourceID) {
  ResourceState &Resource = *Resources[getResourceStateIndex(ResourceID)];
  Resource.clearReserved();
}

} // namespace mca
