//===--------------------- Scheduler.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A scheduler for processor resource units and processor resource groups.
//
//===----------------------------------------------------------------------===//

#include "Scheduler.h"
#include "Support.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mca {

using namespace llvm;

#define DEBUG_TYPE "llvm-mca"

uint64_t ResourceState::selectNextInSequence() {
  assert(isReady());
  uint64_t Next = getNextInSequence();
  while (!isSubResourceReady(Next)) {
    updateNextInSequence();
    Next = getNextInSequence();
  }
  return Next;
}

#ifndef NDEBUG
void ResourceState::dump() const {
  dbgs() << "MASK: " << ResourceMask << ", SIZE_MASK: " << ResourceSizeMask
         << ", NEXT: " << NextInSequenceMask << ", RDYMASK: " << ReadyMask
         << ", BufferSize=" << BufferSize
         << ", AvailableSlots=" << AvailableSlots
         << ", Reserved=" << Unavailable << '\n';
}
#endif

unsigned getResourceStateIndex(uint64_t Mask) {
  return std::numeric_limits<uint64_t>::digits - llvm::countLeadingZeros(Mask);
}

unsigned ResourceManager::resolveResourceMask(uint64_t Mask) const {
  return Resources[getResourceStateIndex(Mask)]->getProcResourceID();
}

unsigned ResourceManager::getNumUnits(uint64_t ResourceID) const {
  return Resources[getResourceStateIndex(ResourceID)]->getNumUnits();
}

void ResourceManager::initialize(const llvm::MCSchedModel &SM) {
  computeProcResourceMasks(SM, ProcResID2Mask);
  Resources.resize(SM.getNumProcResourceKinds());

  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    uint64_t Mask = ProcResID2Mask[I];
    Resources[getResourceStateIndex(Mask)] =
        llvm::make_unique<ResourceState>(*SM.getProcResource(I), I, Mask);
  }
}

// Returns the actual resource consumed by this Use.
// First, is the primary resource ID.
// Second, is the specific sub-resource ID.
std::pair<uint64_t, uint64_t> ResourceManager::selectPipe(uint64_t ResourceID) {
  ResourceState &RS = *Resources[getResourceStateIndex(ResourceID)];
  uint64_t SubResourceID = RS.selectNextInSequence();
  if (RS.isAResourceGroup())
    return selectPipe(SubResourceID);
  return std::make_pair(ResourceID, SubResourceID);
}

void ResourceState::removeFromNextInSequence(uint64_t ID) {
  assert(NextInSequenceMask);
  assert(countPopulation(ID) == 1);
  if (ID > getNextInSequence())
    RemovedFromNextInSequence |= ID;
  NextInSequenceMask = NextInSequenceMask & (~ID);
  if (!NextInSequenceMask) {
    NextInSequenceMask = ResourceSizeMask;
    assert(NextInSequenceMask != RemovedFromNextInSequence);
    NextInSequenceMask ^= RemovedFromNextInSequence;
    RemovedFromNextInSequence = 0;
  }
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
  for (UniqueResourceState &Res : Resources) {
    ResourceState &Current = *Res;
    if (!Current.isAResourceGroup() || Current.getResourceMask() == RR.first)
      continue;

    if (Current.containsResource(RR.first)) {
      Current.markSubResourceAsUsed(RR.first);
      Current.removeFromNextInSequence(RR.first);
    }
  }
}

void ResourceManager::release(const ResourceRef &RR) {
  ResourceState &RS = *Resources[getResourceStateIndex(RR.first)];
  bool WasFullyUsed = !RS.isReady();
  RS.releaseSubResource(RR.second);
  if (!WasFullyUsed)
    return;

  for (UniqueResourceState &Res : Resources) {
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
bool ResourceManager::mustIssueImmediately(const InstrDesc &Desc) {
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

#ifndef NDEBUG
void Scheduler::dump() const {
  dbgs() << "[SCHEDULER]: WaitSet size is: " << WaitSet.size() << '\n';
  dbgs() << "[SCHEDULER]: ReadySet size is: " << ReadySet.size() << '\n';
  dbgs() << "[SCHEDULER]: IssuedSet size is: " << IssuedSet.size() << '\n';
  Resources->dump();
}
#endif

bool Scheduler::canBeDispatched(const InstRef &IR,
                                Scheduler::StallKind &Event) const {
  Event = StallKind::NoStall;
  const InstrDesc &Desc = IR.getInstruction()->getDesc();

  // Give lower priority to these stall events.
  if (Desc.MayStore && LSU->isSQFull())
    Event = StallKind::StoreQueueFull;
  if (Desc.MayLoad && LSU->isLQFull())
    Event = StallKind::LoadQueueFull;
    
  switch (Resources->canBeDispatched(Desc.Buffers)) {
  case ResourceStateEvent::RS_BUFFER_UNAVAILABLE:
    Event = StallKind::SchedulerQueueFull;
    break;
  case ResourceStateEvent::RS_RESERVED:
    Event = StallKind::DispatchGroupStall;
    break;
  default:
    break;
  }

  return Event == StallKind::NoStall;
}

void Scheduler::issueInstructionImpl(
    InstRef &IR,
    SmallVectorImpl<std::pair<ResourceRef, double>> &UsedResources) {
  Instruction *IS = IR.getInstruction();
  const InstrDesc &D = IS->getDesc();

  // Issue the instruction and collect all the consumed resources
  // into a vector. That vector is then used to notify the listener.
  Resources->issueInstruction(D, UsedResources);

  // Notify the instruction that it started executing.
  // This updates the internal state of each write.
  IS->execute();

  if (IS->isExecuting())
    IssuedSet.emplace_back(IR);
}

// Release the buffered resources and issue the instruction.
void Scheduler::issueInstruction(
    InstRef &IR,
    SmallVectorImpl<std::pair<ResourceRef, double>> &UsedResources) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  releaseBuffers(Desc.Buffers);
  issueInstructionImpl(IR, UsedResources);
}

void Scheduler::promoteToReadySet(SmallVectorImpl<InstRef> &Ready) {
  // Scan the set of waiting instructions and promote them to the
  // ready queue if operands are all ready.
  unsigned RemovedElements = 0;
  for (auto I = WaitSet.begin(), E = WaitSet.end(); I != E;) {
    InstRef &IR = *I;
    if (!IR.isValid())
      break;

    // Check if this instruction is now ready. In case, force
    // a transition in state using method 'update()'.
    Instruction &IS = *IR.getInstruction();
    if (!IS.isReady())
      IS.update();

    const InstrDesc &Desc = IS.getDesc();
    bool IsMemOp = Desc.MayLoad || Desc.MayStore;
    if (!IS.isReady() || (IsMemOp && !LSU->isReady(IR))) {
      ++I;
      continue;
    }

    Ready.emplace_back(IR);
    ReadySet.emplace_back(IR);

    IR.invalidate();
    ++RemovedElements;
    std::iter_swap(I, E - RemovedElements);
  }

  WaitSet.resize(WaitSet.size() - RemovedElements);
}

InstRef Scheduler::select() {
  unsigned QueueIndex = ReadySet.size();
  int Rank = std::numeric_limits<int>::max();

  for (unsigned I = 0, E = ReadySet.size(); I != E; ++I) {
    const InstRef &IR = ReadySet[I];
    const unsigned IID = IR.getSourceIndex();
    const Instruction &IS = *IR.getInstruction();

    // Compute a rank value based on the age of an instruction (i.e. its source
    // index) and its number of users. The lower the rank value, the better.
    int CurrentRank = IID - IS.getNumUsers();

    // We want to prioritize older instructions over younger instructions to
    // minimize the pressure on the reorder buffer.  We also want to
    // rank higher the instructions with more users to better expose ILP.
    if (CurrentRank == Rank)
      if (IID > ReadySet[QueueIndex].getSourceIndex())
        continue;

    if (CurrentRank <= Rank) {
      const InstrDesc &D = IS.getDesc();
      if (Resources->canBeIssued(D)) {
        Rank = CurrentRank;
        QueueIndex = I;
      }
    }
  }

  if (QueueIndex == ReadySet.size())
    return InstRef();

  // We found an instruction to issue.

  InstRef IR = ReadySet[QueueIndex];
  std::swap(ReadySet[QueueIndex], ReadySet[ReadySet.size() - 1]);
  ReadySet.pop_back();
  return IR;
}

void Scheduler::updatePendingQueue(SmallVectorImpl<InstRef> &Ready) {
  // Notify to instructions in the pending queue that a new cycle just
  // started.
  for (InstRef &Entry : WaitSet)
    Entry.getInstruction()->cycleEvent();
  promoteToReadySet(Ready);
}

void Scheduler::updateIssuedSet(SmallVectorImpl<InstRef> &Executed) {
  unsigned RemovedElements = 0;
  for (auto I = IssuedSet.begin(), E = IssuedSet.end(); I != E;) {
    InstRef &IR = *I;
    if (!IR.isValid())
      break;
    Instruction &IS = *IR.getInstruction();
    IS.cycleEvent();
    if (!IS.isExecuted()) {
      LLVM_DEBUG(dbgs() << "[SCHEDULER]: Instruction #" << IR
                        << " is still executing.\n");
      ++I;
      continue;
    }

    Executed.emplace_back(IR);
    ++RemovedElements;
    IR.invalidate();
    std::iter_swap(I, E - RemovedElements);
  }

  IssuedSet.resize(IssuedSet.size() - RemovedElements);
}

void Scheduler::onInstructionExecuted(const InstRef &IR) {
  LSU->onInstructionExecuted(IR);
}

void Scheduler::reclaimSimulatedResources(SmallVectorImpl<ResourceRef> &Freed) {
  Resources->cycleEvent(Freed);
}

bool Scheduler::reserveResources(InstRef &IR) {
  // If necessary, reserve queue entries in the load-store unit (LSU).
  const bool Reserved = LSU->reserve(IR);
  if (!IR.getInstruction()->isReady() || (Reserved && !LSU->isReady(IR))) {
    LLVM_DEBUG(dbgs() << "[SCHEDULER] Adding #" << IR << " to the WaitSet\n");
    WaitSet.push_back(IR);
    return false;
  }
  return true;
}

bool Scheduler::issueImmediately(InstRef &IR) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  if (!Desc.isZeroLatency() && !Resources->mustIssueImmediately(Desc)) {
    LLVM_DEBUG(dbgs() << "[SCHEDULER] Adding #" << IR << " to the ReadySet\n");
    ReadySet.push_back(IR);
    return false;
  }
  return true;
}

} // namespace mca
