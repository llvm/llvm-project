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

void ResourceManager::initialize(const llvm::MCSchedModel &SM) {
  computeProcResourceMasks(SM, ProcResID2Mask);
  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I)
    addResource(*SM.getProcResource(I), I, ProcResID2Mask[I]);
}

// Adds a new resource state in Resources, as well as a new descriptor in
// ResourceDescriptor. Map 'Resources' allows to quickly obtain ResourceState
// objects from resource mask identifiers.
void ResourceManager::addResource(const MCProcResourceDesc &Desc,
                                  unsigned Index, uint64_t Mask) {
  assert(Resources.find(Mask) == Resources.end() && "Resource already added!");
  Resources[Mask] = llvm::make_unique<ResourceState>(Desc, Index, Mask);
}

// Returns the actual resource consumed by this Use.
// First, is the primary resource ID.
// Second, is the specific sub-resource ID.
std::pair<uint64_t, uint64_t> ResourceManager::selectPipe(uint64_t ResourceID) {
  ResourceState &RS = *Resources[ResourceID];
  uint64_t SubResourceID = RS.selectNextInSequence();
  if (RS.isAResourceGroup())
    return selectPipe(SubResourceID);
  return std::pair<uint64_t, uint64_t>(ResourceID, SubResourceID);
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

void ResourceManager::use(ResourceRef RR) {
  // Mark the sub-resource referenced by RR as used.
  ResourceState &RS = *Resources[RR.first];
  RS.markSubResourceAsUsed(RR.second);
  // If there are still available units in RR.first,
  // then we are done.
  if (RS.isReady())
    return;

  // Notify to other resources that RR.first is no longer available.
  for (const std::pair<uint64_t, UniqueResourceState> &Res : Resources) {
    ResourceState &Current = *Res.second.get();
    if (!Current.isAResourceGroup() || Current.getResourceMask() == RR.first)
      continue;

    if (Current.containsResource(RR.first)) {
      Current.markSubResourceAsUsed(RR.first);
      Current.removeFromNextInSequence(RR.first);
    }
  }
}

void ResourceManager::release(ResourceRef RR) {
  ResourceState &RS = *Resources[RR.first];
  bool WasFullyUsed = !RS.isReady();
  RS.releaseSubResource(RR.second);
  if (!WasFullyUsed)
    return;

  for (const std::pair<uint64_t, UniqueResourceState> &Res : Resources) {
    ResourceState &Current = *Res.second.get();
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
    Result = isBufferAvailable(Buffer);
    if (Result != ResourceStateEvent::RS_BUFFER_AVAILABLE)
      break;
  }
  return Result;
}

void ResourceManager::reserveBuffers(ArrayRef<uint64_t> Buffers) {
  for (const uint64_t R : Buffers) {
    reserveBuffer(R);
    ResourceState &Resource = *Resources[R];
    if (Resource.isADispatchHazard()) {
      assert(!Resource.isReserved());
      Resource.setReserved();
    }
  }
}

void ResourceManager::releaseBuffers(ArrayRef<uint64_t> Buffers) {
  for (const uint64_t R : Buffers)
    releaseBuffer(R);
}

bool ResourceManager::canBeIssued(const InstrDesc &Desc) const {
  return std::all_of(Desc.Resources.begin(), Desc.Resources.end(),
                     [&](const std::pair<uint64_t, const ResourceUsage> &E) {
                       unsigned NumUnits =
                           E.second.isReserved() ? 0U : E.second.NumUnits;
                       return isReady(E.first, NumUnits);
                     });
}

// Returns true if all resources are in-order, and there is at least one
// resource which is a dispatch hazard (BufferSize = 0).
bool ResourceManager::mustIssueImmediately(const InstrDesc &Desc) {
  if (!canBeIssued(Desc))
    return false;
  bool AllInOrderResources = all_of(Desc.Buffers, [&](uint64_t BufferMask) {
    const ResourceState &Resource = *Resources[BufferMask];
    return Resource.isInOrder() || Resource.isADispatchHazard();
  });
  if (!AllInOrderResources)
    return false;

  return any_of(Desc.Buffers, [&](uint64_t BufferMask) {
    return Resources[BufferMask]->isADispatchHazard();
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
      const ResourceState &RS = *Resources[Pipe.first];
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

#ifndef NDEBUG
void Scheduler::dump() const {
  dbgs() << "[SCHEDULER]: WaitQueue size is: " << WaitQueue.size() << '\n';
  dbgs() << "[SCHEDULER]: ReadyQueue size is: " << ReadyQueue.size() << '\n';
  dbgs() << "[SCHEDULER]: IssuedQueue size is: " << IssuedQueue.size() << '\n';
  Resources->dump();
}
#endif

bool Scheduler::canBeDispatched(const InstRef &IR,
                                HWStallEvent::GenericEventType &Event) const {
  Event = HWStallEvent::Invalid;
  const InstrDesc &Desc = IR.getInstruction()->getDesc();

  if (Desc.MayLoad && LSU->isLQFull())
    Event = HWStallEvent::LoadQueueFull;
  else if (Desc.MayStore && LSU->isSQFull())
    Event = HWStallEvent::StoreQueueFull;
  else {
    switch (Resources->canBeDispatched(Desc.Buffers)) {
    default:
      return true;
    case ResourceStateEvent::RS_BUFFER_UNAVAILABLE:
      Event = HWStallEvent::SchedulerQueueFull;
      break;
    case ResourceStateEvent::RS_RESERVED:
      Event = HWStallEvent::DispatchGroupStall;
    }
  }

  return false;
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
    IssuedQueue[IR.getSourceIndex()] = IS;
}

// Release the buffered resources and issue the instruction.
void Scheduler::issueInstruction(
    InstRef &IR,
    SmallVectorImpl<std::pair<ResourceRef, double>> &UsedResources) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  releaseBuffers(Desc.Buffers);
  issueInstructionImpl(IR, UsedResources);
}

void Scheduler::promoteToReadyQueue(SmallVectorImpl<InstRef> &Ready) {
  // Scan the set of waiting instructions and promote them to the
  // ready queue if operands are all ready.
  for (auto I = WaitQueue.begin(), E = WaitQueue.end(); I != E;) {
    const unsigned IID = I->first;
    Instruction *IS = I->second;

    // Check if this instruction is now ready. In case, force
    // a transition in state using method 'update()'.
    if (!IS->isReady())
      IS->update();

    const InstrDesc &Desc = IS->getDesc();
    bool IsMemOp = Desc.MayLoad || Desc.MayStore;
    if (!IS->isReady() || (IsMemOp && !LSU->isReady({IID, IS}))) {
      ++I;
      continue;
    }

    Ready.emplace_back(IID, IS);
    ReadyQueue[IID] = IS;
    auto ToRemove = I;
    ++I;
    WaitQueue.erase(ToRemove);
  }
}

InstRef Scheduler::select() {
  // Find the oldest ready-to-issue instruction in the ReadyQueue.
  auto It = std::find_if(ReadyQueue.begin(), ReadyQueue.end(),
                         [&](const QueueEntryTy &Entry) {
                           const InstrDesc &D = Entry.second->getDesc();
                           return Resources->canBeIssued(D);
                         });

  if (It == ReadyQueue.end())
    return {0, nullptr};

  // We want to prioritize older instructions over younger instructions to
  // minimize the pressure on the reorder buffer.  We also want to
  // rank higher the instructions with more users to better expose ILP.

  // Compute a rank value based on the age of an instruction (i.e. its source
  // index) and its number of users. The lower the rank value, the better.
  int Rank = It->first - It->second->getNumUsers();
  for (auto I = It, E = ReadyQueue.end(); I != E; ++I) {
    int CurrentRank = I->first - I->second->getNumUsers();
    if (CurrentRank < Rank) {
      const InstrDesc &D = I->second->getDesc();
      if (Resources->canBeIssued(D))
        It = I;
    }
  }

  // We found an instruction to issue.
  InstRef IR(It->first, It->second);
  ReadyQueue.erase(It);
  return IR;
}

void Scheduler::updatePendingQueue(SmallVectorImpl<InstRef> &Ready) {
  // Notify to instructions in the pending queue that a new cycle just
  // started.
  for (QueueEntryTy Entry : WaitQueue)
    Entry.second->cycleEvent();
  promoteToReadyQueue(Ready);
}

void Scheduler::updateIssuedQueue(SmallVectorImpl<InstRef> &Executed) {
  for (auto I = IssuedQueue.begin(), E = IssuedQueue.end(); I != E;) {
    const QueueEntryTy Entry = *I;
    Instruction *IS = Entry.second;
    IS->cycleEvent();
    if (IS->isExecuted()) {
      Executed.push_back({Entry.first, Entry.second});
      auto ToRemove = I;
      ++I;
      IssuedQueue.erase(ToRemove);
    } else {
      LLVM_DEBUG(dbgs() << "[SCHEDULER]: Instruction #" << Entry.first
                        << " is still executing.\n");
      ++I;
    }
  }
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
    LLVM_DEBUG(dbgs() << "[SCHEDULER] Adding #" << IR << " to the Wait Queue\n");
    WaitQueue[IR.getSourceIndex()] = IR.getInstruction();
    return false;
  }
  return true;
}

bool Scheduler::issueImmediately(InstRef &IR) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  if (!Desc.isZeroLatency() && !Resources->mustIssueImmediately(Desc)) {
    LLVM_DEBUG(dbgs() << "[SCHEDULER] Adding #" << IR
                      << " to the Ready Queue\n");
    ReadyQueue[IR.getSourceIndex()] = IR.getInstruction();
    return false;
  }
  return true;
}

} // namespace mca
