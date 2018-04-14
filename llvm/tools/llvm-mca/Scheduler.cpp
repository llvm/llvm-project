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
#include "Backend.h"
#include "HWEventListener.h"
#include "Support.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "llvm-mca"

namespace mca {

using namespace llvm;

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
  bool AllInOrderResources = std::all_of(
      Desc.Buffers.begin(), Desc.Buffers.end(), [&](const unsigned BufferMask) {
        const ResourceState &Resource = *Resources[BufferMask];
        return Resource.isInOrder() || Resource.isADispatchHazard();
      });
  if (!AllInOrderResources)
    return false;

  return std::any_of(Desc.Buffers.begin(), Desc.Buffers.end(),
                     [&](const unsigned BufferMask) {
                       return Resources[BufferMask]->isADispatchHazard();
                     });
}

void ResourceManager::issueInstruction(
    unsigned Index, const InstrDesc &Desc,
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

void Scheduler::scheduleInstruction(unsigned Idx, Instruction &MCIS) {
  assert(WaitQueue.find(Idx) == WaitQueue.end());
  assert(ReadyQueue.find(Idx) == ReadyQueue.end());
  assert(IssuedQueue.find(Idx) == IssuedQueue.end());

  // Special case where MCIS is a zero-latency instruction.  A zero-latency
  // instruction doesn't consume any scheduler resources.  That is because it
  // doesn't need to be executed.  Most of the times, zero latency instructions
  // are removed at register renaming stage. For example, register-register
  // moves can be removed at register renaming stage by creating new aliases.
  // Zero-idiom instruction (for example: a `xor reg, reg`) can also be
  // eliminated at register renaming stage, since we know in advance that those
  // clear their output register.
  if (MCIS.isZeroLatency()) {
    assert(MCIS.isReady() && "data dependent zero-latency instruction?");
    notifyInstructionReady(Idx);
    MCIS.execute();
    notifyInstructionIssued(Idx, {});
    assert(MCIS.isExecuted() && "Unexpected non-zero latency!");
    notifyInstructionExecuted(Idx);
    return;
  }

  const InstrDesc &Desc = MCIS.getDesc();
  if (!Desc.Buffers.empty()) {
    // Reserve a slot in each buffered resource. Also, mark units with
    // BufferSize=0 as reserved. Resources with a buffer size of zero will only
    // be released after MCIS is issued, and all the ResourceCycles for those
    // units have been consumed.
    Resources->reserveBuffers(Desc.Buffers);
    notifyReservedBuffers(Desc.Buffers);
  }

  // If necessary, reserve queue entries in the load-store unit (LSU).
  bool Reserved = LSU->reserve(Idx, Desc);
  if (!MCIS.isReady() || (Reserved && !LSU->isReady(Idx))) {
    DEBUG(dbgs() << "[SCHEDULER] Adding " << Idx << " to the Wait Queue\n");
    WaitQueue[Idx] = &MCIS;
    return;
  }
  notifyInstructionReady(Idx);

  // Special case where the instruction is ready, and it uses an in-order
  // dispatch/issue processor resource. The instruction is issued immediately to
  // the pipelines. Any other in-order buffered resources (i.e. BufferSize=1)
  // are consumed.
  if (Resources->mustIssueImmediately(Desc)) {
    DEBUG(dbgs() << "[SCHEDULER] Instruction " << Idx
                 << " issued immediately\n");
    return issueInstruction(Idx, MCIS);
  }

  DEBUG(dbgs() << "[SCHEDULER] Adding " << Idx << " to the Ready Queue\n");
  ReadyQueue[Idx] = &MCIS;
}

void Scheduler::cycleEvent() {
  SmallVector<ResourceRef, 8> ResourcesFreed;
  Resources->cycleEvent(ResourcesFreed);

  for (const ResourceRef &RR : ResourcesFreed)
    notifyResourceAvailable(RR);

  updateIssuedQueue();
  updatePendingQueue();

  while (issue()) {
    // Instructions that have been issued during this cycle might have unblocked
    // other dependent instructions. Dependent instructions may be issued during
    // this same cycle if operands have ReadAdvance entries.  Promote those
    // instructions to the ReadyQueue and tell to the caller that we need
    // another round of 'issue()'.
    promoteToReadyQueue();
  }
}

#ifndef NDEBUG
void Scheduler::dump() const {
  dbgs() << "[SCHEDULER]: WaitQueue size is: " << WaitQueue.size() << '\n';
  dbgs() << "[SCHEDULER]: ReadyQueue size is: " << ReadyQueue.size() << '\n';
  dbgs() << "[SCHEDULER]: IssuedQueue size is: " << IssuedQueue.size() << '\n';
  Resources->dump();
}
#endif

bool Scheduler::canBeDispatched(unsigned Index, const InstrDesc &Desc) const {
  HWStallEvent::GenericEventType Type = HWStallEvent::Invalid;

  if (Desc.MayLoad && LSU->isLQFull())
    Type = HWStallEvent::LoadQueueFull;
  else if (Desc.MayStore && LSU->isSQFull())
    Type = HWStallEvent::StoreQueueFull;
  else {
    switch (Resources->canBeDispatched(Desc.Buffers)) {
    default:
      return true;
    case ResourceStateEvent::RS_BUFFER_UNAVAILABLE:
      Type = HWStallEvent::SchedulerQueueFull;
      break;
    case ResourceStateEvent::RS_RESERVED:
      Type = HWStallEvent::DispatchGroupStall;
    }
  }

  Owner->notifyStallEvent(HWStallEvent(Type, Index));
  return false;
}

void Scheduler::issueInstruction(unsigned InstrIndex, Instruction &IS) {
  const InstrDesc &D = IS.getDesc();

  if (!D.Buffers.empty()) {
    Resources->releaseBuffers(D.Buffers);
    notifyReleasedBuffers(D.Buffers);
  }

  // Issue the instruction and collect all the consumed resources
  // into a vector. That vector is then used to notify the listener.
  // Most instructions consume very few resurces (typically one or
  // two resources). We use a small vector here, and conservatively
  // initialize its capacity to 4. This should address the majority of
  // the cases.
  SmallVector<std::pair<ResourceRef, double>, 4> UsedResources;
  Resources->issueInstruction(InstrIndex, D, UsedResources);
  // Notify the instruction that it started executing.
  // This updates the internal state of each write.
  IS.execute();

  notifyInstructionIssued(InstrIndex, UsedResources);
  if (D.MaxLatency) {
    assert(IS.isExecuting() && "A zero latency instruction?");
    IssuedQueue[InstrIndex] = &IS;
    return;
  }

  // A zero latency instruction which reads and/or updates registers.
  assert(IS.isExecuted() && "Instruction still executing!");
  notifyInstructionExecuted(InstrIndex);
}

void Scheduler::promoteToReadyQueue() {
  // Scan the set of waiting instructions and promote them to the
  // ready queue if operands are all ready.
  for (auto I = WaitQueue.begin(), E = WaitQueue.end(); I != E;) {
    const QueueEntryTy &Entry = *I;
    unsigned IID = Entry.first;
    Instruction &Inst = *Entry.second;

    // Check if this instruction is now ready. In case, force
    // a transition in state using method 'update()'.
    Inst.update();

    const InstrDesc &Desc = Inst.getDesc();
    bool IsMemOp = Desc.MayLoad || Desc.MayStore;
    if (!Inst.isReady() || (IsMemOp && !LSU->isReady(IID))) {
      ++I;
      continue;
    }

    notifyInstructionReady(IID);
    ReadyQueue[IID] = &Inst;
    auto ToRemove = I;
    ++I;
    WaitQueue.erase(ToRemove);
  }
}

bool Scheduler::issue() {
  // Give priority to older instructions in the ReadyQueue. Since the ready
  // queue is ordered by key, this will always prioritize older instructions.
  const auto It = std::find_if(ReadyQueue.begin(), ReadyQueue.end(),
                               [&](const QueueEntryTy &Entry) {
                                 const Instruction &IS = *Entry.second;
                                 const InstrDesc &D = IS.getDesc();
                                 return Resources->canBeIssued(D);
                               });

  if (It == ReadyQueue.end())
    return false;

  // We found an instruction. Issue it, and update the ready queue.
  const QueueEntryTy &Entry = *It;
  issueInstruction(Entry.first, *Entry.second);
  ReadyQueue.erase(Entry.first);
  return true;
}

void Scheduler::updatePendingQueue() {
  // Notify to instructions in the pending queue that a new cycle just
  // started.
  for (QueueEntryTy Entry : WaitQueue)
    Entry.second->cycleEvent();
  promoteToReadyQueue();
}

void Scheduler::updateIssuedQueue() {
  for (auto I = IssuedQueue.begin(), E = IssuedQueue.end(); I != E;) {
    const QueueEntryTy Entry = *I;
    Entry.second->cycleEvent();
    if (Entry.second->isExecuted()) {
      notifyInstructionExecuted(Entry.first);
      auto ToRemove = I;
      ++I;
      IssuedQueue.erase(ToRemove);
    } else {
      DEBUG(dbgs() << "[SCHEDULER]: Instruction " << Entry.first
                   << " is still executing.\n");
      ++I;
    }
  }
}

void Scheduler::notifyInstructionIssued(
    unsigned Index, ArrayRef<std::pair<ResourceRef, double>> Used) {
  DEBUG({
    dbgs() << "[E] Instruction Issued: " << Index << '\n';
    for (const std::pair<ResourceRef, unsigned> &Resource : Used) {
      dbgs() << "[E] Resource Used: [" << Resource.first.first << '.'
             << Resource.first.second << "]\n";
      dbgs() << "           cycles: " << Resource.second << '\n';
    }
  });
  Owner->notifyInstructionEvent(HWInstructionIssuedEvent(Index, Used));
}

void Scheduler::notifyInstructionExecuted(unsigned Index) {
  LSU->onInstructionExecuted(Index);
  DEBUG(dbgs() << "[E] Instruction Executed: " << Index << '\n');
  Owner->notifyInstructionEvent(
      HWInstructionEvent(HWInstructionEvent::Executed, Index));

  const Instruction &IS = Owner->getInstruction(Index);
  DU->onInstructionExecuted(IS.getRCUTokenID());
}

void Scheduler::notifyInstructionReady(unsigned Index) {
  DEBUG(dbgs() << "[E] Instruction Ready: " << Index << '\n');
  Owner->notifyInstructionEvent(
      HWInstructionEvent(HWInstructionEvent::Ready, Index));
}

void Scheduler::notifyResourceAvailable(const ResourceRef &RR) {
  Owner->notifyResourceAvailable(RR);
}

void Scheduler::notifyReservedBuffers(ArrayRef<uint64_t> Buffers) {
  SmallVector<unsigned, 4> BufferIDs(Buffers.begin(), Buffers.end());
  std::transform(
      Buffers.begin(), Buffers.end(), BufferIDs.begin(),
      [&](uint64_t Op) { return Resources->resolveResourceMask(Op); });
  Owner->notifyReservedBuffers(BufferIDs);
}

void Scheduler::notifyReleasedBuffers(ArrayRef<uint64_t> Buffers) {
  SmallVector<unsigned, 4> BufferIDs(Buffers.begin(), Buffers.end());
  std::transform(
      Buffers.begin(), Buffers.end(), BufferIDs.begin(),
      [&](uint64_t Op) { return Resources->resolveResourceMask(Op); });
  Owner->notifyReleasedBuffers(BufferIDs);
}
} // namespace mca
