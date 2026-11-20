//===----------------------- LSUnit.cpp --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// A Load-Store Unit for the llvm-mca tool.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/HardwareUnits/LSUnit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "llvm-mca"

namespace llvm {
namespace mca {

LSUnitBase::LSUnitBase(const MCSchedModel &SM, unsigned LQ, unsigned SQ,
                       bool AssumeNoAlias)
    : LQSize(LQ), SQSize(SQ), UsedLQEntries(0), UsedSQEntries(0),
      NoAlias(AssumeNoAlias) {
  if (SM.hasExtraProcessorInfo()) {
    const MCExtraProcessorInfo &EPI = SM.getExtraProcessorInfo();
    if (!LQSize && EPI.LoadQueueID) {
      const MCProcResourceDesc &LdQDesc = *SM.getProcResource(EPI.LoadQueueID);
      LQSize = std::max(0, LdQDesc.BufferSize);
    }

    if (!SQSize && EPI.StoreQueueID) {
      const MCProcResourceDesc &StQDesc = *SM.getProcResource(EPI.StoreQueueID);
      SQSize = std::max(0, StQDesc.BufferSize);
    }
  }
}

LSUnitBase::~LSUnitBase() = default;

void LSUnit::cycleEvent() {
  for (const auto &G : Groups)
    G.second->cycleEvent();
}

#ifndef NDEBUG
void LSUnit::dump() const {
  dbgs() << "[LSUnit] LQ_Size = " << getLoadQueueSize() << '\n';
  dbgs() << "[LSUnit] SQ_Size = " << getStoreQueueSize() << '\n';
  dbgs() << "[LSUnit] NextLQSlotIdx = " << getUsedLQEntries() << '\n';
  dbgs() << "[LSUnit] NextSQSlotIdx = " << getUsedSQEntries() << '\n';
  dbgs() << "\n";
  for (const auto &GroupIt : Groups) {
    const MemoryGroup &Group = *GroupIt.second;
    dbgs() << "[LSUnit] Group (" << GroupIt.first << "): "
           << "[ #Preds = " << Group.getNumPredecessors()
           << ", #GIssued = " << Group.getNumExecutingPredecessors()
           << ", #GExecuted = " << Group.getNumExecutedPredecessors()
           << ", #Inst = " << Group.getNumInstructions()
           << ", #IIssued = " << Group.getNumExecuting()
           << ", #IExecuted = " << Group.getNumExecuted() << '\n';
  }
}
#endif

std::optional<LSUnit::ResolvedStackSlot>
LSUnit::resolveStackAccess(const StackAccessInfo &Access) const {
  if (!Access.isValid())
    return std::nullopt;
  if (Access.Base == StackAccessInfo::BaseKind::SP) {
    int64_t Start = SPDelta + Access.Offset;
    return ResolvedStackSlot{SPSpaceID, Start, Start + Access.Size};
  }
  if (Access.Base == StackAccessInfo::BaseKind::FP) {
    int64_t Start = FPDelta + Access.Offset;
    return ResolvedStackSlot{FPSpaceID, Start, Start + Access.Size};
  }
  return std::nullopt;
}

void LSUnit::applyStackAdjustments(const Instruction &IS) {
  const StackPointerAdjustment &SPAdj = IS.getSPAdjust();
  const StackPointerAdjustment &FPAdj = IS.getFPAdjust();
  if (SPAdj.isNone() && FPAdj.isNone())
    return;

  unsigned OldSPSpace = SPSpaceID;
  int64_t OldSPDelta = SPDelta;
  unsigned OldFPSpace = FPSpaceID;
  int64_t OldFPDelta = FPDelta;

  switch (SPAdj.AdjustKind) {
  case StackPointerAdjustment::Kind::None:
    break;
  case StackPointerAdjustment::Kind::Delta:
    SPDelta = OldSPDelta + SPAdj.Delta;
    break;
  case StackPointerAdjustment::Kind::FromOther:
    SPSpaceID = OldFPSpace;
    SPDelta = OldFPDelta + SPAdj.Delta;
    break;
  case StackPointerAdjustment::Kind::Unknown:
    SPSpaceID = NextSpaceID++;
    SPDelta = 0;
    break;
  }

  switch (FPAdj.AdjustKind) {
  case StackPointerAdjustment::Kind::None:
    break;
  case StackPointerAdjustment::Kind::Delta:
    FPDelta = OldFPDelta + FPAdj.Delta;
    break;
  case StackPointerAdjustment::Kind::FromOther:
    FPSpaceID = OldSPSpace;
    FPDelta = OldSPDelta + FPAdj.Delta;
    break;
  case StackPointerAdjustment::Kind::Unknown:
    FPSpaceID = NextSpaceID++;
    FPDelta = 0;
    break;
  }
}

void LSUnit::onInstructionDispatched(const InstRef &IR) {
  applyStackAdjustments(*IR.getInstruction());
}

unsigned LSUnit::dispatch(const InstRef &IR) {
  const Instruction &IS = *IR.getInstruction();
  bool IsStoreBarrier = IS.isAStoreBarrier();
  bool IsLoadBarrier = IS.isALoadBarrier();
  assert((IS.getMayLoad() || IS.getMayStore()) && "Not a memory operation!");

  std::optional<ResolvedStackSlot> Slot =
      resolveStackAccess(IS.getStackAccess());

  if (IS.getMayLoad())
    acquireLQSlot();
  if (IS.getMayStore())
    acquireSQSlot();

  if (IS.getMayStore()) {
    unsigned NewGID = createMemoryGroup();
    MemoryGroup &NewGroup = getGroup(NewGID);
    NewGroup.addInstruction();

    // A store may not pass a previous load or load barrier.
    unsigned ImmediateLoadDominator =
        std::max(CurrentLoadGroupID, CurrentLoadBarrierGroupID);
    if (ImmediateLoadDominator) {
      MemoryGroup &IDom = getGroup(ImmediateLoadDominator);
      LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: (" << ImmediateLoadDominator
                        << ") --> (" << NewGID << ")\n");
      IDom.addSuccessor(&NewGroup, !assumeNoAlias());
    }

    // A store may not pass a previous store barrier.
    if (CurrentStoreBarrierGroupID) {
      MemoryGroup &StoreGroup = getGroup(CurrentStoreBarrierGroupID);
      LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: ("
                        << CurrentStoreBarrierGroupID << ") --> (" << NewGID
                        << ")\n");
      StoreGroup.addSuccessor(&NewGroup, true);
    }

    // A store may not pass a previous store.
    if (CurrentStoreGroupID &&
        (CurrentStoreGroupID != CurrentStoreBarrierGroupID)) {
      MemoryGroup &StoreGroup = getGroup(CurrentStoreGroupID);
      LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: (" << CurrentStoreGroupID
                        << ") --> (" << NewGID << ")\n");
      StoreGroup.addSuccessor(&NewGroup, !assumeNoAlias());
    }

    // For a read-modify-write accessing a known stack slot under -noalias=true,
    // enforce dependencies on overlapping in-flight stack stores (especially
    // prior RMW stores whose written data is only ready at execute).
    if (IS.getMayLoad() && Slot && assumeNoAlias()) {
      for (const StackSlotRecord &Rec : ActiveStackStores) {
        if (!Slot->overlaps(Rec.Slot))
          continue;
        MemoryGroup &StoreGroup = getGroup(Rec.GroupID);
        MemoryGroup::DepKind Kind = Rec.IsRMW ? MemoryGroup::DepKind::Data
                                              : MemoryGroup::DepKind::Forwarding;
        LLVM_DEBUG(dbgs() << "[LSUnit]: STACK RMW DEP: (" << Rec.GroupID
                          << ") --> (" << NewGID << ")\n");
        StoreGroup.addSuccessor(&NewGroup, Kind);
      }
    }

    if (Slot)
      ActiveStackStores.push_back(
          StackSlotRecord{NewGID, *Slot, IS.getMayLoad()});

    CurrentStoreGroupID = NewGID;
    if (IsStoreBarrier)
      CurrentStoreBarrierGroupID = NewGID;

    if (IS.getMayLoad()) {
      CurrentLoadGroupID = NewGID;
      if (IsLoadBarrier)
        CurrentLoadBarrierGroupID = NewGID;
    }

    applyStackAdjustments(IS);
    return NewGID;
  }

  assert(IS.getMayLoad() && "Expected a load!");

  SmallVector<const StackSlotRecord *, 4> OverlappingStores;
  if (assumeNoAlias() && Slot) {
    for (const StackSlotRecord &Rec : ActiveStackStores) {
      if (!Slot->overlaps(Rec.Slot))
        continue;
      if (Rec.IsRMW || !getGroup(Rec.GroupID).isExecuting())
        OverlappingStores.push_back(&Rec);
    }
  }

  unsigned ImmediateLoadDominator =
      std::max(CurrentLoadGroupID, CurrentLoadBarrierGroupID);

  // Do not coalesce a load into a previous load group that already has
  // stack-store dependencies under -noalias=true.
  bool PrevLoadHasStackDep =
      ImmediateLoadDominator && assumeNoAlias() &&
      getGroup(ImmediateLoadDominator).getNumPredecessors() > 0;

  // A new load group is created if we are in one of the following situations:
  // 1) This is a load barrier (by construction, a load barrier is always
  //    assigned to a different memory group).
  // 2) There is no load in flight (by construction we always keep loads and
  //    stores into separate memory groups).
  // 3) There is a load barrier in flight. This load depends on it.
  // 4) There is an intervening store between the last load dispatched to the
  //    LSU and this load. We always create a new group even if this load
  //    does not alias the last dispatched store.
  // 5) There is no intervening store and there is an active load group.
  //    However that group has already started execution, so we cannot add
  //    this load to it.
  // 6) This load aliases an in-flight stack store, or the active load group
  //    already carries a stack-store dependency.
  bool ShouldCreateANewGroup =
      IsLoadBarrier || !ImmediateLoadDominator ||
      CurrentLoadBarrierGroupID == ImmediateLoadDominator ||
      ImmediateLoadDominator <= CurrentStoreGroupID ||
      getGroup(ImmediateLoadDominator).isExecuting() ||
      !OverlappingStores.empty() || PrevLoadHasStackDep;

  if (ShouldCreateANewGroup) {
    unsigned NewGID = createMemoryGroup();
    MemoryGroup &NewGroup = getGroup(NewGID);
    NewGroup.addInstruction();

    // A load may not pass a previous store or store barrier
    // unless flag 'NoAlias' is set.
    if (!assumeNoAlias() && CurrentStoreGroupID) {
      MemoryGroup &StoreGroup = getGroup(CurrentStoreGroupID);
      LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: (" << CurrentStoreGroupID
                        << ") --> (" << NewGID << ")\n");
      StoreGroup.addSuccessor(&NewGroup, true);
    } else if (!OverlappingStores.empty()) {
      for (const StackSlotRecord *Rec : OverlappingStores) {
        MemoryGroup &StoreGroup = getGroup(Rec->GroupID);
        MemoryGroup::DepKind Kind = Rec->IsRMW
                                        ? MemoryGroup::DepKind::Data
                                        : MemoryGroup::DepKind::Forwarding;
        LLVM_DEBUG(dbgs() << "[LSUnit]: STACK DEP: (" << Rec->GroupID
                          << ") --> (" << NewGID << ")\n");
        StoreGroup.addSuccessor(&NewGroup, Kind);
      }
    }

    // A load barrier may not pass a previous load or load barrier.
    if (IsLoadBarrier) {
      if (ImmediateLoadDominator) {
        MemoryGroup &LoadGroup = getGroup(ImmediateLoadDominator);
        LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: (" << ImmediateLoadDominator
                          << ") --> (" << NewGID << ")\n");
        LoadGroup.addSuccessor(&NewGroup, true);
      }
    } else {
      // A younger load cannot pass a older load barrier.
      if (CurrentLoadBarrierGroupID) {
        MemoryGroup &LoadGroup = getGroup(CurrentLoadBarrierGroupID);
        LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: ("
                          << CurrentLoadBarrierGroupID << ") --> (" << NewGID
                          << ")\n");
        LoadGroup.addSuccessor(&NewGroup, true);
      }
    }

    CurrentLoadGroupID = NewGID;
    if (IsLoadBarrier)
      CurrentLoadBarrierGroupID = NewGID;
    applyStackAdjustments(IS);
    return NewGID;
  }

  // A load may pass a previous load.
  MemoryGroup &Group = getGroup(CurrentLoadGroupID);
  Group.addInstruction();
  applyStackAdjustments(IS);
  return CurrentLoadGroupID;
}

LSUnit::Status LSUnit::isAvailable(const InstRef &IR) const {
  const Instruction &IS = *IR.getInstruction();
  if (IS.getMayLoad() && isLQFull())
    return LSUnit::LSU_LQUEUE_FULL;
  if (IS.getMayStore() && isSQFull())
    return LSUnit::LSU_SQUEUE_FULL;
  return LSUnit::LSU_AVAILABLE;
}

void LSUnit::onInstructionRetired(const InstRef &IR) {
  const Instruction &IS = *IR.getInstruction();
  bool IsALoad = IS.getMayLoad();
  bool IsAStore = IS.getMayStore();
  assert((IsALoad || IsAStore) && "Expected a memory operation!");

  if (IsALoad) {
    releaseLQSlot();
    LLVM_DEBUG(dbgs() << "[LSUnit]: Instruction idx=" << IR.getSourceIndex()
                      << " has been removed from the load queue.\n");
  }

  if (IsAStore) {
    releaseSQSlot();
    LLVM_DEBUG(dbgs() << "[LSUnit]: Instruction idx=" << IR.getSourceIndex()
                      << " has been removed from the store queue.\n");
  }
}

void LSUnit::onInstructionExecuted(const InstRef &IR) {
  const Instruction &IS = *IR.getInstruction();
  if (!IS.isMemOp())
    return;

  unsigned GroupID = IS.getLSUTokenID();
  auto It = Groups.find(GroupID);
  assert(It != Groups.end() && "Instruction not dispatched to the LS unit");
  It->second->onInstructionExecuted(IR);
  if (It->second->isExecuted())
    Groups.erase(It);

  if (!isValidGroupID(GroupID)) {
    if (GroupID == CurrentLoadGroupID)
      CurrentLoadGroupID = 0;
    if (GroupID == CurrentStoreGroupID)
      CurrentStoreGroupID = 0;
    if (GroupID == CurrentLoadBarrierGroupID)
      CurrentLoadBarrierGroupID = 0;
    if (GroupID == CurrentStoreBarrierGroupID)
      CurrentStoreBarrierGroupID = 0;
    llvm::erase_if(ActiveStackStores, [GroupID](const StackSlotRecord &Rec) {
      return Rec.GroupID == GroupID;
    });
  }
}

} // namespace mca
} // namespace llvm
