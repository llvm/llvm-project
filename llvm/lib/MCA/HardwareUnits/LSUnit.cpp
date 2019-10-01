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
#include "llvm/MCA/Instruction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "llvm-mca"

namespace llvm {
namespace mca {

LSUnitBase::LSUnitBase(const MCSchedModel &SM, unsigned LQ, unsigned SQ,
                       bool AssumeNoAlias)
    : LQSize(LQ), SQSize(SQ), UsedLQEntries(0), UsedSQEntries(0),
      NoAlias(AssumeNoAlias), NextGroupID(1) {
  if (SM.hasExtraProcessorInfo()) {
    const MCExtraProcessorInfo &EPI = SM.getExtraProcessorInfo();
    if (!LQSize && EPI.LoadQueueID) {
      const MCProcResourceDesc &LdQDesc = *SM.getProcResource(EPI.LoadQueueID);
      LQSize = LdQDesc.BufferSize;
    }

    if (!SQSize && EPI.StoreQueueID) {
      const MCProcResourceDesc &StQDesc = *SM.getProcResource(EPI.StoreQueueID);
      SQSize = StQDesc.BufferSize;
    }
  }
}

LSUnitBase::~LSUnitBase() {}

void LSUnitBase::cycleEvent() {
  for (const std::pair<unsigned, std::unique_ptr<MemoryGroup>> &G : Groups)
    G.second->cycleEvent();
}

#ifndef NDEBUG
void LSUnitBase::dump() const {
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

unsigned LSUnit::dispatch(const InstRef &IR) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  unsigned IsMemBarrier = Desc.HasSideEffects;
  assert((Desc.MayLoad || Desc.MayStore) && "Not a memory operation!");

  if (Desc.MayLoad)
    assignLQSlot();
  if (Desc.MayStore)
    assignSQSlot();

  if (Desc.MayStore) {
    // Always create a new group for store operations.

    // A store may not pass a previous store or store barrier.
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
      IDom.addSuccessor(&NewGroup);
    }
    if (CurrentStoreGroupID) {
      MemoryGroup &StoreGroup = getGroup(CurrentStoreGroupID);
      LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: (" << CurrentStoreGroupID
                        << ") --> (" << NewGID << ")\n");
      StoreGroup.addSuccessor(&NewGroup);
    }

    CurrentStoreGroupID = NewGID;
    if (Desc.MayLoad) {
      CurrentLoadGroupID = NewGID;
      if (IsMemBarrier)
        CurrentLoadBarrierGroupID = NewGID;
    }

    return NewGID;
  }

  assert(Desc.MayLoad && "Expected a load!");

  // Always create a new memory group if this is the first load of the sequence.

  // A load may not pass a previous store unless flag 'NoAlias' is set.
  // A load may pass a previous load.
  // A younger load cannot pass a older load barrier.
  // A load barrier cannot pass a older load.
  bool ShouldCreateANewGroup = !CurrentLoadGroupID || IsMemBarrier ||
                               CurrentLoadGroupID <= CurrentStoreGroupID ||
                               CurrentLoadGroupID <= CurrentLoadBarrierGroupID;
  if (ShouldCreateANewGroup) {
    unsigned NewGID = createMemoryGroup();
    MemoryGroup &NewGroup = getGroup(NewGID);
    NewGroup.addInstruction();

    if (!assumeNoAlias() && CurrentStoreGroupID) {
      MemoryGroup &StGroup = getGroup(CurrentStoreGroupID);
      LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: (" << CurrentStoreGroupID
                        << ") --> (" << NewGID << ")\n");
      StGroup.addSuccessor(&NewGroup);
    }
    if (CurrentLoadBarrierGroupID) {
      MemoryGroup &LdGroup = getGroup(CurrentLoadBarrierGroupID);
      LLVM_DEBUG(dbgs() << "[LSUnit]: GROUP DEP: (" << CurrentLoadBarrierGroupID
                        << ") --> (" << NewGID << ")\n");
      LdGroup.addSuccessor(&NewGroup);
    }

    CurrentLoadGroupID = NewGID;
    if (IsMemBarrier)
      CurrentLoadBarrierGroupID = NewGID;
    return NewGID;
  }

  MemoryGroup &Group = getGroup(CurrentLoadGroupID);
  Group.addInstruction();
  return CurrentLoadGroupID;
}

LSUnit::Status LSUnit::isAvailable(const InstRef &IR) const {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  if (Desc.MayLoad && isLQFull())
    return LSUnit::LSU_LQUEUE_FULL;
  if (Desc.MayStore && isSQFull())
    return LSUnit::LSU_SQUEUE_FULL;
  return LSUnit::LSU_AVAILABLE;
}

void LSUnitBase::onInstructionExecuted(const InstRef &IR) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  bool IsALoad = Desc.MayLoad;
  bool IsAStore = Desc.MayStore;
  assert((IsALoad || IsAStore) && "Expected a memory operation!");

  unsigned GroupID = IR.getInstruction()->getLSUTokenID();
  auto It = Groups.find(GroupID);
  It->second->onInstructionExecuted();
  if (It->second->isExecuted()) {
    Groups.erase(It);
  }

  if (IsALoad) {
    UsedLQEntries--;
    LLVM_DEBUG(dbgs() << "[LSUnit]: Instruction idx=" << IR.getSourceIndex()
                      << " has been removed from the load queue.\n");
  }

  if (IsAStore) {
    UsedSQEntries--;
    LLVM_DEBUG(dbgs() << "[LSUnit]: Instruction idx=" << IR.getSourceIndex()
                      << " has been removed from the store queue.\n");
  }
}

void LSUnit::onInstructionExecuted(const InstRef &IR) {
  const Instruction &IS = *IR.getInstruction();
  if (!IS.isMemOp())
    return;

  LSUnitBase::onInstructionExecuted(IR);
  unsigned GroupID = IS.getLSUTokenID();
  if (!isValidGroupID(GroupID)) {
    if (GroupID == CurrentLoadGroupID)
      CurrentLoadGroupID = 0;
    if (GroupID == CurrentStoreGroupID)
      CurrentStoreGroupID = 0;
    if (GroupID == CurrentLoadBarrierGroupID)
      CurrentLoadBarrierGroupID = 0;
  }
}

} // namespace mca
} // namespace llvm
