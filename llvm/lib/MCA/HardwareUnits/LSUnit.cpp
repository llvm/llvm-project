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
    : LQSize(LQ), SQSize(SQ), NoAlias(AssumeNoAlias) {
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

#ifndef NDEBUG
void LSUnit::dump() const {
  dbgs() << "[LSUnit] LQ_Size = " << getLoadQueueSize() << '\n';
  dbgs() << "[LSUnit] SQ_Size = " << getStoreQueueSize() << '\n';
  dbgs() << "[LSUnit] NextLQSlotIdx = " << LoadQueue.size() << '\n';
  dbgs() << "[LSUnit] NextSQSlotIdx = " << StoreQueue.size() << '\n';
}
#endif

void LSUnit::assignLQSlot(const InstRef &IR) {
  assert(!isLQFull() && "Load Queue is full!");

  LLVM_DEBUG(dbgs() << "[LSUnit] - AssignLQSlot <Idx=" << IR.getSourceIndex()
                    << ",slot=" << LoadQueue.size() << ">\n");
  LoadQueue.insert(IR);
}

void LSUnit::assignSQSlot(const InstRef &IR) {
  assert(!isSQFull() && "Store Queue is full!");

  LLVM_DEBUG(dbgs() << "[LSUnit] - AssignSQSlot <Idx=" << IR.getSourceIndex()
                    << ",slot=" << StoreQueue.size() << ">\n");
  StoreQueue.insert(IR);
}

void LSUnit::dispatch(const InstRef &IR) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  unsigned IsMemBarrier = Desc.HasSideEffects;
  assert((Desc.MayLoad || Desc.MayStore) && "Not a memory operation!");

  if (Desc.MayLoad) {
    if (IsMemBarrier)
      LoadBarriers.insert(IR);
    assignLQSlot(IR);
  }

  if (Desc.MayStore) {
    if (IsMemBarrier)
      StoreBarriers.insert(IR);
    assignSQSlot(IR);
  }
}

LSUnit::Status LSUnit::isAvailable(const InstRef &IR) const {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  if (Desc.MayLoad && isLQFull())
    return LSUnit::LSU_LQUEUE_FULL;
  if (Desc.MayStore && isSQFull())
    return LSUnit::LSU_SQUEUE_FULL;
  return LSUnit::LSU_AVAILABLE;
}

const InstRef &LSUnit::isReady(const InstRef &IR) const {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  const unsigned Index = IR.getSourceIndex();
  bool IsALoad = Desc.MayLoad;
  bool IsAStore = Desc.MayStore;
  assert((IsALoad || IsAStore) && "Not a memory operation!");

  if (IsALoad && !LoadBarriers.empty()) {
    const InstRef &LoadBarrier = *LoadBarriers.begin();
    // A younger load cannot pass a older load barrier.
    if (Index > LoadBarrier.getSourceIndex())
      return LoadBarrier;
    // A load barrier cannot pass a older load.
    if (Index == LoadBarrier.getSourceIndex()) {
      const InstRef &Load = *LoadQueue.begin();
      if (Index != Load.getSourceIndex())
        return Load;
    }
  }

  if (IsAStore && !StoreBarriers.empty()) {
    const InstRef &StoreBarrier = *StoreBarriers.begin();
    // A younger store cannot pass a older store barrier.
    if (Index > StoreBarrier.getSourceIndex())
      return StoreBarrier;
    // A store barrier cannot pass a older store.
    if (Index == StoreBarrier.getSourceIndex()) {
      const InstRef &Store = *StoreQueue.begin();
      if (Index != Store.getSourceIndex())
        return Store;
    }
  }

  // A load may not pass a previous store unless flag 'NoAlias' is set.
  // A load may pass a previous load.
  if (assumeNoAlias() && IsALoad)
    return IR;

  if (StoreQueue.size()) {
    // A load may not pass a previous store.
    // A store may not pass a previous store.
    const InstRef &Store = *StoreQueue.begin();
    if (Index > Store.getSourceIndex())
      return Store;
  }

  // Okay, we are older than the oldest store in the queue.
  if (isLQEmpty())
    return IR;

  // Check if there are no older loads.
  const InstRef &Load = *LoadQueue.begin();
  if (Index <= Load.getSourceIndex())
    return IR;

  // A load may pass a previous load.
  if (IsALoad)
    return IR;

  // A store may not pass a previous load.
  return Load;
}

void LSUnit::onInstructionExecuted(const InstRef &IR) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  const unsigned Index = IR.getSourceIndex();
  bool IsALoad = Desc.MayLoad;
  bool IsAStore = Desc.MayStore;

  if (IsALoad) {
    if (LoadQueue.erase(IR)) {
      LLVM_DEBUG(dbgs() << "[LSUnit]: Instruction idx=" << Index
                        << " has been removed from the load queue.\n");
    }
    if (!LoadBarriers.empty()) {
      const InstRef &LoadBarrier = *LoadBarriers.begin();
      if (Index == LoadBarrier.getSourceIndex()) {
        LLVM_DEBUG(
            dbgs() << "[LSUnit]: Instruction idx=" << Index
                   << " has been removed from the set of load barriers.\n");
        LoadBarriers.erase(IR);
      }
    }
  }

  if (IsAStore) {
    if (StoreQueue.erase(IR)) {
      LLVM_DEBUG(dbgs() << "[LSUnit]: Instruction idx=" << Index
                        << " has been removed from the store queue.\n");
    }

    if (!StoreBarriers.empty()) {
      const InstRef &StoreBarrier = *StoreBarriers.begin();
      if (Index == StoreBarrier.getSourceIndex()) {
        LLVM_DEBUG(
            dbgs() << "[LSUnit]: Instruction idx=" << Index
                   << " has been removed from the set of store barriers.\n");
        StoreBarriers.erase(IR);
      }
    }
  }
}

} // namespace mca
} // namespace llvm
