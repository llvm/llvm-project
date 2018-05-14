//===----------------------- LSUnit.cpp --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// A Load-Store Unit for the llvm-mca tool.
///
//===----------------------------------------------------------------------===//

#include "LSUnit.h"
#include "Instruction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "llvm-mca"

namespace mca {

#ifndef NDEBUG
void LSUnit::dump() const {
  dbgs() << "[LSUnit] LQ_Size = " << LQ_Size << '\n';
  dbgs() << "[LSUnit] SQ_Size = " << SQ_Size << '\n';
  dbgs() << "[LSUnit] NextLQSlotIdx = " << LoadQueue.size() << '\n';
  dbgs() << "[LSUnit] NextSQSlotIdx = " << StoreQueue.size() << '\n';
}
#endif

void LSUnit::assignLQSlot(unsigned Index) {
  assert(!isLQFull());
  assert(LoadQueue.count(Index) == 0);

  LLVM_DEBUG(dbgs() << "[LSUnit] - AssignLQSlot <Idx=" << Index
                    << ",slot=" << LoadQueue.size() << ">\n");
  LoadQueue.insert(Index);
}

void LSUnit::assignSQSlot(unsigned Index) {
  assert(!isSQFull());
  assert(StoreQueue.count(Index) == 0);

  LLVM_DEBUG(dbgs() << "[LSUnit] - AssignSQSlot <Idx=" << Index
                    << ",slot=" << StoreQueue.size() << ">\n");
  StoreQueue.insert(Index);
}

bool LSUnit::reserve(const InstRef &IR) {
  const InstrDesc Desc = IR.getInstruction()->getDesc();
  unsigned MayLoad = Desc.MayLoad;
  unsigned MayStore = Desc.MayStore;
  unsigned IsMemBarrier = Desc.HasSideEffects;
  if (!MayLoad && !MayStore)
    return false;

  const unsigned Index = IR.getSourceIndex();
  if (MayLoad) {
    if (IsMemBarrier)
      LoadBarriers.insert(Index);
    assignLQSlot(Index);
  }
  if (MayStore) {
    if (IsMemBarrier)
      StoreBarriers.insert(Index);
    assignSQSlot(Index);
  }
  return true;
}

bool LSUnit::isReady(const InstRef &IR) const {
  const unsigned Index = IR.getSourceIndex();
  bool IsALoad = LoadQueue.count(Index) != 0;
  bool IsAStore = StoreQueue.count(Index) != 0;
  assert((IsALoad || IsAStore) && "Instruction is not in queue!");

  unsigned LoadBarrierIndex = LoadBarriers.empty() ? 0 : *LoadBarriers.begin();
  unsigned StoreBarrierIndex =
      StoreBarriers.empty() ? 0 : *StoreBarriers.begin();

  if (IsALoad && LoadBarrierIndex) {
    if (Index > LoadBarrierIndex)
      return false;
    if (Index == LoadBarrierIndex && Index != *LoadQueue.begin())
      return false;
  }

  if (IsAStore && StoreBarrierIndex) {
    if (Index > StoreBarrierIndex)
      return false;
    if (Index == StoreBarrierIndex && Index != *StoreQueue.begin())
      return false;
  }

  if (NoAlias && IsALoad)
    return true;

  if (StoreQueue.size()) {
    // Check if this memory operation is younger than the older store.
    if (Index > *StoreQueue.begin())
      return false;
  }

  // Okay, we are older than the oldest store in the queue.
  // If there are no pending loads, then we can say for sure that this
  // instruction is ready.
  if (isLQEmpty())
    return true;

  // Check if there are no older loads.
  if (Index <= *LoadQueue.begin())
    return true;

  // There is at least one younger load.
  return !IsAStore;
}

void LSUnit::onInstructionExecuted(const InstRef &IR) {
  const unsigned Index = IR.getSourceIndex();
  std::set<unsigned>::iterator it = LoadQueue.find(Index);
  if (it != LoadQueue.end()) {
    LLVM_DEBUG(dbgs() << "[LSUnit]: Instruction idx=" << Index
                      << " has been removed from the load queue.\n");
    LoadQueue.erase(it);
  }

  it = StoreQueue.find(Index);
  if (it != StoreQueue.end()) {
    LLVM_DEBUG(dbgs() << "[LSUnit]: Instruction idx=" << Index
                      << " has been removed from the store queue.\n");
    StoreQueue.erase(it);
  }

  if (!StoreBarriers.empty() && Index == *StoreBarriers.begin())
    StoreBarriers.erase(StoreBarriers.begin());
  if (!LoadBarriers.empty() && Index == *LoadBarriers.begin())
    LoadBarriers.erase(LoadBarriers.begin());
}
} // namespace mca
