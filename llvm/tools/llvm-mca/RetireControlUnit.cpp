//===---------------------- RetireControlUnit.cpp ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file simulates the hardware responsible for retiring instructions.
///
//===----------------------------------------------------------------------===//

#include "RetireControlUnit.h"
#include "DispatchStage.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "llvm-mca"

namespace mca {

RetireControlUnit::RetireControlUnit(const llvm::MCSchedModel &SM)
    : NextAvailableSlotIdx(0), CurrentInstructionSlotIdx(0),
      AvailableSlots(SM.MicroOpBufferSize), MaxRetirePerCycle(0) {
  // Check if the scheduling model provides extra information about the machine
  // processor. If so, then use that information to set the reorder buffer size
  // and the maximum number of instructions retired per cycle.
  if (SM.hasExtraProcessorInfo()) {
    const MCExtraProcessorInfo &EPI = SM.getExtraProcessorInfo();
    if (EPI.ReorderBufferSize)
      AvailableSlots = EPI.ReorderBufferSize;
    MaxRetirePerCycle = EPI.MaxRetirePerCycle;
  }

  assert(AvailableSlots && "Invalid reorder buffer size!");
  Queue.resize(AvailableSlots);
}

// Reserves a number of slots, and returns a new token.
unsigned RetireControlUnit::reserveSlot(const InstRef &IR,
                                        unsigned NumMicroOps) {
  assert(isAvailable(NumMicroOps));
  unsigned NormalizedQuantity =
      std::min(NumMicroOps, static_cast<unsigned>(Queue.size()));
  // Zero latency instructions may have zero mOps. Artificially bump this
  // value to 1. Although zero latency instructions don't consume scheduler
  // resources, they still consume one slot in the retire queue.
  NormalizedQuantity = std::max(NormalizedQuantity, 1U);
  unsigned TokenID = NextAvailableSlotIdx;
  Queue[NextAvailableSlotIdx] = {IR, NormalizedQuantity, false};
  NextAvailableSlotIdx += NormalizedQuantity;
  NextAvailableSlotIdx %= Queue.size();
  AvailableSlots -= NormalizedQuantity;
  return TokenID;
}

const RetireControlUnit::RUToken &RetireControlUnit::peekCurrentToken() const {
  return Queue[CurrentInstructionSlotIdx];
}

void RetireControlUnit::consumeCurrentToken() {
  const RetireControlUnit::RUToken &Current = peekCurrentToken();
  assert(Current.NumSlots && "Reserved zero slots?");
  assert(Current.IR.isValid() && "Invalid RUToken in the RCU queue.");

  // Update the slot index to be the next item in the circular queue.
  CurrentInstructionSlotIdx += Current.NumSlots;
  CurrentInstructionSlotIdx %= Queue.size();
  AvailableSlots += Current.NumSlots;
}

void RetireControlUnit::onInstructionExecuted(unsigned TokenID) {
  assert(Queue.size() > TokenID);
  assert(Queue[TokenID].Executed == false && Queue[TokenID].IR.isValid());
  Queue[TokenID].Executed = true;
}

#ifndef NDEBUG
void RetireControlUnit::dump() const {
  dbgs() << "Retire Unit: { Total Slots=" << Queue.size()
         << ", Available Slots=" << AvailableSlots << " }\n";
}
#endif

} // namespace mca
