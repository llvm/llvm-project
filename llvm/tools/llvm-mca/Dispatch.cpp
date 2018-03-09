//===--------------------- Dispatch.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods declared by class RegisterFile, DispatchUnit
/// and RetireControlUnit.
///
//===----------------------------------------------------------------------===//

#include "Dispatch.h"
#include "Backend.h"
#include "Scheduler.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "llvm-mca"

namespace mca {

void RegisterFile::addRegisterMapping(WriteState &WS) {
  unsigned RegID = WS.getRegisterID();
  assert(RegID && "Adding an invalid register definition?");

  RegisterMappings[RegID] = &WS;
  for (MCSubRegIterator I(RegID, &MRI); I.isValid(); ++I)
    RegisterMappings[*I] = &WS;
  if (MaxUsedMappings == NumUsedMappings)
    MaxUsedMappings++;
  NumUsedMappings++;
  TotalMappingsCreated++;
  // If this is a partial update, then we are done.
  if (!WS.fullyUpdatesSuperRegs())
    return;

  for (MCSuperRegIterator I(RegID, &MRI); I.isValid(); ++I)
    RegisterMappings[*I] = &WS;
}

void RegisterFile::invalidateRegisterMapping(const WriteState &WS) {
  unsigned RegID = WS.getRegisterID();
  bool ShouldInvalidateSuperRegs = WS.fullyUpdatesSuperRegs();

  assert(RegID != 0 && "Invalidating an already invalid register?");
  assert(WS.getCyclesLeft() != -512 &&
         "Invalidating a write of unknown cycles!");
  assert(WS.getCyclesLeft() <= 0 && "Invalid cycles left for this write!");
  if (!RegisterMappings[RegID])
    return;

  assert(NumUsedMappings);
  NumUsedMappings--;

  if (RegisterMappings[RegID] == &WS)
    RegisterMappings[RegID] = nullptr;

  for (MCSubRegIterator I(RegID, &MRI); I.isValid(); ++I)
    if (RegisterMappings[*I] == &WS)
      RegisterMappings[*I] = nullptr;

  if (!ShouldInvalidateSuperRegs)
    return;

  for (MCSuperRegIterator I(RegID, &MRI); I.isValid(); ++I)
    if (RegisterMappings[*I] == &WS)
      RegisterMappings[*I] = nullptr;
}

// Update the number of used mappings in the event of instruction retired.
// This mehod delegates to the register file the task of invalidating
// register mappings that were created for instruction IS.
void DispatchUnit::invalidateRegisterMappings(const Instruction &IS) {
  for (const std::unique_ptr<WriteState> &WS : IS.getDefs()) {
    DEBUG(dbgs() << "[RAT] Invalidating mapping for: ");
    DEBUG(WS->dump());
    RAT->invalidateRegisterMapping(*WS.get());
  }
}

void RegisterFile::collectWrites(SmallVectorImpl<WriteState *> &Writes,
                                 unsigned RegID) const {
  assert(RegID && RegID < RegisterMappings.size());
  WriteState *WS = RegisterMappings[RegID];
  if (WS) {
    DEBUG(dbgs() << "Found a dependent use of RegID=" << RegID << '\n');
    Writes.push_back(WS);
  }

  // Handle potential partial register updates.
  for (MCSubRegIterator I(RegID, &MRI); I.isValid(); ++I) {
    WS = RegisterMappings[*I];
    if (WS && std::find(Writes.begin(), Writes.end(), WS) == Writes.end()) {
      DEBUG(dbgs() << "Found a dependent use of subReg " << *I << " (part of "
                   << RegID << ")\n");
      Writes.push_back(WS);
    }
  }
}

bool RegisterFile::isAvailable(unsigned NumRegWrites) {
  if (!TotalMappings)
    return true;
  if (NumRegWrites > TotalMappings) {
    // The user specified a too small number of registers.
    // Artificially set the number of temporaries to NumRegWrites.
    errs() << "warning: not enough temporaries in the register file. "
           << "The register file size has been automatically increased to "
           << NumRegWrites << '\n';
    TotalMappings = NumRegWrites;
  }

  return NumRegWrites + NumUsedMappings <= TotalMappings;
}

#ifndef NDEBUG
void RegisterFile::dump() const {
  for (unsigned I = 0, E = MRI.getNumRegs(); I < E; ++I)
    if (RegisterMappings[I]) {
      dbgs() << MRI.getName(I) << ", " << I << ", ";
      RegisterMappings[I]->dump();
    }

  dbgs() << "TotalMappingsCreated: " << TotalMappingsCreated
         << ", MaxUsedMappings: " << MaxUsedMappings
         << ", NumUsedMappings: " << NumUsedMappings << '\n';
}
#endif

// Reserves a number of slots, and returns a new token.
unsigned RetireControlUnit::reserveSlot(unsigned Index, unsigned NumMicroOps) {
  assert(isAvailable(NumMicroOps));
  unsigned NormalizedQuantity =
      std::min(NumMicroOps, static_cast<unsigned>(Queue.size()));
  // Zero latency instructions may have zero mOps. Artificially bump this
  // value to 1. Although zero latency instructions don't consume scheduler
  // resources, they still consume one slot in the retire queue.
  NormalizedQuantity = std::max(NormalizedQuantity, 1U);
  unsigned TokenID = NextAvailableSlotIdx;
  Queue[NextAvailableSlotIdx] = {Index, NormalizedQuantity, false};
  NextAvailableSlotIdx += NormalizedQuantity;
  NextAvailableSlotIdx %= Queue.size();
  AvailableSlots -= NormalizedQuantity;
  return TokenID;
}

void DispatchUnit::notifyInstructionDispatched(unsigned Index) {
  Owner->notifyInstructionDispatched(Index);
}

void DispatchUnit::notifyInstructionRetired(unsigned Index) {
  Owner->notifyInstructionRetired(Index);
}

void RetireControlUnit::cycleEvent() {
  if (isEmpty())
    return;

  unsigned NumRetired = 0;
  while (!isEmpty()) {
    if (MaxRetirePerCycle != 0 && NumRetired == MaxRetirePerCycle)
      break;
    RUToken &Current = Queue[CurrentInstructionSlotIdx];
    assert(Current.NumSlots && "Reserved zero slots?");
    if (!Current.Executed)
      break;
    Owner->notifyInstructionRetired(Current.Index);
    CurrentInstructionSlotIdx += Current.NumSlots;
    CurrentInstructionSlotIdx %= Queue.size();
    AvailableSlots += Current.NumSlots;
    NumRetired++;
  }
}

void RetireControlUnit::onInstructionExecuted(unsigned TokenID) {
  assert(Queue.size() > TokenID);
  assert(Queue[TokenID].Executed == false && Queue[TokenID].Index != ~0U);
  Queue[TokenID].Executed = true;
}

#ifndef NDEBUG
void RetireControlUnit::dump() const {
  dbgs() << "Retire Unit: { Total Slots=" << Queue.size()
         << ", Available Slots=" << AvailableSlots << " }\n";
}
#endif

bool DispatchUnit::checkRAT(const InstrDesc &Desc) {
  unsigned NumWrites = Desc.Writes.size();
  if (RAT->isAvailable(NumWrites))
    return true;
  DispatchStalls[DS_RAT_REG_UNAVAILABLE]++;
  return false;
}

bool DispatchUnit::checkRCU(const InstrDesc &Desc) {
  unsigned NumMicroOps = Desc.NumMicroOps;
  if (RCU->isAvailable(NumMicroOps))
    return true;
  DispatchStalls[DS_RCU_TOKEN_UNAVAILABLE]++;
  return false;
}

bool DispatchUnit::checkScheduler(const InstrDesc &Desc) {
  // If this is a zero-latency instruction, then it bypasses
  // the scheduler.
  switch (SC->canBeDispatched(Desc)) {
  case Scheduler::HWS_AVAILABLE:
    return true;
  case Scheduler::HWS_QUEUE_UNAVAILABLE:
    DispatchStalls[DS_SQ_TOKEN_UNAVAILABLE]++;
    break;
  case Scheduler::HWS_LD_QUEUE_UNAVAILABLE:
    DispatchStalls[DS_LDQ_TOKEN_UNAVAILABLE]++;
    break;
  case Scheduler::HWS_ST_QUEUE_UNAVAILABLE:
    DispatchStalls[DS_STQ_TOKEN_UNAVAILABLE]++;
    break;
  case Scheduler::HWS_DISPATCH_GROUP_RESTRICTION:
    DispatchStalls[DS_DISPATCH_GROUP_RESTRICTION]++;
  }

  return false;
}

unsigned DispatchUnit::dispatch(unsigned IID, Instruction *NewInst) {
  assert(!CarryOver && "Cannot dispatch another instruction!");
  unsigned NumMicroOps = NewInst->getDesc().NumMicroOps;
  if (NumMicroOps > DispatchWidth) {
    assert(AvailableEntries == DispatchWidth);
    AvailableEntries = 0;
    CarryOver = NumMicroOps - DispatchWidth;
  } else {
    assert(AvailableEntries >= NumMicroOps);
    AvailableEntries -= NumMicroOps;
  }

  // Reserve slots in the RCU.
  unsigned RCUTokenID = RCU->reserveSlot(IID, NumMicroOps);
  NewInst->setRCUTokenID(RCUTokenID);
  Owner->notifyInstructionDispatched(IID);

  SC->scheduleInstruction(IID, NewInst);
  return RCUTokenID;
}

#ifndef NDEBUG
void DispatchUnit::dump() const {
  RAT->dump();
  RCU->dump();

  unsigned DSRAT = DispatchStalls[DS_RAT_REG_UNAVAILABLE];
  unsigned DSRCU = DispatchStalls[DS_RCU_TOKEN_UNAVAILABLE];
  unsigned DSSCHEDQ = DispatchStalls[DS_SQ_TOKEN_UNAVAILABLE];
  unsigned DSLQ = DispatchStalls[DS_LDQ_TOKEN_UNAVAILABLE];
  unsigned DSSQ = DispatchStalls[DS_STQ_TOKEN_UNAVAILABLE];

  dbgs() << "STALLS --- RAT: " << DSRAT << ", RCU: " << DSRCU
         << ", SCHED_QUEUE: " << DSSCHEDQ << ", LOAD_QUEUE: " << DSLQ
         << ", STORE_QUEUE: " << DSSQ << '\n';
}
#endif

} // namespace mca
