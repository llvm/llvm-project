//===--------------------- DispatchStage.cpp --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods declared by the DispatchStage class.
///
//===----------------------------------------------------------------------===//

#include "DispatchStage.h"
#include "Backend.h"
#include "HWEventListener.h"
#include "Scheduler.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "llvm-mca"

namespace mca {

void DispatchStage::notifyInstructionDispatched(const InstRef &IR,
                                                ArrayRef<unsigned> UsedRegs) {
  LLVM_DEBUG(dbgs() << "[E] Instruction Dispatched: " << IR << '\n');
  Owner->notifyInstructionEvent(HWInstructionDispatchedEvent(IR, UsedRegs));
}

void DispatchStage::notifyInstructionRetired(const InstRef &IR) {
  LLVM_DEBUG(dbgs() << "[E] Instruction Retired: " << IR << '\n');
  SmallVector<unsigned, 4> FreedRegs(RAT->getNumRegisterFiles());
  const InstrDesc &Desc = IR.getInstruction()->getDesc();

  for (const std::unique_ptr<WriteState> &WS : IR.getInstruction()->getDefs())
    RAT->removeRegisterWrite(*WS.get(), FreedRegs, !Desc.isZeroLatency());
  Owner->notifyInstructionEvent(HWInstructionRetiredEvent(IR, FreedRegs));
}

bool DispatchStage::checkRAT(const InstRef &IR) {
  SmallVector<unsigned, 4> RegDefs;
  for (const std::unique_ptr<WriteState> &RegDef :
       IR.getInstruction()->getDefs())
    RegDefs.emplace_back(RegDef->getRegisterID());

  unsigned RegisterMask = RAT->isAvailable(RegDefs);
  // A mask with all zeroes means: register files are available.
  if (RegisterMask) {
    Owner->notifyStallEvent(HWStallEvent(HWStallEvent::RegisterFileStall, IR));
    return false;
  }

  return true;
}

bool DispatchStage::checkRCU(const InstRef &IR) {
  const unsigned NumMicroOps = IR.getInstruction()->getDesc().NumMicroOps;
  if (RCU->isAvailable(NumMicroOps))
    return true;
  Owner->notifyStallEvent(
      HWStallEvent(HWStallEvent::RetireControlUnitStall, IR));
  return false;
}

bool DispatchStage::checkScheduler(const InstRef &IR) {
  return SC->canBeDispatched(IR);
}

void DispatchStage::updateRAWDependencies(ReadState &RS,
                                          const MCSubtargetInfo &STI) {
  SmallVector<WriteState *, 4> DependentWrites;

  collectWrites(DependentWrites, RS.getRegisterID());
  RS.setDependentWrites(DependentWrites.size());
  LLVM_DEBUG(dbgs() << "Found " << DependentWrites.size()
                    << " dependent writes\n");
  // We know that this read depends on all the writes in DependentWrites.
  // For each write, check if we have ReadAdvance information, and use it
  // to figure out in how many cycles this read becomes available.
  const ReadDescriptor &RD = RS.getDescriptor();
  if (!RD.HasReadAdvanceEntries) {
    for (WriteState *WS : DependentWrites)
      WS->addUser(&RS, /* ReadAdvance */ 0);
    return;
  }

  const MCSchedModel &SM = STI.getSchedModel();
  const MCSchedClassDesc *SC = SM.getSchedClassDesc(RD.SchedClassID);
  for (WriteState *WS : DependentWrites) {
    unsigned WriteResID = WS->getWriteResourceID();
    int ReadAdvance = STI.getReadAdvanceCycles(SC, RD.UseIndex, WriteResID);
    WS->addUser(&RS, ReadAdvance);
  }
  // Prepare the set for another round.
  DependentWrites.clear();
}

void DispatchStage::dispatch(InstRef IR) {
  assert(!CarryOver && "Cannot dispatch another instruction!");
  Instruction &IS = *IR.getInstruction();
  const InstrDesc &Desc = IS.getDesc();
  const unsigned NumMicroOps = Desc.NumMicroOps;
  if (NumMicroOps > DispatchWidth) {
    assert(AvailableEntries == DispatchWidth);
    AvailableEntries = 0;
    CarryOver = NumMicroOps - DispatchWidth;
  } else {
    assert(AvailableEntries >= NumMicroOps);
    AvailableEntries -= NumMicroOps;
  }

  // A dependency-breaking instruction doesn't have to wait on the register
  // input operands, and it is often optimized at register renaming stage.
  // Update RAW dependencies if this instruction is not a dependency-breaking
  // instruction. A dependency-breaking instruction is a zero-latency
  // instruction that doesn't consume hardware resources.
  // An example of dependency-breaking instruction on X86 is a zero-idiom XOR.
  if (!Desc.isZeroLatency())
    for (std::unique_ptr<ReadState> &RS : IS.getUses())
      updateRAWDependencies(*RS, STI);

  // By default, a dependency-breaking zero-latency instruction is expected to
  // be optimized at register renaming stage. That means, no physical register
  // is allocated to the instruction.
  SmallVector<unsigned, 4> RegisterFiles(RAT->getNumRegisterFiles());
  for (std::unique_ptr<WriteState> &WS : IS.getDefs())
    RAT->addRegisterWrite(*WS, RegisterFiles, !Desc.isZeroLatency());

  // Reserve slots in the RCU, and notify the instruction that it has been
  // dispatched to the schedulers for execution.
  IS.dispatch(RCU->reserveSlot(IR, NumMicroOps));

  // Notify listeners of the "instruction dispatched" event.
  notifyInstructionDispatched(IR, RegisterFiles);

  // Now move the instruction into the scheduler's queue.
  // The scheduler is responsible for checking if this is a zero-latency
  // instruction that doesn't consume pipeline/scheduler resources.
  SC->scheduleInstruction(IR);
}

void DispatchStage::preExecute(const InstRef &IR) {
  RCU->cycleEvent();
  AvailableEntries = CarryOver >= DispatchWidth ? 0 : DispatchWidth - CarryOver;
  CarryOver = CarryOver >= DispatchWidth ? CarryOver - DispatchWidth : 0U;
}

bool DispatchStage::execute(InstRef &IR) {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  if (!isAvailable(Desc.NumMicroOps) || !canDispatch(IR))
    return false;
  dispatch(IR);
  return true;
}

#ifndef NDEBUG
void DispatchStage::dump() const {
  RAT->dump();
  RCU->dump();
}
#endif
} // namespace mca
