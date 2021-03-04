//===---------------------- InOrderIssueStage.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// InOrderIssueStage implements an in-order execution pipeline.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/Stages/InOrderIssueStage.h"

#include "llvm/MC/MCSchedule.h"
#include "llvm/MCA/HWEventListener.h"
#include "llvm/MCA/HardwareUnits/RegisterFile.h"
#include "llvm/MCA/HardwareUnits/ResourceManager.h"
#include "llvm/MCA/HardwareUnits/RetireControlUnit.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"

#include <algorithm>

#define DEBUG_TYPE "llvm-mca"
namespace llvm {
namespace mca {

bool InOrderIssueStage::hasWorkToComplete() const {
  return !IssuedInst.empty() || StalledInst;
}

bool InOrderIssueStage::isAvailable(const InstRef &IR) const {
  const Instruction &Inst = *IR.getInstruction();
  unsigned NumMicroOps = Inst.getNumMicroOps();
  const InstrDesc &Desc = Inst.getDesc();

  if (Bandwidth < NumMicroOps)
    return false;

  // Instruction with BeginGroup must be the first instruction to be issued in a
  // cycle.
  if (Desc.BeginGroup && NumIssued != 0)
    return false;

  return true;
}

static bool hasResourceHazard(const ResourceManager &RM, const InstRef &IR) {
  if (RM.checkAvailability(IR.getInstruction()->getDesc())) {
    LLVM_DEBUG(dbgs() << "[E] Stall #" << IR << '\n');
    return true;
  }

  return false;
}

/// Return a number of cycles left until register requirements of the
/// instructions are met.
static unsigned checkRegisterHazard(const RegisterFile &PRF,
                                    const MCSchedModel &SM,
                                    const MCSubtargetInfo &STI,
                                    const InstRef &IR) {
  unsigned StallCycles = 0;
  SmallVector<WriteRef, 4> Writes;

  for (const ReadState &RS : IR.getInstruction()->getUses()) {
    const ReadDescriptor &RD = RS.getDescriptor();
    const MCSchedClassDesc *SC = SM.getSchedClassDesc(RD.SchedClassID);

    PRF.collectWrites(RS, Writes);
    for (const WriteRef &WR : Writes) {
      const WriteState *WS = WR.getWriteState();
      unsigned WriteResID = WS->getWriteResourceID();
      int ReadAdvance = STI.getReadAdvanceCycles(SC, RD.UseIndex, WriteResID);
      LLVM_DEBUG(dbgs() << "[E] ReadAdvance for #" << IR << ": " << ReadAdvance
                        << '\n');

      if (WS->getCyclesLeft() == UNKNOWN_CYCLES) {
        // Try again in the next cycle until the value is known
        StallCycles = std::max(StallCycles, 1U);
        continue;
      }

      int CyclesLeft = WS->getCyclesLeft() - ReadAdvance;
      if (CyclesLeft > 0) {
        LLVM_DEBUG(dbgs() << "[E] Register hazard: " << WS->getRegisterID()
                          << '\n');
        StallCycles = std::max(StallCycles, (unsigned)CyclesLeft);
      }
    }
    Writes.clear();
  }

  return StallCycles;
}

bool InOrderIssueStage::canExecute(const InstRef &IR,
                                   unsigned *StallCycles) const {
  *StallCycles = 0;

  if (unsigned RegStall = checkRegisterHazard(PRF, SM, STI, IR)) {
    *StallCycles = RegStall;
    // FIXME: add a parameter to HWStallEvent to indicate a number of cycles.
    for (unsigned I = 0; I < RegStall; ++I) {
      notifyEvent<HWStallEvent>(
          HWStallEvent(HWStallEvent::RegisterFileStall, IR));
      notifyEvent<HWPressureEvent>(
          HWPressureEvent(HWPressureEvent::REGISTER_DEPS, IR));
    }
  } else if (hasResourceHazard(*RM, IR)) {
    *StallCycles = 1;
    notifyEvent<HWStallEvent>(
        HWStallEvent(HWStallEvent::DispatchGroupStall, IR));
    notifyEvent<HWPressureEvent>(
        HWPressureEvent(HWPressureEvent::RESOURCES, IR));
  }

  return *StallCycles == 0;
}

static void addRegisterReadWrite(RegisterFile &PRF, Instruction &IS,
                                 unsigned SourceIndex,
                                 const MCSubtargetInfo &STI,
                                 SmallVectorImpl<unsigned> &UsedRegs) {
  assert(!IS.isEliminated());

  for (ReadState &RS : IS.getUses())
    PRF.addRegisterRead(RS, STI);

  for (WriteState &WS : IS.getDefs())
    PRF.addRegisterWrite(WriteRef(SourceIndex, &WS), UsedRegs);
}

static void notifyInstructionExecute(
    const InstRef &IR,
    const SmallVectorImpl<std::pair<ResourceRef, ResourceCycles>> &UsedRes,
    const Stage &S) {

  S.notifyEvent<HWInstructionEvent>(
      HWInstructionEvent(HWInstructionEvent::Ready, IR));
  S.notifyEvent<HWInstructionEvent>(HWInstructionIssuedEvent(IR, UsedRes));

  LLVM_DEBUG(dbgs() << "[E] Issued #" << IR << "\n");
}

static void notifyInstructionDispatch(const InstRef &IR, unsigned Ops,
                                      const SmallVectorImpl<unsigned> &UsedRegs,
                                      const Stage &S) {

  S.notifyEvent<HWInstructionEvent>(
      HWInstructionDispatchedEvent(IR, UsedRegs, Ops));

  LLVM_DEBUG(dbgs() << "[E] Dispatched #" << IR << "\n");
}

llvm::Error InOrderIssueStage::execute(InstRef &IR) {
  Instruction &IS = *IR.getInstruction();
  const InstrDesc &Desc = IS.getDesc();

  unsigned RCUTokenID = RetireControlUnit::UnhandledTokenID;
  if (!Desc.RetireOOO)
    RCUTokenID = RCU.dispatch(IR);
  IS.dispatch(RCUTokenID);

  if (Desc.EndGroup) {
    Bandwidth = 0;
  } else {
    unsigned NumMicroOps = IR.getInstruction()->getNumMicroOps();
    assert(Bandwidth >= NumMicroOps);
    Bandwidth -= NumMicroOps;
  }

  if (llvm::Error E = tryIssue(IR, &StallCyclesLeft))
    return E;

  if (StallCyclesLeft) {
    StalledInst = IR;
    Bandwidth = 0;
  }

  return llvm::ErrorSuccess();
}

llvm::Error InOrderIssueStage::tryIssue(InstRef &IR, unsigned *StallCycles) {
  Instruction &IS = *IR.getInstruction();
  unsigned SourceIndex = IR.getSourceIndex();

  if (!canExecute(IR, StallCycles)) {
    LLVM_DEBUG(dbgs() << "[E] Stalled #" << IR << " for " << *StallCycles
                      << " cycles\n");
    return llvm::ErrorSuccess();
  }

  SmallVector<unsigned, 4> UsedRegs(PRF.getNumRegisterFiles());
  addRegisterReadWrite(PRF, IS, SourceIndex, STI, UsedRegs);

  notifyInstructionDispatch(IR, IS.getDesc().NumMicroOps, UsedRegs, *this);

  SmallVector<std::pair<ResourceRef, ResourceCycles>, 4> UsedResources;
  RM->issueInstruction(IS.getDesc(), UsedResources);
  IS.execute(SourceIndex);

  // Replace resource masks with valid resource processor IDs.
  for (std::pair<ResourceRef, ResourceCycles> &Use : UsedResources) {
    uint64_t Mask = Use.first.first;
    Use.first.first = RM->resolveResourceMask(Mask);
  }
  notifyInstructionExecute(IR, UsedResources, *this);

  IssuedInst.push_back(IR);
  ++NumIssued;

  return llvm::ErrorSuccess();
}

llvm::Error InOrderIssueStage::updateIssuedInst() {
  // Update other instructions. Executed instructions will be retired during the
  // next cycle.
  unsigned NumExecuted = 0;
  for (auto I = IssuedInst.begin(), E = IssuedInst.end();
       I != (E - NumExecuted);) {
    InstRef &IR = *I;
    Instruction &IS = *IR.getInstruction();

    IS.cycleEvent();
    if (!IS.isExecuted()) {
      LLVM_DEBUG(dbgs() << "[E] Instruction #" << IR
                        << " is still executing\n");
      ++I;
      continue;
    }
    notifyEvent<HWInstructionEvent>(
        HWInstructionEvent(HWInstructionEvent::Executed, IR));

    LLVM_DEBUG(dbgs() << "[E] Instruction #" << IR << " is executed\n");
    ++NumExecuted;
    std::iter_swap(I, E - NumExecuted);
  }

  // Retire instructions in the next cycle
  if (NumExecuted) {
    for (auto I = IssuedInst.end() - NumExecuted, E = IssuedInst.end(); I != E;
         ++I) {
      if (llvm::Error E = moveToTheNextStage(*I))
        return E;
    }
    IssuedInst.resize(IssuedInst.size() - NumExecuted);
  }

  return llvm::ErrorSuccess();
}

llvm::Error InOrderIssueStage::cycleStart() {
  NumIssued = 0;

  // Release consumed resources.
  SmallVector<ResourceRef, 4> Freed;
  RM->cycleEvent(Freed);

  if (llvm::Error E = updateIssuedInst())
    return E;

  // Issue instructions scheduled for this cycle
  if (!StallCyclesLeft && StalledInst) {
    if (llvm::Error E = tryIssue(StalledInst, &StallCyclesLeft))
      return E;
  }

  if (!StallCyclesLeft) {
    StalledInst.invalidate();
    assert(NumIssued <= SM.IssueWidth && "Overflow.");
    Bandwidth = SM.IssueWidth - NumIssued;
  } else {
    // The instruction is still stalled, cannot issue any new instructions in
    // this cycle.
    Bandwidth = 0;
  }

  return llvm::ErrorSuccess();
}

llvm::Error InOrderIssueStage::cycleEnd() {
  if (StallCyclesLeft > 0)
    --StallCyclesLeft;
  return llvm::ErrorSuccess();
}

} // namespace mca
} // namespace llvm
