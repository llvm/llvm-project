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

static unsigned findLastWriteBackCycle(const InstRef &IR) {
  unsigned LastWBCycle = 0;
  for (const WriteState &WS : IR.getInstruction()->getDefs()) {
    int CyclesLeft = WS.getCyclesLeft();
    if (CyclesLeft == UNKNOWN_CYCLES)
      CyclesLeft = WS.getLatency();
    if (CyclesLeft < 0)
      CyclesLeft = 0;
    LastWBCycle = std::max(LastWBCycle, (unsigned)CyclesLeft);
  }
  return LastWBCycle;
}

static unsigned findFirstWriteBackCycle(const InstRef &IR) {
  unsigned FirstWBCycle = ~0U;
  for (const WriteState &WS : IR.getInstruction()->getDefs()) {
    int CyclesLeft = WS.getCyclesLeft();
    if (CyclesLeft == UNKNOWN_CYCLES)
      CyclesLeft = WS.getLatency();
    if (CyclesLeft < 0)
      CyclesLeft = 0;
    FirstWBCycle = std::min(FirstWBCycle, (unsigned)CyclesLeft);
  }
  return FirstWBCycle;
}

/// Return a number of cycles left until register requirements of the
/// instructions are met.
static unsigned checkRegisterHazard(const RegisterFile &PRF,
                                    const MCSchedModel &SM,
                                    const MCSubtargetInfo &STI,
                                    const InstRef &IR) {
  unsigned StallCycles = 0;
  SmallVector<WriteRef, 4> Writes;
  SmallVector<WriteRef, 4> CommittedWrites;

  for (const ReadState &RS : IR.getInstruction()->getUses()) {
    const ReadDescriptor &RD = RS.getDescriptor();
    const MCSchedClassDesc *SC = SM.getSchedClassDesc(RD.SchedClassID);

    PRF.collectWrites(STI, RS, Writes, CommittedWrites);
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

    for (const WriteRef &WR : CommittedWrites) {
      unsigned WriteResID = WR.getWriteResourceID();
      assert(!WR.getWriteState() && "Should be already committed!");
      assert(WR.hasKnownWriteBackCycle() && "Invalid write!");
      assert(STI.getReadAdvanceCycles(SC, RD.UseIndex, WriteResID) < 0);
      unsigned ReadAdvance = static_cast<unsigned>(
          -STI.getReadAdvanceCycles(SC, RD.UseIndex, WriteResID));
      unsigned Elapsed = PRF.getElapsedCyclesFromWriteBack(WR);
      assert(Elapsed < ReadAdvance && "Should not have been added to the set!");
      unsigned CyclesLeft = (ReadAdvance - Elapsed);
      StallCycles = std::max(StallCycles, CyclesLeft);
    }
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
  } else if (LastWriteBackCycle) {
    if (!IR.getInstruction()->getDesc().RetireOOO) {
      unsigned NextWriteBackCycle = findFirstWriteBackCycle(IR);
      // Delay the instruction to ensure that writes occur in program order
      if (NextWriteBackCycle < LastWriteBackCycle) {
        *StallCycles = LastWriteBackCycle - NextWriteBackCycle;
      }
    }
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

static void notifyInstructionIssue(
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
  if (llvm::Error E = tryIssue(IR, &StallCyclesLeft))
    return E;

  if (StallCyclesLeft) {
    StalledInst = IR;
  }

  return llvm::ErrorSuccess();
}

llvm::Error InOrderIssueStage::tryIssue(InstRef &IR, unsigned *StallCycles) {
  Instruction &IS = *IR.getInstruction();
  unsigned SourceIndex = IR.getSourceIndex();
  const InstrDesc &Desc = IS.getDesc();

  if (!canExecute(IR, StallCycles)) {
    LLVM_DEBUG(dbgs() << "[E] Stalled #" << IR << " for " << *StallCycles
                      << " cycles\n");
    Bandwidth = 0;
    return llvm::ErrorSuccess();
  }

  unsigned RCUTokenID = RetireControlUnit::UnhandledTokenID;
  IS.dispatch(RCUTokenID);

  SmallVector<unsigned, 4> UsedRegs(PRF.getNumRegisterFiles());
  addRegisterReadWrite(PRF, IS, SourceIndex, STI, UsedRegs);

  unsigned NumMicroOps = IS.getNumMicroOps();
  notifyInstructionDispatch(IR, NumMicroOps, UsedRegs, *this);

  SmallVector<std::pair<ResourceRef, ResourceCycles>, 4> UsedResources;
  RM->issueInstruction(Desc, UsedResources);
  IS.execute(SourceIndex);

  // Replace resource masks with valid resource processor IDs.
  for (std::pair<ResourceRef, ResourceCycles> &Use : UsedResources) {
    uint64_t Mask = Use.first.first;
    Use.first.first = RM->resolveResourceMask(Mask);
  }
  notifyInstructionIssue(IR, UsedResources, *this);

  if (Desc.EndGroup) {
    Bandwidth = 0;
  } else {
    assert(Bandwidth >= NumMicroOps);
    Bandwidth -= NumMicroOps;
  }

  IssuedInst.push_back(IR);
  NumIssued += NumMicroOps;

  if (!IR.getInstruction()->getDesc().RetireOOO)
    LastWriteBackCycle = findLastWriteBackCycle(IR);

  return llvm::ErrorSuccess();
}

void InOrderIssueStage::updateIssuedInst() {
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

    PRF.onInstructionExecuted(&IS);
    notifyEvent<HWInstructionEvent>(
        HWInstructionEvent(HWInstructionEvent::Executed, IR));
    LLVM_DEBUG(dbgs() << "[E] Instruction #" << IR << " is executed\n");
    ++NumExecuted;

    retireInstruction(*I);

    std::iter_swap(I, E - NumExecuted);
  }

  if (NumExecuted)
    IssuedInst.resize(IssuedInst.size() - NumExecuted);
}

void InOrderIssueStage::retireInstruction(InstRef &IR) {
  Instruction &IS = *IR.getInstruction();
  IS.retire();

  llvm::SmallVector<unsigned, 4> FreedRegs(PRF.getNumRegisterFiles());
  for (const WriteState &WS : IS.getDefs())
    PRF.removeRegisterWrite(WS, FreedRegs);

  notifyEvent<HWInstructionEvent>(HWInstructionRetiredEvent(IR, FreedRegs));
  LLVM_DEBUG(dbgs() << "[E] Retired #" << IR << " \n");
}

llvm::Error InOrderIssueStage::cycleStart() {
  NumIssued = 0;
  Bandwidth = SM.IssueWidth;

  PRF.cycleStart();

  // Release consumed resources.
  SmallVector<ResourceRef, 4> Freed;
  RM->cycleEvent(Freed);

  updateIssuedInst();

  // Issue instructions scheduled for this cycle
  if (!StallCyclesLeft && StalledInst) {
    if (llvm::Error E = tryIssue(StalledInst, &StallCyclesLeft))
      return E;
  }

  if (!StallCyclesLeft) {
    StalledInst.invalidate();
    assert(NumIssued <= SM.IssueWidth && "Overflow.");
  } else {
    // The instruction is still stalled, cannot issue any new instructions in
    // this cycle.
    Bandwidth = 0;
  }

  return llvm::ErrorSuccess();
}

llvm::Error InOrderIssueStage::cycleEnd() {
  PRF.cycleEnd();

  if (StallCyclesLeft > 0)
    --StallCyclesLeft;

  if (LastWriteBackCycle > 0)
    --LastWriteBackCycle;

  return llvm::ErrorSuccess();
}

} // namespace mca
} // namespace llvm
