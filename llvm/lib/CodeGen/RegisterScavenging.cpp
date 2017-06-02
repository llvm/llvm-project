//===- RegisterScavenging.cpp - Machine register scavenging ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the machine register scavenger. It can provide
/// information, such as unused registers, at any point in a machine basic
/// block. It also provides a mechanism to make registers available by evicting
/// them to spill slots.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <cassert>
#include <iterator>
#include <limits>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "reg-scavenging"

STATISTIC(NumScavengedRegs, "Number of frame index regs scavenged");

void RegScavenger::setRegUsed(unsigned Reg, LaneBitmask LaneMask) {
  LiveUnits.addRegMasked(Reg, LaneMask);
}

void RegScavenger::init(MachineBasicBlock &MBB) {
  MachineFunction &MF = *MBB.getParent();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  LiveUnits.init(*TRI);

  assert((NumRegUnits == 0 || NumRegUnits == TRI->getNumRegUnits()) &&
         "Target changed?");

  // Self-initialize.
  if (!this->MBB) {
    NumRegUnits = TRI->getNumRegUnits();
    KillRegUnits.resize(NumRegUnits);
    DefRegUnits.resize(NumRegUnits);
    TmpRegUnits.resize(NumRegUnits);
  }
  this->MBB = &MBB;

  for (ScavengedInfo &SI : Scavenged) {
    SI.Reg = 0;
    SI.Restore = nullptr;
  }

  Tracking = false;
}

void RegScavenger::enterBasicBlock(MachineBasicBlock &MBB) {
  init(MBB);
  LiveUnits.addLiveIns(MBB);
}

void RegScavenger::enterBasicBlockEnd(MachineBasicBlock &MBB) {
  init(MBB);
  LiveUnits.addLiveOuts(MBB);

  // Move internal iterator at the last instruction of the block.
  if (MBB.begin() != MBB.end()) {
    MBBI = std::prev(MBB.end());
    Tracking = true;
  }
}

void RegScavenger::addRegUnits(BitVector &BV, unsigned Reg) {
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI)
    BV.set(*RUI);
}

void RegScavenger::removeRegUnits(BitVector &BV, unsigned Reg) {
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI)
    BV.reset(*RUI);
}

void RegScavenger::determineKillsAndDefs() {
  assert(Tracking && "Must be tracking to determine kills and defs");

  MachineInstr &MI = *MBBI;
  assert(!MI.isDebugValue() && "Debug values have no kills or defs");

  // Find out which registers are early clobbered, killed, defined, and marked
  // def-dead in this instruction.
  KillRegUnits.reset();
  DefRegUnits.reset();
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isRegMask()) {
      TmpRegUnits.clear();
      for (unsigned RU = 0, RUEnd = TRI->getNumRegUnits(); RU != RUEnd; ++RU) {
        for (MCRegUnitRootIterator RURI(RU, TRI); RURI.isValid(); ++RURI) {
          if (MO.clobbersPhysReg(*RURI)) {
            TmpRegUnits.set(RU);
            break;
          }
        }
      }

      // Apply the mask.
      KillRegUnits |= TmpRegUnits;
    }
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isPhysicalRegister(Reg) || isReserved(Reg))
      continue;

    if (MO.isUse()) {
      // Ignore undef uses.
      if (MO.isUndef())
        continue;
      if (MO.isKill())
        addRegUnits(KillRegUnits, Reg);
    } else {
      assert(MO.isDef());
      if (MO.isDead())
        addRegUnits(KillRegUnits, Reg);
      else
        addRegUnits(DefRegUnits, Reg);
    }
  }
}

void RegScavenger::unprocess() {
  assert(Tracking && "Cannot unprocess because we're not tracking");

  MachineInstr &MI = *MBBI;
  if (!MI.isDebugValue()) {
    determineKillsAndDefs();

    // Commit the changes.
    setUsed(KillRegUnits);
    setUnused(DefRegUnits);
  }

  if (MBBI == MBB->begin()) {
    MBBI = MachineBasicBlock::iterator(nullptr);
    Tracking = false;
  } else
    --MBBI;
}

void RegScavenger::forward() {
  // Move ptr forward.
  if (!Tracking) {
    MBBI = MBB->begin();
    Tracking = true;
  } else {
    assert(MBBI != MBB->end() && "Already past the end of the basic block!");
    MBBI = std::next(MBBI);
  }
  assert(MBBI != MBB->end() && "Already at the end of the basic block!");

  MachineInstr &MI = *MBBI;

  for (SmallVectorImpl<ScavengedInfo>::iterator I = Scavenged.begin(),
         IE = Scavenged.end(); I != IE; ++I) {
    if (I->Restore != &MI)
      continue;

    I->Reg = 0;
    I->Restore = nullptr;
  }

  if (MI.isDebugValue())
    return;

  determineKillsAndDefs();

  // Verify uses and defs.
#ifndef NDEBUG
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isPhysicalRegister(Reg) || isReserved(Reg))
      continue;
    if (MO.isUse()) {
      if (MO.isUndef())
        continue;
      if (!isRegUsed(Reg)) {
        // Check if it's partial live: e.g.
        // D0 = insert_subreg D0<undef>, S0
        // ... D0
        // The problem is the insert_subreg could be eliminated. The use of
        // D0 is using a partially undef value. This is not *incorrect* since
        // S1 is can be freely clobbered.
        // Ideally we would like a way to model this, but leaving the
        // insert_subreg around causes both correctness and performance issues.
        bool SubUsed = false;
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
          if (isRegUsed(*SubRegs)) {
            SubUsed = true;
            break;
          }
        bool SuperUsed = false;
        for (MCSuperRegIterator SR(Reg, TRI); SR.isValid(); ++SR) {
          if (isRegUsed(*SR)) {
            SuperUsed = true;
            break;
          }
        }
        if (!SubUsed && !SuperUsed) {
          MBB->getParent()->verify(nullptr, "In Register Scavenger");
          llvm_unreachable("Using an undefined register!");
        }
        (void)SubUsed;
        (void)SuperUsed;
      }
    } else {
      assert(MO.isDef());
#if 0
      // FIXME: Enable this once we've figured out how to correctly transfer
      // implicit kills during codegen passes like the coalescer.
      assert((KillRegs.test(Reg) || isUnused(Reg) ||
              isLiveInButUnusedBefore(Reg, MI, MBB, TRI, MRI)) &&
             "Re-defining a live register!");
#endif
    }
  }
#endif // NDEBUG

  // Commit the changes.
  setUnused(KillRegUnits);
  setUsed(DefRegUnits);
}

void RegScavenger::backward() {
  assert(Tracking && "Must be tracking to determine kills and defs");

  const MachineInstr &MI = *MBBI;
  LiveUnits.stepBackward(MI);

  if (MBBI == MBB->begin()) {
    MBBI = MachineBasicBlock::iterator(nullptr);
    Tracking = false;
  } else
    --MBBI;
}

bool RegScavenger::isRegUsed(unsigned Reg, bool includeReserved) const {
  if (isReserved(Reg))
    return includeReserved;
  return !LiveUnits.available(Reg);
}

unsigned RegScavenger::FindUnusedReg(const TargetRegisterClass *RC) const {
  for (unsigned Reg : *RC) {
    if (!isRegUsed(Reg)) {
      DEBUG(dbgs() << "Scavenger found unused reg: " << TRI->getName(Reg) <<
            "\n");
      return Reg;
    }
  }
  return 0;
}

BitVector RegScavenger::getRegsAvailable(const TargetRegisterClass *RC) {
  BitVector Mask(TRI->getNumRegs());
  for (unsigned Reg : *RC)
    if (!isRegUsed(Reg))
      Mask.set(Reg);
  return Mask;
}

unsigned RegScavenger::findSurvivorReg(MachineBasicBlock::iterator StartMI,
                                       BitVector &Candidates,
                                       unsigned InstrLimit,
                                       MachineBasicBlock::iterator &UseMI) {
  int Survivor = Candidates.find_first();
  assert(Survivor > 0 && "No candidates for scavenging");

  MachineBasicBlock::iterator ME = MBB->getFirstTerminator();
  assert(StartMI != ME && "MI already at terminator");
  MachineBasicBlock::iterator RestorePointMI = StartMI;
  MachineBasicBlock::iterator MI = StartMI;

  bool inVirtLiveRange = false;
  for (++MI; InstrLimit > 0 && MI != ME; ++MI, --InstrLimit) {
    if (MI->isDebugValue()) {
      ++InstrLimit; // Don't count debug instructions
      continue;
    }
    bool isVirtKillInsn = false;
    bool isVirtDefInsn = false;
    // Remove any candidates touched by instruction.
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isRegMask())
        Candidates.clearBitsNotInMask(MO.getRegMask());
      if (!MO.isReg() || MO.isUndef() || !MO.getReg())
        continue;
      if (TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        if (MO.isDef())
          isVirtDefInsn = true;
        else if (MO.isKill())
          isVirtKillInsn = true;
        continue;
      }
      for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid(); ++AI)
        Candidates.reset(*AI);
    }
    // If we're not in a virtual reg's live range, this is a valid
    // restore point.
    if (!inVirtLiveRange) RestorePointMI = MI;

    // Update whether we're in the live range of a virtual register
    if (isVirtKillInsn) inVirtLiveRange = false;
    if (isVirtDefInsn) inVirtLiveRange = true;

    // Was our survivor untouched by this instruction?
    if (Candidates.test(Survivor))
      continue;

    // All candidates gone?
    if (Candidates.none())
      break;

    Survivor = Candidates.find_first();
  }
  // If we ran off the end, that's where we want to restore.
  if (MI == ME) RestorePointMI = ME;
  assert(RestorePointMI != StartMI &&
         "No available scavenger restore location!");

  // We ran out of candidates, so stop the search.
  UseMI = RestorePointMI;
  return Survivor;
}

static unsigned getFrameIndexOperandNum(MachineInstr &MI) {
  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  return i;
}

unsigned RegScavenger::scavengeRegister(const TargetRegisterClass *RC,
                                        MachineBasicBlock::iterator I,
                                        int SPAdj) {
  MachineInstr &MI = *I;
  const MachineFunction &MF = *MI.getParent()->getParent();
  // Consider all allocatable registers in the register class initially
  BitVector Candidates = TRI->getAllocatableSet(MF, RC);

  // Exclude all the registers being used by the instruction.
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg() != 0 && !(MO.isUse() && MO.isUndef()) &&
        !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid(); ++AI)
        Candidates.reset(*AI);
  }

  // Try to find a register that's unused if there is one, as then we won't
  // have to spill.
  BitVector Available = getRegsAvailable(RC);
  Available &= Candidates;
  if (Available.any())
    Candidates = Available;

  // Find the register whose use is furthest away.
  MachineBasicBlock::iterator UseMI;
  unsigned SReg = findSurvivorReg(I, Candidates, 25, UseMI);

  // If we found an unused register there is no reason to spill it.
  if (!isRegUsed(SReg)) {
    DEBUG(dbgs() << "Scavenged register: " << TRI->getName(SReg) << "\n");
    return SReg;
  }

  // Find an available scavenging slot with size and alignment matching
  // the requirements of the class RC.
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned NeedSize = TRI->getSpillSize(*RC);
  unsigned NeedAlign = TRI->getSpillAlignment(*RC);

  unsigned SI = Scavenged.size(), Diff = std::numeric_limits<unsigned>::max();
  int FIB = MFI.getObjectIndexBegin(), FIE = MFI.getObjectIndexEnd();
  for (unsigned I = 0; I < Scavenged.size(); ++I) {
    if (Scavenged[I].Reg != 0)
      continue;
    // Verify that this slot is valid for this register.
    int FI = Scavenged[I].FrameIndex;
    if (FI < FIB || FI >= FIE)
      continue;
    unsigned S = MFI.getObjectSize(FI);
    unsigned A = MFI.getObjectAlignment(FI);
    if (NeedSize > S || NeedAlign > A)
      continue;
    // Avoid wasting slots with large size and/or large alignment. Pick one
    // that is the best fit for this register class (in street metric).
    // Picking a larger slot than necessary could happen if a slot for a
    // larger register is reserved before a slot for a smaller one. When
    // trying to spill a smaller register, the large slot would be found
    // first, thus making it impossible to spill the larger register later.
    unsigned D = (S-NeedSize) + (A-NeedAlign);
    if (D < Diff) {
      SI = I;
      Diff = D;
    }
  }

  if (SI == Scavenged.size()) {
    // We need to scavenge a register but have no spill slot, the target
    // must know how to do it (if not, we'll assert below).
    Scavenged.push_back(ScavengedInfo(FIE));
  }

  // Avoid infinite regress
  Scavenged[SI].Reg = SReg;

  // If the target knows how to save/restore the register, let it do so;
  // otherwise, use the emergency stack spill slot.
  if (!TRI->saveScavengerRegister(*MBB, I, UseMI, RC, SReg)) {
    // Spill the scavenged register before I.
    int FI = Scavenged[SI].FrameIndex;
    if (FI < FIB || FI >= FIE) {
      std::string Msg = std::string("Error while trying to spill ") +
          TRI->getName(SReg) + " from class " + TRI->getRegClassName(RC) +
          ": Cannot scavenge register without an emergency spill slot!";
      report_fatal_error(Msg.c_str());
    }
    TII->storeRegToStackSlot(*MBB, I, SReg, true, Scavenged[SI].FrameIndex,
                             RC, TRI);
    MachineBasicBlock::iterator II = std::prev(I);

    unsigned FIOperandNum = getFrameIndexOperandNum(*II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, this);

    // Restore the scavenged register before its use (or first terminator).
    TII->loadRegFromStackSlot(*MBB, UseMI, SReg, Scavenged[SI].FrameIndex,
                              RC, TRI);
    II = std::prev(UseMI);

    FIOperandNum = getFrameIndexOperandNum(*II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, this);
  }

  Scavenged[SI].Restore = &*std::prev(UseMI);

  // Doing this here leads to infinite regress.
  // Scavenged[SI].Reg = SReg;

  DEBUG(dbgs() << "Scavenged register (with spill): " << TRI->getName(SReg) <<
        "\n");

  return SReg;
}

void llvm::scavengeFrameVirtualRegs(MachineFunction &MF, RegScavenger &RS) {
  // FIXME: Iterating over the instruction stream is unnecessary. We can simply
  // iterate over the vreg use list, which at this point only contains machine
  // operands for which eliminateFrameIndex need a new scratch reg.

  // Run through the instructions and find any virtual registers.
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF) {
    RS.enterBasicBlock(MBB);

    int SPAdj = 0;

    // The instruction stream may change in the loop, so check MBB.end()
    // directly.
    for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ) {
      // We might end up here again with a NULL iterator if we scavenged a
      // register for which we inserted spill code for definition by what was
      // originally the first instruction in MBB.
      if (I == MachineBasicBlock::iterator(nullptr))
        I = MBB.begin();

      const MachineInstr &MI = *I;
      MachineBasicBlock::iterator J = std::next(I);
      MachineBasicBlock::iterator P =
                         I == MBB.begin() ? MachineBasicBlock::iterator(nullptr)
                                          : std::prev(I);

      // RS should process this instruction before we might scavenge at this
      // location. This is because we might be replacing a virtual register
      // defined by this instruction, and if so, registers killed by this
      // instruction are available, and defined registers are not.
      RS.forward(I);

      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        unsigned Reg = MO.getReg();
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          continue;

        // When we first encounter a new virtual register, it
        // must be a definition.
        assert(MO.isDef() && "frame index virtual missing def!");
        // Scavenge a new scratch register
        const TargetRegisterClass *RC = MRI.getRegClass(Reg);
        unsigned ScratchReg = RS.scavengeRegister(RC, J, SPAdj);

        ++NumScavengedRegs;

        // Replace this reference to the virtual register with the
        // scratch register.
        assert(ScratchReg && "Missing scratch register!");
        MRI.replaceRegWith(Reg, ScratchReg);

        // Because this instruction was processed by the RS before this
        // register was allocated, make sure that the RS now records the
        // register as being used.
        RS.setRegUsed(ScratchReg);
      }

      // If the scavenger needed to use one of its spill slots, the
      // spill code will have been inserted in between I and J. This is a
      // problem because we need the spill code before I: Move I to just
      // prior to J.
      if (I != std::prev(J)) {
        MBB.splice(J, &MBB, I);

        // Before we move I, we need to prepare the RS to visit I again.
        // Specifically, RS will assert if it sees uses of registers that
        // it believes are undefined. Because we have already processed
        // register kills in I, when it visits I again, it will believe that
        // those registers are undefined. To avoid this situation, unprocess
        // the instruction I.
        assert(RS.getCurrentPosition() == I &&
          "The register scavenger has an unexpected position");
        I = P;
        RS.unprocess(P);
      } else
        ++I;
    }
  }

  MF.getProperties().set(MachineFunctionProperties::Property::NoVRegs);
}
