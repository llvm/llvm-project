//===-- RegisterScavenging.cpp - Machine register scavenging --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the machine register scavenger. It can provide
// information, such as unused registers, at any point in a machine basic block.
// It also provides a mechanism to make registers available by evicting them to
// spill slots.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg-scavenging"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

/// setUsed - Set the register and its sub-registers as being used.
void RegScavenger::setUsed(unsigned Reg) {
  RegsAvailable.reset(Reg);

  for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
    RegsAvailable.reset(*SubRegs);
}

bool RegScavenger::isAliasUsed(unsigned Reg) const {
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
    if (isUsed(*AI))
      return true;
  return false;
}

void RegScavenger::initRegState() {
  ScavengedReg = 0;
  ScavengedRC = NULL;
  ScavengeRestore = NULL;

  // All registers started out unused.
  RegsAvailable.set();

  if (!MBB)
    return;

  // Live-in registers are in use.
  for (MachineBasicBlock::livein_iterator I = MBB->livein_begin(),
         E = MBB->livein_end(); I != E; ++I)
    setUsed(*I);

  // Pristine CSRs are also unavailable.
  BitVector PR = MBB->getParent()->getFrameInfo()->getPristineRegs(MBB);
  for (int I = PR.find_first(); I>0; I = PR.find_next(I))
    setUsed(I);
}

void RegScavenger::enterBasicBlock(MachineBasicBlock *mbb) {
  MachineFunction &MF = *mbb->getParent();
  const TargetMachine &TM = MF.getTarget();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  MRI = &MF.getRegInfo();

  assert((NumPhysRegs == 0 || NumPhysRegs == TRI->getNumRegs()) &&
         "Target changed?");

  // It is not possible to use the register scavenger after late optimization
  // passes that don't preserve accurate liveness information.
  assert(MRI->tracksLiveness() &&
         "Cannot use register scavenger with inaccurate liveness");

  // Self-initialize.
  if (!MBB) {
    NumPhysRegs = TRI->getNumRegs();
    RegsAvailable.resize(NumPhysRegs);
    KillRegs.resize(NumPhysRegs);
    DefRegs.resize(NumPhysRegs);

    // Create callee-saved registers bitvector.
    CalleeSavedRegs.resize(NumPhysRegs);
    const uint16_t *CSRegs = TRI->getCalleeSavedRegs(&MF);
    if (CSRegs != NULL)
      for (unsigned i = 0; CSRegs[i]; ++i)
        CalleeSavedRegs.set(CSRegs[i]);
  }

  MBB = mbb;
  initRegState();

  Tracking = false;
}

void RegScavenger::addRegWithSubRegs(BitVector &BV, unsigned Reg) {
  BV.set(Reg);
  for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
    BV.set(*SubRegs);
}

void RegScavenger::forward() {
  // Move ptr forward.
  if (!Tracking) {
    MBBI = MBB->begin();
    Tracking = true;
  } else {
    assert(MBBI != MBB->end() && "Already past the end of the basic block!");
    MBBI = llvm::next(MBBI);
  }
  assert(MBBI != MBB->end() && "Already at the end of the basic block!");

  MachineInstr *MI = MBBI;

  if (MI == ScavengeRestore) {
    ScavengedReg = 0;
    ScavengedRC = NULL;
    ScavengeRestore = NULL;
  }

  if (MI->isDebugValue())
    return;

  // Find out which registers are early clobbered, killed, defined, and marked
  // def-dead in this instruction.
  // FIXME: The scavenger is not predication aware. If the instruction is
  // predicated, conservatively assume "kill" markers do not actually kill the
  // register. Similarly ignores "dead" markers.
  bool isPred = TII->isPredicated(MI);
  KillRegs.reset();
  DefRegs.reset();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegMask())
      (isPred ? DefRegs : KillRegs).setBitsNotInMask(MO.getRegMask());
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || isReserved(Reg))
      continue;

    if (MO.isUse()) {
      // Ignore undef uses.
      if (MO.isUndef())
        continue;
      if (!isPred && MO.isKill())
        addRegWithSubRegs(KillRegs, Reg);
    } else {
      assert(MO.isDef());
      if (!isPred && MO.isDead())
        addRegWithSubRegs(KillRegs, Reg);
      else
        addRegWithSubRegs(DefRegs, Reg);
    }
  }

  // Verify uses and defs.
#ifndef NDEBUG
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || isReserved(Reg))
      continue;
    if (MO.isUse()) {
      if (MO.isUndef())
        continue;
      if (!isUsed(Reg)) {
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
          if (isUsed(*SubRegs)) {
            SubUsed = true;
            break;
          }
        if (!SubUsed) {
          MBB->getParent()->verify(NULL, "In Register Scavenger");
          llvm_unreachable("Using an undefined register!");
        }
        (void)SubUsed;
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
  setUnused(KillRegs);
  setUsed(DefRegs);
}

void RegScavenger::getRegsUsed(BitVector &used, bool includeReserved) {
  used = RegsAvailable;
  used.flip();
  if (includeReserved)
    used |= MRI->getReservedRegs();
  else
    used.reset(MRI->getReservedRegs());
}

unsigned RegScavenger::FindUnusedReg(const TargetRegisterClass *RC) const {
  for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
       I != E; ++I)
    if (!isAliasUsed(*I)) {
      DEBUG(dbgs() << "Scavenger found unused reg: " << TRI->getName(*I) <<
            "\n");
      return *I;
    }
  return 0;
}

/// getRegsAvailable - Return all available registers in the register class
/// in Mask.
BitVector RegScavenger::getRegsAvailable(const TargetRegisterClass *RC) {
  BitVector Mask(TRI->getNumRegs());
  for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
       I != E; ++I)
    if (!isAliasUsed(*I))
      Mask.set(*I);
  return Mask;
}

/// findSurvivorReg - Return the candidate register that is unused for the
/// longest after StargMII. UseMI is set to the instruction where the search
/// stopped.
///
/// No more than InstrLimit instructions are inspected.
///
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
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
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
  assert (RestorePointMI != StartMI &&
          "No available scavenger restore location!");

  // We ran out of candidates, so stop the search.
  UseMI = RestorePointMI;
  return Survivor;
}

unsigned RegScavenger::scavengeRegister(const TargetRegisterClass *RC,
                                        MachineBasicBlock::iterator I,
                                        int SPAdj) {
  // Consider all allocatable registers in the register class initially
  BitVector Candidates =
    TRI->getAllocatableSet(*I->getParent()->getParent(), RC);

  // Exclude all the registers being used by the instruction.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = I->getOperand(i);
    if (MO.isReg() && MO.getReg() != 0 &&
        !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      Candidates.reset(MO.getReg());
  }

  // Try to find a register that's unused if there is one, as then we won't
  // have to spill. Search explicitly rather than masking out based on
  // RegsAvailable, as RegsAvailable does not take aliases into account.
  // That's what getRegsAvailable() is for.
  BitVector Available = getRegsAvailable(RC);
  Available &= Candidates;
  if (Available.any())
    Candidates = Available;

  // Find the register whose use is furthest away.
  MachineBasicBlock::iterator UseMI;
  unsigned SReg = findSurvivorReg(I, Candidates, 25, UseMI);

  // If we found an unused register there is no reason to spill it.
  if (!isAliasUsed(SReg)) {
    DEBUG(dbgs() << "Scavenged register: " << TRI->getName(SReg) << "\n");
    return SReg;
  }

  assert(ScavengedReg == 0 &&
         "Scavenger slot is live, unable to scavenge another register!");

  // Avoid infinite regress
  ScavengedReg = SReg;

  // If the target knows how to save/restore the register, let it do so;
  // otherwise, use the emergency stack spill slot.
  if (!TRI->saveScavengerRegister(*MBB, I, UseMI, RC, SReg)) {
    // Spill the scavenged register before I.
    assert(ScavengingFrameIndex >= 0 &&
           "Cannot scavenge register without an emergency spill slot!");
    TII->storeRegToStackSlot(*MBB, I, SReg, true, ScavengingFrameIndex, RC,TRI);
    MachineBasicBlock::iterator II = prior(I);
    TRI->eliminateFrameIndex(II, SPAdj, this);

    // Restore the scavenged register before its use (or first terminator).
    TII->loadRegFromStackSlot(*MBB, UseMI, SReg, ScavengingFrameIndex, RC, TRI);
    II = prior(UseMI);
    TRI->eliminateFrameIndex(II, SPAdj, this);
  }

  ScavengeRestore = prior(UseMI);

  // Doing this here leads to infinite regress.
  // ScavengedReg = SReg;
  ScavengedRC = RC;

  DEBUG(dbgs() << "Scavenged register (with spill): " << TRI->getName(SReg) <<
        "\n");

  return SReg;
}
