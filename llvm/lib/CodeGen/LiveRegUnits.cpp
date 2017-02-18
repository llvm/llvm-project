//===- LiveRegUnits.cpp - Register Unit Set -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file imlements the LiveRegUnits set.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

void LiveRegUnits::removeRegsNotPreserved(const uint32_t *RegMask) {
  for (unsigned U = 0, E = TRI->getNumRegUnits(); U != E; ++U) {
    for (MCRegUnitRootIterator RootReg(U, TRI); RootReg.isValid(); ++RootReg) {
      if (MachineOperand::clobbersPhysReg(RegMask, *RootReg))
        Units.reset(U);
    }
  }
}

void LiveRegUnits::addRegsInMask(const uint32_t *RegMask) {
  for (unsigned U = 0, E = TRI->getNumRegUnits(); U != E; ++U) {
    for (MCRegUnitRootIterator RootReg(U, TRI); RootReg.isValid(); ++RootReg) {
      if (MachineOperand::clobbersPhysReg(RegMask, *RootReg))
        Units.set(U);
    }
  }
}

void LiveRegUnits::stepBackward(const MachineInstr &MI) {
  // Remove defined registers and regmask kills from the set.
  for (ConstMIBundleOperands O(MI); O.isValid(); ++O) {
    if (O->isReg()) {
      if (!O->isDef())
        continue;
      unsigned Reg = O->getReg();
      if (!TargetRegisterInfo::isPhysicalRegister(Reg))
        continue;
      removeReg(Reg);
    } else if (O->isRegMask())
      removeRegsNotPreserved(O->getRegMask());
  }

  // Add uses to the set.
  for (ConstMIBundleOperands O(MI); O.isValid(); ++O) {
    if (!O->isReg() || !O->readsReg())
      continue;
    unsigned Reg = O->getReg();
    if (!TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    addReg(Reg);
  }
}

void LiveRegUnits::accumulateBackward(const MachineInstr &MI) {
  // Add defs, uses and regmask clobbers to the set.
  for (ConstMIBundleOperands O(MI); O.isValid(); ++O) {
    if (O->isReg()) {
      unsigned Reg = O->getReg();
      if (!TargetRegisterInfo::isPhysicalRegister(Reg))
        continue;
      if (!O->isDef() && !O->readsReg())
        continue;
      addReg(Reg);
    } else if (O->isRegMask())
      addRegsInMask(O->getRegMask());
  }
}

/// Add live-in registers of basic block \p MBB to \p LiveUnits.
static void addLiveIns(LiveRegUnits &LiveUnits, const MachineBasicBlock &MBB) {
  for (const auto &LI : MBB.liveins())
    LiveUnits.addRegMasked(LI.PhysReg, LI.LaneMask);
}

static void addLiveOuts(LiveRegUnits &LiveUnits, const MachineBasicBlock &MBB) {
  // To get the live-outs we simply merge the live-ins of all successors.
  for (const MachineBasicBlock *Succ : MBB.successors())
    addLiveIns(LiveUnits, *Succ);
}

/// Add pristine registers to the given \p LiveUnits. This function removes
/// actually saved callee save registers when \p InPrologueEpilogue is false.
static void removeSavedRegs(LiveRegUnits &LiveUnits, const MachineFunction &MF,
                            const MachineFrameInfo &MFI,
                            const TargetRegisterInfo &TRI) {
  for (const CalleeSavedInfo &Info : MFI.getCalleeSavedInfo())
    LiveUnits.removeReg(Info.getReg());
}

void LiveRegUnits::addLiveOuts(const MachineBasicBlock &MBB) {
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (MFI.isCalleeSavedInfoValid()) {
    for (const MCPhysReg *I = TRI->getCalleeSavedRegs(&MF); *I; ++I)
      addReg(*I);
    if (!MBB.isReturnBlock())
      removeSavedRegs(*this, MF, MFI, *TRI);
  }
  ::addLiveOuts(*this, MBB);
}

void LiveRegUnits::addLiveIns(const MachineBasicBlock &MBB) {
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (MFI.isCalleeSavedInfoValid()) {
    for (const MCPhysReg *I = TRI->getCalleeSavedRegs(&MF); *I; ++I)
      addReg(*I);
    if (&MBB != &MF.front())
      removeSavedRegs(*this, MF, MFI, *TRI);
  }
  ::addLiveIns(*this, MBB);
}
