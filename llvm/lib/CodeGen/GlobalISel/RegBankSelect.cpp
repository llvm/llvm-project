//===- llvm/CodeGen/GlobalISel/RegBankSelect.cpp - RegBankSelect -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the RegBankSelect class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define DEBUG_TYPE "regbankselect"

using namespace llvm;

char RegBankSelect::ID = 0;
INITIALIZE_PASS(RegBankSelect, "regbankselect",
                "Assign register bank of generic virtual registers",
                false, false);

RegBankSelect::RegBankSelect()
    : MachineFunctionPass(ID), RBI(nullptr), MRI(nullptr) {
  initializeRegBankSelectPass(*PassRegistry::getPassRegistry());
}

void RegBankSelect::init(MachineFunction &MF) {
  RBI = MF.getSubtarget().getRegBankInfo();
  assert(RBI && "Cannot work without RegisterBankInfo");
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MIRBuilder.setMF(MF);
}

bool RegBankSelect::assignmentMatch(
    unsigned Reg, const RegisterBankInfo::ValueMapping &ValMapping) const {
  // Each part of a break down needs to end up in a different register.
  // In other word, Reg assignement does not match.
  if (ValMapping.BreakDown.size() > 1)
    return false;

  return RBI->getRegBank(Reg, *MRI, *TRI) == ValMapping.BreakDown[0].RegBank;
}

unsigned
RegBankSelect::repairReg(unsigned Reg,
                         const RegisterBankInfo::ValueMapping &ValMapping) {
  assert(ValMapping.BreakDown.size() == 1 &&
         "Support for complex break down not supported yet");
  const RegisterBankInfo::PartialMapping &PartialMap = ValMapping.BreakDown[0];
  assert(PartialMap.Mask.getBitWidth() == MRI->getSize(Reg) &&
         "Repairing other than copy not implemented yet");
  unsigned NewReg =
      MRI->createGenericVirtualRegister(PartialMap.Mask.getBitWidth());
  (void)MIRBuilder.buildInstr(TargetOpcode::COPY, NewReg, Reg);
  DEBUG(dbgs() << "Repair: " << PrintReg(Reg) << " with: "
        << PrintReg(NewReg) << '\n');
  return NewReg;
}

void RegBankSelect::assignInstr(MachineInstr &MI) {
  DEBUG(dbgs() << "Assign: " << MI);
  const RegisterBankInfo::InstructionMapping DefaultMapping =
      RBI->getInstrMapping(MI);
  // Make sure the mapping is valid for MI.
  DefaultMapping.verify(MI);

  DEBUG(dbgs() << "Mapping: " << DefaultMapping << '\n');

  // Set the insertion point before MI.
  // This is where we are going to insert the repairing code if any.
  MIRBuilder.setInstr(MI, /*Before*/ true);

  // For now, do not look for alternative mappings.
  // Alternative mapping may require to rewrite MI and we do not support
  // that yet.
  // Walk the operands and assign then to the chosen mapping, possibly with
  // the insertion of repair code for uses.
  for (unsigned OpIdx = 0, EndIdx = MI.getNumOperands(); OpIdx != EndIdx;
       ++OpIdx) {
    MachineOperand &MO = MI.getOperand(OpIdx);
    // Nothing to be done for non-register operands.
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;

    const RegisterBankInfo::ValueMapping &ValMapping =
        DefaultMapping.getOperandMapping(OpIdx);
    // If Reg is already properly mapped, move on.
    if (assignmentMatch(Reg, ValMapping))
      continue;

    // For uses, we may need to create a new temporary.
    // Indeed, if Reg is already assigned a register bank, at this
    // point, we know it is different from the one defined by the
    // chosen mapping, we need to adjust for that.
    assert(ValMapping.BreakDown.size() == 1 &&
           "Support for complex break down not supported yet");
    if (!MO.isDef() && MRI->getRegClassOrRegBank(Reg)) {
      // For phis, we need to change the insertion point to the end of
      // the related predecessor block.
      assert(!MI.isPHI() && "PHI support not implemented yet");
      Reg = repairReg(Reg, ValMapping);
    }
    // If we end up here, MO should be free of encoding constraints,
    // i.e., we do not have to constrained the RegBank of Reg to
    // the requirement of the operands.
    // If that is not the case, this means the code was broken before
    // hands because we should have found that the assignment match.
    // This will not hold when we will consider alternative mappings.
    MRI->setRegBank(Reg, *ValMapping.BreakDown[0].RegBank);
    MO.setReg(Reg);
  }
  DEBUG(dbgs() << "Assigned: " << MI);
}

bool RegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "Assign register banks for: " << MF.getName() << '\n');
  init(MF);
  // Walk the function and assign register banks to all operands.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      assignInstr(MI);
  return false;
}
