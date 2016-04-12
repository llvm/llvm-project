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

#include "llvm/ADT/PostOrderIterator.h"
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

  const RegisterBank *CurRegBank = RBI->getRegBank(Reg, *MRI, *TRI);
  const RegisterBank *DesiredRegBrank = ValMapping.BreakDown[0].RegBank;
  DEBUG(dbgs() << "Does assignment already match: ";
        if (CurRegBank) dbgs() << *CurRegBank; else dbgs() << "none";
        dbgs() << " against ";
        assert(DesiredRegBrank && "The mapping must be valid");
        dbgs() << *DesiredRegBrank << '\n';);
  return CurRegBank == DesiredRegBrank;
}

unsigned
RegBankSelect::repairReg(unsigned Reg,
                         const RegisterBankInfo::ValueMapping &ValMapping,
                         MachineInstr &DefUseMI, bool IsDef) {
  assert(ValMapping.BreakDown.size() == 1 &&
         "Support for complex break down not supported yet");
  const RegisterBankInfo::PartialMapping &PartialMap = ValMapping.BreakDown[0];
  assert(PartialMap.Mask.getBitWidth() ==
             (TargetRegisterInfo::isPhysicalRegister(Reg)
                  ? TRI->getMinimalPhysRegClass(Reg)->getSize() * 8
                  : MRI->getSize(Reg)) &&
         "Repairing other than copy not implemented yet");
  // If the MIRBuilder is configured to insert somewhere else than
  // DefUseMI, we may not use this function like was it first
  // internded (local repairing), so make sure we pay attention before
  // we remove the assert.
  // In particular, it is likely that we will have to properly save
  // the insertion point of the MIRBuilder and restore it at the end
  // of this method.
  assert(&DefUseMI == &(*MIRBuilder.getInsertPt()) &&
         "Need to save and restore the insertion point");
  // For use, we will add a copy just in front of the instruction.
  // For def, we will add a copy just after the instruction.
  // In either case, the insertion point must be valid. In particular,
  // make sure we do not insert in the middle of terminators or phis.
  bool Before = !IsDef;
  setSafeInsertionPoint(DefUseMI, Before);
  if (DefUseMI.isTerminator() && Before) {
    // Check that the insertion point does not happen
    // before the definition of Reg.
    // This can happen if Reg is defined by a terminator
    // and used by another one.
    // In that case the repairing code is actually more involved
    // because we have to split the block.

    // Assert that this is not a physical register.
    // The target independent code does not insert physical registers
    // on terminators, so if we end up in this situation, this is
    // likely a bug in the target.
    assert(!TargetRegisterInfo::isPhysicalRegister(Reg) &&
           "Check for physical register not implemented");
    const MachineInstr *RegDef = MRI->getVRegDef(Reg);
    assert(RegDef && "Reg has more than one definition?");
    // Assert to make the code more readable; Reg is used by DefUseMI, i.e.,
    // (Before == !IsDef == true), so DefUseMI != RegDef otherwise we have
    // a use (that is not a PHI) that is not dominated by its def.
    assert(&DefUseMI != RegDef && "Def does not dominate all of its uses");
    if (RegDef->isTerminator() && RegDef->getParent() == DefUseMI.getParent())
      // By construction, the repairing should happen between two
      // terminators: RegDef and DefUseMI.
      // This is not implemented.
      report_fatal_error("Repairing between terminators not implemented yet");
  }

  // Create a new temporary to hold the repaired value.
  unsigned NewReg =
      MRI->createGenericVirtualRegister(PartialMap.Mask.getBitWidth());
  // Set the registers for the source and destination of the copy.
  unsigned Src = Reg, Dst = NewReg;
  // If this is a definition that we repair, the copy will be
  // inverted.
  if (IsDef)
    std::swap(Src, Dst);
  (void)MIRBuilder.buildInstr(TargetOpcode::COPY, Dst, Src);

  DEBUG(dbgs() << "Repair: " << PrintReg(Reg) << " with: "
        << PrintReg(NewReg) << '\n');

  // Restore the insertion point of the MIRBuilder.
  MIRBuilder.setInstr(DefUseMI, Before);
  return NewReg;
}

void RegBankSelect::setSafeInsertionPoint(MachineInstr &InsertPt, bool Before) {
  // Check that we are not looking to insert before a phi.
  // Indeed, we would need more information on what to do.
  // By default that should be all the predecessors, but this is
  // probably not what we want in general.
  assert((!Before || !InsertPt.isPHI()) &&
         "Insertion before phis not implemented");
  // The same kind of observation hold for terminators if we try to
  // insert after them.
  assert((Before || !InsertPt.isTerminator()) &&
         "Insertion after terminatos not implemented");
  if (InsertPt.isPHI()) {
    assert(!Before && "Not supported!!");
    MachineBasicBlock *MBB = InsertPt.getParent();
    assert(MBB && "Insertion point is not in a basic block");
    MachineBasicBlock::iterator FirstNonPHIPt = MBB->getFirstNonPHI();
    if (FirstNonPHIPt == MBB->end()) {
      // If there is not any non-phi instruction, insert at the end of MBB.
      MIRBuilder.setMBB(*MBB, /*Beginning*/ false);
      return;
    }
    // The insertion point before the first non-phi instruction.
    MIRBuilder.setInstr(*FirstNonPHIPt, /*Before*/ true);
    return;
  }
  if (InsertPt.isTerminator()) {
    MachineBasicBlock *MBB = InsertPt.getParent();
    assert(MBB && "Insertion point is not in a basic block");
    MIRBuilder.setInstr(*MBB->getFirstTerminator(), /*Before*/ true);
    return;
  }
  MIRBuilder.setInstr(InsertPt, /*Before*/ Before);
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
    // For definitions, changing the register bank will affect all
    // its uses, and in particular the ones we already visited.
    // Although this is correct, since with the RPO traversal of the
    // basic blocks the only uses that we already visisted for this
    // definition are PHIs (i.e., copies), this may not be the best
    // solution according to the cost model.
    // Therefore, create a new temporary for Reg.
    assert(ValMapping.BreakDown.size() == 1 &&
           "Support for complex break down not supported yet");
    if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
        MRI->getRegClassOrRegBank(Reg)) {
      if (!MO.isDef() && MI.isPHI()) {
        // Phis are already copies, so there is nothing to repair.
        // Note: This will not hold when we support break downs with
        // more than one segment.
        DEBUG(dbgs() << "Skip PHI use\n");
        continue;
      }
      // If MO is a definition, since repairing after a terminator is
      // painful, do not repair. Indeed, this is probably not worse
      // saving the move in the PHIs that will get reassigned.
      if (!MO.isDef() || !MI.isTerminator())
        Reg = repairReg(Reg, ValMapping, MI, MO.isDef());
    }

    // If we end up here, MO should be free of encoding constraints,
    // i.e., we do not have to constrained the RegBank of Reg to
    // the requirement of the operands.
    // If that is not the case, this means the code was broken before
    // hands because we should have found that the assignment match.
    // This will not hold when we will consider alternative mappings.
    DEBUG(dbgs() << "Assign: " << *ValMapping.BreakDown[0].RegBank << " to "
                 << PrintReg(Reg) << '\n');

    MRI->setRegBank(Reg, *ValMapping.BreakDown[0].RegBank);
    MO.setReg(Reg);
  }
  DEBUG(dbgs() << "Assigned: " << MI);
}

bool RegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "Assign register banks for: " << MF.getName() << '\n');
  init(MF);
  // Walk the function and assign register banks to all operands.
  // Use a RPOT to make sure all registers are assigned before we choose
  // the best mapping of the current instruction.
  ReversePostOrderTraversal<MachineFunction*> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT)
    for (MachineInstr &MI : *MBB)
      assignInstr(MI);
  return false;
}
