//===- llvm/CodeGen/GlobalISel/InstructionSelector.cpp -----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the InstructionSelector class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define DEBUG_TYPE "instructionselector"

using namespace llvm;

InstructionSelector::InstructionSelector() {}

bool InstructionSelector::constrainSelectedInstRegOperands(
    MachineInstr &I, const TargetInstrInfo &TII, const TargetRegisterInfo &TRI,
    const RegisterBankInfo &RBI) const {
  MachineBasicBlock &MBB = *I.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  for (unsigned OpI = 0, OpE = I.getNumExplicitOperands(); OpI != OpE; ++OpI) {
    MachineOperand &MO = I.getOperand(OpI);

    // There's nothing to be done on non-register operands.
    if (!MO.isReg())
      continue;

    DEBUG(dbgs() << "Converting operand: " << MO << '\n');
    assert(MO.isReg() && "Unsupported non-reg operand");

    unsigned Reg = MO.getReg();
    // Physical registers don't need to be constrained.
    if (TRI.isPhysicalRegister(Reg))
      continue;

    // Register operands with a value of 0 (e.g. predicate operands) don't need
    // to be constrained.
    if (Reg == 0)
      continue;

    // If the operand is a vreg, we should constrain its regclass, and only
    // insert COPYs if that's impossible.
    // constrainOperandRegClass does that for us.
    MO.setReg(constrainOperandRegClass(MF, TRI, MRI, TII, RBI, I, I.getDesc(),
                                       Reg, OpI));

    // Tie uses to defs as indicated in MCInstrDesc.
    if (MO.isUse()) {
      int DefIdx = I.getDesc().getOperandConstraint(OpI, MCOI::TIED_TO);
      if (DefIdx != -1)
        I.tieOperands(DefIdx, OpI);
    }
  }
  return true;
}

Optional<int64_t>
InstructionSelector::getConstantVRegVal(unsigned VReg,
                                        const MachineRegisterInfo &MRI) const {
  MachineInstr *MI = MRI.getVRegDef(VReg);
  if (MI->getOpcode() != TargetOpcode::G_CONSTANT)
    return None;

  if (MI->getOperand(1).isImm())
    return MI->getOperand(1).getImm();

  if (MI->getOperand(1).isCImm() &&
      MI->getOperand(1).getCImm()->getBitWidth() <= 64)
    return MI->getOperand(1).getCImm()->getSExtValue();

  return None;
}

bool InstructionSelector::isOperandImmEqual(
    const MachineOperand &MO, int64_t Value,
    const MachineRegisterInfo &MRI) const {

  if (MO.getReg())
    if (auto VRegVal = getConstantVRegVal(MO.getReg(), MRI))
      return *VRegVal == Value;
  return false;
}
