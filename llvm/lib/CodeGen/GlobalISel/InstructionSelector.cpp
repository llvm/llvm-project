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
#include "llvm/CodeGen/MachineInstr.h"
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

    // There's nothing to be done on immediates and frame indexes.
    if (MO.isImm() || MO.isFI())
      continue;

    DEBUG(dbgs() << "Converting operand: " << MO << '\n');
    assert(MO.isReg() && "Unsupported non-reg operand");

    // Physical registers don't need to be constrained.
    if (TRI.isPhysicalRegister(MO.getReg()))
      continue;

    const TargetRegisterClass *RC = TII.getRegClass(I.getDesc(), OpI, &TRI, MF);
    assert(RC && "Selected inst should have regclass operand");

    // If the operand is a vreg, we should constrain its regclass, and only
    // insert COPYs if that's impossible.
    // If the operand is a physreg, we only insert COPYs if the register class
    // doesn't contain the register.
    if (RBI.constrainGenericRegister(MO.getReg(), *RC, MRI))
      continue;

    DEBUG(dbgs() << "Constraining with COPYs isn't implemented yet");
    return false;
  }
  return true;
}
