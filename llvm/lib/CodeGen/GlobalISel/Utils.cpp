//===- llvm/CodeGen/GlobalISel/Utils.cpp -------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file This file implements the utility functions used by the GlobalISel
/// pipeline.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define DEBUG_TYPE "globalisel-utils"

using namespace llvm;

unsigned llvm::constrainOperandRegClass(
    const MachineFunction &MF, const TargetRegisterInfo &TRI,
    MachineRegisterInfo &MRI, const TargetInstrInfo &TII,
    const RegisterBankInfo &RBI, MachineInstr &InsertPt, const MCInstrDesc &II,
    unsigned Reg, unsigned OpIdx) {
  // Assume physical registers are properly constrained.
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "PhysReg not implemented");

  const TargetRegisterClass *RegClass = TII.getRegClass(II, OpIdx, &TRI, MF);

  if (!RBI.constrainGenericRegister(Reg, *RegClass, MRI)) {
    unsigned NewReg = MRI.createVirtualRegister(RegClass);
    BuildMI(*InsertPt.getParent(), InsertPt, InsertPt.getDebugLoc(),
            TII.get(TargetOpcode::COPY), NewReg)
        .addReg(Reg);
    return NewReg;
  }

  return Reg;
}
