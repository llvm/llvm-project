//===-- Nios2InstrInfo.cpp - Nios2 Instruction Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Nios2 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Nios2InstrInfo.h"
#include "Nios2TargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "Nios2GenInstrInfo.inc"

// Pin the vtable to this file.
void Nios2InstrInfo::anchor() {}

Nios2InstrInfo::Nios2InstrInfo(Nios2Subtarget &ST)
    : Nios2GenInstrInfo(), RI(ST), Subtarget(ST) {}

/// Expand Pseudo instructions into real backend instructions
bool Nios2InstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();

  switch (MI.getDesc().getOpcode()) {
  default:
    return false;
  case Nios2::RetRA:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(Nios2::RET_R1)).addReg(Nios2::RA);
    break;
  }

  MBB.erase(MI);
  return true;
}
