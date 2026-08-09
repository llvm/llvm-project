//===-- SuperHInstrInfo.cpp - SuperH Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SuperHInstrInfo.h"
#include "SuperHRegisterInfo.h"
#include "SuperHSubtarget.h"
#include "SuperHTargetMachine.h"
#include "SuperH.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "sh-instrinfo"

#define GET_INSTRINFO_CTOR_DTOR
#include "SuperHGenInstrInfo.inc"

SuperHInstrInfo::SuperHInstrInfo(const SuperHSubtarget &ST)
    : SuperHGenInstrInfo(ST, RI, SH::ADJCALLSTACKDOWN, SH::ADJCALLSTACKUP),
      RI(ST), Subtarget(ST) { }

void SuperHInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, const DebugLoc &DL,
                           Register DestReg, Register SrcReg, bool KillSrc,
                           bool RenamableDest,
                           bool RenamableSrc) const {

  // If the targets are GPR registers, use MOV Rm, Rn.
  if (SH::GPRRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, MI, DL, get(SH::MOVRmRn), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Otherwise this is not possible.
  llvm_unreachable("Impossible reg-to-reg copy");
}

bool SuperHInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  unsigned OpCode = MI.getOpcode();
  switch(OpCode) {
  case SH::DIVURmRn:
  case SH::DIVSRmRn:
    return expandDIV(OpCode, MI);
  default:
    return false;
  }
}





//===----------------------------------------------------------------------===//
//                        Pseudo Instruction Expansion
//===----------------------------------------------------------------------===//

// Expands the division psuedo instructions into valid SuperH sequences.
// SuperH sets the division mode with an struction inserted before.
bool SuperHInstrInfo::expandDIV(unsigned Opcode, MachineInstr &MI) const {
  assert(MI.getOperand(0).isReg() && "Expected register in op0 for expansion!");
  assert(MI.getOperand(1).isReg() && "Expected register in op1 for expansion!");
  auto &MBB = *MI.getParent();
  auto Lhs = MI.getOperand(0).getReg();
  auto Rhs = MI.getOperand(1).getReg();
  auto DL = MI.getDebugLoc();

  switch(Opcode) {

  // Expand DIVURmRn to the following sequence:
  // div0u
  // div1 Rm, Rn
  case SH::DIVURmRn: {
    BuildMI(MBB, MI, DL, get(SH::DIV0U));
    BuildMI(MBB, MI, DL, get(SH::DIV1RmRn), Rhs)
      .addReg(Lhs);
    MI.removeFromParent();
    return true;
  }

  // Expand DIVSRmRn to the following sequence:
  // div0s Rm, Rn
  // div1 Rm, Rn
  case SH::DIVSRmRn: {
    BuildMI(MBB, MI, DL, get(SH::DIV0SRmRn))
      .addReg(Rhs)
      .addReg(Lhs);
    BuildMI(MBB, MI, DL, get(SH::DIV1RmRn), Rhs)
      .addReg(Lhs);
    MI.removeFromParent();
    return true;
  }

  // This shouldn't be reached.
  default: {
    llvm_unreachable("expandDIV was wrongfully called on a non-div pseudo!");
    return false; 
  }
  }
}