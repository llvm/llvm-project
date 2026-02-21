//===-- ConnexRegisterInfo.cpp - Connex Register Information ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Connex implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ConnexRegisterInfo.h"
#include "Connex.h"
#include "ConnexSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_REGINFO_TARGET_DESC
#include "ConnexGenRegisterInfo.inc"
using namespace llvm;

#include "llvm/Support/Debug.h" // for dbgs and LLVM_DEBUG() macro
#define DEBUG_TYPE "mc-inst-lower"

ConnexRegisterInfo::ConnexRegisterInfo() : ConnexGenRegisterInfo(Connex::R0) {}

// Inspired from lib/Target/Mips/MipsRegisterInfo.cpp
const TargetRegisterClass *
ConnexRegisterInfo::getPointerRegClass(const MachineFunction &MF,
                                       unsigned Kind) const {
  return &Connex::GPRRegClass;
}

const MCPhysReg *
ConnexRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_SaveList;
}

BitVector ConnexRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  int numRegs = getNumRegs();

  LLVM_DEBUG(dbgs() << "getReservedRegs(): numRegs = " << numRegs << "\n");

  BitVector Reserved(numRegs);
  // We reserve scalar registers
  Reserved.set(Connex::R10); // R10 is read only frame pointer
  Reserved.set(Connex::R11); // R11 is pseudo stack pointer

  /*
  We now reserve vector registers.
  Wh30, vector register R(30), is used by me to codegen:
    - LLVM's VSELECT on Connex in ConnexTargetMachine.cpp
                                                      - PassAfterPostRAScheduler
      (NO longer: in ConnexISelLowering::Lower() for VSELECT to be
          lowered to WHERE*).
       Doing so we avoid errors like:
         <<*** Bad machine code: Using an undefined physical register ***
         - function:    IfConversion
         - basic block: BB#6 vector.body (0x1501fd8)
         - instruction: %vreg47<def> = COPY
         - operand 1:   %Wh31>>

    - in ConnexInstrInfo::copyPhysReg() .
  */
  /*
  LLVM_DEBUG(dbgs() << "getReservedRegs(): CONNEX_RESERVED_REGISTER_01/02/03 = "
                    << CONNEX_RESERVED_REGISTER_01 << "(normally Wh30)/"
                    << CONNEX_RESERVED_REGISTER_02 << "(normally Wh31)/"
                    << CONNEX_RESERVED_REGISTER_03 << "(normally Wh31)"
                    << "\n");
  */
  Reserved.set(CONNEX_RESERVED_REGISTER_01);
  Reserved.set(CONNEX_RESERVED_REGISTER_02);
  Reserved.set(CONNEX_RESERVED_REGISTER_03);

  return Reserved;
}

// From book Lopes_2014:
//  eliminateFrameIndex
//  "implements this replacement by converting each frame index to a real stack
//   offset for all machine instructions that contain stack references (usually
//   loads and stores). Extra instructions are also generated whenever
//   additional stack offset arithmetic is necessary".
bool ConnexRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                             int SPAdj, unsigned FIOperandNum,
                                             RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  DebugLoc DL = MI.getDebugLoc();

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  unsigned FrameReg = getFrameRegister(MF);
  int FrameIndex = MI.getOperand(i).getIndex();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineBasicBlock &MBB = *MI.getParent();

  if (MI.getOpcode() == Connex::MOV_rr) {
    MI.getOperand(i).ChangeToRegister(FrameReg, false);

    // TODO MAYBE: we took out the scalar ADD and therefore we have to comment
    // this
    int Offset = MF.getFrameInfo().getObjectOffset(FrameIndex);
    unsigned reg = MI.getOperand(i - 1).getReg();

    BuildMI(MBB, ++II, DL, TII.get(Connex::ADD_ri), reg)
        .addReg(reg)
        .addImm(Offset);

    return false;
  }

  int Offset = MF.getFrameInfo().getObjectOffset(FrameIndex) +
               MI.getOperand(i + 1).getImm();

  if (!isInt<32>(Offset))
    llvm_unreachable("bug in frame offset");

  if (MI.getOpcode() == Connex::FI_ri) {
    // architecture does not really support FI_ri, replace it with
    //    MOV_rr <target_reg>, frame_reg
    //    ADD_ri <target_reg>, imm
    unsigned reg = MI.getOperand(i - 1).getReg();

    BuildMI(MBB, ++II, DL, TII.get(Connex::MOV_rr), reg).addReg(FrameReg);

    // TODO MAYBE: we took out the scalar ADD and therefore we have to comment
    // this
    BuildMI(MBB, II, DL, TII.get(Connex::ADD_ri), reg)
        .addReg(reg)
        .addImm(Offset);

    // Remove FI_ri instruction
    MI.eraseFromParent();
  } else {
    MI.getOperand(i).ChangeToRegister(FrameReg, false);
    MI.getOperand(i + 1).ChangeToImmediate(Offset);
  }

  return false;
}

Register ConnexRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  // MEGA-TODO: in principle we should return also for the Connex vector
  // processor a vector register like: Connex::Wh28
  return Connex::R10;
}
