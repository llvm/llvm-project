//=- LoongArchInstrInfo.cpp - LoongArch Instruction Information -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the LoongArch implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "LoongArchInstrInfo.h"
#include "LoongArch.h"
#include "LoongArchMachineFunctionInfo.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "LoongArchGenInstrInfo.inc"

LoongArchInstrInfo::LoongArchInstrInfo(LoongArchSubtarget &STI)
    : LoongArchGenInstrInfo(LoongArch::ADJCALLSTACKDOWN,
                            LoongArch::ADJCALLSTACKUP) {}

void LoongArchInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     const DebugLoc &DL, MCRegister DstReg,
                                     MCRegister SrcReg, bool KillSrc) const {
  if (LoongArch::GPRRegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(LoongArch::OR), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(LoongArch::R0);
    return;
  }

  // FPR->FPR copies.
  unsigned Opc;
  if (LoongArch::FPR32RegClass.contains(DstReg, SrcReg)) {
    Opc = LoongArch::FMOV_S;
  } else if (LoongArch::FPR64RegClass.contains(DstReg, SrcReg)) {
    Opc = LoongArch::FMOV_D;
  } else {
    // TODO: support other copies.
    llvm_unreachable("Impossible reg-to-reg copy");
  }

  BuildMI(MBB, MBBI, DL, get(Opc), DstReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}

void LoongArchInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, Register SrcReg,
    bool IsKill, int FI, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();
  MachineFunction *MF = MBB.getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  unsigned Opcode;
  if (LoongArch::GPRRegClass.hasSubClassEq(RC))
    Opcode = TRI->getRegSizeInBits(LoongArch::GPRRegClass) == 32
                 ? LoongArch::ST_W
                 : LoongArch::ST_D;
  else if (LoongArch::FPR32RegClass.hasSubClassEq(RC))
    Opcode = LoongArch::FST_S;
  else if (LoongArch::FPR64RegClass.hasSubClassEq(RC))
    Opcode = LoongArch::FST_D;
  else
    llvm_unreachable("Can't store this register to stack slot");

  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  BuildMI(MBB, I, DL, get(Opcode))
      .addReg(SrcReg, getKillRegState(IsKill))
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
}

void LoongArchInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, Register DstReg,
    int FI, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();
  MachineFunction *MF = MBB.getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  unsigned Opcode;
  if (LoongArch::GPRRegClass.hasSubClassEq(RC))
    Opcode = TRI->getRegSizeInBits(LoongArch::GPRRegClass) == 32
                 ? LoongArch::LD_W
                 : LoongArch::LD_D;
  else if (LoongArch::FPR32RegClass.hasSubClassEq(RC))
    Opcode = LoongArch::FLD_S;
  else if (LoongArch::FPR64RegClass.hasSubClassEq(RC))
    Opcode = LoongArch::FLD_D;
  else
    llvm_unreachable("Can't load this register from stack slot");

  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  BuildMI(MBB, I, DL, get(Opcode), DstReg)
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
}
