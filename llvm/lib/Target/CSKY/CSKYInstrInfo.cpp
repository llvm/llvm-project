//===-- CSKYInstrInfo.h - CSKY Instruction Information --------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the CSKY implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "CSKYInstrInfo.h"
#include "CSKYMachineFunctionInfo.h"
#include "CSKYTargetMachine.h"
#include "llvm/MC/MCContext.h"

#define DEBUG_TYPE "csky-instr-info"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "CSKYGenInstrInfo.inc"

CSKYInstrInfo::CSKYInstrInfo(CSKYSubtarget &STI)
    : CSKYGenInstrInfo(CSKY::ADJCALLSTACKDOWN, CSKY::ADJCALLSTACKUP), STI(STI) {
}

Register CSKYInstrInfo::movImm(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI,
                               const DebugLoc &DL, int64_t Val,
                               MachineInstr::MIFlag Flag) const {
  assert(isUInt<32>(Val) && "should be uint32");

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  Register DstReg;
  if (STI.hasE2()) {
    DstReg = MRI.createVirtualRegister(&CSKY::GPRRegClass);

    if (isUInt<16>(Val)) {
      BuildMI(MBB, MBBI, DL, get(CSKY::MOVI32), DstReg)
          .addImm(Val & 0xFFFF)
          .setMIFlags(Flag);
    } else if (isShiftedUInt<16, 16>(Val)) {
      BuildMI(MBB, MBBI, DL, get(CSKY::MOVIH32), DstReg)
          .addImm((Val >> 16) & 0xFFFF)
          .setMIFlags(Flag);
    } else {
      BuildMI(MBB, MBBI, DL, get(CSKY::MOVIH32), DstReg)
          .addImm((Val >> 16) & 0xFFFF)
          .setMIFlags(Flag);
      BuildMI(MBB, MBBI, DL, get(CSKY::ORI32), DstReg)
          .addReg(DstReg)
          .addImm(Val & 0xFFFF)
          .setMIFlags(Flag);
    }

  } else {
    DstReg = MRI.createVirtualRegister(&CSKY::mGPRRegClass);
    if (isUInt<8>(Val)) {
      BuildMI(MBB, MBBI, DL, get(CSKY::MOVI16), DstReg)
          .addImm(Val & 0xFF)
          .setMIFlags(Flag);
    } else if (isUInt<16>(Val)) {
      BuildMI(MBB, MBBI, DL, get(CSKY::MOVI16), DstReg)
          .addImm((Val >> 8) & 0xFF)
          .setMIFlags(Flag);
      BuildMI(MBB, MBBI, DL, get(CSKY::LSLI16), DstReg)
          .addReg(DstReg)
          .addImm(8)
          .setMIFlags(Flag);
      if ((Val & 0xFF) != 0)
        BuildMI(MBB, MBBI, DL, get(CSKY::ADDI16), DstReg)
            .addReg(DstReg)
            .addImm(Val & 0xFF)
            .setMIFlags(Flag);
    } else if (isUInt<24>(Val)) {
      BuildMI(MBB, MBBI, DL, get(CSKY::MOVI16), DstReg)
          .addImm((Val >> 16) & 0xFF)
          .setMIFlags(Flag);
      BuildMI(MBB, MBBI, DL, get(CSKY::LSLI16), DstReg)
          .addReg(DstReg)
          .addImm(8)
          .setMIFlags(Flag);
      if (((Val >> 8) & 0xFF) != 0)
        BuildMI(MBB, MBBI, DL, get(CSKY::ADDI16), DstReg)
            .addReg(DstReg)
            .addImm((Val >> 8) & 0xFF)
            .setMIFlags(Flag);
      BuildMI(MBB, MBBI, DL, get(CSKY::LSLI16), DstReg)
          .addReg(DstReg)
          .addImm(8)
          .setMIFlags(Flag);
      if ((Val & 0xFF) != 0)
        BuildMI(MBB, MBBI, DL, get(CSKY::ADDI16), DstReg)
            .addReg(DstReg)
            .addImm(Val & 0xFF)
            .setMIFlags(Flag);
    } else {
      BuildMI(MBB, MBBI, DL, get(CSKY::MOVI16), DstReg)
          .addImm((Val >> 24) & 0xFF)
          .setMIFlags(Flag);
      BuildMI(MBB, MBBI, DL, get(CSKY::LSLI16), DstReg)
          .addReg(DstReg)
          .addImm(8)
          .setMIFlags(Flag);
      if (((Val >> 16) & 0xFF) != 0)
        BuildMI(MBB, MBBI, DL, get(CSKY::ADDI16), DstReg)
            .addReg(DstReg)
            .addImm((Val >> 16) & 0xFF)
            .setMIFlags(Flag);
      BuildMI(MBB, MBBI, DL, get(CSKY::LSLI16), DstReg)
          .addReg(DstReg)
          .addImm(8)
          .setMIFlags(Flag);
      if (((Val >> 8) & 0xFF) != 0)
        BuildMI(MBB, MBBI, DL, get(CSKY::ADDI16), DstReg)
            .addReg(DstReg)
            .addImm((Val >> 8) & 0xFF)
            .setMIFlags(Flag);
      BuildMI(MBB, MBBI, DL, get(CSKY::LSLI16), DstReg)
          .addReg(DstReg)
          .addImm(8)
          .setMIFlags(Flag);
      if ((Val & 0xFF) != 0)
        BuildMI(MBB, MBBI, DL, get(CSKY::ADDI16), DstReg)
            .addReg(DstReg)
            .addImm(Val & 0xFF)
            .setMIFlags(Flag);
    }
  }

  return DstReg;
}

unsigned CSKYInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                            int &FrameIndex) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
  case CSKY::LD16B:
  case CSKY::LD16H:
  case CSKY::LD16W:
  case CSKY::LD32B:
  case CSKY::LD32BS:
  case CSKY::LD32H:
  case CSKY::LD32HS:
  case CSKY::LD32W:
  case CSKY::RESTORE_CARRY:
    break;
  }

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }

  return 0;
}

unsigned CSKYInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                           int &FrameIndex) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
  case CSKY::ST16B:
  case CSKY::ST16H:
  case CSKY::ST16W:
  case CSKY::ST32B:
  case CSKY::ST32H:
  case CSKY::ST32W:
  case CSKY::SPILL_CARRY:
    break;
  }

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }

  return 0;
}

void CSKYInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        Register SrcReg, bool IsKill, int FI,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  CSKYMachineFunctionInfo *CFI = MF.getInfo<CSKYMachineFunctionInfo>();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  unsigned Opcode = 0;

  if (CSKY::GPRRegClass.hasSubClassEq(RC)) {
    Opcode = CSKY::ST32W; // Optimize for 16bit
  } else if (CSKY::CARRYRegClass.hasSubClassEq(RC)) {
    Opcode = CSKY::SPILL_CARRY;
    CFI->setSpillsCR();
  } else {
    llvm_unreachable("Unknown RegisterClass");
  }

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FI), MachineMemOperand::MOStore,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  BuildMI(MBB, I, DL, get(Opcode))
      .addReg(SrcReg, getKillRegState(IsKill))
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
}

void CSKYInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         Register DestReg, int FI,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  CSKYMachineFunctionInfo *CFI = MF.getInfo<CSKYMachineFunctionInfo>();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  unsigned Opcode = 0;

  if (CSKY::GPRRegClass.hasSubClassEq(RC)) {
    Opcode = CSKY::LD32W;
  } else if (CSKY::CARRYRegClass.hasSubClassEq(RC)) {
    Opcode = CSKY::RESTORE_CARRY;
    CFI->setSpillsCR();
  } else {
    llvm_unreachable("Unknown RegisterClass");
  }

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FI), MachineMemOperand::MOLoad,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  BuildMI(MBB, I, DL, get(Opcode), DestReg)
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
}

void CSKYInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I,
                                const DebugLoc &DL, MCRegister DestReg,
                                MCRegister SrcReg, bool KillSrc) const {

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  if (CSKY::GPRRegClass.contains(SrcReg) &&
      CSKY::CARRYRegClass.contains(DestReg)) {
    if (STI.hasE2()) {
      BuildMI(MBB, I, DL, get(CSKY::BTSTI32), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc))
          .addImm(0);
    } else {
      assert(SrcReg < CSKY::R8);
      BuildMI(MBB, I, DL, get(CSKY::BTSTI16), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc))
          .addImm(0);
    }
    return;
  }

  if (CSKY::CARRYRegClass.contains(SrcReg) &&
      CSKY::GPRRegClass.contains(DestReg)) {

    if (STI.hasE2()) {
      BuildMI(MBB, I, DL, get(CSKY::MVC32), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      assert(DestReg < CSKY::R16);
      assert(DestReg < CSKY::R8);
      BuildMI(MBB, I, DL, get(CSKY::MOVI16), DestReg).addImm(0);
      BuildMI(MBB, I, DL, get(CSKY::ADDC16))
          .addReg(DestReg, RegState::Define)
          .addReg(SrcReg, RegState::Define)
          .addReg(DestReg, getKillRegState(true))
          .addReg(DestReg, getKillRegState(true))
          .addReg(SrcReg, getKillRegState(true));
      BuildMI(MBB, I, DL, get(CSKY::BTSTI16))
          .addReg(SrcReg, RegState::Define | getDeadRegState(KillSrc))
          .addReg(DestReg)
          .addImm(0);
    }
    return;
  }

  unsigned Opcode = 0;
  if (CSKY::GPRRegClass.contains(DestReg, SrcReg))
    Opcode = CSKY::MOV32;
  else {
    LLVM_DEBUG(dbgs() << "src = " << SrcReg << ", dst = " << DestReg);
    LLVM_DEBUG(I->dump());
    llvm_unreachable("Unknown RegisterClass");
  }

  BuildMI(MBB, I, DL, get(Opcode), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}
