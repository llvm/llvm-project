//===-- M88kInstrInfo.cpp - M88k instruction information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the M88k implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "M88kInstrInfo.h"
#include "M88k.h"
#include "MCTargetDesc/M88kBaseInfo.h"
#include "MCTargetDesc/M88kMCTargetDesc.h"
//#include "M88kInstrBuilder.h"
#include "M88kSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <cstdint>
#include <iterator>

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRMAP_INFO
#include "M88kGenInstrInfo.inc"

#define DEBUG_TYPE "m88k-ii"

// Pin the vtable to this file.
void M88kInstrInfo::anchor() {}

M88kInstrInfo::M88kInstrInfo(M88kSubtarget &STI)
    : M88kGenInstrInfo(), RI(), STI(STI) {}

std::pair<unsigned, unsigned>
M88kInstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  return std::make_pair(TF, 0u);
}

ArrayRef<std::pair<unsigned, const char *>>
M88kInstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
  using namespace M88kII;

  static const std::pair<unsigned, const char *> Flags[] = {
      {MO_ABS_HI, "m88k-abs-hi"},
      {MO_ABS_LO, "m88k-abs-lo"},
  };
  return makeArrayRef(Flags);
}

static MachineMemOperand *getMachineMemOperand(MachineBasicBlock &MBB, int FI,
                                               MachineMemOperand::Flags Flags) {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  return MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                                 Flags, MFI.getObjectSize(FI),
                                 MFI.getObjectAlign(FI));
}

void M88kInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        Register SrcReg, bool isKill,
                                        int FrameIndex,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  MachineMemOperand *MMO =
      getMachineMemOperand(MBB, FrameIndex, MachineMemOperand::MOStore);

  // Build an STriw instruction.
  BuildMI(MBB, MBBI, DL, get(M88k::STriw))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FrameIndex)
      .addMemOperand(MMO);
}

void M88kInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI,
                                         Register DestReg, int FrameIndex,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  MachineMemOperand *MMO =
      getMachineMemOperand(MBB, FrameIndex, MachineMemOperand::MOStore);

  // Build an LDriw instruction.
  BuildMI(MBB, MBBI, DL, get(M88k::LDriw))
      .addReg(DestReg)
      .addFrameIndex(FrameIndex)
      .addMemOperand(MMO);
}

void M88kInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL, MCRegister DestReg,
                                   MCRegister SrcReg, bool KillSrc) const {
  // Split 64-bit GPR moves into two 64-bit moves. Add implicit uses of the
  // super register in case one of the subregs is undefined.
  if (M88k::GPR64RCRegClass.contains(DestReg, SrcReg)) {
    copyPhysReg(MBB, MBBI, DL, RI.getSubReg(DestReg, M88k::sub_hi),
                RI.getSubReg(SrcReg, M88k::sub_hi), KillSrc);
    MachineInstrBuilder(*MBB.getParent(), std::prev(MBBI))
      .addReg(SrcReg, RegState::Implicit);
    copyPhysReg(MBB, MBBI, DL, RI.getSubReg(DestReg, M88k::sub_lo),
                RI.getSubReg(SrcReg, M88k::sub_lo), KillSrc);
    MachineInstrBuilder(*MBB.getParent(), std::prev(MBBI))
      .addReg(SrcReg, (getKillRegState(KillSrc) | RegState::Implicit));
    return;
  }

  if (M88k::GPRRCRegClass.contains(DestReg, SrcReg) ||
      M88k::FPR32RCRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(M88k::ORrr), DestReg)
        .addReg(M88k::R0)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  unsigned Opc;
  if (M88k::XRRCRegClass.contains(DestReg, SrcReg))
    Opc = M88k::MOVxx;
  else if (M88k::FPR32RCRegClass.contains(DestReg) &&
           M88k::XRRCRegClass.contains(SrcReg))
    Opc = M88k::MOVrxs;
  else if (M88k::GPR64RCRegClass.contains(DestReg) &&
           M88k::XRRCRegClass.contains(SrcReg))
    Opc = M88k::MOVrxd;
  else if (M88k::XRRCRegClass.contains(DestReg) &&
           M88k::FPR32RCRegClass.contains(SrcReg))
    Opc = M88k::MOVxrs;
  else if (M88k::XRRCRegClass.contains(DestReg) &&
           M88k::GPR64RCRegClass.contains(SrcReg))
    Opc = M88k::MOVxrd;
  else
    llvm_unreachable("m88: Impossible reg-to-reg copy");

  BuildMI(MBB, MBBI, DL, get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}
