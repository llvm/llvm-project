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
#include "llvm/CodeGen/RegisterScavenging.h"
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
    : M88kGenInstrInfo(M88k::ADJCALLSTACKDOWN, M88k::ADJCALLSTACKUP), RI(),
      STI(STI) {}

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

bool M88kInstrInfo::isBranchOffsetInRange(unsigned BranchOpc,
                                          int64_t BrOffset) const {
  assert(BranchOpc == M88k::BR || BranchOpc == M88k::BSR ||
         BranchOpc == M88k::BB0 ||
         BranchOpc == M88k::BB1 && "Unexpected branch opcode");
  int Bits = (BranchOpc == M88k::BR || BranchOpc == M88k::BSR) ? 26 : 16;
  return isIntN(Bits, BrOffset / 4);
}

MachineBasicBlock *
M88kInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  assert(MI.getDesc().isBranch() && "Unexpected opcode!");
  // The branch target is always the last operand.
  int NumOp = MI.getNumExplicitOperands();
  return MI.getOperand(NumOp - 1).getMBB();
}

void M88kInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                                         MachineBasicBlock &NewDestBB,
                                         MachineBasicBlock &RestoreBB,
                                         const DebugLoc &DL, int64_t BrOffset,
                                         RegScavenger *RS) const {
  assert(RS && "RegScavenger required for long branching");
  assert(MBB.empty() &&
         "new block should be inserted for expanding unconditional branch");
  assert(MBB.pred_size() == 1);

  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  // FIXME: A virtual register must be used initially, as the register
  // scavenger won't work with empty blocks (SIInstrInfo::insertIndirectBranch
  // uses the same workaround).
  Register ScratchReg = MRI.createVirtualRegister(&M88k::GPRRCRegClass);
  auto I = MBB.end();

  // Load address of destination BB.
  BuildMI(MBB, I, DL, get(M88k::ORriu))
      .addReg(ScratchReg, RegState::Define | RegState::Dead)
      .addReg(M88k::R0)
      .addMBB(&NewDestBB, M88kII::MO_ABS_HI);
  BuildMI(MBB, I, DL, get(M88k::ORri))
      .addReg(ScratchReg)
      .addReg(ScratchReg)
      .addMBB(&NewDestBB, M88kII::MO_ABS_LO);

  MachineInstr *MI = BuildMI(MBB, I, DL, get(M88k::JMP)).addReg(ScratchReg);

  RS->enterBasicBlockEnd(MBB);
  unsigned Scav = RS->scavengeRegisterBackwards(M88k::GPRRCRegClass,
                                                MI->getIterator(), false, 0);

  // TODO: The case when there is no scavenged register needs special handling.
  assert(Scav != M88k::NoRegister && "No register is scavenged!");
  MRI.replaceRegWith(ScratchReg, Scav);
  MRI.clearVirtRegs();
  RS->setRegUsed(Scav);
}

unsigned M88kInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                            int &FrameIndex) const {
  switch (MI.getOpcode()) {
  // TODO Check which LD instructions are really selected.
  case M88k::LDrib:
  case M88k::LDrih:
  case M88k::LDriw:
  case M88k::LDrid:
  case M88k::LDrrsb:
  case M88k::LDrrsbu:
  case M88k::LDrrsd:
  case M88k::LDrrsdu:
  case M88k::LDrrsh:
  case M88k::LDrrshu:
  case M88k::LDrrsw:
  case M88k::LDrrswu:
  case M88k::LDrrub:
  case M88k::LDrrubu:
  case M88k::LDrrud:
  case M88k::LDrrudu:
  case M88k::LDrruh:
  case M88k::LDrruhu:
  case M88k::LDrruw:
  case M88k::LDrruwu:
    if (MI.getOperand(1).isFI()) {
      FrameIndex = MI.getOperand(1).getIndex();
      return MI.getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned M88kInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                           int &FrameIndex) const {
  switch (MI.getOpcode()) {
  // TODO Check which ST instructions are really selected.
  case M88k::STrib:
  case M88k::STrih:
  case M88k::STriw:
  case M88k::STrid:
  case M88k::STrrsb:
  case M88k::STrrsbt:
  case M88k::STrrsbu:
  case M88k::STrrsbut:
  case M88k::STrrsd:
  case M88k::STrrsdt:
  case M88k::STrrsdu:
  case M88k::STrrsdut:
  case M88k::STrrsh:
  case M88k::STrrsht:
  case M88k::STrrshu:
  case M88k::STrrshut:
  case M88k::STrrsw:
  case M88k::STrrswt:
  case M88k::STrrswu:
  case M88k::STrrswut:
  case M88k::STrrub:
  case M88k::STrrubt:
  case M88k::STrrubu:
  case M88k::STrrubut:
  case M88k::STrrud:
  case M88k::STrrudt:
  case M88k::STrrudu:
  case M88k::STrrudut:
  case M88k::STrruh:
  case M88k::STrruht:
  case M88k::STrruhu:
  case M88k::STrruhut:
  case M88k::STrruw:
  case M88k::STrruwt:
  case M88k::STrruwu:
  case M88k::STrruwut:
  case M88k::STxid:
  case M88k::STxis:
  case M88k::STxix:
  case M88k::STxrd:
  case M88k::STxrdt:
  case M88k::STxrdu:
  case M88k::STxrdut:
  case M88k::STxrss:
  case M88k::STxrsst:
  case M88k::STxrssu:
  case M88k::STxrssut:
  case M88k::STxrsx:
  case M88k::STxrsxt:
  case M88k::STxrsxu:
  case M88k::STxrsxut:
  case M88k::STxrud:
  case M88k::STxrudt:
  case M88k::STxrudu:
  case M88k::STxrudut:
  case M88k::STxrus:
  case M88k::STxrust:
  case M88k::STxrusu:
  case M88k::STxrusut:
  case M88k::STxrux:
  case M88k::STxruxt:
  case M88k::STxruxu:
  case M88k::STxruxut:
    if (MI.getOperand(1).isFI()) {
      FrameIndex = MI.getOperand(1).getIndex();
      return MI.getOperand(0).getReg();
    }
    break;
  }
  return 0;
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
      .addImm(0)
      .addMemOperand(MMO);
}

void M88kInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI,
                                         Register DestReg, int FrameIndex,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  MachineMemOperand *MMO =
      getMachineMemOperand(MBB, FrameIndex, MachineMemOperand::MOLoad);

  // Build an LDriw instruction.
  BuildMI(MBB, MBBI, DL, get(M88k::LDriw))
      .addReg(DestReg, RegState::Define)
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addMemOperand(MMO);
}

unsigned M88kInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  if (MI.isInlineAsm()) {
    const MachineFunction *MF = MI.getParent()->getParent();
    const char *AsmStr = MI.getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  }
  return MI.getDesc().getSize();
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

  if (M88k::GPRRCRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(M88k::ORrr), DestReg)
        .addReg(M88k::R0)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  unsigned Opc;
  if (M88k::XRRCRegClass.contains(DestReg, SrcReg))
    Opc = M88k::MOVxx;
  else if (M88k::GPRRCRegClass.contains(DestReg) &&
           M88k::XRRCRegClass.contains(SrcReg))
    Opc = M88k::MOVrxs;
  else if (M88k::GPR64RCRegClass.contains(DestReg) &&
           M88k::XRRCRegClass.contains(SrcReg))
    Opc = M88k::MOVrxd;
  else if (M88k::XRRCRegClass.contains(DestReg) &&
           M88k::GPRRCRegClass.contains(SrcReg))
    Opc = M88k::MOVxrs;
  else if (M88k::XRRCRegClass.contains(DestReg) &&
           M88k::GPR64RCRegClass.contains(SrcReg))
    Opc = M88k::MOVxrd;
  else
    llvm_unreachable("m88: Impossible reg-to-reg copy");

  BuildMI(MBB, MBBI, DL, get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}
