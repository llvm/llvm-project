//===-- BPFInstrInfo.cpp - BPF Instruction Information ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the BPF implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "BPFInstrInfo.h"
#include "BPF.h"
#include "BPFMachineFunctionInfo.h"
#include "BPFSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <iterator>

#define GET_INSTRINFO_CTOR_DTOR
#include "BPFGenInstrInfo.inc"

using namespace llvm;

BPFInstrInfo::BPFInstrInfo(const BPFSubtarget &STI)
    : BPFGenInstrInfo(STI, RI, BPF::ADJCALLSTACKDOWN, BPF::ADJCALLSTACKUP),
      STI(STI) {}

// Return the inverse of a conditional branch, or 0 if there is none.
static unsigned getInverseBranchOpcode(unsigned Opcode) {
  switch (Opcode) {
#define BPF_INVERT(A, B)                                                       \
  case BPF::A##_rr:                                                            \
    return BPF::B##_rr;                                                        \
  case BPF::A##_ri:                                                            \
    return BPF::B##_ri;                                                        \
  case BPF::A##_rr_32:                                                         \
    return BPF::B##_rr_32;                                                     \
  case BPF::A##_ri_32:                                                         \
    return BPF::B##_ri_32;
#define BPF_INVERT_PAIR(A, B) BPF_INVERT(A, B) BPF_INVERT(B, A)
    BPF_INVERT_PAIR(JEQ, JNE)
    BPF_INVERT_PAIR(JUGT, JULE)
    BPF_INVERT_PAIR(JUGE, JULT)
    BPF_INVERT_PAIR(JSGT, JSLE)
    BPF_INVERT_PAIR(JSGE, JSLT)
#undef BPF_INVERT_PAIR
#undef BPF_INVERT
  default:
    return 0;
  }
}

static bool isCondBranch(unsigned Opcode) {
  return getInverseBranchOpcode(Opcode) || Opcode == BPF::JSET_rr ||
         Opcode == BPF::JSET_ri || Opcode == BPF::JSET_rr_32 ||
         Opcode == BPF::JSET_ri_32;
}

bool BPFInstrInfo::canAnalyzeConditionalBranches(MachineFunction &MF) const {
  // Before RA, splitting critical edges can spill past the stack limit.
  if (!MF.getProperties().hasNoVRegs() || !STI.getHasJmp32())
    return false;
  if (STI.hasGotol())
    return true;
  // Before v4, a new layout can put a branch out of 16-bit range, so only
  // functions that fit in it with room to spare are analyzed.
  auto *FI = MF.getInfo<BPFMachineFunctionInfo>();
  if (!FI->ShortBranches.has_value()) {
    // Debug instructions must not change code generation.
    unsigned Count = 0;
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB)
        Count += !MI.isMetaInstruction();
    FI->ShortBranches = Count <= INT16_MAX / 2;
  }
  return FI->ShortBranches.value();
}

void BPFInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I,
                               const DebugLoc &DL, Register DestReg,
                               Register SrcReg, bool KillSrc,
                               bool RenamableDest, bool RenamableSrc) const {
  if (BPF::GPRRegClass.contains(DestReg, SrcReg))
    BuildMI(MBB, I, DL, get(BPF::MOV_rr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
  else if (BPF::GPR32RegClass.contains(DestReg, SrcReg))
    BuildMI(MBB, I, DL, get(BPF::MOV_rr_32), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
  else
    llvm_unreachable("Impossible reg-to-reg copy");
}

void BPFInstrInfo::expandMEMCPY(MachineBasicBlock::iterator MI) const {
  Register DstReg = MI->getOperand(0).getReg();
  Register SrcReg = MI->getOperand(1).getReg();
  uint64_t CopyLen = MI->getOperand(2).getImm();
  uint64_t Alignment = MI->getOperand(3).getImm();
  Register ScratchReg = MI->getOperand(4).getReg();
  MachineBasicBlock *BB = MI->getParent();
  DebugLoc dl = MI->getDebugLoc();
  unsigned LdOpc, StOpc;

  switch (Alignment) {
  case 1:
    LdOpc = BPF::LDB;
    StOpc = BPF::STB;
    break;
  case 2:
    LdOpc = BPF::LDH;
    StOpc = BPF::STH;
    break;
  case 4:
    LdOpc = BPF::LDW;
    StOpc = BPF::STW;
    break;
  case 8:
    LdOpc = BPF::LDD;
    StOpc = BPF::STD;
    break;
  default:
    llvm_unreachable("unsupported memcpy alignment");
  }

  unsigned IterationNum = CopyLen >> Log2_64(Alignment);
  for(unsigned I = 0; I < IterationNum; ++I) {
    BuildMI(*BB, MI, dl, get(LdOpc))
            .addReg(ScratchReg, RegState::Define).addReg(SrcReg)
            .addImm(I * Alignment);
    BuildMI(*BB, MI, dl, get(StOpc))
            .addReg(ScratchReg, RegState::Kill).addReg(DstReg)
            .addImm(I * Alignment);
  }

  unsigned BytesLeft = CopyLen & (Alignment - 1);
  unsigned Offset = IterationNum * Alignment;
  bool Hanging4Byte = BytesLeft & 0x4;
  bool Hanging2Byte = BytesLeft & 0x2;
  bool Hanging1Byte = BytesLeft & 0x1;
  if (Hanging4Byte) {
    BuildMI(*BB, MI, dl, get(BPF::LDW))
            .addReg(ScratchReg, RegState::Define).addReg(SrcReg).addImm(Offset);
    BuildMI(*BB, MI, dl, get(BPF::STW))
            .addReg(ScratchReg, RegState::Kill).addReg(DstReg).addImm(Offset);
    Offset += 4;
  }
  if (Hanging2Byte) {
    BuildMI(*BB, MI, dl, get(BPF::LDH))
            .addReg(ScratchReg, RegState::Define).addReg(SrcReg).addImm(Offset);
    BuildMI(*BB, MI, dl, get(BPF::STH))
            .addReg(ScratchReg, RegState::Kill).addReg(DstReg).addImm(Offset);
    Offset += 2;
  }
  if (Hanging1Byte) {
    BuildMI(*BB, MI, dl, get(BPF::LDB))
            .addReg(ScratchReg, RegState::Define).addReg(SrcReg).addImm(Offset);
    BuildMI(*BB, MI, dl, get(BPF::STB))
            .addReg(ScratchReg, RegState::Kill).addReg(DstReg).addImm(Offset);
  }

  BB->erase(MI);
}

bool BPFInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  if (MI.getOpcode() == BPF::MEMCPY) {
    expandMEMCPY(MI);
    return true;
  }

  return false;
}

void BPFInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       Register SrcReg, bool IsKill, int FI,
                                       const TargetRegisterClass *RC,
                                       Register VReg,
                                       MachineInstr::MIFlag Flags) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  if (RC == &BPF::GPRRegClass)
    BuildMI(MBB, I, DL, get(BPF::STD))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI)
        .addImm(0);
  else if (RC == &BPF::GPR32RegClass)
    BuildMI(MBB, I, DL, get(BPF::STW32))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI)
        .addImm(0);
  else
    llvm_unreachable("Can't store this register to stack slot");
}

void BPFInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        Register DestReg, int FI,
                                        const TargetRegisterClass *RC,
                                        Register VReg, unsigned SubReg,
                                        MachineInstr::MIFlag Flags) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  if (RC == &BPF::GPRRegClass)
    BuildMI(MBB, I, DL, get(BPF::LDD), DestReg).addFrameIndex(FI).addImm(0);
  else if (RC == &BPF::GPR32RegClass)
    BuildMI(MBB, I, DL, get(BPF::LDW32), DestReg).addFrameIndex(FI).addImm(0);
  else
    llvm_unreachable("Can't load this register from stack slot");
}

bool BPFInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  TBB = FBB = nullptr;
  Cond.clear();

  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr())
      continue;

    // Working from the bottom, when we see a non-terminator
    // instruction, we're done.
    if (!isUnpredicatedTerminator(*I))
      break;

    // From base method doc: ... returning true if it cannot be understood ...
    // Indirect branch has multiple destinations and no true/false concepts.
    if (I->isIndirectBranch())
      return true;

    // A terminator that isn't a branch can't easily be handled
    // by this analysis.
    if (!I->isBranch())
      return true;

    // Handle unconditional branches.
    if (I->getOpcode() == BPF::JMP) {
      // Anything after a J is dead.
      Cond.clear();
      FBB = nullptr;
      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a J, delete them.
      MBB.erase(std::next(I), MBB.end());

      // Delete the J if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = nullptr;
        I->eraseFromParent();
        I = MBB.end();
        continue;
      }

      // TBB is used to indicate the unconditinal destination.
      TBB = I->getOperand(0).getMBB();
      continue;
    }

    if (!isCondBranch(I->getOpcode()) || !Cond.empty() ||
        !canAnalyzeConditionalBranches(*MBB.getParent()))
      return true;

    FBB = TBB;
    TBB = I->getOperand(2).getMBB();
    Cond.push_back(MachineOperand::CreateImm(I->getOpcode()));
    Cond.push_back(I->getOperand(0));
    Cond.push_back(I->getOperand(1));
  }

  return false;
}

unsigned BPFInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    ArrayRef<MachineOperand> Cond,
                                    const DebugLoc &DL,
                                    int *BytesAdded) const {
  assert(!BytesAdded && "code size not handled");

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");

  if (Cond.empty()) {
    // Unconditional branch
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(BPF::JMP)).addMBB(TBB);
    return 1;
  }

  BuildMI(&MBB, DL, get(Cond[0].getImm()))
      .add(Cond[1])
      .add(Cond[2])
      .addMBB(TBB);
  if (!FBB)
    return 1;
  BuildMI(&MBB, DL, get(BPF::JMP)).addMBB(FBB);
  return 2;
}

unsigned BPFInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                    int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");

  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr())
      continue;
    if (I->getOpcode() != BPF::JMP && !isCondBranch(I->getOpcode()))
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
}

bool BPFInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  unsigned Opcode = getInverseBranchOpcode(Cond[0].getImm());
  if (!Opcode)
    return true;
  Cond[0].setImm(Opcode);
  return false;
}

int BPFInstrInfo::getJumpTableIndex(const MachineInstr &MI) const {
  if (MI.getOpcode() != BPF::JX)
    return -1;

  // The pattern looks like:
  // %0 = LD_imm64 %jump-table.0   ; load jump-table address
  // %1 = ADD_rr %0, $another_reg  ; address + offset
  // %2 = LDD %1, 0                ; load the actual label
  // JX %2
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  Register Reg = MI.getOperand(0).getReg();
  if (!Reg.isVirtual())
    return -1;
  MachineInstr *Ldd = MRI.getUniqueVRegDef(Reg);
  if (Ldd == nullptr || Ldd->getOpcode() != BPF::LDD)
    return -1;

  Reg = Ldd->getOperand(1).getReg();
  if (!Reg.isVirtual())
    return -1;
  MachineInstr *Add = MRI.getUniqueVRegDef(Reg);
  if (Add == nullptr || Add->getOpcode() != BPF::ADD_rr)
    return -1;

  Reg = Add->getOperand(1).getReg();
  if (!Reg.isVirtual())
    return -1;
  MachineInstr *LDimm64 = MRI.getUniqueVRegDef(Reg);
  if (LDimm64 == nullptr || LDimm64->getOpcode() != BPF::LD_imm64)
    return -1;

  const MachineOperand &MO = LDimm64->getOperand(1);
  if (!MO.isJTI())
    return -1;

  return MO.getIndex();
}
