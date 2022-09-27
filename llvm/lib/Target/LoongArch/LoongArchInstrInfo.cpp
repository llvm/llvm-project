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
#include "MCTargetDesc/LoongArchMatInt.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "LoongArchGenInstrInfo.inc"

LoongArchInstrInfo::LoongArchInstrInfo(LoongArchSubtarget &STI)
    : LoongArchGenInstrInfo(LoongArch::ADJCALLSTACKDOWN,
                            LoongArch::ADJCALLSTACKUP),
      STI(STI) {}

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

void LoongArchInstrInfo::movImm(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                const DebugLoc &DL, Register DstReg,
                                uint64_t Val, MachineInstr::MIFlag Flag) const {
  Register SrcReg = LoongArch::R0;

  if (!STI.is64Bit() && !isInt<32>(Val))
    report_fatal_error("Should only materialize 32-bit constants for LA32");

  auto Seq = LoongArchMatInt::generateInstSeq(Val);
  assert(!Seq.empty());

  for (auto &Inst : Seq) {
    switch (Inst.Opc) {
    case LoongArch::LU12I_W:
      BuildMI(MBB, MBBI, DL, get(Inst.Opc), DstReg)
          .addImm(Inst.Imm)
          .setMIFlag(Flag);
      break;
    case LoongArch::ADDI_W:
    case LoongArch::ORI:
    case LoongArch::LU32I_D: // "rj" is needed due to InstrInfo pattern
    case LoongArch::LU52I_D:
      BuildMI(MBB, MBBI, DL, get(Inst.Opc), DstReg)
          .addReg(SrcReg, RegState::Kill)
          .addImm(Inst.Imm)
          .setMIFlag(Flag);
      break;
    default:
      assert(false && "Unknown insn emitted by LoongArchMatInt");
    }

    // Only the first instruction has $zero as its source.
    SrcReg = DstReg;
  }
}

unsigned LoongArchInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  return MI.getDesc().getSize();
}

MachineBasicBlock *
LoongArchInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  assert(MI.getDesc().isBranch() && "Unexpected opcode!");
  // The branch target is always the last operand.
  return MI.getOperand(MI.getNumExplicitOperands() - 1).getMBB();
}

static void parseCondBranch(MachineInstr &LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  assert(LastInst.getDesc().isConditionalBranch() &&
         "Unknown conditional branch");
  int NumOp = LastInst.getNumExplicitOperands();
  Target = LastInst.getOperand(NumOp - 1).getMBB();

  Cond.push_back(MachineOperand::CreateImm(LastInst.getOpcode()));
  for (int i = 0; i < NumOp - 1; i++)
    Cond.push_back(LastInst.getOperand(i));
}

bool LoongArchInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                       MachineBasicBlock *&TBB,
                                       MachineBasicBlock *&FBB,
                                       SmallVectorImpl<MachineOperand> &Cond,
                                       bool AllowModify) const {
  TBB = FBB = nullptr;
  Cond.clear();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end() || !isUnpredicatedTerminator(*I))
    return false;

  // Count the number of terminators and find the first unconditional or
  // indirect branch.
  MachineBasicBlock::iterator FirstUncondOrIndirectBr = MBB.end();
  int NumTerminators = 0;
  for (auto J = I.getReverse(); J != MBB.rend() && isUnpredicatedTerminator(*J);
       J++) {
    NumTerminators++;
    if (J->getDesc().isUnconditionalBranch() ||
        J->getDesc().isIndirectBranch()) {
      FirstUncondOrIndirectBr = J.getReverse();
    }
  }

  // If AllowModify is true, we can erase any terminators after
  // FirstUncondOrIndirectBR.
  if (AllowModify && FirstUncondOrIndirectBr != MBB.end()) {
    while (std::next(FirstUncondOrIndirectBr) != MBB.end()) {
      std::next(FirstUncondOrIndirectBr)->eraseFromParent();
      NumTerminators--;
    }
    I = FirstUncondOrIndirectBr;
  }

  // Handle a single unconditional branch.
  if (NumTerminators == 1 && I->getDesc().isUnconditionalBranch()) {
    TBB = getBranchDestBlock(*I);
    return false;
  }

  // Handle a single conditional branch.
  if (NumTerminators == 1 && I->getDesc().isConditionalBranch()) {
    parseCondBranch(*I, TBB, Cond);
    return false;
  }

  // Handle a conditional branch followed by an unconditional branch.
  if (NumTerminators == 2 && std::prev(I)->getDesc().isConditionalBranch() &&
      I->getDesc().isUnconditionalBranch()) {
    parseCondBranch(*std::prev(I), TBB, Cond);
    FBB = getBranchDestBlock(*I);
    return false;
  }

  // Otherwise, we can't handle this.
  return true;
}

unsigned LoongArchInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                          int *BytesRemoved) const {
  if (BytesRemoved)
    *BytesRemoved = 0;
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (!I->getDesc().isBranch())
    return 0;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin())
    return 1;
  --I;
  if (!I->getDesc().isConditionalBranch())
    return 1;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();
  return 2;
}

// Inserts a branch into the end of the specific MachineBasicBlock, returning
// the number of instructions inserted.
unsigned LoongArchInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert(Cond.size() <= 3 && Cond.size() != 1 &&
         "LoongArch branch conditions have at most two components!");

  // Unconditional branch.
  if (Cond.empty()) {
    MachineInstr &MI = *BuildMI(&MBB, DL, get(LoongArch::PseudoBR)).addMBB(TBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    return 1;
  }

  // Either a one or two-way conditional branch.
  MachineInstrBuilder MIB = BuildMI(&MBB, DL, get(Cond[0].getImm()));
  for (unsigned i = 1; i < Cond.size(); ++i)
    MIB.add(Cond[i]);
  MIB.addMBB(TBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(*MIB);

  // One-way conditional branch.
  if (!FBB)
    return 1;

  // Two-way conditional branch.
  MachineInstr &MI = *BuildMI(&MBB, DL, get(LoongArch::PseudoBR)).addMBB(FBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(MI);
  return 2;
}

static unsigned getOppositeBranchOpc(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Unrecognized conditional branch");
  case LoongArch::BEQ:
    return LoongArch::BNE;
  case LoongArch::BNE:
    return LoongArch::BEQ;
  case LoongArch::BEQZ:
    return LoongArch::BNEZ;
  case LoongArch::BNEZ:
    return LoongArch::BEQZ;
  case LoongArch::BCEQZ:
    return LoongArch::BCNEZ;
  case LoongArch::BCNEZ:
    return LoongArch::BCEQZ;
  case LoongArch::BLT:
    return LoongArch::BGE;
  case LoongArch::BGE:
    return LoongArch::BLT;
  case LoongArch::BLTU:
    return LoongArch::BGEU;
  case LoongArch::BGEU:
    return LoongArch::BLTU;
  }
}

bool LoongArchInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert((Cond.size() && Cond.size() <= 3) && "Invalid branch condition!");
  Cond[0].setImm(getOppositeBranchOpc(Cond[0].getImm()));
  return false;
}
