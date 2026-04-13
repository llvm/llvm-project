//===- NVPTXInstrInfo.cpp - NVPTX Instruction Information -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "NVPTXInstrInfo.h"
#include "NVPTX.h"
#include "NVPTXSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "NVPTXGenInstrInfo.inc"

// Pin the vtable to this file.
void NVPTXInstrInfo::anchor() {}

NVPTXInstrInfo::NVPTXInstrInfo(const NVPTXSubtarget &STI)
    : NVPTXGenInstrInfo(STI, RegInfo), RegInfo() {}

void NVPTXInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I,
                                 const DebugLoc &DL, Register DestReg,
                                 Register SrcReg, bool KillSrc,
                                 bool RenamableDest, bool RenamableSrc) const {
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *DestRC = MRI.getRegClass(DestReg);
  const TargetRegisterClass *SrcRC = MRI.getRegClass(SrcReg);

  if (DestRC != SrcRC)
    report_fatal_error("Copy one register into another with a different width");

  unsigned Op;
  if (DestRC == &NVPTX::B1RegClass)
    Op = NVPTX::MOV_B1_r;
  else if (DestRC == &NVPTX::B16RegClass)
    Op = NVPTX::MOV_B16_r;
  else if (DestRC == &NVPTX::B32RegClass)
    Op = NVPTX::MOV_B32_r;
  else if (DestRC == &NVPTX::B64RegClass)
    Op = NVPTX::MOV_B64_r;
  else if (DestRC == &NVPTX::B128RegClass)
    Op = NVPTX::MOV_B128_r;
  else
    llvm_unreachable("Bad register copy");

  BuildMI(MBB, I, DL, get(Op), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}

/// analyzeBranch - Analyze the branching code at the end of MBB, returning
/// true if it cannot be understood (e.g. it's a switch dispatch or isn't
/// implemented for a target).  Upon success, this returns false and returns
/// with the following information in various cases:
///
/// 1. If this block ends with no branches (it just falls through to its succ)
///    just return false, leaving TBB/FBB null.
/// 2. If this block ends with only an unconditional branch, it sets TBB to be
///    the destination block.
/// 3. If this block ends with an conditional branch and it falls through to
///    an successor block, it sets TBB to be the branch destination block and a
///    list of operands that evaluate the condition. These
///    operands can be passed to other TargetInstrInfo methods to create new
///    branches.
/// 4. If this block ends with an conditional branch and an unconditional
///    block, it returns the 'true' destination in TBB, the 'false' destination
///    in FBB, and a list of operands that evaluate the condition. These
///    operands can be passed to other TargetInstrInfo methods to create new
///    branches.
///
/// Note that removeBranch and insertBranch must be implemented to support
/// cases where this method returns success.
///
bool NVPTXInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(*--I))
    return false;

  // Get the last instruction in the block.
  MachineInstr &LastInst = *I;

  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(*--I)) {
    if (LastInst.getOpcode() == NVPTX::GOTO) {
      TBB = LastInst.getOperand(0).getMBB();
      return false;
    } else if (LastInst.getOpcode() == NVPTX::CBranch) {
      // Block ends with fall-through condbranch.
      TBB = LastInst.getOperand(1).getMBB();
      Cond.push_back(LastInst.getOperand(0));
      Cond.push_back(LastInst.getOperand(2));
      return false;
    }
    // Otherwise, don't know what this is.
    return true;
  }

  // Get the instruction before it if it's a terminator.
  MachineInstr &SecondLastInst = *I;

  // If there are three terminators, we don't know what sort of block this is.
  if (I != MBB.begin() && isUnpredicatedTerminator(*--I))
    return true;

  // If the block ends with NVPTX::GOTO and NVPTX:CBranch, handle it.
  if (SecondLastInst.getOpcode() == NVPTX::CBranch &&
      LastInst.getOpcode() == NVPTX::GOTO) {
    TBB = SecondLastInst.getOperand(1).getMBB();
    Cond.push_back(SecondLastInst.getOperand(0));
    Cond.push_back(SecondLastInst.getOperand(2));
    FBB = LastInst.getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two NVPTX:GOTOs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst.getOpcode() == NVPTX::GOTO &&
      LastInst.getOpcode() == NVPTX::GOTO) {
    TBB = SecondLastInst.getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned NVPTXInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                      int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return 0;
  --I;
  if (I->getOpcode() != NVPTX::GOTO && I->getOpcode() != NVPTX::CBranch)
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin())
    return 1;
  --I;
  if (I->getOpcode() != NVPTX::CBranch)
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned NVPTXInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                      MachineBasicBlock *TBB,
                                      MachineBasicBlock *FBB,
                                      ArrayRef<MachineOperand> Cond,
                                      const DebugLoc &DL,
                                      int *BytesAdded) const {
  assert(!BytesAdded && "code size not handled");

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "NVPTX branch conditions have two components!");

  // One-way branch.
  if (!FBB) {
    if (Cond.empty()) // Unconditional branch
      BuildMI(&MBB, DL, get(NVPTX::GOTO)).addMBB(TBB);
    else // Conditional branch
      BuildMI(&MBB, DL, get(NVPTX::CBranch))
          .add(Cond[0])
          .addMBB(TBB)
          .add(Cond[1]);
    return 1;
  }

  // Two-way Conditional Branch.
  BuildMI(&MBB, DL, get(NVPTX::CBranch)).add(Cond[0]).addMBB(TBB).add(Cond[1]);
  BuildMI(&MBB, DL, get(NVPTX::GOTO)).addMBB(FBB);
  return 2;
}

bool NVPTXInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid NVPTX branch condition!");
  Cond[1].setImm(!Cond[1].getImm());
  return false;
}

bool NVPTXInstrInfo::invertPredicateBranchInstr(MachineBasicBlock &MBB) const {
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
    return false;
  if (Cond.empty())
    return false;
  if (reverseBranchCondition(Cond))
    return false;
  DebugLoc DL = MBB.findBranchDebugLoc();
  removeBranch(MBB);
  insertBranch(MBB, TBB, FBB, Cond, DL);
  return true;
}

static bool isIntegerSetp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case NVPTX::SETP_i16rr:
  case NVPTX::SETP_i16ri:
  case NVPTX::SETP_i16ir:
  case NVPTX::SETP_i32rr:
  case NVPTX::SETP_i32ri:
  case NVPTX::SETP_i32ir:
  case NVPTX::SETP_i64rr:
  case NVPTX::SETP_i64ri:
  case NVPTX::SETP_i64ir:
    return true;
  default:
    return false;
  }
}

static bool isScalarFloatSetp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case NVPTX::SETP_bf16rr:
  case NVPTX::SETP_f16rr:
  case NVPTX::SETP_f32rr:
  case NVPTX::SETP_f32ri:
  case NVPTX::SETP_f32ir:
  case NVPTX::SETP_f64rr:
  case NVPTX::SETP_f64ri:
  case NVPTX::SETP_f64ir:
    return true;
  default:
    return false;
  }
}

static int64_t invertIntegerCmpMode(int64_t Mode) {
  switch (Mode) {
  case NVPTX::PTXCmpMode::EQ:
    return NVPTX::PTXCmpMode::NE;
  case NVPTX::PTXCmpMode::NE:
    return NVPTX::PTXCmpMode::EQ;
  case NVPTX::PTXCmpMode::LT:
    return NVPTX::PTXCmpMode::GE;
  case NVPTX::PTXCmpMode::LE:
    return NVPTX::PTXCmpMode::GT;
  case NVPTX::PTXCmpMode::GT:
    return NVPTX::PTXCmpMode::LE;
  case NVPTX::PTXCmpMode::GE:
    return NVPTX::PTXCmpMode::LT;
  case NVPTX::PTXCmpMode::LTU:
    return NVPTX::PTXCmpMode::GEU;
  case NVPTX::PTXCmpMode::LEU:
    return NVPTX::PTXCmpMode::GTU;
  case NVPTX::PTXCmpMode::GTU:
    return NVPTX::PTXCmpMode::LEU;
  case NVPTX::PTXCmpMode::GEU:
    return NVPTX::PTXCmpMode::LTU;
  default:
    llvm_unreachable("Invalid integer comparison mode");
  }
}

static int64_t invertScalarFloatCmpMode(int64_t Mode) {
  switch (Mode) {
  case NVPTX::PTXCmpMode::EQ:
    return NVPTX::PTXCmpMode::NEU;
  case NVPTX::PTXCmpMode::NE:
    return NVPTX::PTXCmpMode::EQU;
  case NVPTX::PTXCmpMode::EQU:
    return NVPTX::PTXCmpMode::NE;
  case NVPTX::PTXCmpMode::NEU:
    return NVPTX::PTXCmpMode::EQ;
  case NVPTX::PTXCmpMode::LT:
    return NVPTX::PTXCmpMode::GEU;
  case NVPTX::PTXCmpMode::LE:
    return NVPTX::PTXCmpMode::GTU;
  case NVPTX::PTXCmpMode::GT:
    return NVPTX::PTXCmpMode::LEU;
  case NVPTX::PTXCmpMode::GE:
    return NVPTX::PTXCmpMode::LTU;
  case NVPTX::PTXCmpMode::LTU:
    return NVPTX::PTXCmpMode::GE;
  case NVPTX::PTXCmpMode::LEU:
    return NVPTX::PTXCmpMode::GT;
  case NVPTX::PTXCmpMode::GTU:
    return NVPTX::PTXCmpMode::LE;
  case NVPTX::PTXCmpMode::GEU:
    return NVPTX::PTXCmpMode::LT;
  case NVPTX::PTXCmpMode::NUM:
    return NVPTX::PTXCmpMode::NotANumber;
  case NVPTX::PTXCmpMode::NotANumber:
    return NVPTX::PTXCmpMode::NUM;
  default:
    llvm_unreachable("Invalid scalar float comparison mode");
  }
}

static void invertScalarCompareInstr(MachineInstr &MI) {
  MachineOperand &ModeOp = MI.getOperand(3);

  if (isIntegerSetp(MI))
    ModeOp.setImm(invertIntegerCmpMode(ModeOp.getImm()));
  else if (isScalarFloatSetp(MI))
    ModeOp.setImm(invertScalarFloatCmpMode(ModeOp.getImm()));
  else
    llvm_unreachable("Invalid SETP instruction");
}

bool NVPTXInstrInfo::findCommutedOpIndices(const MachineInstr &MI,
                                           unsigned &SrcOpIdx1,
                                           unsigned &SrcOpIdx2) const {
  if (isIntegerSetp(MI) || isScalarFloatSetp(MI))
    return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 1, 2);
  return TargetInstrInfo::findCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);
}

MachineInstr *NVPTXInstrInfo::commuteInstructionImpl(MachineInstr &MI,
                                                     bool NewMI,
                                                     unsigned OpIdx1,
                                                     unsigned OpIdx2) const {
  assert(!NewMI && "this should never be used");

  if (!isIntegerSetp(MI) && !isScalarFloatSetp(MI))
    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);

  invertScalarCompareInstr(MI);

  // For now all users must be invertible conditional branches.
  // TODO: Support other users such as selects.
  bool AllInverted = true;
  MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  for (MachineInstr &UseMI :
       MRI.use_nodbg_instructions(MI.getOperand(0).getReg())) {
    if (!(UseMI.isConditionalBranch() &&
          invertPredicateBranchInstr(*UseMI.getParent()))) {
      AllInverted = false;
      break;
    }
  }

  if (!AllInverted) {
    for (MachineInstr &UseMI :
         MRI.use_nodbg_instructions(MI.getOperand(0).getReg())) {
      if (!(UseMI.isConditionalBranch() &&
            invertPredicateBranchInstr(*UseMI.getParent())))
        break;
    }
    invertScalarCompareInstr(MI);
    return nullptr;
  }
  return &MI;
}
