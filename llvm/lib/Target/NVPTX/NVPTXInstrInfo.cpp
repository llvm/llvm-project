//===- NVPTXInstrInfo.cpp - NVPTX Instruction Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXInstrInfo.h"
#include "NVPTXTargetMachine.h"
#define GET_INSTRINFO_CTOR
#include "NVPTXGenInstrInfo.inc"
#include "llvm/Function.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <cstdio>


using namespace llvm;

// FIXME: Add the subtarget support on this constructor.
NVPTXInstrInfo::NVPTXInstrInfo(NVPTXTargetMachine &tm)
: NVPTXGenInstrInfo(),
  TM(tm),
  RegInfo(*this, *TM.getSubtargetImpl()) {}


void NVPTXInstrInfo::copyPhysReg (MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I, DebugLoc DL,
                                  unsigned DestReg, unsigned SrcReg,
                                  bool KillSrc) const {
  if (NVPTX::Int32RegsRegClass.contains(DestReg) &&
      NVPTX::Int32RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::IMOV32rr), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::Int8RegsRegClass.contains(DestReg) &&
      NVPTX::Int8RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::IMOV8rr), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::Int1RegsRegClass.contains(DestReg) &&
      NVPTX::Int1RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::IMOV1rr), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::Float32RegsRegClass.contains(DestReg) &&
      NVPTX::Float32RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::FMOV32rr), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::Int16RegsRegClass.contains(DestReg) &&
      NVPTX::Int16RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::IMOV16rr), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::Int64RegsRegClass.contains(DestReg) &&
      NVPTX::Int64RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::IMOV64rr), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::Float64RegsRegClass.contains(DestReg) &&
      NVPTX::Float64RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::FMOV64rr), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V4F32RegsRegClass.contains(DestReg) &&
      NVPTX::V4F32RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V4f32Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V4I32RegsRegClass.contains(DestReg) &&
      NVPTX::V4I32RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V4i32Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V2F32RegsRegClass.contains(DestReg) &&
      NVPTX::V2F32RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V2f32Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V2I32RegsRegClass.contains(DestReg) &&
      NVPTX::V2I32RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V2i32Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V4I8RegsRegClass.contains(DestReg) &&
      NVPTX::V4I8RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V4i8Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V2I8RegsRegClass.contains(DestReg) &&
      NVPTX::V2I8RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V2i8Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V4I16RegsRegClass.contains(DestReg) &&
      NVPTX::V4I16RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V4i16Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V2I16RegsRegClass.contains(DestReg) &&
      NVPTX::V2I16RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V2i16Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V2I64RegsRegClass.contains(DestReg) &&
      NVPTX::V2I64RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V2i64Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else if (NVPTX::V2F64RegsRegClass.contains(DestReg) &&
      NVPTX::V2F64RegsRegClass.contains(SrcReg))
    BuildMI(MBB, I, DL, get(NVPTX::V2f64Mov), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
  else {
    llvm_unreachable("Don't know how to copy a register");
  }
}

bool NVPTXInstrInfo::isMoveInstr(const MachineInstr &MI,
                                 unsigned &SrcReg,
                                 unsigned &DestReg) const {
  // Look for the appropriate part of TSFlags
  bool isMove = false;

  unsigned TSFlags = (MI.getDesc().TSFlags & NVPTX::SimpleMoveMask) >>
      NVPTX::SimpleMoveShift;
  isMove = (TSFlags == 1);

  if (isMove) {
    MachineOperand dest = MI.getOperand(0);
    MachineOperand src = MI.getOperand(1);
    assert(dest.isReg() && "dest of a movrr is not a reg");
    assert(src.isReg() && "src of a movrr is not a reg");

    SrcReg = src.getReg();
    DestReg = dest.getReg();
    return true;
  }

  return false;
}

bool  NVPTXInstrInfo::isReadSpecialReg(MachineInstr &MI) const
{
  switch (MI.getOpcode()) {
  default: return false;
  case NVPTX::INT_PTX_SREG_NTID_X:
  case NVPTX::INT_PTX_SREG_NTID_Y:
  case NVPTX::INT_PTX_SREG_NTID_Z:
  case NVPTX::INT_PTX_SREG_TID_X:
  case NVPTX::INT_PTX_SREG_TID_Y:
  case NVPTX::INT_PTX_SREG_TID_Z:
  case NVPTX::INT_PTX_SREG_CTAID_X:
  case NVPTX::INT_PTX_SREG_CTAID_Y:
  case NVPTX::INT_PTX_SREG_CTAID_Z:
  case NVPTX::INT_PTX_SREG_NCTAID_X:
  case NVPTX::INT_PTX_SREG_NCTAID_Y:
  case NVPTX::INT_PTX_SREG_NCTAID_Z:
  case NVPTX::INT_PTX_SREG_WARPSIZE:
    return true;
  }
}


bool NVPTXInstrInfo::isLoadInstr(const MachineInstr &MI,
                                 unsigned &AddrSpace) const {
  bool isLoad = false;
  unsigned TSFlags = (MI.getDesc().TSFlags & NVPTX::isLoadMask) >>
      NVPTX::isLoadShift;
  isLoad = (TSFlags == 1);
  if (isLoad)
    AddrSpace = getLdStCodeAddrSpace(MI);
  return isLoad;
}

bool NVPTXInstrInfo::isStoreInstr(const MachineInstr &MI,
                                  unsigned &AddrSpace) const {
  bool isStore = false;
  unsigned TSFlags = (MI.getDesc().TSFlags & NVPTX::isStoreMask) >>
      NVPTX::isStoreShift;
  isStore = (TSFlags == 1);
  if (isStore)
    AddrSpace = getLdStCodeAddrSpace(MI);
  return isStore;
}


bool NVPTXInstrInfo::CanTailMerge(const MachineInstr *MI) const {
  unsigned addrspace = 0;
  if (MI->getOpcode() == NVPTX::INT_CUDA_SYNCTHREADS)
    return false;
  if (isLoadInstr(*MI, addrspace))
    if (addrspace == NVPTX::PTXLdStInstCode::SHARED)
      return false;
  if (isStoreInstr(*MI, addrspace))
    if (addrspace == NVPTX::PTXLdStInstCode::SHARED)
      return false;
  return true;
}


/// AnalyzeBranch - Analyze the branching code at the end of MBB, returning
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
/// Note that RemoveBranch and InsertBranch must be implemented to support
/// cases where this method returns success.
///
bool NVPTXInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastInst->getOpcode() == NVPTX::GOTO) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (LastInst->getOpcode() == NVPTX::CBranch) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(1).getMBB();
      Cond.push_back(LastInst->getOperand(0));
      return false;
    }
    // Otherwise, don't know what this is.
    return true;
  }

  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with NVPTX::GOTO and NVPTX:CBranch, handle it.
  if (SecondLastInst->getOpcode() == NVPTX::CBranch &&
      LastInst->getOpcode() == NVPTX::GOTO) {
    TBB =  SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two NVPTX:GOTOs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst->getOpcode() == NVPTX::GOTO &&
      LastInst->getOpcode() == NVPTX::GOTO) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned NVPTXInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != NVPTX::GOTO && I->getOpcode() != NVPTX::CBranch)
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != NVPTX::CBranch)
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
NVPTXInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                             MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond,
                             DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "NVPTX branch conditions have two components!");

  // One-way branch.
  if (FBB == 0) {
    if (Cond.empty())   // Unconditional branch
      BuildMI(&MBB, DL, get(NVPTX::GOTO)).addMBB(TBB);
    else                // Conditional branch
      BuildMI(&MBB, DL, get(NVPTX::CBranch))
      .addReg(Cond[0].getReg()).addMBB(TBB);
    return 1;
  }

  // Two-way Conditional Branch.
  BuildMI(&MBB, DL, get(NVPTX::CBranch))
  .addReg(Cond[0].getReg()).addMBB(TBB);
  BuildMI(&MBB, DL, get(NVPTX::GOTO)).addMBB(FBB);
  return 2;
}
