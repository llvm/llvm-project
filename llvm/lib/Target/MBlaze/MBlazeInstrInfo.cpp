//===-- MBlazeInstrInfo.cpp - MBlaze Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MBlazeInstrInfo.h"
#include "MBlazeTargetMachine.h"
#include "MBlazeMachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScoreboardHazardRecognizer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/ADT/STLExtras.h"

#define GET_INSTRINFO_CTOR
#include "MBlazeGenInstrInfo.inc"

using namespace llvm;

MBlazeInstrInfo::MBlazeInstrInfo(MBlazeTargetMachine &tm)
  : MBlazeGenInstrInfo(MBlaze::ADJCALLSTACKDOWN, MBlaze::ADJCALLSTACKUP),
    TM(tm), RI(*TM.getSubtargetImpl(), *this) {}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImm() && op.getImm() == 0;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned MBlazeInstrInfo::
isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const {
  if (MI->getOpcode() == MBlaze::LWI) {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2)))) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
  }

  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned MBlazeInstrInfo::
isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const {
  if (MI->getOpcode() == MBlaze::SWI) {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2)))) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

/// insertNoop - If data hazard condition is found insert the target nop
/// instruction.
void MBlazeInstrInfo::
insertNoop(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI) const {
  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(MBlaze::NOP));
}

void MBlazeInstrInfo::
copyPhysReg(MachineBasicBlock &MBB,
            MachineBasicBlock::iterator I, DebugLoc DL,
            unsigned DestReg, unsigned SrcReg,
            bool KillSrc) const {
  llvm::BuildMI(MBB, I, DL, get(MBlaze::ADDK), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc)).addReg(MBlaze::R0);
}

void MBlazeInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  BuildMI(MBB, I, DL, get(MBlaze::SWI)).addReg(SrcReg,getKillRegState(isKill))
    .addFrameIndex(FI).addImm(0); //.addFrameIndex(FI);
}

void MBlazeInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  BuildMI(MBB, I, DL, get(MBlaze::LWI), DestReg)
      .addFrameIndex(FI).addImm(0); //.addFrameIndex(FI);
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//
bool MBlazeInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *&TBB,
                                    MachineBasicBlock *&FBB,
                                    SmallVectorImpl<MachineOperand> &Cond,
                                    bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return false;
    --I;
  }
  if (!isUnpredicatedTerminator(I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (MBlaze::isUncondBranchOpcode(LastOpc)) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (MBlaze::isCondBranchOpcode(LastOpc)) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
      Cond.push_back(LastInst->getOperand(0));
      return false;
    }
    // Otherwise, don't know what this is.
    return true;
  }

  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with something like BEQID then BRID, handle it.
  if (MBlaze::isCondBranchOpcode(SecondLastInst->getOpcode()) &&
      MBlaze::isUncondBranchOpcode(LastInst->getOpcode())) {
    TBB = SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(SecondLastInst->getOpcode()));
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two unconditional branches, handle it.
  // The second one is not executed, so remove it.
  if (MBlaze::isUncondBranchOpcode(SecondLastInst->getOpcode()) &&
      MBlaze::isUncondBranchOpcode(LastInst->getOpcode())) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned MBlazeInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond,
             DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "MBlaze branch conditions have two components!");

  unsigned Opc = MBlaze::BRID;
  if (!Cond.empty())
    Opc = (unsigned)Cond[0].getImm();

  if (FBB == 0) {
    if (Cond.empty()) // Unconditional branch
      BuildMI(&MBB, DL, get(Opc)).addMBB(TBB);
    else              // Conditional branch
      BuildMI(&MBB, DL, get(Opc)).addReg(Cond[1].getReg()).addMBB(TBB);
    return 1;
  }

  BuildMI(&MBB, DL, get(Opc)).addReg(Cond[1].getReg()).addMBB(TBB);
  BuildMI(&MBB, DL, get(MBlaze::BRID)).addMBB(FBB);
  return 2;
}

unsigned MBlazeInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }

  if (!MBlaze::isUncondBranchOpcode(I->getOpcode()) &&
      !MBlaze::isCondBranchOpcode(I->getOpcode()))
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (!MBlaze::isCondBranchOpcode(I->getOpcode()))
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

bool MBlazeInstrInfo::ReverseBranchCondition(SmallVectorImpl<MachineOperand>
                                               &Cond) const {
  assert(Cond.size() == 2 && "Invalid MBlaze branch opcode!");
  switch (Cond[0].getImm()) {
  default:            return true;
  case MBlaze::BEQ:   Cond[0].setImm(MBlaze::BNE); return false;
  case MBlaze::BNE:   Cond[0].setImm(MBlaze::BEQ); return false;
  case MBlaze::BGT:   Cond[0].setImm(MBlaze::BLE); return false;
  case MBlaze::BGE:   Cond[0].setImm(MBlaze::BLT); return false;
  case MBlaze::BLT:   Cond[0].setImm(MBlaze::BGE); return false;
  case MBlaze::BLE:   Cond[0].setImm(MBlaze::BGT); return false;
  case MBlaze::BEQI:  Cond[0].setImm(MBlaze::BNEI); return false;
  case MBlaze::BNEI:  Cond[0].setImm(MBlaze::BEQI); return false;
  case MBlaze::BGTI:  Cond[0].setImm(MBlaze::BLEI); return false;
  case MBlaze::BGEI:  Cond[0].setImm(MBlaze::BLTI); return false;
  case MBlaze::BLTI:  Cond[0].setImm(MBlaze::BGEI); return false;
  case MBlaze::BLEI:  Cond[0].setImm(MBlaze::BGTI); return false;
  case MBlaze::BEQD:  Cond[0].setImm(MBlaze::BNED); return false;
  case MBlaze::BNED:  Cond[0].setImm(MBlaze::BEQD); return false;
  case MBlaze::BGTD:  Cond[0].setImm(MBlaze::BLED); return false;
  case MBlaze::BGED:  Cond[0].setImm(MBlaze::BLTD); return false;
  case MBlaze::BLTD:  Cond[0].setImm(MBlaze::BGED); return false;
  case MBlaze::BLED:  Cond[0].setImm(MBlaze::BGTD); return false;
  case MBlaze::BEQID: Cond[0].setImm(MBlaze::BNEID); return false;
  case MBlaze::BNEID: Cond[0].setImm(MBlaze::BEQID); return false;
  case MBlaze::BGTID: Cond[0].setImm(MBlaze::BLEID); return false;
  case MBlaze::BGEID: Cond[0].setImm(MBlaze::BLTID); return false;
  case MBlaze::BLTID: Cond[0].setImm(MBlaze::BGEID); return false;
  case MBlaze::BLEID: Cond[0].setImm(MBlaze::BGTID); return false;
  }
}

/// getGlobalBaseReg - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
unsigned MBlazeInstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  MBlazeFunctionInfo *MBlazeFI = MF->getInfo<MBlazeFunctionInfo>();
  unsigned GlobalBaseReg = MBlazeFI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // Insert the set of GlobalBaseReg into the first MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  GlobalBaseReg = RegInfo.createVirtualRegister(&MBlaze::GPRRegClass);
  BuildMI(FirstMBB, MBBI, DebugLoc(), TII->get(TargetOpcode::COPY),
          GlobalBaseReg).addReg(MBlaze::R20);
  RegInfo.addLiveIn(MBlaze::R20);

  MBlazeFI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}
