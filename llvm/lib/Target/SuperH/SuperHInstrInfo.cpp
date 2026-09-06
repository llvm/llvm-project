//===-- SuperHInstrInfo.cpp - SuperH Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SuperHInstrInfo.h"
#include "SuperHRegisterInfo.h"
#include "SuperHSubtarget.h"
#include "SuperHTargetMachine.h"
#include "MCTargetDesc/SuperHInstPrinter.h"
#include "SuperH.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "sh-instrinfo"

#define GET_INSTRINFO_CTOR_DTOR
#include "SuperHGenInstrInfo.inc"

SuperHInstrInfo::SuperHInstrInfo(const SuperHSubtarget &ST)
    : SuperHGenInstrInfo(ST, RI, SH::ADJCALLSTACKDOWN, SH::ADJCALLSTACKUP),
      RI(ST), Subtarget(ST) { }

// Pin the vtable to this file.
void SuperHInstrInfo::anchor() {}


// Gets whether a given opcode can fill a delay slot.
// 
// SuperH does not allow branch instructions of any kind to be situated 
// in a delay slot, nor does it allow instructions with delay slots
// to be chained together.
bool SuperHInstrInfo::canFillDelaySlot(unsigned Opcode) const {
  auto Desc = this->get(Opcode);
  return !Desc.hasDelaySlot() && 
         !Desc.isBranch() && 
         !Desc.isCall() && !Desc.isReturn() &&
         !(Desc.TSFlags & 0x1);
}

/// Return the noop instruction to use for a noop.
MCInst SuperHInstrInfo::getNop() const {
  MCInst I = MCInst();
  I.setOpcode(SH::NOP);
  return I;
}

void SuperHInstrInfo::insertNoop(MachineBasicBlock &MBB, 
                                 MachineBasicBlock::iterator MI) const {
  BuildMI(&MBB, MI->getDebugLoc(), get(SH::NOP));
}

ISD::CondCode SuperHInstrInfo::getCondFromBranchOp(unsigned Op) const {
  switch (Op) {
  default:
    return ISD::SETCC_INVALID;
  case SH::NOP:
  case SH::BT:
  case SH::BTS:
    return ISD::SETTRUE;
  case SH::BF:
  case SH::BFS:
    return ISD::SETFALSE;
  }
}

ISD::CondCode SuperHInstrInfo::getOppositeCondCode(ISD::CondCode Op) const {
  switch (Op) {
  default:
    return ISD::SETCC_INVALID;
  case ISD::SETTRUE:
    return ISD::SETFALSE;
  case ISD::SETFALSE:
    return ISD::SETTRUE;
  }
}

const MCInstrDesc &SuperHInstrInfo::getBrCond(ISD::CondCode CC) const {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case ISD::SETTRUE:
  case ISD::SETEQ:
  case ISD::SETGT:
  case ISD::SETGE:
    return get(SH::BT);
  case ISD::SETFALSE:
  case ISD::SETNE:
  case ISD::SETLT:
  case ISD::SETLE:
    return get(SH::BF);
  }
}

unsigned SuperHInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();

  switch (Opcode) {
  // A regular instruction
  default: {
    const MCInstrDesc &Desc = get(Opcode);
    return Desc.getSize();
  }
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::DBG_VALUE:
    return 0;
  case TargetOpcode::INLINEASM:
  case TargetOpcode::INLINEASM_BR: {
    // TODO: Add inline ASM support.
    return 0;
  }
  }
}





//===----------------------------------------------------------------------===//
//                             Register Managment.
//===----------------------------------------------------------------------===//

void SuperHInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, const DebugLoc &DL,
                           Register DestReg, Register SrcReg, bool KillSrc,
                           bool RenamableDest,
                           bool RenamableSrc) const {
  // Do nothing, self copy.
  if (SrcReg == DestReg)
    return;

  // Load from MACL
  if (SrcReg == SH::MACLO && SH::GPRRegClass.contains(DestReg)) {
    BuildMI(MBB, MI, DL, get(SH::STSMACL), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Store to MACL
  if (SH::GPRRegClass.contains(SrcReg) && DestReg == SH::MACLO) {
    BuildMI(MBB, MI, DL, get(SH::LDSMACL), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // If the targets are GPR registers, use MOV Rm, Rn.
  if (SH::GPRRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, MI, DL, get(SH::MOV), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  };

  // Otherwise this is not possible.
  llvm_unreachable("Impossible reg-to-reg copy");
}





//===----------------------------------------------------------------------===//
//                              Stack Frames
//===----------------------------------------------------------------------===//

void SuperHInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator II, 
                                          Register SrcReg, bool isKill, int FrameIndex, 
                                          const TargetRegisterClass *RC, Register VReg, 
                                          MachineInstr::MIFlag Flags) const {
  LLVM_DEBUG(dbgs() << "Store " << RI.getName(SrcReg) << " to slot " << FrameIndex << "\n");
  
  MachineInstr &MI = *II;
  DebugLoc DL = MI.getDebugLoc();
  if (RI.isTypeLegalForClass(*RC, MVT::i8)) {
    BuildMI(MBB, II, DebugLoc(), get(SH::MOVBSFR))
        .addReg(SrcReg, getKillRegState(isKill))
        .addFrameIndex(FrameIndex)
        .addImm(0)
        .addDef(SH::R0)
        .addUse(SH::R0);
  } else if (RI.isTypeLegalForClass(*RC, MVT::i16)) {
    BuildMI(MBB, II, DebugLoc(), get(SH::MOVWSFR))
        .addReg(SrcReg, getKillRegState(isKill))
        .addFrameIndex(FrameIndex)
        .addImm(0)
        .addDef(SH::R0)
        .addUse(SH::R0);
  } else if (RI.isTypeLegalForClass(*RC, MVT::i32)) {
    BuildMI(MBB, II, DebugLoc(), get(SH::MOVLSFR))
        .addReg(SrcReg, getKillRegState(isKill))
        .addFrameIndex(FrameIndex)
        .addImm(0);
  } else {
    llvm_unreachable("Cannot store this register into stack slot!");
  }
}

void SuperHInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator II, 
                                           Register DestReg, int FrameIndex, 
                                           const TargetRegisterClass *RC, Register VReg,
                                           unsigned SubReg, MachineInstr::MIFlag Flags) const {
  LLVM_DEBUG(dbgs() << "Load " << RI.getName(DestReg) << " from slot " << FrameIndex << "\n");

  MachineInstr &MI = *II;
  DebugLoc DL = MI.getDebugLoc();
  if (RI.isTypeLegalForClass(*RC, MVT::i8)) {
    BuildMI(MBB, II, DebugLoc(), get(SH::MOVBLFR), DestReg)
        .addFrameIndex(FrameIndex)
        .addImm(0)
        .addDef(SH::R0)
        .addUse(SH::R0);

  } else if (RI.isTypeLegalForClass(*RC, MVT::i16)) {
    BuildMI(MBB, II, DebugLoc(), get(SH::MOVWLFR), DestReg)
        .addFrameIndex(FrameIndex)
        .addImm(0)
        .addDef(SH::R0)
        .addUse(SH::R0);

  } else if (RI.isTypeLegalForClass(*RC, MVT::i32)) {
    BuildMI(MBB, II, DebugLoc(), get(SH::MOVLLFR), DestReg)
        .addFrameIndex(FrameIndex)
        .addImm(0);
  } else {
    llvm_unreachable("Cannot load this register from stack slot!");
  }
}

Register SuperHInstrInfo::isLoadFromStackSlot(const MachineInstr &MI, int &FrameIndex) const {
  LLVM_DEBUG(dbgs() << "isLoadFromStackSlot\n");

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }
  return 0;
}

Register SuperHInstrInfo::isStoreToStackSlot(const MachineInstr &MI, int &FrameIndex) const {
  LLVM_DEBUG(dbgs() << "isStoreToStackSlot\n");

  if (MI.getOperand(0).isFI() && MI.getOperand(1).isImm() &&
      MI.getOperand(1).getImm() == 0) {
    FrameIndex = MI.getOperand(0).getIndex();
    return MI.getOperand(2).getReg();
  }
  return 0;
}




//===----------------------------------------------------------------------===//
//                              Branch Analysis
//===----------------------------------------------------------------------===//

bool SuperHInstrInfo::analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                   MachineBasicBlock *&FBB,
                   SmallVectorImpl<MachineOperand> &Cond,
                   bool AllowModify) const {
  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  MachineBasicBlock::iterator UnCondBrIter = MBB.end();

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr()) {
      continue;
    }

    // Working from the bottom, when we see a non-terminator
    // instruction, we're done.
    if (!isUnpredicatedTerminator(*I)) {
      break;
    }

    // A terminator that isn't a branch can't easily be handled
    // by this analysis.
    if (!I->getDesc().isBranch()) {
      return true;
    }

    // Handle unconditional branches.
    if (I->getOpcode() == SH::BRA) {
      UnCondBrIter = I;

      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a BRA, delete them.
      MBB.erase(std::next(I), MBB.end());
      Cond.clear();
      FBB = nullptr;

      // Delete the BRA if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = nullptr;
        I->eraseFromParent();
        I = MBB.end();
        UnCondBrIter = MBB.end();
        continue;
      }

      // TBB is used to indicate the unconditinal destination.
      TBB = I->getOperand(0).getMBB();
      continue;
    }

    // Handle conditional branches.
    ISD::CondCode BranchCode = getCondFromBranchOp(I->getOpcode());
    if (BranchCode == ISD::SETCC_INVALID) {
      return true; // Can't handle indirect branch.
    }

    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      MachineBasicBlock *TargetBB = I->getOperand(0).getMBB();
      if (AllowModify && UnCondBrIter != MBB.end() &&
          MBB.isLayoutSuccessor(TargetBB)) {

        BranchCode = getOppositeCondCode(BranchCode);
        auto JNCC = getBrCond(BranchCode);

        MachineBasicBlock::iterator OldInst = I;
        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), JNCC)
            .addMBB(UnCondBrIter->getOperand(0).getMBB());
        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(SH::BRA))
            .addMBB(TargetBB);

        OldInst->eraseFromParent();
        UnCondBrIter->eraseFromParent();

        // Restart the analysis.
        UnCondBrIter = MBB.end();
        I = MBB.end();
        continue;
      }

      FBB = TBB;
      TBB = I->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(BranchCode));
      continue;
    }

    // Handle subsequent conditional branches. Only handle the case where all
    // conditional branches branch to the same destination.
    assert(Cond.size() == 1);
    assert(TBB);

    // Only handle the case where all conditional branches branch to
    // the same destination.
    if (TBB != I->getOperand(0).getMBB()) {
      return true;
    }

    ISD::CondCode OldBranchCode = (ISD::CondCode)Cond[0].getImm();
    // If the conditions are the same, we can leave them alone.
    if (OldBranchCode == BranchCode) {
      continue;
    }

    return true;
  }

  return false;
}

unsigned SuperHInstrInfo::insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                      MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                      const DebugLoc &DL,
                      int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "SH branch conditions have one component!");

  if (Cond.empty()) {
    assert(!FBB && "Unconditional branch with multiple successors!");
    auto &MI = *BuildMI(&MBB, DL, get(SH::BRA)).addMBB(TBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  ISD::CondCode CC = (ISD::CondCode)Cond[0].getImm();
  auto &CondMI = *BuildMI(&MBB, DL, getBrCond(CC)).addMBB(TBB);

  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(CondMI);
  ++Count;

  if (FBB) {
    // Two-way Conditional branch. Insert the second branch.
    auto &MI = *BuildMI(&MBB, DL, get(SH::BRA)).addMBB(FBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    ++Count;
  }

  return Count;
}

unsigned SuperHInstrInfo::removeBranch(MachineBasicBlock &MBB,
                      int *BytesRemoved) const {
  if (BytesRemoved)
    *BytesRemoved = 0;

  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr()) {
      continue;
    }

    if (I->getOpcode() != SH::BRA &&
        getCondFromBranchOp(I->getOpcode()) == ISD::SETCC_INVALID) {
      break;
    }

    // Remove the branch.
    if (BytesRemoved)
      *BytesRemoved += getInstSizeInBytes(*I);
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
}

bool
SuperHInstrInfo::reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid SH branch condition!");

  ISD::CondCode CC = static_cast<ISD::CondCode>(Cond[0].getImm());
  Cond[0].setImm(getOppositeCondCode(CC));
  return false;
}

MachineBasicBlock *SuperHInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  if (MI.isBranch())
    return MI.getOperand(0).getMBB();

  llvm_unreachable("unimplemented branch instructions");
}

bool SuperHInstrInfo::isBranchOffsetInRange(unsigned BranchOp,
                           int64_t BrOffset) const {
  switch (BranchOp) {
  default:
    llvm_unreachable("unexpected opcode!");
  case SH::BF:
  case SH::BT:
  case SH::BFS:
  case SH::BTS:
  case SH::BRA:
    return isIntN(8, BrOffset);
  case SH::BSR:
    return isIntN(12, BrOffset);
  }
}

