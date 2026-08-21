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
    return ISD::SETFALSE;
  case SH::BRA:
  case SH::NOP:
    return ISD::SETTRUE;
  case SH::BT:
  case SH::BTS:
    return ISD::SETEQ;
  case SH::BF:
  case SH::BFS:
    return ISD::SETNE;
  }
}

const MCInstrDesc &SuperHInstrInfo::getBrCond(ISD::CondCode CC) const {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case ISD::SETEQ:
  case ISD::SETGE:
  case ISD::SETGT:
    return get(SH::BT);
  case ISD::SETNE:
  case ISD::SETLE:
  case ISD::SETLT:
    return get(SH::BF);
  }
}





//===----------------------------------------------------------------------===//
//                              Stack Frames
//===----------------------------------------------------------------------===//

void SuperHInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, const DebugLoc &DL,
                           Register DestReg, Register SrcReg, bool KillSrc,
                           bool RenamableDest,
                           bool RenamableSrc) const {

  // Do nothing, self copy.
  if (SrcReg == DestReg)
    return;

  // If the targets are GPR registers, use MOV Rm, Rn.
  if (SH::GPRRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, MI, DL, get(SH::MOVRmRn), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  };

  // Otherwise this is not possible.
  llvm_unreachable("Impossible reg-to-reg copy");
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

    LLVM_DEBUG(dbgs() << "analyzeBranch " << getName(I->getOpcode()) << "\n");

    // Working from the bottom, when we see a non-terminator
    // instruction, we're done.
    if (!isUnpredicatedTerminator(*I)) {
      break;
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
  }

  return false;
}

unsigned SuperHInstrInfo::insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                      MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                      const DebugLoc &DL,
                      int *BytesAdded) const {
  return 0;
}

unsigned SuperHInstrInfo::removeBranch(MachineBasicBlock &MBB,
                      int *BytesRemoved) const {
  return 0;
}

bool
SuperHInstrInfo::reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {

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

void SuperHInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                          MachineBasicBlock &NewDestBB,
                          MachineBasicBlock &RestoreBB, const DebugLoc &DL,
                          int64_t BrOffset, RegScavenger *RS) const {

}


