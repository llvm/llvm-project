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
    return get(SH::BT);
  case ISD::SETFALSE:
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
    if (BranchCode == ISD::SETFALSE) {
      return true; // Can't handle indirect branch.
    }

    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      MachineBasicBlock *TargetBB = I->getOperand(0).getMBB();
      if (AllowModify && UnCondBrIter != MBB.end() &&
          MBB.isLayoutSuccessor(TargetBB)) {

        BranchCode = getOppositeCondCode(BranchCode);
        unsigned JNCC = getBrCond(BranchCode).getOpcode();
        MachineBasicBlock::iterator OldInst = I;

        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(JNCC))
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

void SuperHInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                          MachineBasicBlock &NewDestBB,
                          MachineBasicBlock &RestoreBB, const DebugLoc &DL,
                          int64_t BrOffset, RegScavenger *RS) const {

}


