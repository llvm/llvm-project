//===-- AVRInstrInfo.cpp - AVR Instruction Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AVR implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "AVRInstrInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#include "AVR.h"
#include "AVRMachineFunctionInfo.h"
#include "AVRTargetMachine.h"
#include "MCTargetDesc/AVRMCTargetDesc.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "AVRGenInstrInfo.inc"

namespace llvm {

AVRInstrInfo::AVRInstrInfo()
    : AVRGenInstrInfo(AVR::ADJCALLSTACKDOWN, AVR::ADJCALLSTACKUP), RI() {}

void AVRInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI,
                               const DebugLoc &DL, unsigned DestReg,
                               unsigned SrcReg, bool KillSrc) const {
  unsigned Opc;

  if (AVR::GPR8RegClass.contains(DestReg, SrcReg)) {
    Opc = AVR::MOVRdRr;
  } else if (AVR::DREGSRegClass.contains(DestReg, SrcReg)) {
    Opc = AVR::MOVWRdRr;
  } else if (SrcReg == AVR::SP && AVR::DREGSRegClass.contains(DestReg)) {
    Opc = AVR::SPREAD;
  } else if (DestReg == AVR::SP && AVR::DREGSRegClass.contains(SrcReg)) {
    Opc = AVR::SPWRITE;
  } else {
    llvm_unreachable("Impossible reg-to-reg copy");
  }

  BuildMI(MBB, MI, DL, get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}

unsigned AVRInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  case AVR::LDDRdPtrQ:
  case AVR::LDDWRdYQ: { //:FIXME: remove this once PR13375 gets fixed
    if ((MI->getOperand(1).isFI()) && (MI->getOperand(2).isImm()) &&
        (MI->getOperand(2).getImm() == 0)) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  default:
    break;
  }

  return 0;
}

unsigned AVRInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  case AVR::STDPtrQRr:
  case AVR::STDWPtrQRr: {
    if ((MI->getOperand(0).isFI()) && (MI->getOperand(1).isImm()) &&
        (MI->getOperand(1).getImm() == 0)) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(2).getReg();
    }
    break;
  }
  default:
    break;
  }

  return 0;
}

void AVRInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned SrcReg, bool isKill,
                                       int FrameIndex,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();

  DebugLoc DL;
  if (MI != MBB.end()) {
    DL = MI->getDebugLoc();
  }

  const MachineFrameInfo &MFI = *MF.getFrameInfo();

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FrameIndex),
      MachineMemOperand::MOStore, MFI.getObjectSize(FrameIndex),
      MFI.getObjectAlignment(FrameIndex));

  unsigned Opcode = 0;
  if (RC->hasType(MVT::i8)) {
    Opcode = AVR::STDPtrQRr;
  } else if (RC->hasType(MVT::i16)) {
    Opcode = AVR::STDWPtrQRr;
  } else {
    llvm_unreachable("Cannot store this register into a stack slot!");
  }

  BuildMI(MBB, MI, DL, get(Opcode))
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addReg(SrcReg, getKillRegState(isKill))
      .addMemOperand(MMO);
}

void AVRInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIndex,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (MI != MBB.end()) {
    DL = MI->getDebugLoc();
  }

  MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = *MF.getFrameInfo();

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FrameIndex),
      MachineMemOperand::MOLoad, MFI.getObjectSize(FrameIndex),
      MFI.getObjectAlignment(FrameIndex));

  unsigned Opcode = 0;
  if (RC->hasType(MVT::i8)) {
    Opcode = AVR::LDDRdPtrQ;
  } else if (RC->hasType(MVT::i16)) {
    // Opcode = AVR::LDDWRdPtrQ;
    //:FIXME: remove this once PR13375 gets fixed
    Opcode = AVR::LDDWRdYQ;
  } else {
    llvm_unreachable("Cannot load this register from a stack slot!");
  }

  BuildMI(MBB, MI, DL, get(Opcode), DestReg)
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addMemOperand(MMO);
}

const MCInstrDesc &AVRInstrInfo::getBrCond(AVRCC::CondCodes CC) const {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case AVRCC::COND_EQ:
    return get(AVR::BREQk);
  case AVRCC::COND_NE:
    return get(AVR::BRNEk);
  case AVRCC::COND_GE:
    return get(AVR::BRGEk);
  case AVRCC::COND_LT:
    return get(AVR::BRLTk);
  case AVRCC::COND_SH:
    return get(AVR::BRSHk);
  case AVRCC::COND_LO:
    return get(AVR::BRLOk);
  case AVRCC::COND_MI:
    return get(AVR::BRMIk);
  case AVRCC::COND_PL:
    return get(AVR::BRPLk);
  }
}

AVRCC::CondCodes AVRInstrInfo::getCondFromBranchOpc(unsigned Opc) const {
  switch (Opc) {
  default:
    return AVRCC::COND_INVALID;
  case AVR::BREQk:
    return AVRCC::COND_EQ;
  case AVR::BRNEk:
    return AVRCC::COND_NE;
  case AVR::BRSHk:
    return AVRCC::COND_SH;
  case AVR::BRLOk:
    return AVRCC::COND_LO;
  case AVR::BRMIk:
    return AVRCC::COND_MI;
  case AVR::BRPLk:
    return AVRCC::COND_PL;
  case AVR::BRGEk:
    return AVRCC::COND_GE;
  case AVR::BRLTk:
    return AVRCC::COND_LT;
  }
}

AVRCC::CondCodes AVRInstrInfo::getOppositeCondition(AVRCC::CondCodes CC) const {
  switch (CC) {
  default:
    llvm_unreachable("Invalid condition!");
  case AVRCC::COND_EQ:
    return AVRCC::COND_NE;
  case AVRCC::COND_NE:
    return AVRCC::COND_EQ;
  case AVRCC::COND_SH:
    return AVRCC::COND_LO;
  case AVRCC::COND_LO:
    return AVRCC::COND_SH;
  case AVRCC::COND_GE:
    return AVRCC::COND_LT;
  case AVRCC::COND_LT:
    return AVRCC::COND_GE;
  case AVRCC::COND_MI:
    return AVRCC::COND_PL;
  case AVRCC::COND_PL:
    return AVRCC::COND_MI;
  }
}

bool AVRInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  MachineBasicBlock::iterator UnCondBrIter = MBB.end();

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue()) {
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
    //:TODO: add here jmp
    if (I->getOpcode() == AVR::RJMPk) {
      UnCondBrIter = I;

      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a JMP, delete them.
      while (std::next(I) != MBB.end()) {
        std::next(I)->eraseFromParent();
      }

      Cond.clear();
      FBB = 0;

      // Delete the JMP if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = 0;
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
    AVRCC::CondCodes BranchCode = getCondFromBranchOpc(I->getOpcode());
    if (BranchCode == AVRCC::COND_INVALID) {
      return true; // Can't handle indirect branch.
    }

    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      MachineBasicBlock *TargetBB = I->getOperand(0).getMBB();
      if (AllowModify && UnCondBrIter != MBB.end() &&
          MBB.isLayoutSuccessor(TargetBB)) {
        // If we can modify the code and it ends in something like:
        //
        //     jCC L1
        //     jmp L2
        //   L1:
        //     ...
        //   L2:
        //
        // Then we can change this to:
        //
        //     jnCC L2
        //   L1:
        //     ...
        //   L2:
        //
        // Which is a bit more efficient.
        // We conditionally jump to the fall-through block.
        BranchCode = getOppositeCondition(BranchCode);
        unsigned JNCC = getBrCond(BranchCode).getOpcode();
        MachineBasicBlock::iterator OldInst = I;

        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(JNCC))
            .addMBB(UnCondBrIter->getOperand(0).getMBB());
        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(AVR::RJMPk))
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

    AVRCC::CondCodes OldBranchCode = (AVRCC::CondCodes)Cond[0].getImm();
    // If the conditions are the same, we can leave them alone.
    if (OldBranchCode == BranchCode) {
      continue;
    }

    return true;
  }

  return false;
}

unsigned AVRInstrInfo::InsertBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    ArrayRef<MachineOperand> Cond,
                                    const DebugLoc &DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "AVR branch conditions have one component!");

  if (Cond.empty()) {
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(AVR::RJMPk)).addMBB(TBB);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  AVRCC::CondCodes CC = (AVRCC::CondCodes)Cond[0].getImm();
  BuildMI(&MBB, DL, getBrCond(CC)).addMBB(TBB);
  ++Count;

  if (FBB) {
    // Two-way Conditional branch. Insert the second branch.
    BuildMI(&MBB, DL, get(AVR::RJMPk)).addMBB(FBB);
    ++Count;
  }

  return Count;
}

unsigned AVRInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue()) {
      continue;
    }
    //:TODO: add here the missing jmp instructions once they are implemented
    // like jmp, {e}ijmp, and other cond branches, ...
    if (I->getOpcode() != AVR::RJMPk &&
        getCondFromBranchOpc(I->getOpcode()) == AVRCC::COND_INVALID) {
      break;
    }

    // Remove the branch.
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
}

bool AVRInstrInfo::ReverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid AVR branch condition!");

  AVRCC::CondCodes CC = static_cast<AVRCC::CondCodes>(Cond[0].getImm());
  Cond[0].setImm(getOppositeCondition(CC));

  return false;
}

unsigned AVRInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  unsigned Opcode = MI->getOpcode();

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
  case TargetOpcode::INLINEASM: {
    const MachineFunction *MF = MI->getParent()->getParent();
    const AVRTargetMachine &TM = static_cast<const AVRTargetMachine&>(MF->getTarget());
    const TargetInstrInfo &TII = *TM.getSubtargetImpl()->getInstrInfo();
    return TII.getInlineAsmLength(MI->getOperand(0).getSymbolName(),
                                  *TM.getMCAsmInfo());
  }
  }
}

} // end of namespace llvm
