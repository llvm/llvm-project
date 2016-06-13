//===-- LanaiInstrInfo.cpp - Lanai Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Lanai implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Lanai.h"
#include "LanaiInstrInfo.h"
#include "LanaiMachineFunctionInfo.h"
#include "LanaiTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "LanaiGenInstrInfo.inc"

LanaiInstrInfo::LanaiInstrInfo()
    : LanaiGenInstrInfo(Lanai::ADJCALLSTACKDOWN, Lanai::ADJCALLSTACKUP),
      RegisterInfo() {}

void LanaiInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator Position,
                                 const DebugLoc &DL,
                                 unsigned DestinationRegister,
                                 unsigned SourceRegister,
                                 bool KillSource) const {
  if (!Lanai::GPRRegClass.contains(DestinationRegister, SourceRegister)) {
    llvm_unreachable("Impossible reg-to-reg copy");
  }

  BuildMI(MBB, Position, DL, get(Lanai::OR_I_LO), DestinationRegister)
      .addReg(SourceRegister, getKillRegState(KillSource))
      .addImm(0);
}

void LanaiInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
    unsigned SourceRegister, bool IsKill, int FrameIndex,
    const TargetRegisterClass *RegisterClass,
    const TargetRegisterInfo *RegisterInfo) const {
  DebugLoc DL;
  if (Position != MBB.end()) {
    DL = Position->getDebugLoc();
  }

  if (!Lanai::GPRRegClass.hasSubClassEq(RegisterClass)) {
    llvm_unreachable("Can't store this register to stack slot");
  }
  BuildMI(MBB, Position, DL, get(Lanai::SW_RI))
      .addReg(SourceRegister, getKillRegState(IsKill))
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addImm(LPAC::ADD);
}

void LanaiInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
    unsigned DestinationRegister, int FrameIndex,
    const TargetRegisterClass *RegisterClass,
    const TargetRegisterInfo *RegisterInfo) const {
  DebugLoc DL;
  if (Position != MBB.end()) {
    DL = Position->getDebugLoc();
  }

  if (!Lanai::GPRRegClass.hasSubClassEq(RegisterClass)) {
    llvm_unreachable("Can't load this register from stack slot");
  }
  BuildMI(MBB, Position, DL, get(Lanai::LDW_RI), DestinationRegister)
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addImm(LPAC::ADD);
}

bool LanaiInstrInfo::areMemAccessesTriviallyDisjoint(MachineInstr *MIa,
                                                     MachineInstr *MIb,
                                                     AliasAnalysis *AA) const {
  assert(MIa && MIa->mayLoadOrStore() && "MIa must be a load or store.");
  assert(MIb && MIb->mayLoadOrStore() && "MIb must be a load or store.");

  if (MIa->hasUnmodeledSideEffects() || MIb->hasUnmodeledSideEffects() ||
      MIa->hasOrderedMemoryRef() || MIb->hasOrderedMemoryRef())
    return false;

  // Retrieve the base register, offset from the base register and width. Width
  // is the size of memory that is being loaded/stored (e.g. 1, 2, 4).  If
  // base registers are identical, and the offset of a lower memory access +
  // the width doesn't overlap the offset of a higher memory access,
  // then the memory accesses are different.
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  unsigned BaseRegA = 0, BaseRegB = 0;
  int64_t OffsetA = 0, OffsetB = 0;
  unsigned int WidthA = 0, WidthB = 0;
  if (getMemOpBaseRegImmOfsWidth(MIa, BaseRegA, OffsetA, WidthA, TRI) &&
      getMemOpBaseRegImmOfsWidth(MIb, BaseRegB, OffsetB, WidthB, TRI)) {
    if (BaseRegA == BaseRegB) {
      int LowOffset = std::min(OffsetA, OffsetB);
      int HighOffset = std::max(OffsetA, OffsetB);
      int LowWidth = (LowOffset == OffsetA) ? WidthA : WidthB;
      if (LowOffset + LowWidth <= HighOffset)
        return true;
    }
  }
  return false;
}

bool LanaiInstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
  return false;
}

static LPCC::CondCode GetOppositeBranchCondition(LPCC::CondCode CC) {
  switch (CC) {
  case LPCC::ICC_T: //  true
    return LPCC::ICC_F;
  case LPCC::ICC_F: //  false
    return LPCC::ICC_T;
  case LPCC::ICC_HI: //  high
    return LPCC::ICC_LS;
  case LPCC::ICC_LS: //  low or same
    return LPCC::ICC_HI;
  case LPCC::ICC_CC: //  carry cleared
    return LPCC::ICC_CS;
  case LPCC::ICC_CS: //  carry set
    return LPCC::ICC_CC;
  case LPCC::ICC_NE: //  not equal
    return LPCC::ICC_EQ;
  case LPCC::ICC_EQ: //  equal
    return LPCC::ICC_NE;
  case LPCC::ICC_VC: //  oVerflow cleared
    return LPCC::ICC_VS;
  case LPCC::ICC_VS: //  oVerflow set
    return LPCC::ICC_VC;
  case LPCC::ICC_PL: //  plus (note: 0 is "minus" too here)
    return LPCC::ICC_MI;
  case LPCC::ICC_MI: //  minus
    return LPCC::ICC_PL;
  case LPCC::ICC_GE: //  greater than or equal
    return LPCC::ICC_LT;
  case LPCC::ICC_LT: //  less than
    return LPCC::ICC_GE;
  case LPCC::ICC_GT: //  greater than
    return LPCC::ICC_LE;
  case LPCC::ICC_LE: //  less than or equal
    return LPCC::ICC_GT;
  default:
    llvm_unreachable("Invalid condtional code");
  }
}

// The AnalyzeBranch function is used to examine conditional instructions and
// remove unnecessary instructions. This method is used by BranchFolder and
// IfConverter machine function passes to improve the CFG.
// - TrueBlock is set to the destination if condition evaluates true (it is the
//   nullptr if the destination is the fall-through branch);
// - FalseBlock is set to the destination if condition evaluates to false (it
//   is the nullptr if the branch is unconditional);
// - condition is populated with machine operands needed to generate the branch
//   to insert in InsertBranch;
// Returns: false if branch could successfully be analyzed.
bool LanaiInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TrueBlock,
                                   MachineBasicBlock *&FalseBlock,
                                   SmallVectorImpl<MachineOperand> &Condition,
                                   bool AllowModify) const {
  // Iterator to current instruction being considered.
  MachineBasicBlock::iterator Instruction = MBB.end();

  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  while (Instruction != MBB.begin()) {
    --Instruction;

    // Skip over debug values.
    if (Instruction->isDebugValue())
      continue;

    // Working from the bottom, when we see a non-terminator
    // instruction, we're done.
    if (!isUnpredicatedTerminator(*Instruction))
      break;

    // A terminator that isn't a branch can't easily be handled
    // by this analysis.
    if (!Instruction->isBranch())
      return true;

    // Handle unconditional branches.
    if (Instruction->getOpcode() == Lanai::BT) {
      if (!AllowModify) {
        TrueBlock = Instruction->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a branch, delete them.
      while (std::next(Instruction) != MBB.end()) {
        std::next(Instruction)->eraseFromParent();
      }

      Condition.clear();
      FalseBlock = nullptr;

      // Delete the jump if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(Instruction->getOperand(0).getMBB())) {
        TrueBlock = nullptr;
        Instruction->eraseFromParent();
        Instruction = MBB.end();
        continue;
      }

      // TrueBlock is used to indicate the unconditional destination.
      TrueBlock = Instruction->getOperand(0).getMBB();
      continue;
    }

    // Handle conditional branches
    unsigned Opcode = Instruction->getOpcode();
    if (Opcode != Lanai::BRCC)
      return true; // Unknown opcode.

    // Multiple conditional branches are not handled here so only proceed if
    // there are no conditions enqueued.
    if (Condition.empty()) {
      LPCC::CondCode BranchCond =
          static_cast<LPCC::CondCode>(Instruction->getOperand(1).getImm());

      // TrueBlock is the target of the previously seen unconditional branch.
      FalseBlock = TrueBlock;
      TrueBlock = Instruction->getOperand(0).getMBB();
      Condition.push_back(MachineOperand::CreateImm(BranchCond));
      continue;
    }

    // Multiple conditional branches are not handled.
    return true;
  }

  // Return false indicating branch successfully analyzed.
  return false;
}

// ReverseBranchCondition - Reverses the branch condition of the specified
// condition list, returning false on success and true if it cannot be
// reversed.
bool LanaiInstrInfo::ReverseBranchCondition(
    SmallVectorImpl<llvm::MachineOperand> &Condition) const {
  assert((Condition.size() == 1) &&
         "Lanai branch conditions should have one component.");

  LPCC::CondCode BranchCond =
      static_cast<LPCC::CondCode>(Condition[0].getImm());
  Condition[0].setImm(GetOppositeBranchCondition(BranchCond));
  return false;
}

// Insert the branch with condition specified in condition and given targets
// (TrueBlock and FalseBlock). This function returns the number of machine
// instructions inserted.
unsigned LanaiInstrInfo::InsertBranch(MachineBasicBlock &MBB,
                                      MachineBasicBlock *TrueBlock,
                                      MachineBasicBlock *FalseBlock,
                                      ArrayRef<MachineOperand> Condition,
                                      const DebugLoc &DL) const {
  // Shouldn't be a fall through.
  assert(TrueBlock && "InsertBranch must not be told to insert a fallthrough");

  // If condition is empty then an unconditional branch is being inserted.
  if (Condition.empty()) {
    assert(!FalseBlock && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(Lanai::BT)).addMBB(TrueBlock);
    return 1;
  }

  // Else a conditional branch is inserted.
  assert((Condition.size() == 1) &&
         "Lanai branch conditions should have one component.");
  unsigned ConditionalCode = Condition[0].getImm();
  BuildMI(&MBB, DL, get(Lanai::BRCC)).addMBB(TrueBlock).addImm(ConditionalCode);

  // If no false block, then false behavior is fall through and no branch needs
  // to be inserted.
  if (!FalseBlock)
    return 1;

  BuildMI(&MBB, DL, get(Lanai::BT)).addMBB(FalseBlock);
  return 2;
}

unsigned LanaiInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator Instruction = MBB.end();
  unsigned Count = 0;

  while (Instruction != MBB.begin()) {
    --Instruction;
    if (Instruction->isDebugValue())
      continue;
    if (Instruction->getOpcode() != Lanai::BT &&
        Instruction->getOpcode() != Lanai::BRCC) {
      break;
    }

    // Remove the branch.
    Instruction->eraseFromParent();
    Instruction = MBB.end();
    ++Count;
  }

  return Count;
}

unsigned LanaiInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {
  if (MI->getOpcode() == Lanai::LDW_RI)
    if (MI->getOperand(1).isFI() && MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
  return 0;
}

unsigned LanaiInstrInfo::isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                                   int &FrameIndex) const {
  if (MI->getOpcode() == Lanai::LDW_RI) {
    unsigned Reg;
    if ((Reg = isLoadFromStackSlot(MI, FrameIndex)))
      return Reg;
    // Check for post-frame index elimination operations
    const MachineMemOperand *Dummy;
    return hasLoadFromStackSlot(MI, Dummy, FrameIndex);
  }
  return 0;
}

unsigned LanaiInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                            int &FrameIndex) const {
  if (MI->getOpcode() == Lanai::SW_RI)
    if (MI->getOperand(0).isFI() && MI->getOperand(1).isImm() &&
        MI->getOperand(1).getImm() == 0) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(2).getReg();
    }
  return 0;
}

bool LanaiInstrInfo::getMemOpBaseRegImmOfsWidth(
    MachineInstr *LdSt, unsigned &BaseReg, int64_t &Offset, unsigned &Width,
    const TargetRegisterInfo *TRI) const {
  // Handle only loads/stores with base register followed by immediate offset
  // and with add as ALU op.
  if (LdSt->getNumOperands() != 4)
    return false;
  if (!LdSt->getOperand(1).isReg() || !LdSt->getOperand(2).isImm() ||
      !(LdSt->getOperand(3).isImm() &&
        LdSt->getOperand(3).getImm() == LPAC::ADD))
    return false;

  switch (LdSt->getOpcode()) {
  default:
    return false;
  case Lanai::LDW_RI:
  case Lanai::LDW_RR:
  case Lanai::SW_RR:
  case Lanai::SW_RI:
    Width = 4;
    break;
  case Lanai::LDHs_RI:
  case Lanai::LDHz_RI:
  case Lanai::STH_RI:
    Width = 2;
    break;
  case Lanai::LDBs_RI:
  case Lanai::LDBz_RI:
  case Lanai::STB_RI:
    Width = 1;
    break;
  }

  BaseReg = LdSt->getOperand(1).getReg();
  Offset = LdSt->getOperand(2).getImm();
  return true;
}

bool LanaiInstrInfo::getMemOpBaseRegImmOfs(
    MachineInstr *LdSt, unsigned &BaseReg, int64_t &Offset,
    const TargetRegisterInfo *TRI) const {
  switch (LdSt->getOpcode()) {
  default:
    return false;
  case Lanai::LDW_RI:
  case Lanai::LDW_RR:
  case Lanai::SW_RR:
  case Lanai::SW_RI:
  case Lanai::LDHs_RI:
  case Lanai::LDHz_RI:
  case Lanai::STH_RI:
  case Lanai::LDBs_RI:
  case Lanai::LDBz_RI:
    unsigned Width;
    return getMemOpBaseRegImmOfsWidth(LdSt, BaseReg, Offset, Width, TRI);
  }
}
