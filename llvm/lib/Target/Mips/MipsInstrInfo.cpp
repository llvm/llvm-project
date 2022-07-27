//===- MipsInstrInfo.cpp - Mips Instruction Information -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MipsInstrInfo.h"
#include "MipsMachineFunction.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "MipsSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "MipsGenInstrInfo.inc"

// Pin the vtable to this file.
void MipsInstrInfo::anchor() {}

enum MachineOutlinerClass {
  MachineOutlinerDefault,  /// Emit a save, restore, call, and return.
  MachineOutlinerTailCall, /// Only emit a branch.
  MachineOutlinerNoRASave, /// Emit a call and return.
  MachineOutlinerThunk,    /// Emit a call and tail-call.
  MachineOutlinerRegSave   /// Same as default, but save to a register.
};

enum MachineOutlinerMBBFlags {
  LRUnavailableSomewhere = 0x2,
  HasCalls = 0x4,
  UnsafeRegsDead = 0x8
};

MipsInstrInfo::MipsInstrInfo(const MipsSubtarget &STI, unsigned UncondBr)
    : MipsGenInstrInfo(STI.hasNanoMips() ? Mips::ADJCALLSTACKDOWN_NM
                                         : Mips::ADJCALLSTACKDOWN,
                       STI.hasNanoMips() ? Mips::ADJCALLSTACKUP_NM
                                         : Mips::ADJCALLSTACKUP),
      Subtarget(STI), UncondBrOpc(UncondBr) {}

const MipsInstrInfo *MipsInstrInfo::create(MipsSubtarget &STI) {
  if (STI.inMips16Mode())
    return createMips16InstrInfo(STI);

  return createMipsSEInstrInfo(STI);
}

bool MipsInstrInfo::isZeroImm(const MachineOperand &op) const {
  return op.isImm() && op.getImm() == 0;
}

/// insertNoop - If data hazard condition is found insert the target nop
/// instruction.
// FIXME: This appears to be dead code.
void MipsInstrInfo::
insertNoop(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI) const
{
  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(Mips::NOP));
}

MachineMemOperand *
MipsInstrInfo::GetMemOperand(MachineBasicBlock &MBB, int FI,
                             MachineMemOperand::Flags Flags) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  return MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                                 Flags, MFI.getObjectSize(FI),
                                 MFI.getObjectAlign(FI));
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//

void MipsInstrInfo::AnalyzeCondBr(const MachineInstr *Inst, unsigned Opc,
                                  MachineBasicBlock *&BB,
                                  SmallVectorImpl<MachineOperand> &Cond) const {
  assert(getAnalyzableBrOpc(Opc) && "Not an analyzable branch");
  int NumOp = Inst->getNumExplicitOperands();

  // for both int and fp branches, the last explicit operand is the
  // MBB.
  BB = Inst->getOperand(NumOp-1).getMBB();
  Cond.push_back(MachineOperand::CreateImm(Opc));

  for (int i = 0; i < NumOp-1; i++)
    Cond.push_back(Inst->getOperand(i));
}

bool MipsInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond,
                                  bool AllowModify) const {
  SmallVector<MachineInstr*, 2> BranchInstrs;
  BranchType BT = analyzeBranch(MBB, TBB, FBB, Cond, AllowModify, BranchInstrs);

  return (BT == BT_None) || (BT == BT_Indirect);
}

void MipsInstrInfo::BuildCondBr(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                const DebugLoc &DL,
                                ArrayRef<MachineOperand> Cond) const {
  unsigned Opc = Cond[0].getImm();
  const MCInstrDesc &MCID = get(Opc);
  MachineInstrBuilder MIB = BuildMI(&MBB, DL, MCID);

  for (unsigned i = 1; i < Cond.size(); ++i) {
    assert((Cond[i].isImm() || Cond[i].isReg()) &&
           "Cannot copy operand for conditional branch!");
    MIB.add(Cond[i]);
  }
  MIB.addMBB(TBB);
}

unsigned MipsInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock *TBB,
                                     MachineBasicBlock *FBB,
                                     ArrayRef<MachineOperand> Cond,
                                     const DebugLoc &DL,
                                     int *BytesAdded) const {
  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert(!BytesAdded && "code size not handled");

  // # of condition operands:
  //  Unconditional branches: 0
  //  Floating point branches: 1 (opc)
  //  Int BranchZero: 2 (opc, reg)
  //  Int Branch: 3 (opc, reg0, reg1)
  assert((Cond.size() <= 3) &&
         "# of Mips branch conditions must be <= 3!");

  // Two-way Conditional branch.
  if (FBB) {
    BuildCondBr(MBB, TBB, DL, Cond);
    BuildMI(&MBB, DL, get(UncondBrOpc)).addMBB(FBB);
    return 2;
  }

  // One way branch.
  // Unconditional branch.
  if (Cond.empty())
    BuildMI(&MBB, DL, get(UncondBrOpc)).addMBB(TBB);
  else // Conditional branch.
    BuildCondBr(MBB, TBB, DL, Cond);
  return 1;
}

unsigned MipsInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                     int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");

  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();
  unsigned removed = 0;

  // Up to 2 branches are removed.
  // Note that indirect branches are not removed.
  while (I != REnd && removed < 2) {
    // Skip past debug instructions.
    if (I->isDebugInstr()) {
      ++I;
      continue;
    }
    if (!getAnalyzableBrOpc(I->getOpcode()))
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.rbegin();
    ++removed;
  }

  return removed;
}

/// reverseBranchCondition - Return the inverse opcode of the
/// specified Branch instruction.
bool MipsInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert( (Cond.size() && Cond.size() <= 3) &&
          "Invalid Mips branch condition!");
  Cond[0].setImm(getOppositeBranchOpc(Cond[0].getImm()));
  return false;
}

MipsInstrInfo::BranchType MipsInstrInfo::analyzeBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *&TBB, MachineBasicBlock *&FBB,
    SmallVectorImpl<MachineOperand> &Cond, bool AllowModify,
    SmallVectorImpl<MachineInstr *> &BranchInstrs) const {
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugInstr())
    ++I;

  if (I == REnd || !isUnpredicatedTerminator(*I)) {
    // This block ends with no branches (it just falls through to its succ).
    // Leave TBB/FBB null.
    TBB = FBB = nullptr;
    return BT_NoBranch;
  }

  MachineInstr *LastInst = &*I;
  unsigned LastOpc = LastInst->getOpcode();
  BranchInstrs.push_back(LastInst);

  // Not an analyzable branch (e.g., indirect jump).
  if (!getAnalyzableBrOpc(LastOpc))
    return LastInst->isIndirectBranch() ? BT_Indirect : BT_None;

  // Get the second to last instruction in the block.
  unsigned SecondLastOpc = 0;
  MachineInstr *SecondLastInst = nullptr;

  // Skip past any debug instruction to see if the second last actual
  // is a branch.
  ++I;
  while (I != REnd && I->isDebugInstr())
    ++I;

  if (I != REnd) {
    SecondLastInst = &*I;
    SecondLastOpc = getAnalyzableBrOpc(SecondLastInst->getOpcode());

    // Not an analyzable branch (must be an indirect jump).
    if (isUnpredicatedTerminator(*SecondLastInst) && !SecondLastOpc)
      return BT_None;
  }

  // If there is only one terminator instruction, process it.
  if (!SecondLastOpc) {
    // Unconditional branch.
    if (LastInst->isUnconditionalBranch()) {
      TBB = LastInst->getOperand(0).getMBB();
      return BT_Uncond;
    }

    // Conditional branch
    AnalyzeCondBr(LastInst, LastOpc, TBB, Cond);
    return BT_Cond;
  }

  // If we reached here, there are two branches.
  // If there are three terminators, we don't know what sort of block this is.
  if (++I != REnd && isUnpredicatedTerminator(*I))
    return BT_None;

  BranchInstrs.insert(BranchInstrs.begin(), SecondLastInst);

  // If second to last instruction is an unconditional branch,
  // analyze it and remove the last instruction.
  if (SecondLastInst->isUnconditionalBranch()) {
    // Return if the last instruction cannot be removed.
    if (!AllowModify)
      return BT_None;

    TBB = SecondLastInst->getOperand(0).getMBB();
    LastInst->eraseFromParent();
    BranchInstrs.pop_back();
    return BT_Uncond;
  }

  // Conditional branch followed by an unconditional branch.
  // The last one must be unconditional.
  if (!LastInst->isUnconditionalBranch())
    return BT_None;

  AnalyzeCondBr(SecondLastInst, SecondLastOpc, TBB, Cond);
  FBB = LastInst->getOperand(0).getMBB();

  return BT_CondUncond;
}

bool MipsInstrInfo::isBranchOffsetInRange(unsigned BranchOpc,
                                          int64_t BrOffset) const {
  switch (BranchOpc) {
  case Mips::B:
  case Mips::BAL:
  case Mips::BAL_BR:
  case Mips::BAL_BR_MM:
  case Mips::BC1F:
  case Mips::BC1FL:
  case Mips::BC1T:
  case Mips::BC1TL:
  case Mips::BEQ:     case Mips::BEQ64:
  case Mips::BEQL:
  case Mips::BGEZ:    case Mips::BGEZ64:
  case Mips::BGEZL:
  case Mips::BGEZAL:
  case Mips::BGEZALL:
  case Mips::BGTZ:    case Mips::BGTZ64:
  case Mips::BGTZL:
  case Mips::BLEZ:    case Mips::BLEZ64:
  case Mips::BLEZL:
  case Mips::BLTZ:    case Mips::BLTZ64:
  case Mips::BLTZL:
  case Mips::BLTZAL:
  case Mips::BLTZALL:
  case Mips::BNE:     case Mips::BNE64:
  case Mips::BNEL:
    return isInt<18>(BrOffset);

  // microMIPSr3 branches
  case Mips::B_MM:
  case Mips::BC1F_MM:
  case Mips::BC1T_MM:
  case Mips::BEQ_MM:
  case Mips::BGEZ_MM:
  case Mips::BGEZAL_MM:
  case Mips::BGTZ_MM:
  case Mips::BLEZ_MM:
  case Mips::BLTZ_MM:
  case Mips::BLTZAL_MM:
  case Mips::BNE_MM:
  case Mips::BEQZC_MM:
  case Mips::BNEZC_MM:
    return isInt<17>(BrOffset);

  // microMIPSR3 short branches.
  case Mips::B16_MM:
    return isInt<11>(BrOffset);

  case Mips::BEQZ16_MM:
  case Mips::BNEZ16_MM:
    return isInt<8>(BrOffset);

  // MIPSR6 branches.
  case Mips::BALC:
  case Mips::BC:
    return isInt<28>(BrOffset);

  case Mips::BC1EQZ:
  case Mips::BC1NEZ:
  case Mips::BC2EQZ:
  case Mips::BC2NEZ:
  case Mips::BEQC:   case Mips::BEQC64:
  case Mips::BNEC:   case Mips::BNEC64:
  case Mips::BGEC:   case Mips::BGEC64:
  case Mips::BGEUC:  case Mips::BGEUC64:
  case Mips::BGEZC:  case Mips::BGEZC64:
  case Mips::BGTZC:  case Mips::BGTZC64:
  case Mips::BLEZC:  case Mips::BLEZC64:
  case Mips::BLTC:   case Mips::BLTC64:
  case Mips::BLTUC:  case Mips::BLTUC64:
  case Mips::BLTZC:  case Mips::BLTZC64:
  case Mips::BNVC:
  case Mips::BOVC:
  case Mips::BGEZALC:
  case Mips::BEQZALC:
  case Mips::BGTZALC:
  case Mips::BLEZALC:
  case Mips::BLTZALC:
  case Mips::BNEZALC:
    return isInt<18>(BrOffset);

  case Mips::BEQZC:  case Mips::BEQZC64:
  case Mips::BNEZC:  case Mips::BNEZC64:
    return isInt<23>(BrOffset);

  // microMIPSR6 branches
  case Mips::BC16_MMR6:
    return isInt<11>(BrOffset);

  case Mips::BEQZC16_MMR6:
  case Mips::BNEZC16_MMR6:
    return isInt<8>(BrOffset);

  case Mips::BALC_MMR6:
  case Mips::BC_MMR6:
    return isInt<27>(BrOffset);

  case Mips::BC1EQZC_MMR6:
  case Mips::BC1NEZC_MMR6:
  case Mips::BC2EQZC_MMR6:
  case Mips::BC2NEZC_MMR6:
  case Mips::BGEZALC_MMR6:
  case Mips::BEQZALC_MMR6:
  case Mips::BGTZALC_MMR6:
  case Mips::BLEZALC_MMR6:
  case Mips::BLTZALC_MMR6:
  case Mips::BNEZALC_MMR6:
  case Mips::BNVC_MMR6:
  case Mips::BOVC_MMR6:
    return isInt<17>(BrOffset);

  case Mips::BEQC_MMR6:
  case Mips::BNEC_MMR6:
  case Mips::BGEC_MMR6:
  case Mips::BGEUC_MMR6:
  case Mips::BGEZC_MMR6:
  case Mips::BGTZC_MMR6:
  case Mips::BLEZC_MMR6:
  case Mips::BLTC_MMR6:
  case Mips::BLTUC_MMR6:
  case Mips::BLTZC_MMR6:
    return isInt<18>(BrOffset);

  case Mips::BEQZC_MMR6:
  case Mips::BNEZC_MMR6:
    return isInt<23>(BrOffset);

  // DSP branches.
  case Mips::BPOSGE32:
    return isInt<18>(BrOffset);
  case Mips::BPOSGE32_MM:
  case Mips::BPOSGE32C_MMR3:
    return isInt<17>(BrOffset);

  // cnMIPS branches.
  case Mips::BBIT0:
  case Mips::BBIT032:
  case Mips::BBIT1:
  case Mips::BBIT132:
    return isInt<18>(BrOffset);

  // MSA branches.
  case Mips::BZ_B:
  case Mips::BZ_H:
  case Mips::BZ_W:
  case Mips::BZ_D:
  case Mips::BZ_V:
  case Mips::BNZ_B:
  case Mips::BNZ_H:
  case Mips::BNZ_W:
  case Mips::BNZ_D:
  case Mips::BNZ_V:
    return isInt<18>(BrOffset);

  // NanoMips branches
  case Mips::BEQC_NM:
  case Mips::BGEC_NM:
  case Mips::BGEUC_NM:
  case Mips::BLTC_NM:
  case Mips::BLTUC_NM:
  case Mips::BNEC_NM:
    return isInt<15>(BrOffset);

  case Mips::BEQIC_NM:
  case Mips::BGEIC_NM:
  case Mips::BGEIUC_NM:
  case Mips::BLTIC_NM:
  case Mips::BLTIUC_NM:
  case Mips::BNEIC_NM:
    return isInt<12>(BrOffset);

  case Mips::BEQZC_NM:
  case Mips::BNEZC_NM:
    return isInt<8>(BrOffset);

  case Mips::BC_NM:
  case Mips::BALC_NM:
    return isInt<26>(BrOffset);
}

  llvm_unreachable("Unknown branch instruction!");
}

/// Return the corresponding compact (no delay slot) form of a branch.
unsigned MipsInstrInfo::getEquivalentCompactForm(
    const MachineBasicBlock::iterator I) const {
  unsigned Opcode = I->getOpcode();
  bool canUseShortMicroMipsCTI = false;

  if (Subtarget.inMicroMipsMode()) {
    switch (Opcode) {
    case Mips::BNE:
    case Mips::BNE_MM:
    case Mips::BEQ:
    case Mips::BEQ_MM:
    // microMIPS has NE,EQ branches that do not have delay slots provided one
    // of the operands is zero.
      if (I->getOperand(1).getReg() == Subtarget.getABI().GetZeroReg())
        canUseShortMicroMipsCTI = true;
      break;
    // For microMIPS the PseudoReturn and PseudoIndirectBranch are always
    // expanded to JR_MM, so they can be replaced with JRC16_MM.
    case Mips::JR:
    case Mips::PseudoReturn:
    case Mips::PseudoIndirectBranch:
      canUseShortMicroMipsCTI = true;
      break;
    }
  }

  // MIPSR6 forbids both operands being the zero register.
  if (Subtarget.hasMips32r6() && (I->getNumOperands() > 1) &&
      (I->getOperand(0).isReg() &&
       (I->getOperand(0).getReg() == Mips::ZERO ||
        I->getOperand(0).getReg() == Mips::ZERO_64)) &&
      (I->getOperand(1).isReg() &&
       (I->getOperand(1).getReg() == Mips::ZERO ||
        I->getOperand(1).getReg() == Mips::ZERO_64)))
    return 0;

  if (Subtarget.hasMips32r6() || canUseShortMicroMipsCTI) {
    switch (Opcode) {
    case Mips::B:
      return Mips::BC;
    case Mips::BAL:
      return Mips::BALC;
    case Mips::BEQ:
    case Mips::BEQ_MM:
      if (canUseShortMicroMipsCTI)
        return Mips::BEQZC_MM;
      else if (I->getOperand(0).getReg() == I->getOperand(1).getReg())
        return 0;
      return Mips::BEQC;
    case Mips::BNE:
    case Mips::BNE_MM:
      if (canUseShortMicroMipsCTI)
        return Mips::BNEZC_MM;
      else if (I->getOperand(0).getReg() == I->getOperand(1).getReg())
        return 0;
      return Mips::BNEC;
    case Mips::BGE:
      if (I->getOperand(0).getReg() == I->getOperand(1).getReg())
        return 0;
      return Mips::BGEC;
    case Mips::BGEU:
      if (I->getOperand(0).getReg() == I->getOperand(1).getReg())
        return 0;
      return Mips::BGEUC;
    case Mips::BGEZ:
      return Mips::BGEZC;
    case Mips::BGTZ:
      return Mips::BGTZC;
    case Mips::BLEZ:
      return Mips::BLEZC;
    case Mips::BLT:
      if (I->getOperand(0).getReg() == I->getOperand(1).getReg())
        return 0;
      return Mips::BLTC;
    case Mips::BLTU:
      if (I->getOperand(0).getReg() == I->getOperand(1).getReg())
        return 0;
      return Mips::BLTUC;
    case Mips::BLTZ:
      return Mips::BLTZC;
    case Mips::BEQ64:
      if (I->getOperand(0).getReg() == I->getOperand(1).getReg())
        return 0;
      return Mips::BEQC64;
    case Mips::BNE64:
      if (I->getOperand(0).getReg() == I->getOperand(1).getReg())
        return 0;
      return Mips::BNEC64;
    case Mips::BGTZ64:
      return Mips::BGTZC64;
    case Mips::BGEZ64:
      return Mips::BGEZC64;
    case Mips::BLTZ64:
      return Mips::BLTZC64;
    case Mips::BLEZ64:
      return Mips::BLEZC64;
    // For MIPSR6, the instruction 'jic' can be used for these cases. Some
    // tools will accept 'jrc reg' as an alias for 'jic 0, $reg'.
    case Mips::JR:
    case Mips::PseudoIndirectBranchR6:
    case Mips::PseudoReturn:
    case Mips::TAILCALLR6REG:
      if (canUseShortMicroMipsCTI)
        return Mips::JRC16_MM;
      return Mips::JIC;
    case Mips::JALRPseudo:
      return Mips::JIALC;
    case Mips::JR64:
    case Mips::PseudoIndirectBranch64R6:
    case Mips::PseudoReturn64:
    case Mips::TAILCALL64R6REG:
      return Mips::JIC64;
    case Mips::JALR64Pseudo:
      return Mips::JIALC64;
    default:
      return 0;
    }
  }

  return 0;
}

/// Predicate for distingushing between control transfer instructions and all
/// other instructions for handling forbidden slots. Consider inline assembly
/// as unsafe as well.
bool MipsInstrInfo::SafeInForbiddenSlot(const MachineInstr &MI) const {
  if (MI.isInlineAsm())
    return false;

  return (MI.getDesc().TSFlags & MipsII::IsCTI) == 0;
}

/// Predicate for distingushing instructions that have forbidden slots.
bool MipsInstrInfo::HasForbiddenSlot(const MachineInstr &MI) const {
  return (MI.getDesc().TSFlags & MipsII::HasForbiddenSlot) != 0;
}

/// Return the number of bytes of code the specified instruction may be.
unsigned MipsInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    return MI.getDesc().getSize();
  case  TargetOpcode::INLINEASM:
  case  TargetOpcode::INLINEASM_BR: {       // Inline Asm: Variable size.
    const MachineFunction *MF = MI.getParent()->getParent();
    const char *AsmStr = MI.getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  }
  case Mips::CONSTPOOL_ENTRY:
    // If this machine instr is a constant pool entry, its size is recorded as
    // operand #2.
    return MI.getOperand(2).getImm();
  }
}

MachineInstrBuilder
MipsInstrInfo::genInstrWithNewOpc(unsigned NewOpc,
                                  MachineBasicBlock::iterator I) const {
  MachineInstrBuilder MIB;

  // Certain branches have two forms: e.g beq $1, $zero, dest vs beqz $1, dest
  // Pick the zero form of the branch for readable assembly and for greater
  // branch distance in non-microMIPS mode.
  // Additional MIPSR6 does not permit the use of register $zero for compact
  // branches.
  // FIXME: Certain atomic sequences on mips64 generate 32bit references to
  // Mips::ZERO, which is incorrect. This test should be updated to use
  // Subtarget.getABI().GetZeroReg() when those atomic sequences and others
  // are fixed.
  int ZeroOperandPosition = -1;
  bool BranchWithZeroOperand = false;
  if (I->isBranch() && !I->isPseudo()) {
    auto TRI = I->getParent()->getParent()->getSubtarget().getRegisterInfo();
    ZeroOperandPosition = I->findRegisterUseOperandIdx(Mips::ZERO, false, TRI);
    BranchWithZeroOperand = ZeroOperandPosition != -1;
  }

  if (BranchWithZeroOperand) {
    switch (NewOpc) {
    case Mips::BEQC:
      NewOpc = Mips::BEQZC;
      break;
    case Mips::BNEC:
      NewOpc = Mips::BNEZC;
      break;
    case Mips::BGEC:
      NewOpc = Mips::BGEZC;
      break;
    case Mips::BLTC:
      NewOpc = Mips::BLTZC;
      break;
    case Mips::BEQC64:
      NewOpc = Mips::BEQZC64;
      break;
    case Mips::BNEC64:
      NewOpc = Mips::BNEZC64;
      break;
    }
  }

  MIB = BuildMI(*I->getParent(), I, I->getDebugLoc(), get(NewOpc));

  // For MIPSR6 JI*C requires an immediate 0 as an operand, JIALC(64) an
  // immediate 0 as an operand and requires the removal of it's implicit-def %ra
  // implicit operand as copying the implicit operations of the instructio we're
  // looking at will give us the correct flags.
  if (NewOpc == Mips::JIC || NewOpc == Mips::JIALC || NewOpc == Mips::JIC64 ||
      NewOpc == Mips::JIALC64) {

    if (NewOpc == Mips::JIALC || NewOpc == Mips::JIALC64)
      MIB->RemoveOperand(0);

    for (unsigned J = 0, E = I->getDesc().getNumOperands(); J < E; ++J) {
      MIB.add(I->getOperand(J));
    }

    MIB.addImm(0);

    // If I has an MCSymbol operand (used by asm printer, to emit R_MIPS_JALR),
    // add it to the new instruction.
    for (unsigned J = I->getDesc().getNumOperands(), E = I->getNumOperands();
         J < E; ++J) {
      const MachineOperand &MO = I->getOperand(J);
      if (MO.isMCSymbol() && (MO.getTargetFlags() & MipsII::MO_JALR))
        MIB.addSym(MO.getMCSymbol(), MipsII::MO_JALR);
    }


  } else {
    for (unsigned J = 0, E = I->getDesc().getNumOperands(); J < E; ++J) {
      if (BranchWithZeroOperand && (unsigned)ZeroOperandPosition == J)
        continue;

      MIB.add(I->getOperand(J));
    }
  }

  MIB.copyImplicitOps(*I);
  MIB.cloneMemRefs(*I);
  return MIB;
}

bool MipsInstrInfo::findCommutedOpIndices(const MachineInstr &MI,
                                          unsigned &SrcOpIdx1,
                                          unsigned &SrcOpIdx2) const {
  assert(!MI.isBundle() &&
         "TargetInstrInfo::findCommutedOpIndices() can't handle bundles");

  const MCInstrDesc &MCID = MI.getDesc();
  if (!MCID.isCommutable())
    return false;

  switch (MI.getOpcode()) {
  case Mips::DPADD_U_H:
  case Mips::DPADD_U_W:
  case Mips::DPADD_U_D:
  case Mips::DPADD_S_H:
  case Mips::DPADD_S_W:
  case Mips::DPADD_S_D:
    // The first operand is both input and output, so it should not commute
    if (!fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 2, 3))
      return false;

    if (!MI.getOperand(SrcOpIdx1).isReg() || !MI.getOperand(SrcOpIdx2).isReg())
      return false;
    return true;
  }
  return TargetInstrInfo::findCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);
}

// ins, ext, dext*, dins have the following constraints:
// X <= pos      <  Y
// X <  size     <= Y
// X <  pos+size <= Y
//
// dinsm and dinsu have the following constraints:
// X <= pos      <  Y
// X <= size     <= Y
// X <  pos+size <= Y
//
// The callee of verifyInsExtInstruction however gives the bounds of
// dins[um] like the other (d)ins (d)ext(um) instructions, so that this
// function doesn't have to vary it's behaviour based on the instruction
// being checked.
static bool verifyInsExtInstruction(const MachineInstr &MI, StringRef &ErrInfo,
                                    const int64_t PosLow, const int64_t PosHigh,
                                    const int64_t SizeLow,
                                    const int64_t SizeHigh,
                                    const int64_t BothLow,
                                    const int64_t BothHigh) {
  MachineOperand MOPos = MI.getOperand(2);
  if (!MOPos.isImm()) {
    ErrInfo = "Position is not an immediate!";
    return false;
  }
  int64_t Pos = MOPos.getImm();
  if (!((PosLow <= Pos) && (Pos < PosHigh))) {
    ErrInfo = "Position operand is out of range!";
    return false;
  }

  MachineOperand MOSize = MI.getOperand(3);
  if (!MOSize.isImm()) {
    ErrInfo = "Size operand is not an immediate!";
    return false;
  }
  int64_t Size = MOSize.getImm();
  if (!((SizeLow < Size) && (Size <= SizeHigh))) {
    ErrInfo = "Size operand is out of range!";
    return false;
  }

  if (!((BothLow < (Pos + Size)) && ((Pos + Size) <= BothHigh))) {
    ErrInfo = "Position + Size is out of range!";
    return false;
  }

  return true;
}

//  Perform target specific instruction verification.
bool MipsInstrInfo::verifyInstruction(const MachineInstr &MI,
                                      StringRef &ErrInfo) const {
  // Verify that ins and ext instructions are well formed.
  switch (MI.getOpcode()) {
    case Mips::EXT:
    case Mips::EXT_MM:
    case Mips::INS:
    case Mips::INS_MM:
    case Mips::DINS:
      return verifyInsExtInstruction(MI, ErrInfo, 0, 32, 0, 32, 0, 32);
    case Mips::DINSM:
      // The ISA spec has a subtle difference between dinsm and dextm
      // in that it says:
      // 2 <= size <= 64 for 'dinsm' but 'dextm' has 32 < size <= 64.
      // To make the bounds checks similar, the range 1 < size <= 64 is checked
      // for 'dinsm'.
      return verifyInsExtInstruction(MI, ErrInfo, 0, 32, 1, 64, 32, 64);
    case Mips::DINSU:
      // The ISA spec has a subtle difference between dinsu and dextu in that
      // the size range of dinsu is specified as 1 <= size <= 32 whereas size
      // for dextu is 0 < size <= 32. The range checked for dinsu here is
      // 0 < size <= 32, which is equivalent and similar to dextu.
      return verifyInsExtInstruction(MI, ErrInfo, 32, 64, 0, 32, 32, 64);
    case Mips::DEXT:
      return verifyInsExtInstruction(MI, ErrInfo, 0, 32, 0, 32, 0, 63);
    case Mips::DEXTM:
      return verifyInsExtInstruction(MI, ErrInfo, 0, 32, 32, 64, 32, 64);
    case Mips::DEXTU:
      return verifyInsExtInstruction(MI, ErrInfo, 32, 64, 0, 32, 32, 64);
    case Mips::TAILCALLREG:
    case Mips::PseudoIndirectBranch:
    case Mips::JR:
    case Mips::JR64:
    case Mips::JALR:
    case Mips::JALR64:
    case Mips::JALRPseudo:
      if (!Subtarget.useIndirectJumpsHazard())
        return true;

      ErrInfo = "invalid instruction when using jump guards!";
      return false;
    default:
      return true;
  }

  return true;
}

static bool getMemopOffsetRange(unsigned Opcode, int &MinOffset, int &MaxOffset) {
  switch (Opcode) {
    default:
      return false;
    case Mips::LW_NM:
    case Mips::LH_NM:
    case Mips::LHU_NM:
    case Mips::LB_NM:
    case Mips::LBU_NM:
    case Mips::SW_NM:
    case Mips::SH_NM:
    case Mips::SB_NM:
      MinOffset = 0;
      MaxOffset = 4095;
      return true;
    case Mips::LWs9_NM:
    case Mips::LHs9_NM:
    case Mips::LHUs9_NM:
    case Mips::LBs9_NM:
    case Mips::LBUs9_NM:
    case Mips::SWs9_NM:
    case Mips::SHs9_NM:
    case Mips::SBs9_NM:
      MinOffset = -256;
      MaxOffset = 255;
      return true;
  }
}

outliner::OutlinedFunction MipsInstrInfo::getOutliningCandidateInfo(
    std::vector<outliner::Candidate> &RepeatedSequenceLocs) const {

  outliner::Candidate &FirstCand = RepeatedSequenceLocs[0];
  unsigned SequenceSize =
      std::accumulate(FirstCand.front(), std::next(FirstCand.back()), 0,
                      [this](unsigned Sum, const MachineInstr &MI) {
                        return Sum + getInstSizeInBytes(MI);
                      });

  const TargetRegisterInfo &TRI = getRegisterInfo();

  unsigned NumBytesToCreateFrame = 0;

  // We have to check if sp modifying instructions would get outlined.
  auto HasIllegalSPModification = [&TRI](outliner::Candidate &C) {
    MachineBasicBlock::iterator MBBI = C.front();
    for (;;) {
      if (MBBI->modifiesRegister(Mips::SP_NM, &TRI))
        return true;
      if (MBBI == C.back())
        break;
      ++MBBI;
    }
    return false;
  };
  // Remove candidates with illegal stack modifying instructions
  llvm::erase_if(RepeatedSequenceLocs, HasIllegalSPModification);

  // If the sequence doesn't have enough candidates left, then we're done.
  if (RepeatedSequenceLocs.size() < 2)
    return outliner::OutlinedFunction();

  // Properties about candidate MBBs that hold for all of them
  unsigned FlagsSetInAll = 0xF;
  // Compute liveness information for each candidate, and set FlagsSetInAll.
  std::for_each(
      RepeatedSequenceLocs.begin(), RepeatedSequenceLocs.end(),
      [&FlagsSetInAll](outliner::Candidate &C) { FlagsSetInAll &= C.Flags; });

  unsigned LastInstrOpcode = RepeatedSequenceLocs[0].back()->getOpcode();

  auto SetCandidateCallInfo =
      [&RepeatedSequenceLocs](unsigned CallID, unsigned NumBytesForCall) {
        for (outliner::Candidate &C : RepeatedSequenceLocs)
          C.setCallInfo(CallID, NumBytesForCall);
      };

  unsigned FrameID = MachineOutlinerDefault;
  NumBytesToCreateFrame += 4;
  unsigned CFICount = 0;

  // We check to see if CFI Instructions are present, and if they are
  // we find the number of CFI Instructions in the candidates.
  MachineBasicBlock::iterator MBBI = RepeatedSequenceLocs[0].front();
  for (unsigned Loc = RepeatedSequenceLocs[0].getStartIdx();
       Loc < RepeatedSequenceLocs[0].getEndIdx() + 1; Loc++) {
    const std::vector<MCCFIInstruction>
        &CFIInstructions = // Machine Code Control Flow Integrity Instruction
        RepeatedSequenceLocs[0].getMF()->getFrameInstructions();
    if (MBBI->isCFIInstruction()) {
      unsigned CFIIndex = MBBI->getOperand(0).getCFIIndex();
      MCCFIInstruction CFI = CFIInstructions[CFIIndex];
      CFICount++;
    }
    MBBI++;
  }

  for (outliner::Candidate &C : RepeatedSequenceLocs) {
    std::vector<MCCFIInstruction> CFIInstructions =
        C.getMF()->getFrameInstructions();

    if (CFICount > 0 && CFICount != CFIInstructions.size())
      return outliner::OutlinedFunction();
  }

  auto IsSafeToFixup = [this, &TRI](MachineInstr &MI) {
    if (MI.isCall())
      return true;

    if (!MI.modifiesRegister(Mips::SP_NM, &TRI) &&
        !MI.readsRegister(Mips::SP_NM, &TRI))
      return true;

    // Any modification of SP will break our code to save/restore RA.
    if (MI.modifiesRegister(Mips::SP_NM, &TRI))
      return false;

    // At this point, we have a stack instruction that we might need to
    // fix up. We'll handle it if it's a load or store.
    if (MI.mayLoadOrStore()) {
      const MachineOperand *Base; // Filled with the base operand of MI.
      int64_t Offset;             // Filled with the offset of MI.
      bool OffsetIsScalable;

      // Does it allow us to offset the base operand and is the base the
      // register SP?
      if (!getMemOperandWithOffset(MI, Base, Offset, OffsetIsScalable, &TRI) ||
          !Base->isReg() || Base->getReg() != Mips::SP_NM) {
        return false;
      }
      // Fixe-up code below assumes bytes.
      if (OffsetIsScalable)
        return false;

      int MinOffset, MaxOffset;
      if (!getMemopOffsetRange(MI.getOpcode(), MinOffset, MaxOffset))
        return false;

      int64_t NewOffset = MI.getOperand(2).getImm() + 16;
      if (NewOffset < MinOffset || NewOffset > MaxOffset)
        return false;

      // It's in range, so we can outline it.
      return true;
    }
    return false;
  };

  bool AllStackInstrsSafe = std::all_of(
      FirstCand.front(), std::next(FirstCand.back()), IsSafeToFixup);

  for (outliner::Candidate &C : RepeatedSequenceLocs) {
    MachineFunction *MF = C.getMF();
    for (MachineBasicBlock &MBB : *MF)
      recomputeLivenessFlags(MBB);
  }

  // If the last instruction in any candidate is a terminator, then we should
  // tail call all of the candidates.
  if (RepeatedSequenceLocs[0].back()->isTerminator()) {
    FrameID = MachineOutlinerTailCall;
    NumBytesToCreateFrame = 0;
    SetCandidateCallInfo(MachineOutlinerTailCall, 4);
  }

  else if (LastInstrOpcode == Mips::BALC_NM ||
           LastInstrOpcode == Mips::JALRC_NM) {
    FrameID = MachineOutlinerThunk;
    NumBytesToCreateFrame = 0;
    SetCandidateCallInfo(MachineOutlinerThunk, 4);
  }

  else {

    unsigned NumBytesNoStackCalls = 0;
    std::vector<outliner::Candidate> CandidatesWithoutStackFixups;

    for (outliner::Candidate &C : RepeatedSequenceLocs) {

      C.initLRU(TRI);

      // If we have a noreturn caller, then we're going to be conservative and
      // say that we have to save RA. If we don't have a ret at the end of the
      // block, then we can't reason about liveness accurately.
      //
      // FIXME: We can probably do better than always disabling this in
      // noreturn functions by fixing up the liveness info.
      bool IsNoReturn =
          C.getMF()->getFunction().hasFnAttribute(Attribute::NoReturn);

      // Is RA available? If so, we don't need a save.
      if (C.LRU.available(Mips::RA_NM) && !IsNoReturn) {
        NumBytesNoStackCalls += 4;
        C.setCallInfo(MachineOutlinerNoRASave, 4);
        CandidatesWithoutStackFixups.push_back(C);
      }

      // Is an unused register available? If so, we won't modify the stack, so
      // we can outline with the same frame type as those that don't save RA.
      else if (findRegisterToSaveRA(C)) {
        NumBytesNoStackCalls += 12;
        C.setCallInfo(MachineOutlinerRegSave, 12);
        CandidatesWithoutStackFixups.push_back(C);
      }

      // Is SP used in the sequence at all? If not, we don't have to modify
      // the stack, so we are guaranteed to get the same frame.
      else if (C.UsedInSequence.available(Mips::SP_NM)) {
        NumBytesNoStackCalls += 12;
        C.setCallInfo(MachineOutlinerDefault, 12);
        CandidatesWithoutStackFixups.push_back(C);
      }

      // If we outline this, we need to modify the stack. Pretend we don't
      // outline this by saving all of its bytes.
      else {
        NumBytesNoStackCalls += SequenceSize;
      }
    }

    // If there are no places where we have to save RA, then note that we
    // don't have to update the stack. Otherwise, give every candidate the
    // default call type, as long as it's safe to do so.
    if (!AllStackInstrsSafe ||
        NumBytesNoStackCalls <= RepeatedSequenceLocs.size() * 12) {
      RepeatedSequenceLocs = CandidatesWithoutStackFixups;
      FrameID = MachineOutlinerNoRASave;
    } else {
      SetCandidateCallInfo(MachineOutlinerDefault, 12);

      if (FlagsSetInAll & MachineOutlinerMBBFlags::HasCalls) {
        erase_if(RepeatedSequenceLocs, [this](outliner::Candidate &C) {
          return (std::any_of(
                     C.front(), std::next(C.back()),
                     [](const MachineInstr &MI) { return MI.isCall(); })) &&
                 (!C.LRU.available(Mips::RA_NM) || !findRegisterToSaveRA(C));
        });
      }
    }
    // If we dropped all of the candidates, bail out here.
    if (RepeatedSequenceLocs.size() < 2) {
      RepeatedSequenceLocs.clear();
      return outliner::OutlinedFunction();
    }
  }

  // Does every candidate's MBB contain a call? If so, then we might have a
  // call in the range.
  if (FlagsSetInAll & MachineOutlinerMBBFlags::HasCalls) {
    // Check if the range contains a call. These require a save + restore of
    // the RA register.
    bool ModStackToSaveRA = false;
    if (std::any_of(FirstCand.front(), FirstCand.back(),
                    [](const MachineInstr &MI) { return MI.isCall(); }))
      ModStackToSaveRA = true;

    // Handle the last instruction separately. If this is a tail call, then the
    // last instruction is a call. We don't want to save + restore in this
    // case. However, it could be possible that the last instruction is a call
    // without it being valid to tail call this sequence. We should consider
    // this as well.
    else if (FrameID != MachineOutlinerThunk &&
             FrameID != MachineOutlinerTailCall && FirstCand.back()->isCall())
      ModStackToSaveRA = true;

    if (ModStackToSaveRA) {
      // We can't fix up the stack. Bail out.
      if (!AllStackInstrsSafe) {
        RepeatedSequenceLocs.clear();
        return outliner::OutlinedFunction();
      }

      // Save + restore RA.
      NumBytesToCreateFrame += 8;
    }
  }

  auto InstrCount = RepeatedSequenceLocs[0].getLength();
  if (FrameID == MachineOutlinerDefault || FrameID == MachineOutlinerRegSave) {
    if (InstrCount < 4) {
      RepeatedSequenceLocs.clear();
      return outliner::OutlinedFunction();
    }
  } else {
    if (InstrCount < 3) {
      RepeatedSequenceLocs.clear();
      return outliner::OutlinedFunction();
    }
  }

  // If we have CFI instructions, we can only outline if the outlined section
  // can be a tail call
  if (FrameID != MachineOutlinerTailCall && CFICount > 0) {
    RepeatedSequenceLocs.clear();
    return outliner::OutlinedFunction();
  }

  return outliner::OutlinedFunction(RepeatedSequenceLocs, SequenceSize,
                                    NumBytesToCreateFrame, FrameID);
}

bool MipsInstrInfo::isFunctionSafeToOutlineFrom(
    MachineFunction &MF, bool OutlineFromLinksOnceODRs) const {

  const Function &F = MF.getFunction();

  if (!OutlineFromLinksOnceODRs && F.hasLinkOnceODRLinkage())
    return false;

  if (F.hasSection())
    return false;

  return true;
}

void llvm::MipsInstrInfo::buildOutlinedFrame(
    MachineBasicBlock &MBB, MachineFunction &MF,
    const outliner::OutlinedFunction &OF) const {

  // Is there a call in the outlined range?
  auto IsNonTailCall = [](const MachineInstr &MI) {
    return MI.isCall() && !MI.isReturn();
  };

  if (llvm::any_of(MBB.instrs(), IsNonTailCall)) {

    assert(OF.FrameConstructionID != MachineOutlinerDefault &&
           "Can only fix up stack references once");

    fixupPostOutline(MBB);

    if (!MBB.isLiveIn(Mips::RA_NM)) {
      MBB.addLiveIn(Mips::RA_NM);
    }

    MachineBasicBlock::iterator It = MBB.begin();
    MachineBasicBlock::iterator Et = MBB.end();

    if (OF.FrameConstructionID == MachineOutlinerTailCall ||
        OF.FrameConstructionID == MachineOutlinerThunk) {
      Et = std::prev(MBB.end());
    }

    MachineInstr *SaveRAtoStack = BuildMI(MF, DebugLoc(), get(Mips::SAVE_NM))
                                      .addImm(16)
                                      .addReg(Mips::RA_NM);

    It = MBB.insert(It, SaveRAtoStack);

    const TargetSubtargetInfo &STI = MF.getSubtarget();
    const MCRegisterInfo *MRI = STI.getRegisterInfo();
    unsigned DwarfReg = MRI->getDwarfRegNum(Mips::RA_NM, true);

    // Add a CFI saying the stack was moved 8 Bytes down.
    int32_t StackPosEntry =
        MF.addFrameInst(MCCFIInstruction::cfiDefCfaOffset(nullptr, 8));
    BuildMI(MBB, It, DebugLoc(), get(Mips::CFI_INSTRUCTION))
        .addCFIIndex(StackPosEntry)
        .setMIFlags(MachineInstr::FrameSetup);

    // Add a CFI saying that the RA that we want to find is now 8 Bytes higher
    // than before.
    int32_t RAPosEntry =
        MF.addFrameInst(MCCFIInstruction::createOffset(nullptr, DwarfReg, -8));
    BuildMI(MBB, It, DebugLoc(), get(Mips::CFI_INSTRUCTION))
        .addCFIIndex(RAPosEntry)
        .setMIFlags(MachineInstr::FrameSetup);

    // Insert a restore before the terminator for the function.
    MachineInstr *LoadRAfromStack =
        BuildMI(MF, DebugLoc(), get(Mips::RESTORE_NM))
            .addImm(16)
            .addReg(Mips::RA_NM, RegState::Define);

    Et = MBB.insert(Et, LoadRAfromStack);
  }

  if (OF.FrameConstructionID == MachineOutlinerTailCall ||
      OF.FrameConstructionID == MachineOutlinerThunk) {
    return;
  }

  if (!MBB.isLiveIn(Mips::RA_NM)) {
    MBB.addLiveIn(Mips::RA_NM);
  }

  MachineInstr *ret =
      BuildMI(MF, DebugLoc(), get(Mips::JRC_NM)).addReg(Mips::RA_NM);
  MBB.insert(MBB.end(), ret);

  // Did we have to modify the stack by saving the Return Address?
  // We do this only if OF.FrameConstructionID == MachineOutlinerDefault
  // since only in this case fixupPostOutline is required.
  if (OF.FrameConstructionID != MachineOutlinerDefault) {
    return;
  }
  // We modified the stack.
  // Walk over the basic block and fix up all the stack accesses.
  fixupPostOutline(MBB);
}

MachineBasicBlock::iterator MipsInstrInfo::insertOutlinedCall(
    Module &M, MachineBasicBlock &MBB, MachineBasicBlock::iterator &It,
    MachineFunction &MF, const outliner::Candidate &C) const {

  /*MachineOutlinerTailCall implies that the function is being created from
    a sequence of instructions ending in a return. (ending in a JR to RA)*/
  if (C.CallConstructionID == MachineOutlinerTailCall) {
    It = MBB.insert(It, BuildMI(MF, DebugLoc(), get(Mips::TAILCALL_NM))
                            .addGlobalAddress(M.getNamedValue(MF.getName())));
    return It;
  }

  // Are we saving the RA (return address)?
  if (C.CallConstructionID == MachineOutlinerNoRASave ||
      C.CallConstructionID == MachineOutlinerThunk) {
    // No, so just insert the call.

    It = MBB.insert(It, BuildMI(MF, DebugLoc(), get(Mips::BALC_NM))
                            .addGlobalAddress(M.getNamedValue(MF.getName())));
    return It;
  }

  // We want to return the spot where we inserted the call.
  MachineBasicBlock::iterator CallPtr;
  MachineInstr *SaveRA;
  MachineInstr *RestoreRA;

  /* MachineOutlinerRegSave implies that the function should be called with a
   save and restore of RA to an available register. This allows us to avoid
   stack fixups. Note that this outlining variant is compatible with the
   NoRASave case. */
  if (C.CallConstructionID == MachineOutlinerRegSave) {

    unsigned Reg = findRegisterToSaveRA(C);
    assert(Reg != 0 && "No callee-saved register available?");
    // save RA + restore RA from Reg  (available register)

    SaveRA = BuildMI(MF, DebugLoc(), get(Mips::MOVE_NM))
                 .addReg(Reg, RegState::Define)
                 .addReg(Mips::RA_NM);

    RestoreRA = BuildMI(MF, DebugLoc(), get(Mips::MOVE_NM))
                    .addReg(Mips::RA_NM, RegState::Define)
                    .addReg(Reg);

  }

  // Default case. Save and Restore from stack pointer :
  else {

    SaveRA = BuildMI(MF, DebugLoc(), get(Mips::SAVE_NM))
                 .addImm(16)
                 .addReg(Mips::RA_NM);

    RestoreRA = BuildMI(MF, DebugLoc(), get(Mips::RESTORE_NM))
                    .addImm(16)
                    .addReg(Mips::RA_NM, RegState::Define);
  }
  It = MBB.insert(It, SaveRA);
  It++;

  It = MBB.insert(It, BuildMI(MF, DebugLoc(), get(Mips::BALC_NM))
                          .addGlobalAddress(M.getNamedValue(MF.getName())));

  CallPtr = It;
  It++;
  It = MBB.insert(It, RestoreRA);

  return CallPtr;
}

unsigned
MipsInstrInfo::findRegisterToSaveRA(const outliner::Candidate &C) const {

  MachineFunction *MF = C.getMF();

  const MipsRegisterInfo *MRI = static_cast<const MipsRegisterInfo *>(
      MF->getSubtarget().getRegisterInfo());

  for (unsigned Reg : Mips::GPR32NMRegClass) {

    if (!MRI->isReservedReg(*MF, Reg) && Reg != Mips::RA_NM &&
        C.LRU.available(Reg) && C.UsedInSequence.available(Reg) &&
        (Reg == Mips::S0_NM || Reg == Mips::S1_NM || Reg == Mips::S2_NM ||
         Reg == Mips::S3_NM || Reg == Mips::S4_NM || Reg == Mips::S5_NM ||
         Reg == Mips::S6_NM || Reg == Mips::S7_NM))

      return Reg;
  }
  return 0;
}

bool MipsInstrInfo::getMemOperandWithOffsetWidth(
    const MachineInstr &LdSt, const MachineOperand *&BaseOp, int64_t &Offset,
    bool &OffsetIsScalable, unsigned &Width,
    const TargetRegisterInfo *TRI) const {
  assert(LdSt.mayLoadOrStore() && "Expected a memory operation.");
  // Handle only loads/stores with base register followed by immediate offset.
  if (LdSt.getNumExplicitOperands() == 3) {
    // Non-paired instruction (e.g., ldr x1, [x0, #8]).
    if ((!LdSt.getOperand(1).isReg() && !LdSt.getOperand(1).isFI()) ||
        !LdSt.getOperand(2).isImm())
      return false;
  } else if (LdSt.getNumExplicitOperands() == 4) {
    // Paired instruction (e.g., ldp x1, x2, [x0, #8]).
    if (!LdSt.getOperand(1).isReg() ||
        (!LdSt.getOperand(2).isReg() && !LdSt.getOperand(2).isFI()) ||
        !LdSt.getOperand(3).isImm())
      return false;
  } else
    return false;

  // Get the scaling factor for the instruction and set the width for the
  // instruction.
  TypeSize Scale(0U, false);

  // Compute the offset. Offset is calculated as the immediate operand
  // multiplied by the scaling factor. Unscaled instructions have scaling factor
  // set to 1.
  if (LdSt.getNumExplicitOperands() == 3) {
    BaseOp = &LdSt.getOperand(1);
    Offset = LdSt.getOperand(2).getImm() * Scale.getKnownMinSize();
  } else {
    assert(LdSt.getNumExplicitOperands() == 4 && "invalid number of operands");
    BaseOp = &LdSt.getOperand(2);
    Offset = LdSt.getOperand(3).getImm() * Scale.getKnownMinSize();
  }
  OffsetIsScalable = Scale.isScalable();

  if (!BaseOp->isReg() && !BaseOp->isFI())
    return false;

  return true;
}

outliner::InstrType
MipsInstrInfo::getOutliningType(MachineBasicBlock::iterator &MIT,
                                unsigned Flags) const {

  MachineInstr &MI = *MIT;

  if (MI.isCFIInstruction()) {
    return outliner::InstrType::Legal;
  }

  if (MI.isDebugInstr() || MI.isIndirectDebugValue()) {
    return outliner::InstrType::Invisible;
  }

  if (MI.isKill()) {
    return outliner::InstrType::Invisible;
  }

  if (MI.isTerminator()) {

    if (MI.getParent()->succ_empty()) {
      return outliner::InstrType::Legal;
    }

    return outliner::InstrType::Illegal;
  }

  // Make sure none of the operands are un-outlinable.
  for (const MachineOperand &MOP : MI.operands()) {
    if (MOP.isCPI() || MOP.isJTI() || MOP.isCFIIndex() || MOP.isFI() ||
        MOP.isTargetIndex())
      return outliner::InstrType::Illegal;

    // If it uses LR or W30 explicitly, then don't touch it.
    if (MOP.isReg() && !MOP.isImplicit() && (MOP.getReg() == Mips::RA_NM))
      return outliner::InstrType::Illegal;
  }

  // Don't outline positions.
  if (MI.isPosition()) {
    return outliner::InstrType::Illegal;
  }

  if (MI.isCall()) {
    // Get the function associated with the call. Look at each operand and find
    // the one that represents the callee and get its name.
    const Function *Callee = nullptr;
    for (const MachineOperand &MOP : MI.operands()) {
      if (MOP.isGlobal()) {
        Callee = dyn_cast<Function>(MOP.getGlobal());
        break;
      }
    }

    // Never outline calls to mcount.  There isn't any rule that would require
    // this, but the Linux kernel's "ftrace" feature depends on it.
    if (Callee && Callee->getName() == "\01_mcount")
      return outliner::InstrType::Illegal;

    // If we don't know anything about the callee, assume it depends on the
    // stack layout of the caller. In that case, it's only legal to outline
    // as a tail-call. Explicitly list the call instructions we know about so we
    // don't get unexpected results with call pseudo-instructions.

    if (MI.getOpcode() == Mips::JALRC_NM || MI.getOpcode() == Mips::BALC_NM ||
        MI.getOpcode() == Mips::BC_NM)

      return outliner::InstrType::Legal;
  }

  return outliner::InstrType::Legal;
}

MachineOperand &
MipsInstrInfo::getMemOpBaseRegImmOfsOffsetOperand(MachineInstr &LdSt) const {
  assert(LdSt.mayLoadOrStore() && "Expected a memory operation.");
  MachineOperand &OfsOp = LdSt.getOperand(LdSt.getNumExplicitOperands() - 1);
  assert(OfsOp.isImm() && "Offset operand wasn't immediate.");
  return OfsOp;
}

void MipsInstrInfo::fixupPostOutline(MachineBasicBlock &MBB) const {

  const TargetRegisterInfo &TRI = getRegisterInfo();

  for (MachineInstr &MI : MBB) {

    const MachineOperand *Base;
    unsigned Width;
    int64_t Offset;
    bool OffsetIsScalable;

    if (!MI.mayLoadOrStore() ||
        !getMemOperandWithOffsetWidth(MI, Base, Offset, OffsetIsScalable, Width,
                                      &TRI) ||
        (Base->isReg() && Base->getReg() != Mips::SP_NM)) {
      continue;
    }
    TypeSize Scale(0U, false);

    MachineOperand &StackOffsetOperand = getMemOpBaseRegImmOfsOffsetOperand(MI);
    assert(StackOffsetOperand.isImm() && "Stack offset wasn't immediate!");
    assert(Scale != 0 && "Unexpected opcode!");
    assert(!OffsetIsScalable && "Expected offset to be a byte offset");

    // We've pushed the return address to the stack, so add 16 to the offset.
    // This is safe, since we already checked if it would overflow when we
    // checked if this instruction was legal to outline.
    int64_t NewImm = (Offset + 16) / (int64_t)Scale.getFixedSize();
    StackOffsetOperand.setImm(NewImm);
  }
}

std::pair<unsigned, unsigned>
MipsInstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  return std::make_pair(TF, 0u);
}

ArrayRef<std::pair<unsigned, const char*>>
MipsInstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
 using namespace MipsII;

 static const std::pair<unsigned, const char*> Flags[] = {
    {MO_GOT,          "mips-got"},
    {MO_GOT_CALL,     "mips-got-call"},
    {MO_GPREL,        "mips-gprel"},
    {MO_ABS_HI,       "mips-abs-hi"},
    {MO_ABS_LO,       "mips-abs-lo"},
    {MO_TLSGD,        "mips-tlsgd"},
    {MO_TLSLDM,       "mips-tlsldm"},
    {MO_DTPREL_HI,    "mips-dtprel-hi"},
    {MO_DTPREL_LO,    "mips-dtprel-lo"},
    {MO_GOTTPREL,     "mips-gottprel"},
    {MO_TPREL_HI,     "mips-tprel-hi"},
    {MO_TPREL_LO,     "mips-tprel-lo"},
    {MO_GPOFF_HI,     "mips-gpoff-hi"},
    {MO_GPOFF_LO,     "mips-gpoff-lo"},
    {MO_GOT_DISP,     "mips-got-disp"},
    {MO_GOT_PAGE,     "mips-got-page"},
    {MO_GOT_OFST,     "mips-got-ofst"},
    {MO_HIGHER,       "mips-higher"},
    {MO_HIGHEST,      "mips-highest"},
    {MO_GOT_HI16,     "mips-got-hi16"},
    {MO_GOT_LO16,     "mips-got-lo16"},
    {MO_CALL_HI16,    "mips-call-hi16"},
    {MO_CALL_LO16,    "mips-call-lo16"},
    {MO_JALR,         "mips-jalr"}
  };
  return makeArrayRef(Flags);
}

Optional<ParamLoadedValue>
MipsInstrInfo::describeLoadedValue(const MachineInstr &MI, Register Reg) const {
  DIExpression *Expr =
      DIExpression::get(MI.getMF()->getFunction().getContext(), {});

  // TODO: Special MIPS instructions that need to be described separately.
  if (auto RegImm = isAddImmediate(MI, Reg)) {
    Register SrcReg = RegImm->Reg;
    int64_t Offset = RegImm->Imm;
    // When SrcReg is $zero, treat loaded value as immediate only.
    // Ex. $a2 = ADDiu $zero, 10
    if (SrcReg == Mips::ZERO || SrcReg == Mips::ZERO_64) {
      return ParamLoadedValue(MI.getOperand(2), Expr);
    }
    Expr = DIExpression::prepend(Expr, DIExpression::ApplyOffset, Offset);
    return ParamLoadedValue(MachineOperand::CreateReg(SrcReg, false), Expr);
  } else if (auto DestSrc = isCopyInstr(MI)) {
    const MachineFunction *MF = MI.getMF();
    const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
    Register DestReg = DestSrc->Destination->getReg();
    // TODO: Handle cases where the Reg is sub- or super-register of the
    // DestReg.
    if (TRI->isSuperRegister(Reg, DestReg) || TRI->isSubRegister(Reg, DestReg))
      return None;
  }

  return TargetInstrInfo::describeLoadedValue(MI, Reg);
}

Optional<RegImmPair> MipsInstrInfo::isAddImmediate(const MachineInstr &MI,
                                                   Register Reg) const {
  // TODO: Handle cases where Reg is a super- or sub-register of the
  // destination register.
  const MachineOperand &Op0 = MI.getOperand(0);
  if (!Op0.isReg() || Reg != Op0.getReg())
    return None;

  switch (MI.getOpcode()) {
  case Mips::ADDiu:
  case Mips::DADDiu: {
    const MachineOperand &Dop = MI.getOperand(0);
    const MachineOperand &Sop1 = MI.getOperand(1);
    const MachineOperand &Sop2 = MI.getOperand(2);
    // Value is sum of register and immediate. Immediate value could be
    // global string address which is not supported.
    if (Dop.isReg() && Sop1.isReg() && Sop2.isImm())
      return RegImmPair{Sop1.getReg(), Sop2.getImm()};
    // TODO: Handle case where Sop1 is a frame-index.
  }
  }
  return None;
}
