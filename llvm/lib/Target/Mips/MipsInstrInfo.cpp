//===-- MipsInstrInfo.cpp - Mips Instruction Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MipsInstrInfo.h"
#include "InstPrinter/MipsInstPrinter.h"
#include "MipsMachineFunction.h"
#include "MipsSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "MipsGenInstrInfo.inc"

// Pin the vtable to this file.
void MipsInstrInfo::anchor() {}

MipsInstrInfo::MipsInstrInfo(const MipsSubtarget &STI, unsigned UncondBr)
    : MipsGenInstrInfo(Mips::ADJCALLSTACKDOWN, Mips::ADJCALLSTACKUP),
      Subtarget(STI), UncondBrOpc(UncondBr) {}

const MipsInstrInfo *MipsInstrInfo::create(MipsSubtarget &STI) {
  if (STI.inMips16Mode())
    return llvm::createMips16InstrInfo(STI);

  return llvm::createMipsSEInstrInfo(STI);
}

bool MipsInstrInfo::isZeroImm(const MachineOperand &op) const {
  return op.isImm() && op.getImm() == 0;
}

/// insertNoop - If data hazard condition is found insert the target nop
/// instruction.
void MipsInstrInfo::
insertNoop(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI) const
{
  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(Mips::NOP));
}

MachineMemOperand *MipsInstrInfo::GetMemOperand(MachineBasicBlock &MBB, int FI,
                                                unsigned Flag) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  return MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                                 Flag, MFI.getObjectSize(FI), Align);
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

  for (int i=0; i<NumOp-1; i++)
    Cond.push_back(Inst->getOperand(i));
}

bool MipsInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond,
                                  bool AllowModify) const {
  SmallVector<MachineInstr*, 2> BranchInstrs;
  BranchType BT = AnalyzeBranch(MBB, TBB, FBB, Cond, AllowModify, BranchInstrs);

  return (BT == BT_None) || (BT == BT_Indirect);
}

void
MipsInstrInfo::BuildCondBr(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           DebugLoc DL, ArrayRef<MachineOperand> Cond) const {
  unsigned Opc = Cond[0].getImm();
  const MCInstrDesc &MCID = get(Opc);
  MachineInstrBuilder MIB = BuildMI(&MBB, DL, MCID);

  for (unsigned i = 1; i < Cond.size(); ++i) {
    if (Cond[i].isReg())
      MIB.addReg(Cond[i].getReg());
    else if (Cond[i].isImm())
      MIB.addImm(Cond[i].getImm());
    else
       assert(true && "Cannot copy operand");
  }
  MIB.addMBB(TBB);
}

unsigned MipsInstrInfo::InsertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

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

unsigned MipsInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();
  MachineBasicBlock::reverse_iterator FirstBr;
  unsigned removed;

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugValue())
    ++I;

  FirstBr = I;

  // Up to 2 branches are removed.
  // Note that indirect branches are not removed.
  for (removed = 0; I != REnd && removed < 2; ++I, ++removed)
    if (!getAnalyzableBrOpc(I->getOpcode()))
      break;

  MBB.erase(I.base(), FirstBr.base());

  return removed;
}

/// ReverseBranchCondition - Return the inverse opcode of the
/// specified Branch instruction.
bool MipsInstrInfo::ReverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert( (Cond.size() && Cond.size() <= 3) &&
          "Invalid Mips branch condition!");
  Cond[0].setImm(getOppositeBranchOpc(Cond[0].getImm()));
  return false;
}

MipsInstrInfo::BranchType MipsInstrInfo::AnalyzeBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *&TBB, MachineBasicBlock *&FBB,
    SmallVectorImpl<MachineOperand> &Cond, bool AllowModify,
    SmallVectorImpl<MachineInstr *> &BranchInstrs) const {

  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugValue())
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

  if (++I != REnd) {
    SecondLastInst = &*I;
    SecondLastOpc = getAnalyzableBrOpc(SecondLastInst->getOpcode());

    // Not an analyzable branch (must be an indirect jump).
    if (isUnpredicatedTerminator(*SecondLastInst) && !SecondLastOpc)
      return BT_None;
  }

  // If there is only one terminator instruction, process it.
  if (!SecondLastOpc) {
    // Unconditional branch.
    if (LastOpc == UncondBrOpc) {
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
  if (SecondLastOpc == UncondBrOpc) {
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
  if (LastOpc != UncondBrOpc)
    return BT_None;

  AnalyzeCondBr(SecondLastInst, SecondLastOpc, TBB, Cond);
  FBB = LastInst->getOperand(0).getMBB();

  return BT_CondUncond;
}

/// Return the corresponding compact (no delay slot) form of a branch.
unsigned MipsInstrInfo::getEquivalentCompactForm(
    const MachineBasicBlock::iterator I) const {
  unsigned Opcode = I->getOpcode();
  bool canUseShortMMBranches =
      Subtarget.inMicroMipsMode() &&
      (Opcode == Mips::BNE || Opcode == Mips::BEQ) &&
      I->getOperand(1).getReg() == Subtarget.getABI().GetZeroReg();

  if (Subtarget.hasMips32r6() || canUseShortMMBranches) {
    switch (Opcode) {
    case Mips::B:
      return Mips::BC;
    case Mips::BAL:
      return Mips::BALC;
    case Mips::BEQ:
      if (canUseShortMMBranches)
        return Mips::BEQZC_MM;
      else
        return Mips::BEQC;
    case Mips::BNE:
      if (canUseShortMMBranches)
        return Mips::BNEZC_MM;
      else
        return Mips::BNEC;
    case Mips::BGE:
      return Mips::BGEC;
    case Mips::BGEU:
      return Mips::BGEUC;
    case Mips::BGEZ:
      return Mips::BGEZC;
    case Mips::BGTZ:
      return Mips::BGTZC;
    case Mips::BLEZ:
      return Mips::BLEZC;
    case Mips::BLT:
      return Mips::BLTC;
    case Mips::BLTU:
      return Mips::BLTUC;
    case Mips::BLTZ:
      return Mips::BLTZC;
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
unsigned MipsInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    return MI->getDesc().getSize();
  case  TargetOpcode::INLINEASM: {       // Inline Asm: Variable size.
    const MachineFunction *MF = MI->getParent()->getParent();
    const char *AsmStr = MI->getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  }
  case Mips::CONSTPOOL_ENTRY:
    // If this machine instr is a constant pool entry, its size is recorded as
    // operand #2.
    return MI->getOperand(2).getImm();
  }
}

MachineInstrBuilder
MipsInstrInfo::genInstrWithNewOpc(unsigned NewOpc,
                                  MachineBasicBlock::iterator I) const {
  MachineInstrBuilder MIB;
  bool BranchWithZeroOperand = false;

  // Certain branches have two forms: e.g beq $1, $zero, dst vs beqz $1, dest
  // Pick the zero form of the branch for readable assembly and for greater
  // branch distance in non-microMIPS mode.
  if (I->isBranch() && I->getOperand(1).isReg() &&
      // FIXME: Certain atomic sequences on mips64 generate 32bit references to
      // Mips::ZERO, which is incorrect. This test should be updated to use
      // Subtarget.getABI().GetZeroReg() when those atomic sequences and others
      // are fixed.
      (I->getOperand(1).getReg() == Mips::ZERO ||
       I->getOperand(1).getReg() == Mips::ZERO_64)) {
    BranchWithZeroOperand = true;
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
    case Mips::BNEZC_MM:
    case Mips::BEQZC_MM:
      break;
    default:
      BranchWithZeroOperand = false;
      break;
    }
  }

  MIB = BuildMI(*I->getParent(), I, I->getDebugLoc(), get(NewOpc));

  for (unsigned J = 0, E = I->getDesc().getNumOperands(); J < E; ++J)
    if (!(BranchWithZeroOperand && (J == 1)))
      MIB.addOperand(I->getOperand(J));

  MIB.setMemRefs(I->memoperands_begin(), I->memoperands_end());
  return MIB;
}
