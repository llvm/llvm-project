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

#include "MipsAnalyzeImmediate.h"
#include "MipsInstrInfo.h"
#include "MipsTargetMachine.h"
#include "MipsMachineFunction.h"
#include "InstPrinter/MipsInstPrinter.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/ADT/STLExtras.h"

#define GET_INSTRINFO_CTOR
#include "MipsGenInstrInfo.inc"

using namespace llvm;

MipsInstrInfo::MipsInstrInfo(MipsTargetMachine &tm, unsigned UncondBr)
  : MipsGenInstrInfo(Mips::ADJCALLSTACKDOWN, Mips::ADJCALLSTACKUP),
    TM(tm), UncondBrOpc(UncondBr) {}

const MipsInstrInfo *MipsInstrInfo::create(MipsTargetMachine &TM) {
  if (TM.getSubtargetImpl()->inMips16Mode())
    return llvm::createMips16InstrInfo(TM);

  return llvm::createMipsSEInstrInfo(TM);
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

  return MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FI), Flag,
                                 MFI.getObjectSize(FI), Align);
}

MachineInstr*
MipsInstrInfo::emitFrameIndexDebugValue(MachineFunction &MF, int FrameIx,
                                        uint64_t Offset, const MDNode *MDPtr,
                                        DebugLoc DL) const {
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Mips::DBG_VALUE))
    .addFrameIndex(FrameIx).addImm(0).addImm(Offset).addMetadata(MDPtr);
  return &*MIB;
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//

void MipsInstrInfo::AnalyzeCondBr(const MachineInstr *Inst, unsigned Opc,
                                  MachineBasicBlock *&BB,
                                  SmallVectorImpl<MachineOperand> &Cond) const {
  assert(GetAnalyzableBrOpc(Opc) && "Not an analyzable branch");
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
                                  bool AllowModify) const
{

  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugValue())
    ++I;

  if (I == REnd || !isUnpredicatedTerminator(&*I)) {
    // If this block ends with no branches (it just falls through to its succ)
    // just return false, leaving TBB/FBB null.
    TBB = FBB = NULL;
    return false;
  }

  MachineInstr *LastInst = &*I;
  unsigned LastOpc = LastInst->getOpcode();

  // Not an analyzable branch (must be an indirect jump).
  if (!GetAnalyzableBrOpc(LastOpc))
    return true;

  // Get the second to last instruction in the block.
  unsigned SecondLastOpc = 0;
  MachineInstr *SecondLastInst = NULL;

  if (++I != REnd) {
    SecondLastInst = &*I;
    SecondLastOpc = GetAnalyzableBrOpc(SecondLastInst->getOpcode());

    // Not an analyzable branch (must be an indirect jump).
    if (isUnpredicatedTerminator(SecondLastInst) && !SecondLastOpc)
      return true;
  }

  // If there is only one terminator instruction, process it.
  if (!SecondLastOpc) {
    // Unconditional branch
    if (LastOpc == UncondBrOpc) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }

    // Conditional branch
    AnalyzeCondBr(LastInst, LastOpc, TBB, Cond);
    return false;
  }

  // If we reached here, there are two branches.
  // If there are three terminators, we don't know what sort of block this is.
  if (++I != REnd && isUnpredicatedTerminator(&*I))
    return true;

  // If second to last instruction is an unconditional branch,
  // analyze it and remove the last instruction.
  if (SecondLastOpc == UncondBrOpc) {
    // Return if the last instruction cannot be removed.
    if (!AllowModify)
      return true;

    TBB = SecondLastInst->getOperand(0).getMBB();
    LastInst->eraseFromParent();
    return false;
  }

  // Conditional branch followed by an unconditional branch.
  // The last one must be unconditional.
  if (LastOpc != UncondBrOpc)
    return true;

  AnalyzeCondBr(SecondLastInst, SecondLastOpc, TBB, Cond);
  FBB = LastInst->getOperand(0).getMBB();

  return false;
}

void MipsInstrInfo::BuildCondBr(MachineBasicBlock &MBB,
                                MachineBasicBlock *TBB, DebugLoc DL,
                                const SmallVectorImpl<MachineOperand>& Cond)
  const {
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

unsigned MipsInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond,
             DebugLoc DL) const {
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

unsigned MipsInstrInfo::
RemoveBranch(MachineBasicBlock &MBB) const
{
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();
  MachineBasicBlock::reverse_iterator FirstBr;
  unsigned removed;

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugValue())
    ++I;

  FirstBr = I;

  // Up to 2 branches are removed.
  // Note that indirect branches are not removed.
  for(removed = 0; I != REnd && removed < 2; ++I, ++removed)
    if (!GetAnalyzableBrOpc(I->getOpcode()))
      break;

  MBB.erase(I.base(), FirstBr.base());

  return removed;
}

/// ReverseBranchCondition - Return the inverse opcode of the
/// specified Branch instruction.
bool MipsInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const
{
  assert( (Cond.size() && Cond.size() <= 3) &&
          "Invalid Mips branch condition!");
  Cond[0].setImm(GetOppositeBranchOpc(Cond[0].getImm()));
  return false;
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
  }
}
