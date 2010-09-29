//===- ARMBaseInstrInfo.cpp - ARM Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Base ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMBaseInstrInfo.h"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMGenInstrInfo.inc"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalValue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

static cl::opt<bool>
EnableARM3Addr("enable-arm-3-addr-conv", cl::Hidden,
               cl::desc("Enable ARM 2-addr to 3-addr conv"));

ARMBaseInstrInfo::ARMBaseInstrInfo(const ARMSubtarget& STI)
  : TargetInstrInfoImpl(ARMInsts, array_lengthof(ARMInsts)),
    Subtarget(STI) {
}

MachineInstr *
ARMBaseInstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                        MachineBasicBlock::iterator &MBBI,
                                        LiveVariables *LV) const {
  // FIXME: Thumb2 support.

  if (!EnableARM3Addr)
    return NULL;

  MachineInstr *MI = MBBI;
  MachineFunction &MF = *MI->getParent()->getParent();
  uint64_t TSFlags = MI->getDesc().TSFlags;
  bool isPre = false;
  switch ((TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift) {
  default: return NULL;
  case ARMII::IndexModePre:
    isPre = true;
    break;
  case ARMII::IndexModePost:
    break;
  }

  // Try splitting an indexed load/store to an un-indexed one plus an add/sub
  // operation.
  unsigned MemOpc = getUnindexedOpcode(MI->getOpcode());
  if (MemOpc == 0)
    return NULL;

  MachineInstr *UpdateMI = NULL;
  MachineInstr *MemMI = NULL;
  unsigned AddrMode = (TSFlags & ARMII::AddrModeMask);
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned NumOps = TID.getNumOperands();
  bool isLoad = !TID.mayStore();
  const MachineOperand &WB = isLoad ? MI->getOperand(1) : MI->getOperand(0);
  const MachineOperand &Base = MI->getOperand(2);
  const MachineOperand &Offset = MI->getOperand(NumOps-3);
  unsigned WBReg = WB.getReg();
  unsigned BaseReg = Base.getReg();
  unsigned OffReg = Offset.getReg();
  unsigned OffImm = MI->getOperand(NumOps-2).getImm();
  ARMCC::CondCodes Pred = (ARMCC::CondCodes)MI->getOperand(NumOps-1).getImm();
  switch (AddrMode) {
  default:
    assert(false && "Unknown indexed op!");
    return NULL;
  case ARMII::AddrMode2: {
    bool isSub = ARM_AM::getAM2Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM2Offset(OffImm);
    if (OffReg == 0) {
      if (ARM_AM::getSOImmVal(Amt) == -1)
        // Can't encode it in a so_imm operand. This transformation will
        // add more than 1 instruction. Abandon!
        return NULL;
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(Amt)
        .addImm(Pred).addReg(0).addReg(0);
    } else if (Amt != 0) {
      ARM_AM::ShiftOpc ShOpc = ARM_AM::getAM2ShiftOpc(OffImm);
      unsigned SOOpc = ARM_AM::getSORegOpc(ShOpc, Amt);
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrs : ARM::ADDrs), WBReg)
        .addReg(BaseReg).addReg(OffReg).addReg(0).addImm(SOOpc)
        .addImm(Pred).addReg(0).addReg(0);
    } else
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  case ARMII::AddrMode3 : {
    bool isSub = ARM_AM::getAM3Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM3Offset(OffImm);
    if (OffReg == 0)
      // Immediate is 8-bits. It's guaranteed to fit in a so_imm operand.
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(Amt)
        .addImm(Pred).addReg(0).addReg(0);
    else
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  }

  std::vector<MachineInstr*> NewMIs;
  if (isPre) {
    if (isLoad)
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc), MI->getOperand(0).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    NewMIs.push_back(MemMI);
    NewMIs.push_back(UpdateMI);
  } else {
    if (isLoad)
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc), MI->getOperand(0).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    if (WB.isDead())
      UpdateMI->getOperand(0).setIsDead();
    NewMIs.push_back(UpdateMI);
    NewMIs.push_back(MemMI);
  }

  // Transfer LiveVariables states, kill / dead info.
  if (LV) {
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.getReg() &&
          TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        unsigned Reg = MO.getReg();

        LiveVariables::VarInfo &VI = LV->getVarInfo(Reg);
        if (MO.isDef()) {
          MachineInstr *NewMI = (Reg == WBReg) ? UpdateMI : MemMI;
          if (MO.isDead())
            LV->addVirtualRegisterDead(Reg, NewMI);
        }
        if (MO.isUse() && MO.isKill()) {
          for (unsigned j = 0; j < 2; ++j) {
            // Look at the two new MI's in reverse order.
            MachineInstr *NewMI = NewMIs[j];
            if (!NewMI->readsRegister(Reg))
              continue;
            LV->addVirtualRegisterKilled(Reg, NewMI);
            if (VI.removeKill(MI))
              VI.Kills.push_back(NewMI);
            break;
          }
        }
      }
    }
  }

  MFI->insert(MBBI, NewMIs[1]);
  MFI->insert(MBBI, NewMIs[0]);
  return NewMIs[0];
}

bool
ARMBaseInstrInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    bool isKill = true;

    // Add the callee-saved register as live-in unless it's LR and
    // @llvm.returnaddress is called. If LR is returned for @llvm.returnaddress
    // then it's already added to the function and entry block live-in sets.
    if (Reg == ARM::LR) {
      MachineFunction &MF = *MBB.getParent();
      if (MF.getFrameInfo()->isReturnAddressTaken() &&
          MF.getRegInfo().isLiveIn(Reg))
        isKill = false;
    }

    if (isKill)
      MBB.addLiveIn(Reg);

    // Insert the spill to the stack frame. The register is killed at the spill
    // 
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    storeRegToStackSlot(MBB, MI, Reg, isKill,
                        CSI[i].getFrameIdx(), RC, TRI);
  }
  return true;
}

// Branch analysis.
bool
ARMBaseInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
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
    if (isUncondBranchOpcode(LastOpc)) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (isCondBranchOpcode(LastOpc)) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(LastInst->getOperand(1));
      Cond.push_back(LastInst->getOperand(2));
      return false;
    }
    return true;  // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with a B and a Bcc, handle it.
  unsigned SecondLastOpc = SecondLastInst->getOpcode();
  if (isCondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    TBB =  SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(SecondLastInst->getOperand(1));
    Cond.push_back(SecondLastInst->getOperand(2));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed, so remove it.
  if (isUncondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // ...likewise if it ends with a branch table followed by an unconditional
  // branch. The branch folder can create these, and we must get rid of them for
  // correctness of Thumb constant islands.
  if ((isJumpTableBranchOpcode(SecondLastOpc) ||
       isIndirectBranchOpcode(SecondLastOpc)) &&
      isUncondBranchOpcode(LastOpc)) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  }

  // Otherwise, can't handle this.
  return true;
}


unsigned ARMBaseInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }
  if (!isUncondBranchOpcode(I->getOpcode()) &&
      !isCondBranchOpcode(I->getOpcode()))
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (!isCondBranchOpcode(I->getOpcode()))
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
ARMBaseInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                               MachineBasicBlock *FBB,
                               const SmallVectorImpl<MachineOperand> &Cond,
                               DebugLoc DL) const {
  ARMFunctionInfo *AFI = MBB.getParent()->getInfo<ARMFunctionInfo>();
  int BOpc   = !AFI->isThumbFunction()
    ? ARM::B : (AFI->isThumb2Function() ? ARM::t2B : ARM::tB);
  int BccOpc = !AFI->isThumbFunction()
    ? ARM::Bcc : (AFI->isThumb2Function() ? ARM::t2Bcc : ARM::tBcc);

  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "ARM branch conditions have two components!");

  if (FBB == 0) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, DL, get(BOpc)).addMBB(TBB);
    else
      BuildMI(&MBB, DL, get(BccOpc)).addMBB(TBB)
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
    return 1;
  }

  // Two-way conditional branch.
  BuildMI(&MBB, DL, get(BccOpc)).addMBB(TBB)
    .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
  BuildMI(&MBB, DL, get(BOpc)).addMBB(FBB);
  return 2;
}

bool ARMBaseInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)(int)Cond[0].getImm();
  Cond[0].setImm(ARMCC::getOppositeCondition(CC));
  return false;
}

bool ARMBaseInstrInfo::
PredicateInstruction(MachineInstr *MI,
                     const SmallVectorImpl<MachineOperand> &Pred) const {
  unsigned Opc = MI->getOpcode();
  if (isUncondBranchOpcode(Opc)) {
    MI->setDesc(get(getMatchingCondBranchOpcode(Opc)));
    MI->addOperand(MachineOperand::CreateImm(Pred[0].getImm()));
    MI->addOperand(MachineOperand::CreateReg(Pred[1].getReg(), false));
    return true;
  }

  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx != -1) {
    MachineOperand &PMO = MI->getOperand(PIdx);
    PMO.setImm(Pred[0].getImm());
    MI->getOperand(PIdx+1).setReg(Pred[1].getReg());
    return true;
  }
  return false;
}

bool ARMBaseInstrInfo::
SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                  const SmallVectorImpl<MachineOperand> &Pred2) const {
  if (Pred1.size() > 2 || Pred2.size() > 2)
    return false;

  ARMCC::CondCodes CC1 = (ARMCC::CondCodes)Pred1[0].getImm();
  ARMCC::CondCodes CC2 = (ARMCC::CondCodes)Pred2[0].getImm();
  if (CC1 == CC2)
    return true;

  switch (CC1) {
  default:
    return false;
  case ARMCC::AL:
    return true;
  case ARMCC::HS:
    return CC2 == ARMCC::HI;
  case ARMCC::LS:
    return CC2 == ARMCC::LO || CC2 == ARMCC::EQ;
  case ARMCC::GE:
    return CC2 == ARMCC::GT;
  case ARMCC::LE:
    return CC2 == ARMCC::LT;
  }
}

bool ARMBaseInstrInfo::DefinesPredicate(MachineInstr *MI,
                                    std::vector<MachineOperand> &Pred) const {
  // FIXME: This confuses implicit_def with optional CPSR def.
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.getImplicitDefs() && !TID.hasOptionalDef())
    return false;

  bool Found = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == ARM::CPSR) {
      Pred.push_back(MO);
      Found = true;
    }
  }

  return Found;
}

/// isPredicable - Return true if the specified instruction can be predicated.
/// By default, this returns true for every instruction with a
/// PredicateOperand.
bool ARMBaseInstrInfo::isPredicable(MachineInstr *MI) const {
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isPredicable())
    return false;

  if ((TID.TSFlags & ARMII::DomainMask) == ARMII::DomainNEON) {
    ARMFunctionInfo *AFI =
      MI->getParent()->getParent()->getInfo<ARMFunctionInfo>();
    return AFI->isThumb2Function();
  }
  return true;
}

/// FIXME: Works around a gcc miscompilation with -fstrict-aliasing.
DISABLE_INLINE
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI);
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) {
  assert(JTI < JT.size());
  return JT[JTI].MBBs.size();
}

/// GetInstSize - Return the size of the specified MachineInstr.
///
unsigned ARMBaseInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  const MachineBasicBlock &MBB = *MI->getParent();
  const MachineFunction *MF = MBB.getParent();
  const MCAsmInfo *MAI = MF->getTarget().getMCAsmInfo();

  // Basic size info comes from the TSFlags field.
  const TargetInstrDesc &TID = MI->getDesc();
  uint64_t TSFlags = TID.TSFlags;

  unsigned Opc = MI->getOpcode();
  switch ((TSFlags & ARMII::SizeMask) >> ARMII::SizeShift) {
  default: {
    // If this machine instr is an inline asm, measure it.
    if (MI->getOpcode() == ARM::INLINEASM)
      return getInlineAsmLength(MI->getOperand(0).getSymbolName(), *MAI);
    if (MI->isLabel())
      return 0;
    switch (Opc) {
    default:
      llvm_unreachable("Unknown or unset size field for instr!");
    case TargetOpcode::IMPLICIT_DEF:
    case TargetOpcode::KILL:
    case TargetOpcode::PROLOG_LABEL:
    case TargetOpcode::EH_LABEL:
    case TargetOpcode::DBG_VALUE:
      return 0;
    }
    break;
  }
  case ARMII::Size8Bytes: return 8;          // ARM instruction x 2.
  case ARMII::Size4Bytes: return 4;          // ARM / Thumb2 instruction.
  case ARMII::Size2Bytes: return 2;          // Thumb1 instruction.
  case ARMII::SizeSpecial: {
    switch (Opc) {
    case ARM::CONSTPOOL_ENTRY:
      // If this machine instr is a constant pool entry, its size is recorded as
      // operand #2.
      return MI->getOperand(2).getImm();
    case ARM::Int_eh_sjlj_longjmp:
      return 16;
    case ARM::tInt_eh_sjlj_longjmp:
      return 10;
    case ARM::Int_eh_sjlj_setjmp:
    case ARM::Int_eh_sjlj_setjmp_nofp:
      return 20;
    case ARM::tInt_eh_sjlj_setjmp:
    case ARM::t2Int_eh_sjlj_setjmp:
    case ARM::t2Int_eh_sjlj_setjmp_nofp:
      return 12;
    case ARM::BR_JTr:
    case ARM::BR_JTm:
    case ARM::BR_JTadd:
    case ARM::tBR_JTr:
    case ARM::t2BR_JT:
    case ARM::t2TBB:
    case ARM::t2TBH: {
      // These are jumptable branches, i.e. a branch followed by an inlined
      // jumptable. The size is 4 + 4 * number of entries. For TBB, each
      // entry is one byte; TBH two byte each.
      unsigned EntrySize = (Opc == ARM::t2TBB)
        ? 1 : ((Opc == ARM::t2TBH) ? 2 : 4);
      unsigned NumOps = TID.getNumOperands();
      MachineOperand JTOP =
        MI->getOperand(NumOps - (TID.isPredicable() ? 3 : 2));
      unsigned JTI = JTOP.getIndex();
      const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
      assert(MJTI != 0);
      const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
      assert(JTI < JT.size());
      // Thumb instructions are 2 byte aligned, but JT entries are 4 byte
      // 4 aligned. The assembler / linker may add 2 byte padding just before
      // the JT entries.  The size does not include this padding; the
      // constant islands pass does separate bookkeeping for it.
      // FIXME: If we know the size of the function is less than (1 << 16) *2
      // bytes, we can use 16-bit entries instead. Then there won't be an
      // alignment issue.
      unsigned InstSize = (Opc == ARM::tBR_JTr || Opc == ARM::t2BR_JT) ? 2 : 4;
      unsigned NumEntries = getNumJTEntries(JT, JTI);
      if (Opc == ARM::t2TBB && (NumEntries & 1))
        // Make sure the instruction that follows TBB is 2-byte aligned.
        // FIXME: Constant island pass should insert an "ALIGN" instruction
        // instead.
        ++NumEntries;
      return NumEntries * EntrySize + InstSize;
    }
    default:
      // Otherwise, pseudo-instruction sizes are zero.
      return 0;
    }
  }
  }
  return 0; // Not reached
}

unsigned
ARMBaseInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::LDR:
  case ARM::t2LDRs:  // FIXME: don't use t2LDRs to access frame.
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImm() &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::t2LDRi12:
  case ARM::tRestore:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::VLDRD:
  case ARM::VLDRS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
}

unsigned
ARMBaseInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                     int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::STR:
  case ARM::t2STRs: // FIXME: don't use t2STRs to access frame.
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImm() &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::t2STRi12:
  case ARM::tSpill:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::VSTRD:
  case ARM::VSTRS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
}

void ARMBaseInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I, DebugLoc DL,
                                   unsigned DestReg, unsigned SrcReg,
                                   bool KillSrc) const {
  bool GPRDest = ARM::GPRRegClass.contains(DestReg);
  bool GPRSrc  = ARM::GPRRegClass.contains(SrcReg);

  if (GPRDest && GPRSrc) {
    AddDefaultCC(AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::MOVr), DestReg)
                                  .addReg(SrcReg, getKillRegState(KillSrc))));
    return;
  }

  bool SPRDest = ARM::SPRRegClass.contains(DestReg);
  bool SPRSrc  = ARM::SPRRegClass.contains(SrcReg);

  unsigned Opc;
  if (SPRDest && SPRSrc)
    Opc = ARM::VMOVS;
  else if (GPRDest && SPRSrc)
    Opc = ARM::VMOVRS;
  else if (SPRDest && GPRSrc)
    Opc = ARM::VMOVSR;
  else if (ARM::DPRRegClass.contains(DestReg, SrcReg))
    Opc = ARM::VMOVD;
  else if (ARM::QPRRegClass.contains(DestReg, SrcReg))
    Opc = ARM::VMOVQ;
  else if (ARM::QQPRRegClass.contains(DestReg, SrcReg))
    Opc = ARM::VMOVQQ;
  else if (ARM::QQQQPRRegClass.contains(DestReg, SrcReg))
    Opc = ARM::VMOVQQQQ;
  else
    llvm_unreachable("Impossible reg-to-reg copy");

  MachineInstrBuilder MIB = BuildMI(MBB, I, DL, get(Opc), DestReg);
  MIB.addReg(SrcReg, getKillRegState(KillSrc));
  if (Opc != ARM::VMOVQQ && Opc != ARM::VMOVQQQQ)
    AddDefaultPred(MIB);
}

static const
MachineInstrBuilder &AddDReg(MachineInstrBuilder &MIB,
                             unsigned Reg, unsigned SubIdx, unsigned State,
                             const TargetRegisterInfo *TRI) {
  if (!SubIdx)
    return MIB.addReg(Reg, State);

  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return MIB.addReg(TRI->getSubReg(Reg, SubIdx), State);
  return MIB.addReg(Reg, State, SubIdx);
}

void ARMBaseInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PseudoSourceValue::getFixedStack(FI),
                            MachineMemOperand::MOStore, 0,
                            MFI.getObjectSize(FI),
                            Align);

  // tGPR is used sometimes in ARM instructions that need to avoid using
  // certain registers.  Just treat it as GPR here. Likewise, rGPR.
  if (RC == ARM::tGPRRegisterClass || RC == ARM::tcGPRRegisterClass
      || RC == ARM::rGPRRegisterClass)
    RC = ARM::GPRRegisterClass;

  switch (RC->getID()) {
  case ARM::GPRRegClassID:
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::STR))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addReg(0).addImm(0).addMemOperand(MMO));
    break;
  case ARM::SPRRegClassID:
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VSTRS))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
    break;
  case ARM::DPRRegClassID:
  case ARM::DPR_VFP2RegClassID:
  case ARM::DPR_8RegClassID:
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VSTRD))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
    break;
  case ARM::QPRRegClassID:
  case ARM::QPR_VFP2RegClassID:
  case ARM::QPR_8RegClassID:
    // FIXME: Neon instructions should support predicates
    if (Align >= 16 && getRegisterInfo().canRealignStack(MF)) {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VST1q))
                     .addFrameIndex(FI).addImm(16)
                     .addReg(SrcReg, getKillRegState(isKill))
                     .addMemOperand(MMO));
    } else {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VSTMQ))
                     .addReg(SrcReg, getKillRegState(isKill))
                     .addFrameIndex(FI)
                     .addImm(ARM_AM::getAM4ModeImm(ARM_AM::ia))
                     .addMemOperand(MMO));
    }
    break;
  case ARM::QQPRRegClassID:
  case ARM::QQPR_VFP2RegClassID:
    if (Align >= 16 && getRegisterInfo().canRealignStack(MF)) {
      // FIXME: It's possible to only store part of the QQ register if the
      // spilled def has a sub-register index.
      MachineInstrBuilder MIB = BuildMI(MBB, I, DL, get(ARM::VST1d64Q))
        .addFrameIndex(FI).addImm(16);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_0, getKillRegState(isKill), TRI);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_1, 0, TRI);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_2, 0, TRI);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_3, 0, TRI);
      AddDefaultPred(MIB.addMemOperand(MMO));
    } else {
      MachineInstrBuilder MIB =
        AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VSTMD))
                       .addFrameIndex(FI)
                       .addImm(ARM_AM::getAM4ModeImm(ARM_AM::ia)))
        .addMemOperand(MMO);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_0, getKillRegState(isKill), TRI);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_1, 0, TRI);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_2, 0, TRI);
            AddDReg(MIB, SrcReg, ARM::dsub_3, 0, TRI);
    }
    break;
  case ARM::QQQQPRRegClassID: {
    MachineInstrBuilder MIB =
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VSTMD))
                     .addFrameIndex(FI)
                     .addImm(ARM_AM::getAM4ModeImm(ARM_AM::ia)))
      .addMemOperand(MMO);
    MIB = AddDReg(MIB, SrcReg, ARM::dsub_0, getKillRegState(isKill), TRI);
    MIB = AddDReg(MIB, SrcReg, ARM::dsub_1, 0, TRI);
    MIB = AddDReg(MIB, SrcReg, ARM::dsub_2, 0, TRI);
    MIB = AddDReg(MIB, SrcReg, ARM::dsub_3, 0, TRI);
    MIB = AddDReg(MIB, SrcReg, ARM::dsub_4, 0, TRI);
    MIB = AddDReg(MIB, SrcReg, ARM::dsub_5, 0, TRI);
    MIB = AddDReg(MIB, SrcReg, ARM::dsub_6, 0, TRI);
          AddDReg(MIB, SrcReg, ARM::dsub_7, 0, TRI);
    break;
  }
  default:
    llvm_unreachable("Unknown regclass!");
  }
}

void ARMBaseInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PseudoSourceValue::getFixedStack(FI),
                            MachineMemOperand::MOLoad, 0,
                            MFI.getObjectSize(FI),
                            Align);

  // tGPR is used sometimes in ARM instructions that need to avoid using
  // certain registers.  Just treat it as GPR here.
  if (RC == ARM::tGPRRegisterClass || RC == ARM::tcGPRRegisterClass
      || RC == ARM::rGPRRegisterClass)
    RC = ARM::GPRRegisterClass;

  switch (RC->getID()) {
  case ARM::GPRRegClassID:
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::LDR), DestReg)
                   .addFrameIndex(FI).addReg(0).addImm(0).addMemOperand(MMO));
    break;
  case ARM::SPRRegClassID:
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLDRS), DestReg)
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
    break;
  case ARM::DPRRegClassID:
  case ARM::DPR_VFP2RegClassID:
  case ARM::DPR_8RegClassID:
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLDRD), DestReg)
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
    break;
  case ARM::QPRRegClassID:
  case ARM::QPR_VFP2RegClassID:
  case ARM::QPR_8RegClassID:
    if (Align >= 16 && getRegisterInfo().canRealignStack(MF)) {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLD1q), DestReg)
                     .addFrameIndex(FI).addImm(16)
                     .addMemOperand(MMO));
    } else {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLDMQ), DestReg)
                     .addFrameIndex(FI)
                     .addImm(ARM_AM::getAM4ModeImm(ARM_AM::ia))
                     .addMemOperand(MMO));
    }
    break;
  case ARM::QQPRRegClassID:
  case ARM::QQPR_VFP2RegClassID:
    if (Align >= 16 && getRegisterInfo().canRealignStack(MF)) {
      MachineInstrBuilder MIB = BuildMI(MBB, I, DL, get(ARM::VLD1d64Q));
      MIB = AddDReg(MIB, DestReg, ARM::dsub_0, RegState::Define, TRI);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_1, RegState::Define, TRI);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_2, RegState::Define, TRI);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_3, RegState::Define, TRI);
      AddDefaultPred(MIB.addFrameIndex(FI).addImm(16).addMemOperand(MMO));
    } else {
      MachineInstrBuilder MIB =
        AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLDMD))
                       .addFrameIndex(FI)
                       .addImm(ARM_AM::getAM4ModeImm(ARM_AM::ia)))
        .addMemOperand(MMO);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_0, RegState::Define, TRI);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_1, RegState::Define, TRI);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_2, RegState::Define, TRI);
            AddDReg(MIB, DestReg, ARM::dsub_3, RegState::Define, TRI);
    }
    break;
  case ARM::QQQQPRRegClassID: {
    MachineInstrBuilder MIB =
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLDMD))
                     .addFrameIndex(FI)
                     .addImm(ARM_AM::getAM4ModeImm(ARM_AM::ia)))
      .addMemOperand(MMO);
    MIB = AddDReg(MIB, DestReg, ARM::dsub_0, RegState::Define, TRI);
    MIB = AddDReg(MIB, DestReg, ARM::dsub_1, RegState::Define, TRI);
    MIB = AddDReg(MIB, DestReg, ARM::dsub_2, RegState::Define, TRI);
    MIB = AddDReg(MIB, DestReg, ARM::dsub_3, RegState::Define, TRI);
    MIB = AddDReg(MIB, DestReg, ARM::dsub_4, RegState::Define, TRI);
    MIB = AddDReg(MIB, DestReg, ARM::dsub_5, RegState::Define, TRI);
    MIB = AddDReg(MIB, DestReg, ARM::dsub_6, RegState::Define, TRI);
    AddDReg(MIB, DestReg, ARM::dsub_7, RegState::Define, TRI);
    break;
  }
  default:
    llvm_unreachable("Unknown regclass!");
  }
}

MachineInstr*
ARMBaseInstrInfo::emitFrameIndexDebugValue(MachineFunction &MF,
                                           int FrameIx, uint64_t Offset,
                                           const MDNode *MDPtr,
                                           DebugLoc DL) const {
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(ARM::DBG_VALUE))
    .addFrameIndex(FrameIx).addImm(0).addImm(Offset).addMetadata(MDPtr);
  return &*MIB;
}

/// Create a copy of a const pool value. Update CPI to the new index and return
/// the label UID.
static unsigned duplicateCPV(MachineFunction &MF, unsigned &CPI) {
  MachineConstantPool *MCP = MF.getConstantPool();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPI];
  assert(MCPE.isMachineConstantPoolEntry() &&
         "Expecting a machine constantpool entry!");
  ARMConstantPoolValue *ACPV =
    static_cast<ARMConstantPoolValue*>(MCPE.Val.MachineCPVal);

  unsigned PCLabelId = AFI->createConstPoolEntryUId();
  ARMConstantPoolValue *NewCPV = 0;
  // FIXME: The below assumes PIC relocation model and that the function
  // is Thumb mode (t1 or t2). PCAdjustment would be 8 for ARM mode PIC, and
  // zero for non-PIC in ARM or Thumb. The callers are all of thumb LDR
  // instructions, so that's probably OK, but is PIC always correct when
  // we get here?
  if (ACPV->isGlobalValue())
    NewCPV = new ARMConstantPoolValue(ACPV->getGV(), PCLabelId,
                                      ARMCP::CPValue, 4);
  else if (ACPV->isExtSymbol())
    NewCPV = new ARMConstantPoolValue(MF.getFunction()->getContext(),
                                      ACPV->getSymbol(), PCLabelId, 4);
  else if (ACPV->isBlockAddress())
    NewCPV = new ARMConstantPoolValue(ACPV->getBlockAddress(), PCLabelId,
                                      ARMCP::CPBlockAddress, 4);
  else if (ACPV->isLSDA())
    NewCPV = new ARMConstantPoolValue(MF.getFunction(), PCLabelId,
                                      ARMCP::CPLSDA, 4);
  else
    llvm_unreachable("Unexpected ARM constantpool value type!!");
  CPI = MCP->getConstantPoolIndex(NewCPV, MCPE.getAlignment());
  return PCLabelId;
}

void ARMBaseInstrInfo::
reMaterialize(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator I,
              unsigned DestReg, unsigned SubIdx,
              const MachineInstr *Orig,
              const TargetRegisterInfo &TRI) const {
  unsigned Opcode = Orig->getOpcode();
  switch (Opcode) {
  default: {
    MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
    MI->substituteRegister(Orig->getOperand(0).getReg(), DestReg, SubIdx, TRI);
    MBB.insert(I, MI);
    break;
  }
  case ARM::tLDRpci_pic:
  case ARM::t2LDRpci_pic: {
    MachineFunction &MF = *MBB.getParent();
    unsigned CPI = Orig->getOperand(1).getIndex();
    unsigned PCLabelId = duplicateCPV(MF, CPI);
    MachineInstrBuilder MIB = BuildMI(MBB, I, Orig->getDebugLoc(), get(Opcode),
                                      DestReg)
      .addConstantPoolIndex(CPI).addImm(PCLabelId);
    (*MIB).setMemRefs(Orig->memoperands_begin(), Orig->memoperands_end());
    break;
  }
  }
}

MachineInstr *
ARMBaseInstrInfo::duplicate(MachineInstr *Orig, MachineFunction &MF) const {
  MachineInstr *MI = TargetInstrInfoImpl::duplicate(Orig, MF);
  switch(Orig->getOpcode()) {
  case ARM::tLDRpci_pic:
  case ARM::t2LDRpci_pic: {
    unsigned CPI = Orig->getOperand(1).getIndex();
    unsigned PCLabelId = duplicateCPV(MF, CPI);
    Orig->getOperand(1).setIndex(CPI);
    Orig->getOperand(2).setImm(PCLabelId);
    break;
  }
  }
  return MI;
}

bool ARMBaseInstrInfo::produceSameValue(const MachineInstr *MI0,
                                        const MachineInstr *MI1) const {
  int Opcode = MI0->getOpcode();
  if (Opcode == ARM::t2LDRpci ||
      Opcode == ARM::t2LDRpci_pic ||
      Opcode == ARM::tLDRpci ||
      Opcode == ARM::tLDRpci_pic) {
    if (MI1->getOpcode() != Opcode)
      return false;
    if (MI0->getNumOperands() != MI1->getNumOperands())
      return false;

    const MachineOperand &MO0 = MI0->getOperand(1);
    const MachineOperand &MO1 = MI1->getOperand(1);
    if (MO0.getOffset() != MO1.getOffset())
      return false;

    const MachineFunction *MF = MI0->getParent()->getParent();
    const MachineConstantPool *MCP = MF->getConstantPool();
    int CPI0 = MO0.getIndex();
    int CPI1 = MO1.getIndex();
    const MachineConstantPoolEntry &MCPE0 = MCP->getConstants()[CPI0];
    const MachineConstantPoolEntry &MCPE1 = MCP->getConstants()[CPI1];
    ARMConstantPoolValue *ACPV0 =
      static_cast<ARMConstantPoolValue*>(MCPE0.Val.MachineCPVal);
    ARMConstantPoolValue *ACPV1 =
      static_cast<ARMConstantPoolValue*>(MCPE1.Val.MachineCPVal);
    return ACPV0->hasSameValue(ACPV1);
  }

  return MI0->isIdenticalTo(MI1, MachineInstr::IgnoreVRegDefs);
}

/// areLoadsFromSameBasePtr - This is used by the pre-regalloc scheduler to
/// determine if two loads are loading from the same base address. It should
/// only return true if the base pointers are the same and the only differences
/// between the two addresses is the offset. It also returns the offsets by
/// reference.
bool ARMBaseInstrInfo::areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                               int64_t &Offset1,
                                               int64_t &Offset2) const {
  // Don't worry about Thumb: just ARM and Thumb2.
  if (Subtarget.isThumb1Only()) return false;

  if (!Load1->isMachineOpcode() || !Load2->isMachineOpcode())
    return false;

  switch (Load1->getMachineOpcode()) {
  default:
    return false;
  case ARM::LDR:
  case ARM::LDRB:
  case ARM::LDRD:
  case ARM::LDRH:
  case ARM::LDRSB:
  case ARM::LDRSH:
  case ARM::VLDRD:
  case ARM::VLDRS:
  case ARM::t2LDRi8:
  case ARM::t2LDRDi8:
  case ARM::t2LDRSHi8:
  case ARM::t2LDRi12:
  case ARM::t2LDRSHi12:
    break;
  }

  switch (Load2->getMachineOpcode()) {
  default:
    return false;
  case ARM::LDR:
  case ARM::LDRB:
  case ARM::LDRD:
  case ARM::LDRH:
  case ARM::LDRSB:
  case ARM::LDRSH:
  case ARM::VLDRD:
  case ARM::VLDRS:
  case ARM::t2LDRi8:
  case ARM::t2LDRDi8:
  case ARM::t2LDRSHi8:
  case ARM::t2LDRi12:
  case ARM::t2LDRSHi12:
    break;
  }

  // Check if base addresses and chain operands match.
  if (Load1->getOperand(0) != Load2->getOperand(0) ||
      Load1->getOperand(4) != Load2->getOperand(4))
    return false;

  // Index should be Reg0.
  if (Load1->getOperand(3) != Load2->getOperand(3))
    return false;

  // Determine the offsets.
  if (isa<ConstantSDNode>(Load1->getOperand(1)) &&
      isa<ConstantSDNode>(Load2->getOperand(1))) {
    Offset1 = cast<ConstantSDNode>(Load1->getOperand(1))->getSExtValue();
    Offset2 = cast<ConstantSDNode>(Load2->getOperand(1))->getSExtValue();
    return true;
  }

  return false;
}

/// shouldScheduleLoadsNear - This is a used by the pre-regalloc scheduler to
/// determine (in conjuction with areLoadsFromSameBasePtr) if two loads should
/// be scheduled togther. On some targets if two loads are loading from
/// addresses in the same cache line, it's better if they are scheduled
/// together. This function takes two integers that represent the load offsets
/// from the common base address. It returns true if it decides it's desirable
/// to schedule the two loads together. "NumLoads" is the number of loads that
/// have already been scheduled after Load1.
bool ARMBaseInstrInfo::shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                                               int64_t Offset1, int64_t Offset2,
                                               unsigned NumLoads) const {
  // Don't worry about Thumb: just ARM and Thumb2.
  if (Subtarget.isThumb1Only()) return false;

  assert(Offset2 > Offset1);

  if ((Offset2 - Offset1) / 8 > 64)
    return false;

  if (Load1->getMachineOpcode() != Load2->getMachineOpcode())
    return false;  // FIXME: overly conservative?

  // Four loads in a row should be sufficient.
  if (NumLoads >= 3)
    return false;

  return true;
}

bool ARMBaseInstrInfo::isSchedulingBoundary(const MachineInstr *MI,
                                            const MachineBasicBlock *MBB,
                                            const MachineFunction &MF) const {
  // Debug info is never a scheduling boundary. It's necessary to be explicit
  // due to the special treatment of IT instructions below, otherwise a
  // dbg_value followed by an IT will result in the IT instruction being
  // considered a scheduling hazard, which is wrong. It should be the actual
  // instruction preceding the dbg_value instruction(s), just like it is
  // when debug info is not present.
  if (MI->isDebugValue())
    return false;

  // Terminators and labels can't be scheduled around.
  if (MI->getDesc().isTerminator() || MI->isLabel())
    return true;

  // Treat the start of the IT block as a scheduling boundary, but schedule
  // t2IT along with all instructions following it.
  // FIXME: This is a big hammer. But the alternative is to add all potential
  // true and anti dependencies to IT block instructions as implicit operands
  // to the t2IT instruction. The added compile time and complexity does not
  // seem worth it.
  MachineBasicBlock::const_iterator I = MI;
  // Make sure to skip any dbg_value instructions
  while (++I != MBB->end() && I->isDebugValue())
    ;
  if (I != MBB->end() && I->getOpcode() == ARM::t2IT)
    return true;

  // Don't attempt to schedule around any instruction that defines
  // a stack-oriented pointer, as it's unlikely to be profitable. This
  // saves compile time, because it doesn't require every single
  // stack slot reference to depend on the instruction that does the
  // modification.
  if (MI->definesRegister(ARM::SP))
    return true;

  return false;
}

bool ARMBaseInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &MBB, unsigned NumInstrs) const {
  if (!NumInstrs)
    return false;
  if (Subtarget.getCPUString() == "generic")
    // Generic (and overly aggressive) if-conversion limits for testing.
    return NumInstrs <= 10;
  else if (Subtarget.hasV7Ops())
    return NumInstrs <= 3;
  return NumInstrs <= 2;
}
  
bool ARMBaseInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &TMBB, unsigned NumT,
                    MachineBasicBlock &FMBB, unsigned NumF) const {
  return NumT && NumF && NumT <= 2 && NumF <= 2;
}

/// getInstrPredicate - If instruction is predicated, returns its predicate
/// condition, otherwise returns AL. It also returns the condition code
/// register by reference.
ARMCC::CondCodes
llvm::getInstrPredicate(const MachineInstr *MI, unsigned &PredReg) {
  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx == -1) {
    PredReg = 0;
    return ARMCC::AL;
  }

  PredReg = MI->getOperand(PIdx+1).getReg();
  return (ARMCC::CondCodes)MI->getOperand(PIdx).getImm();
}


int llvm::getMatchingCondBranchOpcode(int Opc) {
  if (Opc == ARM::B)
    return ARM::Bcc;
  else if (Opc == ARM::tB)
    return ARM::tBcc;
  else if (Opc == ARM::t2B)
      return ARM::t2Bcc;

  llvm_unreachable("Unknown unconditional branch opcode!");
  return 0;
}


void llvm::emitARMRegPlusImmediate(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                               unsigned DestReg, unsigned BaseReg, int NumBytes,
                               ARMCC::CondCodes Pred, unsigned PredReg,
                               const ARMBaseInstrInfo &TII) {
  bool isSub = NumBytes < 0;
  if (isSub) NumBytes = -NumBytes;

  while (NumBytes) {
    unsigned RotAmt = ARM_AM::getSOImmValRotate(NumBytes);
    unsigned ThisVal = NumBytes & ARM_AM::rotr32(0xFF, RotAmt);
    assert(ThisVal && "Didn't extract field correctly");

    // We will handle these bits from offset, clear them.
    NumBytes &= ~ThisVal;

    assert(ARM_AM::getSOImmVal(ThisVal) != -1 && "Bit extraction didn't work?");

    // Build the new ADD / SUB.
    unsigned Opc = isSub ? ARM::SUBri : ARM::ADDri;
    BuildMI(MBB, MBBI, dl, TII.get(Opc), DestReg)
      .addReg(BaseReg, RegState::Kill).addImm(ThisVal)
      .addImm((unsigned)Pred).addReg(PredReg).addReg(0);
    BaseReg = DestReg;
  }
}

bool llvm::rewriteARMFrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                                unsigned FrameReg, int &Offset,
                                const ARMBaseInstrInfo &TII) {
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = MI.getDesc();
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
  bool isSub = false;

  // Memory operands in inline assembly always use AddrMode2.
  if (Opcode == ARM::INLINEASM)
    AddrMode = ARMII::AddrMode2;

  if (Opcode == ARM::ADDri) {
    Offset += MI.getOperand(FrameRegIdx+1).getImm();
    if (Offset == 0) {
      // Turn it into a move.
      MI.setDesc(TII.get(ARM::MOVr));
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(FrameRegIdx+1);
      Offset = 0;
      return true;
    } else if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
      MI.setDesc(TII.get(ARM::SUBri));
    }

    // Common case: small offset, fits into instruction.
    if (ARM_AM::getSOImmVal(Offset) != -1) {
      // Replace the FrameIndex with sp / fp
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
      MI.getOperand(FrameRegIdx+1).ChangeToImmediate(Offset);
      Offset = 0;
      return true;
    }

    // Otherwise, pull as much of the immedidate into this ADDri/SUBri
    // as possible.
    unsigned RotAmt = ARM_AM::getSOImmValRotate(Offset);
    unsigned ThisImmVal = Offset & ARM_AM::rotr32(0xFF, RotAmt);

    // We will handle these bits from offset, clear them.
    Offset &= ~ThisImmVal;

    // Get the properly encoded SOImmVal field.
    assert(ARM_AM::getSOImmVal(ThisImmVal) != -1 &&
           "Bit extraction didn't work?");
    MI.getOperand(FrameRegIdx+1).ChangeToImmediate(ThisImmVal);
 } else {
    unsigned ImmIdx = 0;
    int InstrOffs = 0;
    unsigned NumBits = 0;
    unsigned Scale = 1;
    switch (AddrMode) {
    case ARMII::AddrMode2: {
      ImmIdx = FrameRegIdx+2;
      InstrOffs = ARM_AM::getAM2Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM2Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 12;
      break;
    }
    case ARMII::AddrMode3: {
      ImmIdx = FrameRegIdx+2;
      InstrOffs = ARM_AM::getAM3Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM3Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      break;
    }
    case ARMII::AddrMode4:
    case ARMII::AddrMode6:
      // Can't fold any offset even if it's zero.
      return false;
    case ARMII::AddrMode5: {
      ImmIdx = FrameRegIdx+1;
      InstrOffs = ARM_AM::getAM5Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM5Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      Scale = 4;
      break;
    }
    default:
      llvm_unreachable("Unsupported addressing mode!");
      break;
    }

    Offset += InstrOffs * Scale;
    assert((Offset & (Scale-1)) == 0 && "Can't encode this offset!");
    if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
    }

    // Attempt to fold address comp. if opcode has offset bits
    if (NumBits > 0) {
      // Common case: small offset, fits into instruction.
      MachineOperand &ImmOp = MI.getOperand(ImmIdx);
      int ImmedOffset = Offset / Scale;
      unsigned Mask = (1 << NumBits) - 1;
      if ((unsigned)Offset <= Mask * Scale) {
        // Replace the FrameIndex with sp
        MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
        if (isSub)
          ImmedOffset |= 1 << NumBits;
        ImmOp.ChangeToImmediate(ImmedOffset);
        Offset = 0;
        return true;
      }

      // Otherwise, it didn't fit. Pull in what we can to simplify the immed.
      ImmedOffset = ImmedOffset & Mask;
      if (isSub)
        ImmedOffset |= 1 << NumBits;
      ImmOp.ChangeToImmediate(ImmedOffset);
      Offset &= ~(Mask*Scale);
    }
  }

  Offset = (isSub) ? -Offset : Offset;
  return Offset == 0;
}

bool ARMBaseInstrInfo::
AnalyzeCompare(const MachineInstr *MI, unsigned &SrcReg, int &CmpValue) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::CMPri:
  case ARM::CMPzri:
  case ARM::t2CMPri:
  case ARM::t2CMPzri:
    SrcReg = MI->getOperand(0).getReg();
    CmpValue = MI->getOperand(1).getImm();
    return true;
  }

  return false;
}

/// ConvertToSetZeroFlag - Convert the instruction to set the "zero" flag so
/// that we can remove a "comparison with zero".
bool ARMBaseInstrInfo::
ConvertToSetZeroFlag(MachineInstr *MI, MachineInstr *CmpInstr) const {
  // Conservatively refuse to convert an instruction which isn't in the same BB
  // as the comparison.
  if (MI->getParent() != CmpInstr->getParent())
    return false;

  // Check that CPSR isn't set between the comparison instruction and the one we
  // want to change.
  MachineBasicBlock::const_iterator I = CmpInstr, E = MI,
    B = MI->getParent()->begin();
  --I;
  for (; I != E; --I) {
    const MachineInstr &Instr = *I;

    for (unsigned IO = 0, EO = Instr.getNumOperands(); IO != EO; ++IO) {
      const MachineOperand &MO = Instr.getOperand(IO);
      if (!MO.isReg() || !MO.isDef()) continue;

      // This instruction modifies CPSR before the one we want to change. We
      // can't do this transformation.
      if (MO.getReg() == ARM::CPSR)
        return false;
    }

    if (I == B)
      // The 'and' is below the comparison instruction.
      return false;
  }

  // Set the "zero" bit in CPSR.
  switch (MI->getOpcode()) {
  default: break;
  case ARM::ADDri:
  case ARM::SUBri:
  case ARM::t2ADDri:
  case ARM::t2SUBri:
    MI->RemoveOperand(5);
    MachineInstrBuilder(MI)
      .addReg(ARM::CPSR, RegState::Define | RegState::Implicit);
    CmpInstr->eraseFromParent();
    return true;
  }

  return false;
}
