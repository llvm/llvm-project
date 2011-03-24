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
#include "ARMHazardRecognizer.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMGenInstrInfo.inc"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalValue.h"
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
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

static cl::opt<bool>
EnableARM3Addr("enable-arm-3-addr-conv", cl::Hidden,
               cl::desc("Enable ARM 2-addr to 3-addr conv"));

/// ARM_MLxEntry - Record information about MLA / MLS instructions.
struct ARM_MLxEntry {
  unsigned MLxOpc;     // MLA / MLS opcode
  unsigned MulOpc;     // Expanded multiplication opcode
  unsigned AddSubOpc;  // Expanded add / sub opcode
  bool NegAcc;         // True if the acc is negated before the add / sub.
  bool HasLane;        // True if instruction has an extra "lane" operand.
};

static const ARM_MLxEntry ARM_MLxTable[] = {
  // MLxOpc,          MulOpc,           AddSubOpc,       NegAcc, HasLane
  // fp scalar ops
  { ARM::VMLAS,       ARM::VMULS,       ARM::VADDS,      false,  false },
  { ARM::VMLSS,       ARM::VMULS,       ARM::VSUBS,      false,  false },
  { ARM::VMLAD,       ARM::VMULD,       ARM::VADDD,      false,  false },
  { ARM::VMLSD,       ARM::VMULD,       ARM::VSUBD,      false,  false },
  { ARM::VNMLAS,      ARM::VNMULS,      ARM::VSUBS,      true,   false },
  { ARM::VNMLSS,      ARM::VMULS,       ARM::VSUBS,      true,   false },
  { ARM::VNMLAD,      ARM::VNMULD,      ARM::VSUBD,      true,   false },
  { ARM::VNMLSD,      ARM::VMULD,       ARM::VSUBD,      true,   false },

  // fp SIMD ops
  { ARM::VMLAfd,      ARM::VMULfd,      ARM::VADDfd,     false,  false },
  { ARM::VMLSfd,      ARM::VMULfd,      ARM::VSUBfd,     false,  false },
  { ARM::VMLAfq,      ARM::VMULfq,      ARM::VADDfq,     false,  false },
  { ARM::VMLSfq,      ARM::VMULfq,      ARM::VSUBfq,     false,  false },
  { ARM::VMLAslfd,    ARM::VMULslfd,    ARM::VADDfd,     false,  true  },
  { ARM::VMLSslfd,    ARM::VMULslfd,    ARM::VSUBfd,     false,  true  },
  { ARM::VMLAslfq,    ARM::VMULslfq,    ARM::VADDfq,     false,  true  },
  { ARM::VMLSslfq,    ARM::VMULslfq,    ARM::VSUBfq,     false,  true  },
};

ARMBaseInstrInfo::ARMBaseInstrInfo(const ARMSubtarget& STI)
  : TargetInstrInfoImpl(ARMInsts, array_lengthof(ARMInsts)),
    Subtarget(STI) {
  for (unsigned i = 0, e = array_lengthof(ARM_MLxTable); i != e; ++i) {
    if (!MLxEntryMap.insert(std::make_pair(ARM_MLxTable[i].MLxOpc, i)).second)
      assert(false && "Duplicated entries?");
    MLxHazardOpcodes.insert(ARM_MLxTable[i].AddSubOpc);
    MLxHazardOpcodes.insert(ARM_MLxTable[i].MulOpc);
  }
}

// Use a ScoreboardHazardRecognizer for prepass ARM scheduling. TargetInstrImpl
// currently defaults to no prepass hazard recognizer.
ScheduleHazardRecognizer *ARMBaseInstrInfo::
CreateTargetHazardRecognizer(const TargetMachine *TM,
                             const ScheduleDAG *DAG) const {
  if (usePreRAHazardRecognizer()) {
    const InstrItineraryData *II = TM->getInstrItineraryData();
    return new ScoreboardHazardRecognizer(II, DAG, "pre-RA-sched");
  }
  return TargetInstrInfoImpl::CreateTargetHazardRecognizer(TM, DAG);
}

ScheduleHazardRecognizer *ARMBaseInstrInfo::
CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                   const ScheduleDAG *DAG) const {
  if (Subtarget.isThumb2() || Subtarget.hasVFP2())
    return (ScheduleHazardRecognizer *)
      new ARMHazardRecognizer(II, *this, getRegisterInfo(), Subtarget, DAG);
  return TargetInstrInfoImpl::CreateTargetPostRAHazardRecognizer(II, DAG);
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
        .addReg(WBReg).addImm(0).addImm(Pred);
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
        .addReg(BaseReg).addImm(0).addImm(Pred);
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
      if (MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
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
  unsigned SecondLastOpc = SecondLastInst->getOpcode();

  // If AllowModify is true and the block ends with two or more unconditional
  // branches, delete all but the first unconditional branch.
  if (AllowModify && isUncondBranchOpcode(LastOpc)) {
    while (isUncondBranchOpcode(SecondLastOpc)) {
      LastInst->eraseFromParent();
      LastInst = SecondLastInst;
      LastOpc = LastInst->getOpcode();
      if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
        // Return now the only terminator is an unconditional branch.
        TBB = LastInst->getOperand(0).getMBB();
        return false;
      } else {
        SecondLastInst = I;
        SecondLastOpc = SecondLastInst->getOpcode();
      }
    }
  }

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with a B and a Bcc, handle it.
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
LLVM_ATTRIBUTE_NOINLINE
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
    case ARM::MOVi16_ga_pcrel:
    case ARM::MOVTi16_ga_pcrel:
    case ARM::t2MOVi16_ga_pcrel:
    case ARM::t2MOVTi16_ga_pcrel:
      return 4;
    case ARM::MOVi32imm:
    case ARM::t2MOVi32imm:
      return 8;
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
    case ARM::t2TBB_JT:
    case ARM::t2TBH_JT: {
      // These are jumptable branches, i.e. a branch followed by an inlined
      // jumptable. The size is 4 + 4 * number of entries. For TBB, each
      // entry is one byte; TBH two byte each.
      unsigned EntrySize = (Opc == ARM::t2TBB_JT)
        ? 1 : ((Opc == ARM::t2TBH_JT) ? 2 : 4);
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
      if (Opc == ARM::t2TBB_JT && (NumEntries & 1))
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
    MF.getMachineMemOperand(MachinePointerInfo(
                                         PseudoSourceValue::getFixedStack(FI)),
                            MachineMemOperand::MOStore,
                            MFI.getObjectSize(FI),
                            Align);

  // tGPR is used sometimes in ARM instructions that need to avoid using
  // certain registers.  Just treat it as GPR here. Likewise, rGPR.
  if (RC == ARM::tGPRRegisterClass || RC == ARM::tcGPRRegisterClass
      || RC == ARM::rGPRRegisterClass)
    RC = ARM::GPRRegisterClass;

  switch (RC->getID()) {
  case ARM::GPRRegClassID:
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::STRi12))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
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
    if (Align >= 16 && getRegisterInfo().needsStackRealignment(MF)) {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VST1q64Pseudo))
                     .addFrameIndex(FI).addImm(16)
                     .addReg(SrcReg, getKillRegState(isKill))
                     .addMemOperand(MMO));
    } else {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VSTMQIA))
                     .addReg(SrcReg, getKillRegState(isKill))
                     .addFrameIndex(FI)
                     .addMemOperand(MMO));
    }
    break;
  case ARM::QQPRRegClassID:
  case ARM::QQPR_VFP2RegClassID:
    if (Align >= 16 && getRegisterInfo().canRealignStack(MF)) {
      // FIXME: It's possible to only store part of the QQ register if the
      // spilled def has a sub-register index.
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VST1d64QPseudo))
                     .addFrameIndex(FI).addImm(16)
                     .addReg(SrcReg, getKillRegState(isKill))
                     .addMemOperand(MMO));
    } else {
      MachineInstrBuilder MIB =
        AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VSTMDIA))
                       .addFrameIndex(FI))
        .addMemOperand(MMO);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_0, getKillRegState(isKill), TRI);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_1, 0, TRI);
      MIB = AddDReg(MIB, SrcReg, ARM::dsub_2, 0, TRI);
            AddDReg(MIB, SrcReg, ARM::dsub_3, 0, TRI);
    }
    break;
  case ARM::QQQQPRRegClassID: {
    MachineInstrBuilder MIB =
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VSTMDIA))
                     .addFrameIndex(FI))
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

unsigned
ARMBaseInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                     int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::STRrs:
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
  case ARM::STRi12:
  case ARM::t2STRi12:
  case ARM::tSpill:
  case ARM::VSTRD:
  case ARM::VSTRS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::VST1q64Pseudo:
    if (MI->getOperand(0).isFI() &&
        MI->getOperand(2).getSubReg() == 0) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(2).getReg();
    }
    break;
  case ARM::VSTMQIA:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(0).getSubReg() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
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
    MF.getMachineMemOperand(
                    MachinePointerInfo(PseudoSourceValue::getFixedStack(FI)),
                            MachineMemOperand::MOLoad,
                            MFI.getObjectSize(FI),
                            Align);

  // tGPR is used sometimes in ARM instructions that need to avoid using
  // certain registers.  Just treat it as GPR here.
  if (RC == ARM::tGPRRegisterClass || RC == ARM::tcGPRRegisterClass
      || RC == ARM::rGPRRegisterClass)
    RC = ARM::GPRRegisterClass;

  switch (RC->getID()) {
  case ARM::GPRRegClassID:
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::LDRi12), DestReg)
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
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
    if (Align >= 16 && getRegisterInfo().needsStackRealignment(MF)) {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLD1q64Pseudo), DestReg)
                     .addFrameIndex(FI).addImm(16)
                     .addMemOperand(MMO));
    } else {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLDMQIA), DestReg)
                     .addFrameIndex(FI)
                     .addMemOperand(MMO));
    }
    break;
  case ARM::QQPRRegClassID:
  case ARM::QQPR_VFP2RegClassID:
    if (Align >= 16 && getRegisterInfo().canRealignStack(MF)) {
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLD1d64QPseudo), DestReg)
                     .addFrameIndex(FI).addImm(16)
                     .addMemOperand(MMO));
    } else {
      MachineInstrBuilder MIB =
        AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLDMDIA))
                       .addFrameIndex(FI))
        .addMemOperand(MMO);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_0, RegState::Define, TRI);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_1, RegState::Define, TRI);
      MIB = AddDReg(MIB, DestReg, ARM::dsub_2, RegState::Define, TRI);
            AddDReg(MIB, DestReg, ARM::dsub_3, RegState::Define, TRI);
    }
    break;
  case ARM::QQQQPRRegClassID: {
    MachineInstrBuilder MIB =
      AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::VLDMDIA))
                     .addFrameIndex(FI))
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

unsigned
ARMBaseInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::LDRrs:
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
  case ARM::LDRi12:
  case ARM::t2LDRi12:
  case ARM::tRestore:
  case ARM::VLDRD:
  case ARM::VLDRS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::VLD1q64Pseudo:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(0).getSubReg() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::VLDMQIA:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(0).getSubReg() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
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

  unsigned PCLabelId = AFI->createPICLabelUId();
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
                                        const MachineInstr *MI1,
                                        const MachineRegisterInfo *MRI) const {
  int Opcode = MI0->getOpcode();
  if (Opcode == ARM::t2LDRpci ||
      Opcode == ARM::t2LDRpci_pic ||
      Opcode == ARM::tLDRpci ||
      Opcode == ARM::tLDRpci_pic ||
      Opcode == ARM::MOV_ga_dyn ||
      Opcode == ARM::MOV_ga_pcrel ||
      Opcode == ARM::MOV_ga_pcrel_ldr ||
      Opcode == ARM::t2MOV_ga_dyn ||
      Opcode == ARM::t2MOV_ga_pcrel) {
    if (MI1->getOpcode() != Opcode)
      return false;
    if (MI0->getNumOperands() != MI1->getNumOperands())
      return false;

    const MachineOperand &MO0 = MI0->getOperand(1);
    const MachineOperand &MO1 = MI1->getOperand(1);
    if (MO0.getOffset() != MO1.getOffset())
      return false;

    if (Opcode == ARM::MOV_ga_dyn ||
        Opcode == ARM::MOV_ga_pcrel ||
        Opcode == ARM::MOV_ga_pcrel_ldr ||
        Opcode == ARM::t2MOV_ga_dyn ||
        Opcode == ARM::t2MOV_ga_pcrel)
      // Ignore the PC labels.
      return MO0.getGlobal() == MO1.getGlobal();

    const MachineFunction *MF = MI0->getParent()->getParent();
    const MachineConstantPool *MCP = MF->getConstantPool();
    int CPI0 = MO0.getIndex();
    int CPI1 = MO1.getIndex();
    const MachineConstantPoolEntry &MCPE0 = MCP->getConstants()[CPI0];
    const MachineConstantPoolEntry &MCPE1 = MCP->getConstants()[CPI1];
    bool isARMCP0 = MCPE0.isMachineConstantPoolEntry();
    bool isARMCP1 = MCPE1.isMachineConstantPoolEntry();
    if (isARMCP0 && isARMCP1) {
      ARMConstantPoolValue *ACPV0 =
        static_cast<ARMConstantPoolValue*>(MCPE0.Val.MachineCPVal);
      ARMConstantPoolValue *ACPV1 =
        static_cast<ARMConstantPoolValue*>(MCPE1.Val.MachineCPVal);
      return ACPV0->hasSameValue(ACPV1);
    } else if (!isARMCP0 && !isARMCP1) {
      return MCPE0.Val.ConstVal == MCPE1.Val.ConstVal;
    }
    return false;
  } else if (Opcode == ARM::PICLDR) {
    if (MI1->getOpcode() != Opcode)
      return false;
    if (MI0->getNumOperands() != MI1->getNumOperands())
      return false;

    unsigned Addr0 = MI0->getOperand(1).getReg();
    unsigned Addr1 = MI1->getOperand(1).getReg();
    if (Addr0 != Addr1) {
      if (!MRI ||
          !TargetRegisterInfo::isVirtualRegister(Addr0) ||
          !TargetRegisterInfo::isVirtualRegister(Addr1))
        return false;

      // This assumes SSA form.
      MachineInstr *Def0 = MRI->getVRegDef(Addr0);
      MachineInstr *Def1 = MRI->getVRegDef(Addr1);
      // Check if the loaded value, e.g. a constantpool of a global address, are
      // the same.
      if (!produceSameValue(Def0, Def1, MRI))
        return false;
    }

    for (unsigned i = 3, e = MI0->getNumOperands(); i != e; ++i) {
      // %vreg12<def> = PICLDR %vreg11, 0, pred:14, pred:%noreg
      const MachineOperand &MO0 = MI0->getOperand(i);
      const MachineOperand &MO1 = MI1->getOperand(i);
      if (!MO0.isIdenticalTo(MO1))
        return false;
    }
    return true;
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
  case ARM::LDRi12:
  case ARM::LDRBi12:
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
  case ARM::LDRi12:
  case ARM::LDRBi12:
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

bool ARMBaseInstrInfo::isProfitableToIfCvt(MachineBasicBlock &MBB,
                                           unsigned NumCyles,
                                           unsigned ExtraPredCycles,
                                           float Probability,
                                           float Confidence) const {
  if (!NumCyles)
    return false;

  // Attempt to estimate the relative costs of predication versus branching.
  float UnpredCost = Probability * NumCyles;
  UnpredCost += 1.0; // The branch itself
  UnpredCost += (1.0 - Confidence) * Subtarget.getMispredictionPenalty();

  return (float)(NumCyles + ExtraPredCycles) < UnpredCost;
}

bool ARMBaseInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &TMBB,
                    unsigned TCycles, unsigned TExtra,
                    MachineBasicBlock &FMBB,
                    unsigned FCycles, unsigned FExtra,
                    float Probability, float Confidence) const {
  if (!TCycles || !FCycles)
    return false;

  // Attempt to estimate the relative costs of predication versus branching.
  float UnpredCost = Probability * TCycles + (1.0 - Probability) * FCycles;
  UnpredCost += 1.0; // The branch itself
  UnpredCost += (1.0 - Confidence) * Subtarget.getMispredictionPenalty();

  return (float)(TCycles + FCycles + TExtra + FExtra) < UnpredCost;
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
                               const ARMBaseInstrInfo &TII, unsigned MIFlags) {
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
      .addImm((unsigned)Pred).addReg(PredReg).addReg(0)
      .setMIFlags(MIFlags);
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
    case ARMII::AddrMode_i12: {
      ImmIdx = FrameRegIdx + 1;
      InstrOffs = MI.getOperand(ImmIdx).getImm();
      NumBits = 12;
      break;
    }
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
        // FIXME: When addrmode2 goes away, this will simplify (like the
        // T2 version), as the LDR.i12 versions don't need the encoding
        // tricks for the offset value.
        if (isSub) {
          if (AddrMode == ARMII::AddrMode_i12)
            ImmedOffset = -ImmedOffset;
          else
            ImmedOffset |= 1 << NumBits;
        }
        ImmOp.ChangeToImmediate(ImmedOffset);
        Offset = 0;
        return true;
      }

      // Otherwise, it didn't fit. Pull in what we can to simplify the immed.
      ImmedOffset = ImmedOffset & Mask;
      if (isSub) {
        if (AddrMode == ARMII::AddrMode_i12)
          ImmedOffset = -ImmedOffset;
        else
          ImmedOffset |= 1 << NumBits;
      }
      ImmOp.ChangeToImmediate(ImmedOffset);
      Offset &= ~(Mask*Scale);
    }
  }

  Offset = (isSub) ? -Offset : Offset;
  return Offset == 0;
}

bool ARMBaseInstrInfo::
AnalyzeCompare(const MachineInstr *MI, unsigned &SrcReg, int &CmpMask,
               int &CmpValue) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::CMPri:
  case ARM::t2CMPri:
    SrcReg = MI->getOperand(0).getReg();
    CmpMask = ~0;
    CmpValue = MI->getOperand(1).getImm();
    return true;
  case ARM::TSTri:
  case ARM::t2TSTri:
    SrcReg = MI->getOperand(0).getReg();
    CmpMask = MI->getOperand(1).getImm();
    CmpValue = 0;
    return true;
  }

  return false;
}

/// isSuitableForMask - Identify a suitable 'and' instruction that
/// operates on the given source register and applies the same mask
/// as a 'tst' instruction. Provide a limited look-through for copies.
/// When successful, MI will hold the found instruction.
static bool isSuitableForMask(MachineInstr *&MI, unsigned SrcReg,
                              int CmpMask, bool CommonUse) {
  switch (MI->getOpcode()) {
    case ARM::ANDri:
    case ARM::t2ANDri:
      if (CmpMask != MI->getOperand(2).getImm())
        return false;
      if (SrcReg == MI->getOperand(CommonUse ? 1 : 0).getReg())
        return true;
      break;
    case ARM::COPY: {
      // Walk down one instruction which is potentially an 'and'.
      const MachineInstr &Copy = *MI;
      MachineBasicBlock::iterator AND(
        llvm::next(MachineBasicBlock::iterator(MI)));
      if (AND == MI->getParent()->end()) return false;
      MI = AND;
      return isSuitableForMask(MI, Copy.getOperand(0).getReg(),
                               CmpMask, true);
    }
  }

  return false;
}

/// OptimizeCompareInstr - Convert the instruction supplying the argument to the
/// comparison into one that sets the zero bit in the flags register.
bool ARMBaseInstrInfo::
OptimizeCompareInstr(MachineInstr *CmpInstr, unsigned SrcReg, int CmpMask,
                     int CmpValue, const MachineRegisterInfo *MRI) const {
  if (CmpValue != 0)
    return false;

  MachineRegisterInfo::def_iterator DI = MRI->def_begin(SrcReg);
  if (llvm::next(DI) != MRI->def_end())
    // Only support one definition.
    return false;

  MachineInstr *MI = &*DI;

  // Masked compares sometimes use the same register as the corresponding 'and'.
  if (CmpMask != ~0) {
    if (!isSuitableForMask(MI, SrcReg, CmpMask, false)) {
      MI = 0;
      for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(SrcReg),
           UE = MRI->use_end(); UI != UE; ++UI) {
        if (UI->getParent() != CmpInstr->getParent()) continue;
        MachineInstr *PotentialAND = &*UI;
        if (!isSuitableForMask(PotentialAND, SrcReg, CmpMask, true))
          continue;
        MI = PotentialAND;
        break;
      }
      if (!MI) return false;
    }
  }

  // Conservatively refuse to convert an instruction which isn't in the same BB
  // as the comparison.
  if (MI->getParent() != CmpInstr->getParent())
    return false;

  // Check that CPSR isn't set between the comparison instruction and the one we
  // want to change.
  MachineBasicBlock::const_iterator I = CmpInstr, E = MI,
    B = MI->getParent()->begin();

  // Early exit if CmpInstr is at the beginning of the BB.
  if (I == B) return false;

  --I;
  for (; I != E; --I) {
    const MachineInstr &Instr = *I;

    for (unsigned IO = 0, EO = Instr.getNumOperands(); IO != EO; ++IO) {
      const MachineOperand &MO = Instr.getOperand(IO);
      if (!MO.isReg()) continue;

      // This instruction modifies or uses CPSR after the one we want to
      // change. We can't do this transformation.
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
  case ARM::ANDri:
  case ARM::t2ANDri:
  case ARM::SUBri:
  case ARM::t2ADDri:
  case ARM::t2SUBri:
    // Toggle the optional operand to CPSR.
    MI->getOperand(5).setReg(ARM::CPSR);
    MI->getOperand(5).setIsDef(true);
    CmpInstr->eraseFromParent();
    return true;
  }

  return false;
}

bool ARMBaseInstrInfo::FoldImmediate(MachineInstr *UseMI,
                                     MachineInstr *DefMI, unsigned Reg,
                                     MachineRegisterInfo *MRI) const {
  // Fold large immediates into add, sub, or, xor.
  unsigned DefOpc = DefMI->getOpcode();
  if (DefOpc != ARM::t2MOVi32imm && DefOpc != ARM::MOVi32imm)
    return false;
  if (!DefMI->getOperand(1).isImm())
    // Could be t2MOVi32imm <ga:xx>
    return false;

  if (!MRI->hasOneNonDBGUse(Reg))
    return false;

  unsigned UseOpc = UseMI->getOpcode();
  unsigned NewUseOpc = 0;
  uint32_t ImmVal = (uint32_t)DefMI->getOperand(1).getImm();
  uint32_t SOImmValV1 = 0, SOImmValV2 = 0;
  bool Commute = false;
  switch (UseOpc) {
  default: return false;
  case ARM::SUBrr:
  case ARM::ADDrr:
  case ARM::ORRrr:
  case ARM::EORrr:
  case ARM::t2SUBrr:
  case ARM::t2ADDrr:
  case ARM::t2ORRrr:
  case ARM::t2EORrr: {
    Commute = UseMI->getOperand(2).getReg() != Reg;
    switch (UseOpc) {
    default: break;
    case ARM::SUBrr: {
      if (Commute)
        return false;
      ImmVal = -ImmVal;
      NewUseOpc = ARM::SUBri;
      // Fallthrough
    }
    case ARM::ADDrr:
    case ARM::ORRrr:
    case ARM::EORrr: {
      if (!ARM_AM::isSOImmTwoPartVal(ImmVal))
        return false;
      SOImmValV1 = (uint32_t)ARM_AM::getSOImmTwoPartFirst(ImmVal);
      SOImmValV2 = (uint32_t)ARM_AM::getSOImmTwoPartSecond(ImmVal);
      switch (UseOpc) {
      default: break;
      case ARM::ADDrr: NewUseOpc = ARM::ADDri; break;
      case ARM::ORRrr: NewUseOpc = ARM::ORRri; break;
      case ARM::EORrr: NewUseOpc = ARM::EORri; break;
      }
      break;
    }
    case ARM::t2SUBrr: {
      if (Commute)
        return false;
      ImmVal = -ImmVal;
      NewUseOpc = ARM::t2SUBri;
      // Fallthrough
    }
    case ARM::t2ADDrr:
    case ARM::t2ORRrr:
    case ARM::t2EORrr: {
      if (!ARM_AM::isT2SOImmTwoPartVal(ImmVal))
        return false;
      SOImmValV1 = (uint32_t)ARM_AM::getT2SOImmTwoPartFirst(ImmVal);
      SOImmValV2 = (uint32_t)ARM_AM::getT2SOImmTwoPartSecond(ImmVal);
      switch (UseOpc) {
      default: break;
      case ARM::t2ADDrr: NewUseOpc = ARM::t2ADDri; break;
      case ARM::t2ORRrr: NewUseOpc = ARM::t2ORRri; break;
      case ARM::t2EORrr: NewUseOpc = ARM::t2EORri; break;
      }
      break;
    }
    }
  }
  }

  unsigned OpIdx = Commute ? 2 : 1;
  unsigned Reg1 = UseMI->getOperand(OpIdx).getReg();
  bool isKill = UseMI->getOperand(OpIdx).isKill();
  unsigned NewReg = MRI->createVirtualRegister(MRI->getRegClass(Reg));
  AddDefaultCC(AddDefaultPred(BuildMI(*UseMI->getParent(),
                                      *UseMI, UseMI->getDebugLoc(),
                                      get(NewUseOpc), NewReg)
                              .addReg(Reg1, getKillRegState(isKill))
                              .addImm(SOImmValV1)));
  UseMI->setDesc(get(NewUseOpc));
  UseMI->getOperand(1).setReg(NewReg);
  UseMI->getOperand(1).setIsKill();
  UseMI->getOperand(2).ChangeToImmediate(SOImmValV2);
  DefMI->eraseFromParent();
  return true;
}

unsigned
ARMBaseInstrInfo::getNumMicroOps(const InstrItineraryData *ItinData,
                                 const MachineInstr *MI) const {
  if (!ItinData || ItinData->isEmpty())
    return 1;

  const TargetInstrDesc &Desc = MI->getDesc();
  unsigned Class = Desc.getSchedClass();
  unsigned UOps = ItinData->Itineraries[Class].NumMicroOps;
  if (UOps)
    return UOps;

  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default:
    llvm_unreachable("Unexpected multi-uops instruction!");
    break;
  case ARM::VLDMQIA:
  case ARM::VLDMQDB:
  case ARM::VSTMQIA:
  case ARM::VSTMQDB:
    return 2;

  // The number of uOps for load / store multiple are determined by the number
  // registers.
  //
  // On Cortex-A8, each pair of register loads / stores can be scheduled on the
  // same cycle. The scheduling for the first load / store must be done
  // separately by assuming the the address is not 64-bit aligned.
  //
  // On Cortex-A9, the formula is simply (#reg / 2) + (#reg % 2). If the address
  // is not 64-bit aligned, then AGU would take an extra cycle.  For VFP / NEON
  // load / store multiple, the formula is (#reg / 2) + (#reg % 2) + 1.
  case ARM::VLDMDIA:
  case ARM::VLDMDDB:
  case ARM::VLDMDIA_UPD:
  case ARM::VLDMDDB_UPD:
  case ARM::VLDMSIA:
  case ARM::VLDMSDB:
  case ARM::VLDMSIA_UPD:
  case ARM::VLDMSDB_UPD:
  case ARM::VSTMDIA:
  case ARM::VSTMDDB:
  case ARM::VSTMDIA_UPD:
  case ARM::VSTMDDB_UPD:
  case ARM::VSTMSIA:
  case ARM::VSTMSDB:
  case ARM::VSTMSIA_UPD:
  case ARM::VSTMSDB_UPD: {
    unsigned NumRegs = MI->getNumOperands() - Desc.getNumOperands();
    return (NumRegs / 2) + (NumRegs % 2) + 1;
  }

  case ARM::LDMIA_RET:
  case ARM::LDMIA:
  case ARM::LDMDA:
  case ARM::LDMDB:
  case ARM::LDMIB:
  case ARM::LDMIA_UPD:
  case ARM::LDMDA_UPD:
  case ARM::LDMDB_UPD:
  case ARM::LDMIB_UPD:
  case ARM::STMIA:
  case ARM::STMDA:
  case ARM::STMDB:
  case ARM::STMIB:
  case ARM::STMIA_UPD:
  case ARM::STMDA_UPD:
  case ARM::STMDB_UPD:
  case ARM::STMIB_UPD:
  case ARM::tLDMIA:
  case ARM::tLDMIA_UPD:
  case ARM::tSTMIA:
  case ARM::tSTMIA_UPD:
  case ARM::tPOP_RET:
  case ARM::tPOP:
  case ARM::tPUSH:
  case ARM::t2LDMIA_RET:
  case ARM::t2LDMIA:
  case ARM::t2LDMDB:
  case ARM::t2LDMIA_UPD:
  case ARM::t2LDMDB_UPD:
  case ARM::t2STMIA:
  case ARM::t2STMDB:
  case ARM::t2STMIA_UPD:
  case ARM::t2STMDB_UPD: {
    unsigned NumRegs = MI->getNumOperands() - Desc.getNumOperands() + 1;
    if (Subtarget.isCortexA8()) {
      if (NumRegs < 4)
        return 2;
      // 4 registers would be issued: 2, 2.
      // 5 registers would be issued: 2, 2, 1.
      UOps = (NumRegs / 2);
      if (NumRegs % 2)
        ++UOps;
      return UOps;
    } else if (Subtarget.isCortexA9()) {
      UOps = (NumRegs / 2);
      // If there are odd number of registers or if it's not 64-bit aligned,
      // then it takes an extra AGU (Address Generation Unit) cycle.
      if ((NumRegs % 2) ||
          !MI->hasOneMemOperand() ||
          (*MI->memoperands_begin())->getAlignment() < 8)
        ++UOps;
      return UOps;
    } else {
      // Assume the worst.
      return NumRegs;
    }
  }
  }
}

int
ARMBaseInstrInfo::getVLDMDefCycle(const InstrItineraryData *ItinData,
                                  const TargetInstrDesc &DefTID,
                                  unsigned DefClass,
                                  unsigned DefIdx, unsigned DefAlign) const {
  int RegNo = (int)(DefIdx+1) - DefTID.getNumOperands() + 1;
  if (RegNo <= 0)
    // Def is the address writeback.
    return ItinData->getOperandCycle(DefClass, DefIdx);

  int DefCycle;
  if (Subtarget.isCortexA8()) {
    // (regno / 2) + (regno % 2) + 1
    DefCycle = RegNo / 2 + 1;
    if (RegNo % 2)
      ++DefCycle;
  } else if (Subtarget.isCortexA9()) {
    DefCycle = RegNo;
    bool isSLoad = false;

    switch (DefTID.getOpcode()) {
    default: break;
    case ARM::VLDMSIA:
    case ARM::VLDMSDB:
    case ARM::VLDMSIA_UPD:
    case ARM::VLDMSDB_UPD:
      isSLoad = true;
      break;
    }

    // If there are odd number of 'S' registers or if it's not 64-bit aligned,
    // then it takes an extra cycle.
    if ((isSLoad && (RegNo % 2)) || DefAlign < 8)
      ++DefCycle;
  } else {
    // Assume the worst.
    DefCycle = RegNo + 2;
  }

  return DefCycle;
}

int
ARMBaseInstrInfo::getLDMDefCycle(const InstrItineraryData *ItinData,
                                 const TargetInstrDesc &DefTID,
                                 unsigned DefClass,
                                 unsigned DefIdx, unsigned DefAlign) const {
  int RegNo = (int)(DefIdx+1) - DefTID.getNumOperands() + 1;
  if (RegNo <= 0)
    // Def is the address writeback.
    return ItinData->getOperandCycle(DefClass, DefIdx);

  int DefCycle;
  if (Subtarget.isCortexA8()) {
    // 4 registers would be issued: 1, 2, 1.
    // 5 registers would be issued: 1, 2, 2.
    DefCycle = RegNo / 2;
    if (DefCycle < 1)
      DefCycle = 1;
    // Result latency is issue cycle + 2: E2.
    DefCycle += 2;
  } else if (Subtarget.isCortexA9()) {
    DefCycle = (RegNo / 2);
    // If there are odd number of registers or if it's not 64-bit aligned,
    // then it takes an extra AGU (Address Generation Unit) cycle.
    if ((RegNo % 2) || DefAlign < 8)
      ++DefCycle;
    // Result latency is AGU cycles + 2.
    DefCycle += 2;
  } else {
    // Assume the worst.
    DefCycle = RegNo + 2;
  }

  return DefCycle;
}

int
ARMBaseInstrInfo::getVSTMUseCycle(const InstrItineraryData *ItinData,
                                  const TargetInstrDesc &UseTID,
                                  unsigned UseClass,
                                  unsigned UseIdx, unsigned UseAlign) const {
  int RegNo = (int)(UseIdx+1) - UseTID.getNumOperands() + 1;
  if (RegNo <= 0)
    return ItinData->getOperandCycle(UseClass, UseIdx);

  int UseCycle;
  if (Subtarget.isCortexA8()) {
    // (regno / 2) + (regno % 2) + 1
    UseCycle = RegNo / 2 + 1;
    if (RegNo % 2)
      ++UseCycle;
  } else if (Subtarget.isCortexA9()) {
    UseCycle = RegNo;
    bool isSStore = false;

    switch (UseTID.getOpcode()) {
    default: break;
    case ARM::VSTMSIA:
    case ARM::VSTMSDB:
    case ARM::VSTMSIA_UPD:
    case ARM::VSTMSDB_UPD:
      isSStore = true;
      break;
    }

    // If there are odd number of 'S' registers or if it's not 64-bit aligned,
    // then it takes an extra cycle.
    if ((isSStore && (RegNo % 2)) || UseAlign < 8)
      ++UseCycle;
  } else {
    // Assume the worst.
    UseCycle = RegNo + 2;
  }

  return UseCycle;
}

int
ARMBaseInstrInfo::getSTMUseCycle(const InstrItineraryData *ItinData,
                                 const TargetInstrDesc &UseTID,
                                 unsigned UseClass,
                                 unsigned UseIdx, unsigned UseAlign) const {
  int RegNo = (int)(UseIdx+1) - UseTID.getNumOperands() + 1;
  if (RegNo <= 0)
    return ItinData->getOperandCycle(UseClass, UseIdx);

  int UseCycle;
  if (Subtarget.isCortexA8()) {
    UseCycle = RegNo / 2;
    if (UseCycle < 2)
      UseCycle = 2;
    // Read in E3.
    UseCycle += 2;
  } else if (Subtarget.isCortexA9()) {
    UseCycle = (RegNo / 2);
    // If there are odd number of registers or if it's not 64-bit aligned,
    // then it takes an extra AGU (Address Generation Unit) cycle.
    if ((RegNo % 2) || UseAlign < 8)
      ++UseCycle;
  } else {
    // Assume the worst.
    UseCycle = 1;
  }
  return UseCycle;
}

int
ARMBaseInstrInfo::getOperandLatency(const InstrItineraryData *ItinData,
                                    const TargetInstrDesc &DefTID,
                                    unsigned DefIdx, unsigned DefAlign,
                                    const TargetInstrDesc &UseTID,
                                    unsigned UseIdx, unsigned UseAlign) const {
  unsigned DefClass = DefTID.getSchedClass();
  unsigned UseClass = UseTID.getSchedClass();

  if (DefIdx < DefTID.getNumDefs() && UseIdx < UseTID.getNumOperands())
    return ItinData->getOperandLatency(DefClass, DefIdx, UseClass, UseIdx);

  // This may be a def / use of a variable_ops instruction, the operand
  // latency might be determinable dynamically. Let the target try to
  // figure it out.
  int DefCycle = -1;
  bool LdmBypass = false;
  switch (DefTID.getOpcode()) {
  default:
    DefCycle = ItinData->getOperandCycle(DefClass, DefIdx);
    break;

  case ARM::VLDMDIA:
  case ARM::VLDMDDB:
  case ARM::VLDMDIA_UPD:
  case ARM::VLDMDDB_UPD:
  case ARM::VLDMSIA:
  case ARM::VLDMSDB:
  case ARM::VLDMSIA_UPD:
  case ARM::VLDMSDB_UPD:
    DefCycle = getVLDMDefCycle(ItinData, DefTID, DefClass, DefIdx, DefAlign);
    break;

  case ARM::LDMIA_RET:
  case ARM::LDMIA:
  case ARM::LDMDA:
  case ARM::LDMDB:
  case ARM::LDMIB:
  case ARM::LDMIA_UPD:
  case ARM::LDMDA_UPD:
  case ARM::LDMDB_UPD:
  case ARM::LDMIB_UPD:
  case ARM::tLDMIA:
  case ARM::tLDMIA_UPD:
  case ARM::tPUSH:
  case ARM::t2LDMIA_RET:
  case ARM::t2LDMIA:
  case ARM::t2LDMDB:
  case ARM::t2LDMIA_UPD:
  case ARM::t2LDMDB_UPD:
    LdmBypass = 1;
    DefCycle = getLDMDefCycle(ItinData, DefTID, DefClass, DefIdx, DefAlign);
    break;
  }

  if (DefCycle == -1)
    // We can't seem to determine the result latency of the def, assume it's 2.
    DefCycle = 2;

  int UseCycle = -1;
  switch (UseTID.getOpcode()) {
  default:
    UseCycle = ItinData->getOperandCycle(UseClass, UseIdx);
    break;

  case ARM::VSTMDIA:
  case ARM::VSTMDDB:
  case ARM::VSTMDIA_UPD:
  case ARM::VSTMDDB_UPD:
  case ARM::VSTMSIA:
  case ARM::VSTMSDB:
  case ARM::VSTMSIA_UPD:
  case ARM::VSTMSDB_UPD:
    UseCycle = getVSTMUseCycle(ItinData, UseTID, UseClass, UseIdx, UseAlign);
    break;

  case ARM::STMIA:
  case ARM::STMDA:
  case ARM::STMDB:
  case ARM::STMIB:
  case ARM::STMIA_UPD:
  case ARM::STMDA_UPD:
  case ARM::STMDB_UPD:
  case ARM::STMIB_UPD:
  case ARM::tSTMIA:
  case ARM::tSTMIA_UPD:
  case ARM::tPOP_RET:
  case ARM::tPOP:
  case ARM::t2STMIA:
  case ARM::t2STMDB:
  case ARM::t2STMIA_UPD:
  case ARM::t2STMDB_UPD:
    UseCycle = getSTMUseCycle(ItinData, UseTID, UseClass, UseIdx, UseAlign);
    break;
  }

  if (UseCycle == -1)
    // Assume it's read in the first stage.
    UseCycle = 1;

  UseCycle = DefCycle - UseCycle + 1;
  if (UseCycle > 0) {
    if (LdmBypass) {
      // It's a variable_ops instruction so we can't use DefIdx here. Just use
      // first def operand.
      if (ItinData->hasPipelineForwarding(DefClass, DefTID.getNumOperands()-1,
                                          UseClass, UseIdx))
        --UseCycle;
    } else if (ItinData->hasPipelineForwarding(DefClass, DefIdx,
                                               UseClass, UseIdx)) {
      --UseCycle;
    }
  }

  return UseCycle;
}

int
ARMBaseInstrInfo::getOperandLatency(const InstrItineraryData *ItinData,
                             const MachineInstr *DefMI, unsigned DefIdx,
                             const MachineInstr *UseMI, unsigned UseIdx) const {
  if (DefMI->isCopyLike() || DefMI->isInsertSubreg() ||
      DefMI->isRegSequence() || DefMI->isImplicitDef())
    return 1;

  const TargetInstrDesc &DefTID = DefMI->getDesc();
  if (!ItinData || ItinData->isEmpty())
    return DefTID.mayLoad() ? 3 : 1;

  const TargetInstrDesc &UseTID = UseMI->getDesc();
  const MachineOperand &DefMO = DefMI->getOperand(DefIdx);
  if (DefMO.getReg() == ARM::CPSR) {
    if (DefMI->getOpcode() == ARM::FMSTAT) {
      // fpscr -> cpsr stalls over 20 cycles on A8 (and earlier?)
      return Subtarget.isCortexA9() ? 1 : 20;
    }

    // CPSR set and branch can be paired in the same cycle.
    if (UseTID.isBranch())
      return 0;
  }

  unsigned DefAlign = DefMI->hasOneMemOperand()
    ? (*DefMI->memoperands_begin())->getAlignment() : 0;
  unsigned UseAlign = UseMI->hasOneMemOperand()
    ? (*UseMI->memoperands_begin())->getAlignment() : 0;
  int Latency = getOperandLatency(ItinData, DefTID, DefIdx, DefAlign,
                                  UseTID, UseIdx, UseAlign);

  if (Latency > 1 &&
      (Subtarget.isCortexA8() || Subtarget.isCortexA9())) {
    // FIXME: Shifter op hack: no shift (i.e. [r +/- r]) or [r + r << 2]
    // variants are one cycle cheaper.
    switch (DefTID.getOpcode()) {
    default: break;
    case ARM::LDRrs:
    case ARM::LDRBrs: {
      unsigned ShOpVal = DefMI->getOperand(3).getImm();
      unsigned ShImm = ARM_AM::getAM2Offset(ShOpVal);
      if (ShImm == 0 ||
          (ShImm == 2 && ARM_AM::getAM2ShiftOpc(ShOpVal) == ARM_AM::lsl))
        --Latency;
      break;
    }
    case ARM::t2LDRs:
    case ARM::t2LDRBs:
    case ARM::t2LDRHs:
    case ARM::t2LDRSHs: {
      // Thumb2 mode: lsl only.
      unsigned ShAmt = DefMI->getOperand(3).getImm();
      if (ShAmt == 0 || ShAmt == 2)
        --Latency;
      break;
    }
    }
  }

  return Latency;
}

int
ARMBaseInstrInfo::getOperandLatency(const InstrItineraryData *ItinData,
                                    SDNode *DefNode, unsigned DefIdx,
                                    SDNode *UseNode, unsigned UseIdx) const {
  if (!DefNode->isMachineOpcode())
    return 1;

  const TargetInstrDesc &DefTID = get(DefNode->getMachineOpcode());

  if (isZeroCost(DefTID.Opcode))
    return 0;

  if (!ItinData || ItinData->isEmpty())
    return DefTID.mayLoad() ? 3 : 1;

  if (!UseNode->isMachineOpcode()) {
    int Latency = ItinData->getOperandCycle(DefTID.getSchedClass(), DefIdx);
    if (Subtarget.isCortexA9())
      return Latency <= 2 ? 1 : Latency - 1;
    else
      return Latency <= 3 ? 1 : Latency - 2;
  }

  const TargetInstrDesc &UseTID = get(UseNode->getMachineOpcode());
  const MachineSDNode *DefMN = dyn_cast<MachineSDNode>(DefNode);
  unsigned DefAlign = !DefMN->memoperands_empty()
    ? (*DefMN->memoperands_begin())->getAlignment() : 0;
  const MachineSDNode *UseMN = dyn_cast<MachineSDNode>(UseNode);
  unsigned UseAlign = !UseMN->memoperands_empty()
    ? (*UseMN->memoperands_begin())->getAlignment() : 0;
  int Latency = getOperandLatency(ItinData, DefTID, DefIdx, DefAlign,
                                  UseTID, UseIdx, UseAlign);

  if (Latency > 1 &&
      (Subtarget.isCortexA8() || Subtarget.isCortexA9())) {
    // FIXME: Shifter op hack: no shift (i.e. [r +/- r]) or [r + r << 2]
    // variants are one cycle cheaper.
    switch (DefTID.getOpcode()) {
    default: break;
    case ARM::LDRrs:
    case ARM::LDRBrs: {
      unsigned ShOpVal =
        cast<ConstantSDNode>(DefNode->getOperand(2))->getZExtValue();
      unsigned ShImm = ARM_AM::getAM2Offset(ShOpVal);
      if (ShImm == 0 ||
          (ShImm == 2 && ARM_AM::getAM2ShiftOpc(ShOpVal) == ARM_AM::lsl))
        --Latency;
      break;
    }
    case ARM::t2LDRs:
    case ARM::t2LDRBs:
    case ARM::t2LDRHs:
    case ARM::t2LDRSHs: {
      // Thumb2 mode: lsl only.
      unsigned ShAmt =
        cast<ConstantSDNode>(DefNode->getOperand(2))->getZExtValue();
      if (ShAmt == 0 || ShAmt == 2)
        --Latency;
      break;
    }
    }
  }

  return Latency;
}

int ARMBaseInstrInfo::getInstrLatency(const InstrItineraryData *ItinData,
                                      const MachineInstr *MI,
                                      unsigned *PredCost) const {
  if (MI->isCopyLike() || MI->isInsertSubreg() ||
      MI->isRegSequence() || MI->isImplicitDef())
    return 1;

  if (!ItinData || ItinData->isEmpty())
    return 1;

  const TargetInstrDesc &TID = MI->getDesc();
  unsigned Class = TID.getSchedClass();
  unsigned UOps = ItinData->Itineraries[Class].NumMicroOps;
  if (PredCost && TID.hasImplicitDefOfPhysReg(ARM::CPSR))
    // When predicated, CPSR is an additional source operand for CPSR updating
    // instructions, this apparently increases their latencies.
    *PredCost = 1;
  if (UOps)
    return ItinData->getStageLatency(Class);
  return getNumMicroOps(ItinData, MI);
}

int ARMBaseInstrInfo::getInstrLatency(const InstrItineraryData *ItinData,
                                      SDNode *Node) const {
  if (!Node->isMachineOpcode())
    return 1;

  if (!ItinData || ItinData->isEmpty())
    return 1;

  unsigned Opcode = Node->getMachineOpcode();
  switch (Opcode) {
  default:
    return ItinData->getStageLatency(get(Opcode).getSchedClass());
  case ARM::VLDMQIA:
  case ARM::VLDMQDB:
  case ARM::VSTMQIA:
  case ARM::VSTMQDB:
    return 2;
  }
}

bool ARMBaseInstrInfo::
hasHighOperandLatency(const InstrItineraryData *ItinData,
                      const MachineRegisterInfo *MRI,
                      const MachineInstr *DefMI, unsigned DefIdx,
                      const MachineInstr *UseMI, unsigned UseIdx) const {
  unsigned DDomain = DefMI->getDesc().TSFlags & ARMII::DomainMask;
  unsigned UDomain = UseMI->getDesc().TSFlags & ARMII::DomainMask;
  if (Subtarget.isCortexA8() &&
      (DDomain == ARMII::DomainVFP || UDomain == ARMII::DomainVFP))
    // CortexA8 VFP instructions are not pipelined.
    return true;

  // Hoist VFP / NEON instructions with 4 or higher latency.
  int Latency = getOperandLatency(ItinData, DefMI, DefIdx, UseMI, UseIdx);
  if (Latency <= 3)
    return false;
  return DDomain == ARMII::DomainVFP || DDomain == ARMII::DomainNEON ||
         UDomain == ARMII::DomainVFP || UDomain == ARMII::DomainNEON;
}

bool ARMBaseInstrInfo::
hasLowDefLatency(const InstrItineraryData *ItinData,
                 const MachineInstr *DefMI, unsigned DefIdx) const {
  if (!ItinData || ItinData->isEmpty())
    return false;

  unsigned DDomain = DefMI->getDesc().TSFlags & ARMII::DomainMask;
  if (DDomain == ARMII::DomainGeneral) {
    unsigned DefClass = DefMI->getDesc().getSchedClass();
    int DefCycle = ItinData->getOperandCycle(DefClass, DefIdx);
    return (DefCycle != -1 && DefCycle <= 2);
  }
  return false;
}

bool
ARMBaseInstrInfo::isFpMLxInstruction(unsigned Opcode, unsigned &MulOpc,
                                     unsigned &AddSubOpc,
                                     bool &NegAcc, bool &HasLane) const {
  DenseMap<unsigned, unsigned>::const_iterator I = MLxEntryMap.find(Opcode);
  if (I == MLxEntryMap.end())
    return false;

  const ARM_MLxEntry &Entry = ARM_MLxTable[I->second];
  MulOpc = Entry.MulOpc;
  AddSubOpc = Entry.AddSubOpc;
  NegAcc = Entry.NegAcc;
  HasLane = Entry.HasLane;
  return true;
}
