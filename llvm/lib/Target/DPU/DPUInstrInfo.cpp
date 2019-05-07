//===-- DPUInstrInfo.cpp - DPU Instruction Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the DPU implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "DPUInstrInfo.h"

#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/StackMaps.h"

//#define GET_INSTRINFO_NAMED_OPS // For getNamedOperandIdx() function
#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRINFO_ENUM

#include "DPUGenInstrInfo.inc"

#define GET_REGINFO_ENUM

#include "DPUCondCodes.h"
#include "DPUGenRegisterInfo.inc"
#include "DPUISelLowering.h"

#define DEBUG_TYPE "asm-printer"

using namespace llvm;

// Specify here that ADJCALLSTACKDOWN/UP are respectively the CF setup/destroy
// opcodes. That way (see PrologEpilogInserter::replaceFrameIndices) we give an
// opportunity to adjust the stack pointer upon function call, via
// DPUFrameLowering::eliminateCallFramePseudoInstr.
DPUInstrInfo::DPUInstrInfo()
    : DPUGenInstrInfo(DPU::ADJCALLSTACKDOWN, DPU::ADJCALLSTACKUP), RI() {}

void DPUInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned SrcReg, bool IsKill, int FI,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  DebugLoc DL = (I != MBB.end()) ? I->getDebugLoc() : DebugLoc();
  unsigned Opcode = (RC == &DPU::GP_REGRegClass) ? DPU::SWrir : DPU::SDrir;

  LLVM_DEBUG({
    dbgs() << "DPU/Instr - storeRegToStackSlot DestReg="
           << std::to_string(SrcReg) << " Opcode= " << std::to_string(Opcode)
           << " BB=\n";
    MBB.dump();
    dbgs() << "!!!! FI = " << std::to_string(FI) << "\n";
  });

  // At this level, we COULD generate a STORErm directly with the frame
  // register, as given by the static method DPURegisterInfo::getFrameRegister
  // However, the generated instruction should go through eliminateFrameIndex
  // then, so we can inject FI.
  BuildMI(MBB, I, DL, get(Opcode))
      .addFrameIndex(FI)
      .addImm(0)
      .addReg(SrcReg, getKillRegState(IsKill))
      .setMIFlag(MachineInstr::MIFlag::FrameSetup);
}

void DPUInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned DestReg, int FI,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();
  unsigned Opcode = (RC == &DPU::GP_REGRegClass) ? DPU::LWrri : DPU::LDrri;
  LLVM_DEBUG({
    dbgs() << "DPU/Instr - loadRegFromStackSlot DestReg="
           << std::to_string(DestReg) << " Opcode= " << std::to_string(Opcode)
           << " BB=\n";
    MBB.dump();
    dbgs() << "!!!! FI = " << std::to_string(FI) << "\n";
  });

  BuildMI(MBB, I, DL, get(Opcode), DestReg).addFrameIndex(FI).addImm(0);
}

bool DPUInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction *MF = MBB.getParent();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

  // todo __sys_thread_nanostack_entry_0 and __sys_thread_nanostack_entry_1
  // should have abstract representations
  switch (MI.getDesc().getOpcode()) {
  default:
    return false;
  case DPU::RETi:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::JUMPr)).addReg(DPU::RADD);
    break;
  case DPU::CALLi:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::CALLri))
        .addReg(DPU::RADD)
        .add(MI.getOperand(0));
    break;
  case DPU::CALLr:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::CALLrr))
        .addReg(DPU::RADD)
        .add(MI.getOperand(0));
    break;
  case DPU::INTRINSIC_CALL:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::MOVErr))
        .addReg(DPU::R18)
        .addReg(DPU::R0);
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::MOVErr))
        .addReg(DPU::R19)
        .addReg(DPU::R1);
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::MOVErr))
        .addReg(DPU::R17)
        .addReg(DPU::R2);
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::CALLri))
        .addReg(DPU::RADD)
        .add(MI.getOperand(0));
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::MOVErr))
        .addReg(DPU::RVAL)
        .addReg(DPU::R18);
    break;
  case DPU::ADJUST_STACK_BEFORE_CALL:
  case DPU::ADJUST_STACK_AFTER_CALL:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::ADDrri))
        .addReg(DPU::STKP)
        .addReg(DPU::STKP)
        .addImm(MI.getOperand(0).getImm());
    break;
  case DPU::PUSH_STACK_POINTER:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::SWrir))
        .addReg(DPU::STKP)
        .addImm(MI.getOperand(0).getImm() - 4)
        .addReg(DPU::STKP);
    break;
  case DPU::STAIN_STACK:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::SWrir))
        .addReg(DPU::STKP)
        .addImm(MI.getOperand(0).getImm() + 4)
        .addImm(-1);
    break;
  case DPU::BSWAP16:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::SHrir))
        .addReg(DPU::ID4)
        .addExternalSymbol("__sys_thread_nanostack_entry_0")
        .add(MI.getOperand(1));
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::LHUerri))
        .add(MI.getOperand(0))
        .addImm(1)
        .addReg(DPU::ID4)
        .addExternalSymbol("__sys_thread_nanostack_entry_0");
    break;
  case DPU::BSWAP32:
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::SWrir))
        .addReg(DPU::ID4)
        .addExternalSymbol("__sys_thread_nanostack_entry_0")
        .add(MI.getOperand(1));
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::LWerri))
        .add(MI.getOperand(0))
        .addImm(1)
        .addReg(DPU::ID4)
        .addExternalSymbol("__sys_thread_nanostack_entry_0");
    break;
  case DPU::BSWAP64: {
    unsigned int LsbDestReg =
        TRI->getSubReg(MI.getOperand(0).getReg(), DPU::sub_32bit);
    unsigned int MsbDestReg =
        TRI->getSubReg(MI.getOperand(0).getReg(), DPU::sub_32bit_hi);
    unsigned int LsbSrcReg =
        TRI->getSubReg(MI.getOperand(1).getReg(), DPU::sub_32bit);
    unsigned int MsbSrcReg =
        TRI->getSubReg(MI.getOperand(1).getReg(), DPU::sub_32bit_hi);

    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::SWrir))
        .addReg(DPU::ID4)
        .addExternalSymbol("__sys_thread_nanostack_entry_0")
        .addReg(LsbSrcReg);
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::SWrir))
        .addReg(DPU::ID4)
        .addExternalSymbol("__sys_thread_nanostack_entry_1")
        .addReg(MsbSrcReg);
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::LWerri))
        .addReg(LsbDestReg)
        .addImm(1)
        .addReg(DPU::ID4)
        .addExternalSymbol("__sys_thread_nanostack_entry_1");
    BuildMI(MBB, MI, MI.getDebugLoc(), get(DPU::LWerri))
        .addReg(MsbDestReg)
        .addImm(1)
        .addReg(DPU::ID4)
        .addExternalSymbol("__sys_thread_nanostack_entry_0");
    break;
  }
  }

  MBB.erase(MI);
  return true;
}

void DPUInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I,
                               const DebugLoc &DL, unsigned DestReg,
                               unsigned SrcReg, bool KillSrc) const {
  if (DPU::GP_REGRegClass.contains(DestReg) &&
      DPU::OP_REGRegClass.contains(SrcReg)) {
    LLVM_DEBUG(dbgs() << "DPU/Instr - copyPhysReg from src=" << SrcReg
                      << " kill= " << KillSrc << " to dest=" << DestReg
                      << "\n");
    BuildMI(MBB, I, DL, get(DPU::MOVErr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
  } else if (DPU::GP64_REGRegClass.contains(DestReg, SrcReg)) {
    LLVM_DEBUG(dbgs() << "DPU/Instr - copyPhysReg from src=" << SrcReg
                      << " kill= " << KillSrc << " to dest=" << DestReg
                      << "\n");
    BuildMI(MBB, I, DL, get(DPU::MOVDrr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
  } else if (DPU::GP64_REGRegClass.contains(SrcReg) &&
             DPU::GP_REGRegClass.contains(DestReg)) {
    // Truncating 64 bits to 32... There's a macro for that.
    LLVM_DEBUG(dbgs() << "DPU/Instr - copyPhysReg from src=" << SrcReg
                      << " kill= " << KillSrc << " to dest=" << DestReg
                      << "\n");
    BuildMI(MBB, I, DL, get(DPU::EXTRACT_SUBREG), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(DPU::sub_32bit);
  } else if (DPU::GP_REGRegClass.contains(SrcReg) &&
             DPU::GP64_REGRegClass.contains(DestReg)) {
    // Expanding 32 bits to 64... There's an instruction for that.
    LLVM_DEBUG(dbgs() << "DPU/Instr - copyPhysReg from src=" << SrcReg
                      << " kill= " << KillSrc << " to dest=" << DestReg
                      << "\n");
    BuildMI(MBB, I, DL, get(DPU::MOVE_Srr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
  } else {
    llvm_unreachable("Impossible reg-to-reg copy");
  }
}

bool DPUInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  unsigned Opc = Cond[0].getImm();

  switch (Opc) {
  case DPU::Jcc:
  case DPU::Jcci:
  case DPU::Jcc64:
    Cond[1].setImm(ISD::getSetCCInverse(ISD::CondCode(Cond[1].getImm()), true));
    return false;
  default:
    break;
  }

  return true;
}

static void
fetchUnconditionalBranchInfo(MachineInstr *Inst,
                             unsigned &targetBasicBlockOperandIndex) {
  switch (Inst->getOpcode()) {
  case DPU::JUMPi:
    targetBasicBlockOperandIndex = 0;
    break;
  default:
    assert(false && "invalid opcode for unconditional branch");
  }
}

static void fetchConditionalBranchInfo(MachineInstr *Inst,
                                       unsigned &targetBasicBlockOperandIndex,
                                       SmallVectorImpl<MachineOperand> &Cond) {
  unsigned Opc = Inst->getOpcode();
  Cond.push_back(MachineOperand::CreateImm(Opc));

  unsigned int NumOp = Inst->getNumExplicitOperands();

  for (unsigned int eachOperandIndex = 0; eachOperandIndex < NumOp;
       eachOperandIndex++) {
    MachineOperand &operand = Inst->getOperand(eachOperandIndex);
    if (operand.isMBB()) {
      targetBasicBlockOperandIndex = eachOperandIndex;
    } else {
      Cond.push_back(operand);
    }
  }
}

static inline bool isAnalyzableBranch(MachineInstr *Inst) {
  return Inst->isBranch() && !Inst->isIndirectBranch();
}

bool DPUInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugValue()) {
    ++I;
  }

  // If this block ends with no branches (it just falls through to its succ),
  // Leave TBB/FBB null.
  if (I == REnd || !isUnpredicatedTerminator(*I)) {
    TBB = FBB = nullptr;
    return false;
  }

  MachineInstr *LastInst = &*I;
  MachineInstr *SecondLastInst = nullptr;

  // If not an analyzable branch (e.g., indirect jump), just leave.
  if (!isAnalyzableBranch(LastInst)) {
    return true;
  }

  if (++I != REnd) {
    SecondLastInst = &*I;
    if (!isUnpredicatedTerminator(*SecondLastInst) ||
        !SecondLastInst->isBranch()) {
      // If not a branch, reset to nullptr.
      SecondLastInst = nullptr;
    } else if (!isAnalyzableBranch(SecondLastInst)) {
      // If not an analyzable branch, just leave.
      return true;
    }
  }

  // If there is only one terminator instruction, process it.
  if (SecondLastInst == nullptr) {
    // Unconditional branch.
    if (LastInst->isUnconditionalBranch()) {
      unsigned int TBBOpIdx;
      fetchUnconditionalBranchInfo(LastInst, TBBOpIdx);
      FBB = nullptr;

      // Delete the Branch if it's equivalent to a fall-through.
      if (AllowModify &&
          MBB.isLayoutSuccessor(LastInst->getOperand(TBBOpIdx).getMBB())) {
        TBB = nullptr;
        LastInst->eraseFromParent();
        return false;
      }

      TBB = LastInst->getOperand(TBBOpIdx).getMBB();
      return false;
    }

    // Conditional branch
    if (LastInst->isConditionalBranch()) {
      unsigned int TBBOpIdx;
      fetchConditionalBranchInfo(LastInst, TBBOpIdx, Cond);
      TBB = LastInst->getOperand(TBBOpIdx).getMBB();
      return false;
    }

    // Unknown branch type
    return true;
  }

  // If we reached here, there are two branches.
  // If there are three terminators, we don't know what sort of block this is.
  if (++I != REnd && isUnpredicatedTerminator(*I)) {
    return true;
  }

  // If second to last instruction is an unconditional branch,
  // analyze it and remove the last instruction.
  if (SecondLastInst->isUnconditionalBranch()) {
    // Return if the last instruction cannot be removed.
    if (!AllowModify) {
      return true;
    }
    unsigned int TBBOpIdx;
    fetchUnconditionalBranchInfo(SecondLastInst, TBBOpIdx);

    TBB = SecondLastInst->getOperand(TBBOpIdx).getMBB();
    LastInst->eraseFromParent();
    return false;
  }

  if (SecondLastInst->isConditionalBranch()) {
    // Conditional branch followed by an unconditional branch.
    // The last one must be unconditional.
    if (!LastInst->isUnconditionalBranch()) {
      return true;
    }
    unsigned int TBBOpIdx;
    unsigned int FTBBOpIdx;

    fetchUnconditionalBranchInfo(LastInst, FTBBOpIdx);
    fetchConditionalBranchInfo(SecondLastInst, TBBOpIdx, Cond);
    TBB = SecondLastInst->getOperand(TBBOpIdx).getMBB();
    FBB = LastInst->getOperand(FTBBOpIdx).getMBB();

    return false;
  }

  // Unknown branch type
  return true;
}

unsigned DPUInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                    int *BytesRemoved) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;
    if (!I->isBranch())
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  // DPU instruction size is constant, meaning that bytes removed is equivalent
  // to instructions removed
  if (BytesRemoved)
    *BytesRemoved = Count;
  return Count;
}

void DPUInstrInfo::buildConditionalBranch(MachineBasicBlock &MBB,
                                          MachineBasicBlock *TBB, DebugLoc DL,
                                          ArrayRef<MachineOperand> Cond) const {
  MachineInstrBuilder MIB;

  unsigned Opc = Cond[0].getImm();

  MIB = BuildMI(&MBB, DL, get(Opc));

  for (unsigned i = 1; i < Cond.size(); ++i) {
    if (Cond[i].isReg())
      MIB.addReg(Cond[i].getReg());
    else if (Cond[i].isImm())
      MIB.addImm(Cond[i].getImm());
    else
      assert(false && "Cannot copy operand");
  }

  MIB.addMBB(TBB);
}

unsigned DPUInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    ArrayRef<MachineOperand> Cond,
                                    const DebugLoc &DL, int *BytesAdded) const {
  unsigned nrOfInsertedMachineInstr = 0;
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

  // Unconditional branch
  if (Cond.empty()) {
    // Be sure that this is a valid unconditional branch
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(DPU::JUMPi)).addMBB(TBB);
    nrOfInsertedMachineInstr++;
  } else {
    // Conditional branch
    buildConditionalBranch(MBB, TBB, DL, Cond);
    nrOfInsertedMachineInstr++;

    if (FBB) {
      BuildMI(&MBB, DL, get(DPU::JUMPi)).addMBB(FBB);
      nrOfInsertedMachineInstr++;
    }
  }

  // DPU instruction size is constant, implying that bytes added is equivalent
  // to instructions added.
  if (BytesAdded)
    *BytesAdded = nrOfInsertedMachineInstr;
  return nrOfInsertedMachineInstr;
}
