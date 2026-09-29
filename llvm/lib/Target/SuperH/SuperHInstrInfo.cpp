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
#include "MCTargetDesc/SuperHInstPrinter.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "SuperH.h"
#include "SuperHRegisterInfo.h"
#include "SuperHSubtarget.h"
#include "SuperHTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "sh-instrinfo"

#define GET_INSTRINFO_CTOR_DTOR
#include "SuperHGenInstrInfo.inc"

SuperHInstrInfo::SuperHInstrInfo(const SuperHSubtarget &ST)
    : SuperHGenInstrInfo(ST, RI, SH::ADJCALLSTACKDOWN, SH::ADJCALLSTACKUP),
      STI(ST), RI(ST) {}

// Pin the vtable to this file.
void SuperHInstrInfo::anchor() {}

// Gets whether a given opcode can fill a delay slot.
//
// SuperH does not allow branch instructions of any kind to be situated
// in a delay slot, nor does it allow instructions with delay slots
// to be chained together.
bool SuperHInstrInfo::canFillDelaySlot(unsigned Opcode) const {
  auto Desc = this->get(Opcode);
  return !Desc.hasDelaySlot() && !Desc.isBranch() && !Desc.isCall() &&
         !Desc.isReturn() && !(Desc.TSFlags & 0x1);
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

SHCC::CondCode SuperHInstrInfo::getCondFromBranchOp(unsigned Op) const {
  switch (Op) {
  default:
    return SHCC::COND_INVALID;
  case SH::NOP:
  case SH::BT:
  case SH::BTS:
    return SHCC::COND_T;
  case SH::BF:
  case SH::BFS:
    return SHCC::COND_F;
  }
}

SHCC::CondCode SuperHInstrInfo::getOppositeCondCode(SHCC::CondCode Op) const {
  switch (Op) {
  default:
    return SHCC::COND_INVALID;
  case SHCC::COND_T:
    return SHCC::COND_F;
  case SHCC::COND_F:
    return SHCC::COND_T;
  }
}

const MCInstrDesc &SuperHInstrInfo::getBrCond(SHCC::CondCode CC,
                                              bool delaySlot) const {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case SHCC::COND_T:
    return get(delaySlot ? SH::BTS : SH::BT);
  case SHCC::COND_F:
    return get(delaySlot ? SH::BFS : SH::BF);
  }
}

unsigned SuperHInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();

  switch (Opcode) {
  // A regular instruction
  default: {
    const MCInstrDesc &Desc = get(Opcode);
    return Desc.getSize();
  }
  case SH::CONSTPOOL_ENTRY:
    // If this machine instr is a constant pool entry, 
    // its size is recorded as operand #2.
    return MI.getOperand(2).getImm();
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::DBG_VALUE:
    return 0;
  case TargetOpcode::INLINEASM:
  case TargetOpcode::INLINEASM_BR: {
    // TODO: Add inline ASM support.
    return 0;
  }
  }
}

unsigned SuperHInstrInfo::getInstSizeInBytes(const MCInst &MI) const {
  unsigned Opcode = MI.getOpcode();

  switch (Opcode) {
  // A regular instruction
  default: {
    const MCInstrDesc &Desc = get(Opcode);
    return Desc.getSize();
  }
  case SH::CONSTPOOL_ENTRY:
    // If this machine instr is a constant pool entry, 
    // its size is recorded as operand #2.
    return MI.getOperand(2).getImm();
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::DBG_VALUE:
    return 0;
  case TargetOpcode::INLINEASM:
  case TargetOpcode::INLINEASM_BR: {
    // TODO: Add inline ASM support.
    return 0;
  }
  }
}





//===----------------------------------------------------------------------===//
//                             Register Managment.
//===----------------------------------------------------------------------===//

void SuperHInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  const DebugLoc &DL, Register DestReg,
                                  Register SrcReg, bool KillSrc,
                                  bool RenamableDest, bool RenamableSrc) const {
  // Do nothing, self copy.
  if (SrcReg == DestReg)
    return;

  // Load from MACL
  if (SrcReg == SH::MACLO && SH::GPRRegClass.contains(DestReg)) {
    BuildMI(MBB, MI, DL, get(SH::STSMACL), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Store to MACL
  if (SH::GPRRegClass.contains(SrcReg) && DestReg == SH::MACLO) {
    BuildMI(MBB, MI, DL, get(SH::LDSMACL), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // If the targets are GPR registers, use MOV Rm, Rn.
  if (SH::GPRRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, MI, DL, get(SH::MOV), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  };

  // GPR -> FR32
  if (SH::GPRRegClass.contains(SrcReg) && 
      SH::FR32RegClass.contains(DestReg)) {

    BuildMI(MBB, MI, DL, get(SH::LDSFPUL))
        .addReg(SrcReg, getKillRegState(KillSrc));
    BuildMI(MBB, MI, DL, get(SH::FSTS), DestReg);
    return;
  }

  // FR32 -> GPR
  if (SH::FR32RegClass.contains(SrcReg) && 
      SH::GPRRegClass.contains(DestReg)) {
    
    BuildMI(MBB, MI, DL, get(SH::FLDS))
        .addReg(SrcReg, getKillRegState(KillSrc));
    BuildMI(MBB, MI, DL, get(SH::STSFPUL), DestReg);
    return;
  }

  // FR32 -> FR32
  if (SH::FR32RegClass.contains(SrcReg, DestReg)) {
    BuildMI(MBB, MI, DL, get(SH::FMOV), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Otherwise this is not possible.
  llvm_unreachable("Impossible reg-to-reg copy");
}

//===----------------------------------------------------------------------===//
//                              Stack Frames
//===----------------------------------------------------------------------===//

Register SuperHInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                             int &FrameIndex) const {
  if (MI.getOperand(0).isReg() && MI.getOperand(1).isFI() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }
  return 0;
}

void SuperHInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator II, Register SrcReg,
    bool isKill, int FrameIndex, const TargetRegisterClass *RC, Register VReg,
    MachineInstr::MIFlag Flags) const {
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t ObjectSize = MFI.getObjectSize(FrameIndex);

  LLVM_DEBUG(dbgs() << "Store "
                    << (SrcReg > RI.getNumRegs() ? "VREG" : RI.getName(SrcReg))
                    << " to slot " << FrameIndex << " size=" << ObjectSize
                    << "\n");

  unsigned Opc;

  if (SH::FR32RegClass.contains(SrcReg)) {

    // F32 Registers.
    Opc = SH::MOVF32SF;
  } else {

    // Integer registers.
    switch (ObjectSize) {
    default:
      llvm_unreachable("Cannot store this register into stack slot!");
    case 1:
      Opc = SH::MOVBSF;
      break;
    case 2:
      Opc = SH::MOVWSF;
      break;
    case 4:
      Opc = SH::MOVLSF;

      break;
    }
  }
  BuildMI(MBB, II, DebugLoc(), get(Opc))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FrameIndex)
      .addImm(0);
}

Register SuperHInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                              int &FrameIndex) const {
  if (MI.getOperand(0).isFI() && MI.getOperand(1).isImm() &&
      MI.getOperand(1).getImm() == 0) {
    FrameIndex = MI.getOperand(0).getIndex();
    return MI.getOperand(2).getReg();
  }
  return 0;
}

void SuperHInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator II,
                                           Register DestReg, int FrameIndex,
                                           const TargetRegisterClass *RC,
                                           Register VReg, unsigned SubReg,
                                           MachineInstr::MIFlag Flags) const {
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t ObjectSize = MFI.getObjectSize(FrameIndex);

  LLVM_DEBUG(
      dbgs() << "Load "
             << (DestReg > RI.getNumRegs() ? "VREG" : RI.getName(DestReg))
             << " from slot " << FrameIndex << " size=" << ObjectSize << "\n");

  unsigned Opc;

  if (SH::FR32RegClass.contains(DestReg)) {

    // F32 Registers.
    Opc = SH::MOVF32LF;
  } else {
    switch (ObjectSize) {
    default:
      llvm_unreachable("Cannot load this register from stack slot!");
    case 1:
      Opc = SH::MOVBLF;
      break;
    case 2:
      Opc = SH::MOVWLF;
      break;
    case 4:
      Opc = SH::MOVLLF;
      break;
    }
  }

  BuildMI(MBB, II, DebugLoc(), get(Opc), DestReg)
      .addFrameIndex(FrameIndex)
      .addImm(0);
}




//===----------------------------------------------------------------------===//
//                              Branch Analysis
//===----------------------------------------------------------------------===//

bool SuperHInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *&TBB,
                                    MachineBasicBlock *&FBB,
                                    SmallVectorImpl<MachineOperand> &Cond,
                                    bool AllowModify) const {
#ifdef SH_ENABLE_BRANCH_FOLDING
  auto UncondBranch =
      std::pair<MachineBasicBlock::reverse_iterator, MachineBasicBlock *>{
          MBB.rend(), nullptr};

  // Erase any instructions if allowed at the end of the scope.
  std::vector<std::reference_wrapper<llvm::MachineInstr>> EraseList;
  llvm::scope_exit FinalizeOnReturn([&EraseList] {
    for (auto &Ref : EraseList)
      Ref.get().eraseFromParent();
  });

  for (auto I = MBB.rbegin(); I != MBB.rend(); I = std::next(I)) {
    unsigned Opcode = I->getOpcode();
    if (I->isDebugInstr()) 
      continue;

    // Skip NOPs
    if (Opcode == SH::NOP)
      continue;

    // Working from the bottom, when we see a non-terminator
    // instruction, we're done.
    if (!isUnpredicatedTerminator(*I))
      break;

    // A terminator that isn't a branch can't easily be handled
    // by this analysis.
    if (!I->isBranch())
      return true;

    // Handle unconditional branches.
    if (Opcode == SH::BRA) {
      if (!I->getOperand(0).isMBB())
        return true;

      UncondBranch = {I, I->getOperand(0).getMBB()};

      // TBB is used to indicate the unconditional destination.
      TBB = UncondBranch.second;

      if (!AllowModify)
        continue;

      // If the block has any instructions after a JMP, erase them.
      EraseList.insert(EraseList.begin(), MBB.rbegin(), I);
      Cond.clear();
      FBB = nullptr;

      // Delete the BRA if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = nullptr;
        EraseList.push_back(*I);
        UncondBranch = {MBB.rend(), nullptr};
        continue;
      }
      continue;
    }

    // Handle conditional branches.
    SHCC::CondCode BranchCode = getCondFromBranchOp(Opcode);

    // Can't handle indirect branch.
    if (BranchCode == SHCC::COND_INVALID)
      return true; 

    if (I->getNumOperands() >= 2 && I->getOperand(1).isUndef())
      return true;

    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      if (!I->getOperand(0).isMBB())
        return true;

      MachineBasicBlock *CondBranchTarget = I->getOperand(0).getMBB();
      if (UncondBranch.first != MBB.rend()) {
        assert(std::next(UncondBranch.first) == I && "Wrong block layout.");

        if (AllowModify && MBB.isLayoutSuccessor(CondBranchTarget)) {

          BranchCode = getOppositeCondCode(BranchCode);
          auto BNCC = getBrCond(BranchCode, false);

          BuildMI(MBB, *UncondBranch.first, MBB.rfindDebugLoc(I), BNCC)
              .addMBB(UncondBranch.second);

          EraseList.push_back(*I);
          EraseList.push_back(*UncondBranch.first);

          TBB = UncondBranch.second;
          FBB = nullptr;
          Cond.push_back(MachineOperand::CreateImm(BranchCode));
        } else {

          // Otherwise preserve TBB, FBB and Cond as requested
          TBB = CondBranchTarget;
          FBB = UncondBranch.second;
          Cond.push_back(MachineOperand::CreateImm(BranchCode));
        }

        UncondBranch = {MBB.rend(), nullptr};
        continue;
      }

      TBB = CondBranchTarget;
      FBB = nullptr;
      Cond.push_back(MachineOperand::CreateImm(BranchCode));
      continue;
    }

    // Handle subsequent conditional branches. Only handle the case where all
    // conditional branches branch to the same destination.
    assert(Cond.size() == 1);
    assert(TBB);

    // If the conditions are the same, we can leave them alone.
    SHCC::CondCode OldBranchCode = static_cast<SHCC::CondCode>(Cond[0].getImm());
    if (!I->getOperand(0).isMBB())
      return true;
    auto *NewTBB = I->getOperand(0).getMBB();
    if (OldBranchCode == BranchCode && TBB == NewTBB)
      continue;

    // If they differ we cannot do much here.
    return true;
  }

  return false;
#else
  return true;
#endif
}

unsigned SuperHInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "SH branch conditions have one component!");

  if (Cond.empty()) {
    assert(!FBB && "Unconditional branch with multiple successors!");
    auto &MI = *BuildMI(&MBB, DL, get(SH::BRA))
      .addMBB(TBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  SHCC::CondCode CC = (SHCC::CondCode)Cond[0].getImm();
  auto &CondMI = *BuildMI(&MBB, DL, getBrCond(CC, false))
    .addMBB(TBB);
  LLVM_DEBUG(dbgs() << "Created cc branch for " << getCondName(CC) << "...\n");

  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(CondMI);
  ++Count;

  if (FBB) {
    // Two-way Conditional branch. Insert the second branch.
    auto &MI = *BuildMI(&MBB, DL, get(SH::BRA))
      .addMBB(FBB);
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
    if (I->isDebugValue())
      continue;

    unsigned Opcode = I->getOpcode();
    if (Opcode != SH::BRA && Opcode != SH::NOP &&
        getCondFromBranchOp(Opcode) == SHCC::COND_INVALID)
      break;

    // Remove the branch.
    if (BytesRemoved)
      *BytesRemoved += getInstSizeInBytes(*I);
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
}

bool SuperHInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid SH branch condition!");

  LLVM_DEBUG(dbgs() << "Reversed branch condition...\n");
  SHCC::CondCode CC = (SHCC::CondCode)Cond[0].getImm();
  Cond[0].setImm(getOppositeCondCode(CC));
  return false;
}

MachineBasicBlock *
SuperHInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
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
