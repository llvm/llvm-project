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
#include "SuperHMachineFunctionInfo.h"
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

// emitAddressAdjust - Emits an adjustment to an address stored in a register,
// said adjustment will, if within the MaxAdd range, be emitted as a series of
// immediate add instructions, otherwise the implementation will try to store
// the offset into the constant pool, load it and add it to the register, 
// clobbering R1.
void SuperHInstrInfo::emitAddressAdjust(Register Reg, Block &MBB, BlockIt MBBI, 
                                        int32_t AdjValue, int32_t MaxAdd, MIFlag Flag) const {
  const MachineFunction &MF = *MBB.getParent();
  const SuperHInstrInfo &TII = *STI.getInstrInfo();
  const SuperHMachineFunctionInfo *MFI = MF.getInfo<SuperHMachineFunctionInfo>();
  DebugLoc DL = (MBBI != MBB.end()) ? MBBI->getDebugLoc() : DebugLoc();
  bool Sign = AdjValue < 0;

  // No stack frame allocation neccesary.
  if (AdjValue == 0)
    return;

  int32_t AbsAdj = std::abs(AdjValue);

  // If adjustment is below the maximum, generate a series of add
  // instructions, AbsAdj is the adjustment made absolute (positive)
  // as such, the sign is reapplied when the add instruction is generated.
  if ((AbsAdj / 128) < MaxAdd) {
    while(AbsAdj != 0) {
      int32_t Adj = std::min<int32_t>(127, AbsAdj);

      BuildMI(MBB, MBBI, DL, TII.get(SH::ADDI), Reg)
          .addReg(Reg)
          .addImm(Sign ? -Adj : Adj)
          .setMIFlag(Flag);

      AbsAdj -= Adj;
    }
    return;
  }

  // If the input register is R1, clobber R2 instead.
  Register TmpReg = SH::R1;
  if (Reg == TmpReg)
    TmpReg = SH::R2;

  // Try promoting the value to the constant pool and load from that.
  auto *CV = ConstantInt::get(MF.getFunction().getContext(), 
      APInt(32, AdjValue, true, true));
  if (CV) {
    if (auto *C = MFI->getOrCreate(CV, MF)) {
      BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLI), TmpReg)
          .addConstantPoolIndex(C->getLabelId())
          .setMIFlag(Flag);
      BuildMI(MBB, MBBI, DL, TII.get(SH::ADD), Reg)
          .addReg(TmpReg, RegState::Kill)
          .setMIFlag(Flag);
      return;
    }
  }

  llvm_unreachable("Failed to emit address adjustment!");
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

  // Load from MACH
  if (SrcReg == SH::MACHI && SH::GPRRegClass.contains(DestReg)) {
    BuildMI(MBB, MI, DL, get(SH::STSMACH), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Store to MACH
  if (SH::GPRRegClass.contains(SrcReg) && DestReg == SH::MACHI) {
    BuildMI(MBB, MI, DL, get(SH::LDSMACH), DestReg)
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
  if (MI.getNumOperands() == 3 && MI.getOperand(0).isReg() && 
      MI.getOperand(1).isFI() && MI.getOperand(2).getImm() == 0) {
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
      .addImm(0)
      .setMIFlags(Flags);
}

Register SuperHInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                              int &FrameIndex) const {
  if (MI.getNumOperands() == 3 && MI.getOperand(0).isFI() && 
      MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 0) {
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
      .addImm(0)
      .setMIFlags(Flags);
}




//===----------------------------------------------------------------------===//
//                              Branch Analysis
//===----------------------------------------------------------------------===//

bool SuperHInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *&TBB,
                                    MachineBasicBlock *&FBB,
                                    SmallVectorImpl<MachineOperand> &Cond,
                                    bool AllowModify) const {
  SmallVector<MachineInstr*, 2> BranchInstrs;
  BranchType BT = analyzeBranch(MBB, TBB, FBB, Cond, AllowModify, BranchInstrs);

  return (BT == BT_None) || (BT == BT_Indirect);
}


SuperHInstrInfo::BranchType 
SuperHInstrInfo::analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                           MachineBasicBlock *&FBB,
                           SmallVectorImpl<MachineOperand> &Cond,
                           bool AllowModify,
                           SmallVectorImpl<MachineInstr *> &BranchInstrs) const {

  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugInstr())
    ++I;

  // Block with no branches, which implies it's a fallthrough.
  if (I == REnd || !isUnpredicatedTerminator(*I)) {
    TBB = FBB = nullptr;
    return BT_NoBranch;
  }

  MachineInstr *LastInst = &*I;
  unsigned LastOpc = LastInst->getOpcode();
  BranchInstrs.push_back(LastInst);

  // Not an analyzable branch (e.g., indirect jump).
  if (!isBranchAnalyzable(LastOpc))
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
    SecondLastOpc = isBranchAnalyzable(SecondLastInst->getOpcode()) ? 
      SecondLastInst->getOpcode() : 0;

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
    analyzeCondBr(LastInst, LastOpc, TBB, Cond);
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

  analyzeCondBr(SecondLastInst, SecondLastOpc, TBB, Cond);
  FBB = LastInst->getOperand(0).getMBB();

  return BT_CondUncond;
}

unsigned SuperHInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert(!BytesAdded && "code size not handled");

  // All branch instructions take at least one cond
  assert((Cond.size() <= 2) && "# of SH branch conditions must be <= 1!");

  // Two-way conditional branch.
  if (FBB) {
    buildCondBr(MBB, TBB, DL, Cond);
    BuildMI(&MBB, DL, get(SH::BR))
        .addMBB(FBB)
        ->setImplicitPhysRegDefsDead();
    return 2;
  }

  // One-way unconditional branch
  if (Cond.empty()) {
    BuildMI(&MBB, DL, get(SH::BR))
        .addMBB(TBB)
        ->setImplicitPhysRegDefsDead();
    return 1;
  }

  // One-way conditional branch.
  buildCondBr(MBB, TBB, DL, Cond);
  return 1;
}

unsigned SuperHInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                       int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");

  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;

    // Skip past debug instructions
    if (I->isDebugInstr())
      continue;

    if (!isBranchAnalyzable(I->getOpcode()))
      break;

    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
}

bool SuperHInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid SH branch condition!");
  Cond[1].setImm(getOppositeCondCode((SHCC::CondCode)Cond[1].getImm()));
  return false;
}

MachineBasicBlock *
SuperHInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  LLVM_DEBUG(dbgs() << "getBranchDestBlock" << "\n");

  if (MI.isBranch())
    return MI.getOperand(0).getMBB();

  llvm_unreachable("unimplemented branch instructions");
}

void SuperHInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                                        MachineBasicBlock &NewDestBB,
                                        MachineBasicBlock &RestoreBB,
                                        const DebugLoc &DL, int64_t BrOffset,
                                        RegScavenger *RS) const {
  // NOTE:  If a BRA instruction would be in range, use that as an
  //        "indirect" jump, otherwise we'll need to use JMP
  //        and load the address from the constant pool.
  if (isBranchOffsetInRange(SH::BRA, BrOffset)) {
    BuildMI(&MBB, DL, get(SH::BRA)).addMBB(&NewDestBB);
  } else {
    BuildMI(&MBB, DL, get(SH::JMP)).addMBB(&NewDestBB);
  }
}

bool SuperHInstrInfo::isBranchOffsetInRange(unsigned BranchOp,
                                            int64_t BrOffset) const {

  LLVM_DEBUG(dbgs() << "isBranchOffsetInRange " << BrOffset << "\n");
  switch (BranchOp) {
  default:
    llvm_unreachable("unexpected opcode!");
  case SH::BR:
  case SH::BRCC:
  case SH::BRSCC:
    return true;
  case SH::BF:
  case SH::BT:
  case SH::BFS:
  case SH::BTS:
    return isIntN(8, BrOffset);
  case SH::BRA:
  case SH::BSR:
    return isIntN(12, BrOffset) && (BrOffset % 2 == 0); 
  }
}


bool SuperHInstrInfo::isBranchAnalyzable(unsigned Opc) const {
  switch(Opc) {
  default:
    return false;
  case SH::BR:
  case SH::BRCC:
  case SH::BRSCC:
      return true;
  }
}

void 
SuperHInstrInfo::analyzeCondBr(const MachineInstr *Inst, unsigned Opc,
                   MachineBasicBlock *&MBB,
                   SmallVectorImpl<MachineOperand> &Cond) const {
  assert(isBranchAnalyzable(Opc) && "Not an analyzable branch");
  int NumOp = Inst->getNumExplicitOperands();

  MBB = Inst->getOperand(0).getMBB();
  Cond.push_back(MachineOperand::CreateImm(Opc));

  for (int i = 1; i < NumOp; i++)
    Cond.push_back(Inst->getOperand(i));
}

void 
SuperHInstrInfo::buildCondBr(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                 const DebugLoc &DL, ArrayRef<MachineOperand> &Cond) const {
  unsigned Opc = Cond[0].getImm();
  const MCInstrDesc &MCID = get(Opc);
  MachineInstrBuilder MIB = BuildMI(&MBB, DL, MCID);

  MIB.addMBB(TBB);
  for (unsigned i = 1; i < Cond.size(); ++i) {
    assert((Cond[i].isImm() || Cond[i].isReg()) &&
           "Cannot copy operand for conditional branch!");
    MIB.add(Cond[i]);
  }
  MIB->setImplicitPhysRegDefsDead();
}