//===-- SystemZInstrInfo.cpp - SystemZ instruction information ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the SystemZ implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SystemZInstrInfo.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "SystemZ.h"
#include "SystemZInstrBuilder.h"
#include "SystemZSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineCombinerPattern.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <cstdint>
#include <iterator>

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRMAP_INFO
#include "SystemZGenInstrInfo.inc"

#define DEBUG_TYPE "systemz-II"

// Return a mask with Count low bits set.
static uint64_t allOnes(unsigned int Count) {
  return Count == 0 ? 0 : (uint64_t(1) << (Count - 1) << 1) - 1;
}

// Pin the vtable to this file.
void SystemZInstrInfo::anchor() {}

SystemZInstrInfo::SystemZInstrInfo(SystemZSubtarget &sti)
    : SystemZGenInstrInfo(-1, -1),
      RI(sti.getSpecialRegisters()->getReturnFunctionAddressRegister()),
      STI(sti) {}

// MI is a 128-bit load or store.  Split it into two 64-bit loads or stores,
// each having the opcode given by NewOpcode.
void SystemZInstrInfo::splitMove(MachineBasicBlock::iterator MI,
                                 unsigned NewOpcode) const {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction &MF = *MBB->getParent();

  // Get two load or store instructions.  Use the original instruction for one
  // of them (arbitrarily the second here) and create a clone for the other.
  MachineInstr *EarlierMI = MF.CloneMachineInstr(&*MI);
  MBB->insert(MI, EarlierMI);

  // Set up the two 64-bit registers and remember super reg and its flags.
  MachineOperand &HighRegOp = EarlierMI->getOperand(0);
  MachineOperand &LowRegOp = MI->getOperand(0);
  Register Reg128 = LowRegOp.getReg();
  unsigned Reg128Killed = getKillRegState(LowRegOp.isKill());
  unsigned Reg128Undef  = getUndefRegState(LowRegOp.isUndef());
  HighRegOp.setReg(RI.getSubReg(HighRegOp.getReg(), SystemZ::subreg_h64));
  LowRegOp.setReg(RI.getSubReg(LowRegOp.getReg(), SystemZ::subreg_l64));

  if (MI->mayStore()) {
    // Add implicit uses of the super register in case one of the subregs is
    // undefined. We could track liveness and skip storing an undefined
    // subreg, but this is hopefully rare (discovered with llvm-stress).
    // If Reg128 was killed, set kill flag on MI.
    unsigned Reg128UndefImpl = (Reg128Undef | RegState::Implicit);
    MachineInstrBuilder(MF, EarlierMI).addReg(Reg128, Reg128UndefImpl);
    MachineInstrBuilder(MF, MI).addReg(Reg128, (Reg128UndefImpl | Reg128Killed));
  }

  // The address in the first (high) instruction is already correct.
  // Adjust the offset in the second (low) instruction.
  MachineOperand &HighOffsetOp = EarlierMI->getOperand(2);
  MachineOperand &LowOffsetOp = MI->getOperand(2);
  LowOffsetOp.setImm(LowOffsetOp.getImm() + 8);

  // Clear the kill flags on the registers in the first instruction.
  if (EarlierMI->getOperand(0).isReg() && EarlierMI->getOperand(0).isUse())
    EarlierMI->getOperand(0).setIsKill(false);
  EarlierMI->getOperand(1).setIsKill(false);
  EarlierMI->getOperand(3).setIsKill(false);

  // Set the opcodes.
  unsigned HighOpcode = getOpcodeForOffset(NewOpcode, HighOffsetOp.getImm());
  unsigned LowOpcode = getOpcodeForOffset(NewOpcode, LowOffsetOp.getImm());
  assert(HighOpcode && LowOpcode && "Both offsets should be in range");

  EarlierMI->setDesc(get(HighOpcode));
  MI->setDesc(get(LowOpcode));
}

// Split ADJDYNALLOC instruction MI.
void SystemZInstrInfo::splitAdjDynAlloc(MachineBasicBlock::iterator MI) const {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction &MF = *MBB->getParent();
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  MachineOperand &OffsetMO = MI->getOperand(2);
  SystemZCallingConventionRegisters *Regs = STI.getSpecialRegisters();

  uint64_t Offset = (MFFrame.getMaxCallFrameSize() +
                     Regs->getCallFrameSize() +
                     Regs->getStackPointerBias() +
                     OffsetMO.getImm());
  unsigned NewOpcode = getOpcodeForOffset(SystemZ::LA, Offset);
  assert(NewOpcode && "No support for huge argument lists yet");
  MI->setDesc(get(NewOpcode));
  OffsetMO.setImm(Offset);
}

// MI is an RI-style pseudo instruction.  Replace it with LowOpcode
// if the first operand is a low GR32 and HighOpcode if the first operand
// is a high GR32.  ConvertHigh is true if LowOpcode takes a signed operand
// and HighOpcode takes an unsigned 32-bit operand.  In those cases,
// MI has the same kind of operand as LowOpcode, so needs to be converted
// if HighOpcode is used.
void SystemZInstrInfo::expandRIPseudo(MachineInstr &MI, unsigned LowOpcode,
                                      unsigned HighOpcode,
                                      bool ConvertHigh) const {
  Register Reg = MI.getOperand(0).getReg();
  bool IsHigh = SystemZ::isHighReg(Reg);
  MI.setDesc(get(IsHigh ? HighOpcode : LowOpcode));
  if (IsHigh && ConvertHigh)
    MI.getOperand(1).setImm(uint32_t(MI.getOperand(1).getImm()));
}

// MI is a three-operand RIE-style pseudo instruction.  Replace it with
// LowOpcodeK if the registers are both low GR32s, otherwise use a move
// followed by HighOpcode or LowOpcode, depending on whether the target
// is a high or low GR32.
void SystemZInstrInfo::expandRIEPseudo(MachineInstr &MI, unsigned LowOpcode,
                                       unsigned LowOpcodeK,
                                       unsigned HighOpcode) const {
  Register DestReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  bool DestIsHigh = SystemZ::isHighReg(DestReg);
  bool SrcIsHigh = SystemZ::isHighReg(SrcReg);
  if (!DestIsHigh && !SrcIsHigh)
    MI.setDesc(get(LowOpcodeK));
  else {
    if (DestReg != SrcReg) {
      emitGRX32Move(*MI.getParent(), MI, MI.getDebugLoc(), DestReg, SrcReg,
                    SystemZ::LR, 32, MI.getOperand(1).isKill(),
                    MI.getOperand(1).isUndef());
      MI.getOperand(1).setReg(DestReg);
    }
    MI.setDesc(get(DestIsHigh ? HighOpcode : LowOpcode));
    MI.tieOperands(0, 1);
  }
}

// MI is an RXY-style pseudo instruction.  Replace it with LowOpcode
// if the first operand is a low GR32 and HighOpcode if the first operand
// is a high GR32.
void SystemZInstrInfo::expandRXYPseudo(MachineInstr &MI, unsigned LowOpcode,
                                       unsigned HighOpcode) const {
  Register Reg = MI.getOperand(0).getReg();
  unsigned Opcode = getOpcodeForOffset(
      SystemZ::isHighReg(Reg) ? HighOpcode : LowOpcode,
      MI.getOperand(2).getImm());
  MI.setDesc(get(Opcode));
}

// MI is a load-on-condition pseudo instruction with a single register
// (source or destination) operand.  Replace it with LowOpcode if the
// register is a low GR32 and HighOpcode if the register is a high GR32.
void SystemZInstrInfo::expandLOCPseudo(MachineInstr &MI, unsigned LowOpcode,
                                       unsigned HighOpcode) const {
  Register Reg = MI.getOperand(0).getReg();
  unsigned Opcode = SystemZ::isHighReg(Reg) ? HighOpcode : LowOpcode;
  MI.setDesc(get(Opcode));
}

// MI is an RR-style pseudo instruction that zero-extends the low Size bits
// of one GRX32 into another.  Replace it with LowOpcode if both operands
// are low registers, otherwise use RISB[LH]G.
void SystemZInstrInfo::expandZExtPseudo(MachineInstr &MI, unsigned LowOpcode,
                                        unsigned Size) const {
  MachineInstrBuilder MIB =
    emitGRX32Move(*MI.getParent(), MI, MI.getDebugLoc(),
               MI.getOperand(0).getReg(), MI.getOperand(1).getReg(), LowOpcode,
               Size, MI.getOperand(1).isKill(), MI.getOperand(1).isUndef());

  // Keep the remaining operands as-is.
  for (const MachineOperand &MO : llvm::drop_begin(MI.operands(), 2))
    MIB.add(MO);

  MI.eraseFromParent();
}

void SystemZInstrInfo::expandLoadStackGuard(MachineInstr *MI) const {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction &MF = *MBB->getParent();
  const Register Reg64 = MI->getOperand(0).getReg();
  const Register Reg32 = RI.getSubReg(Reg64, SystemZ::subreg_l32);

  // EAR can only load the low subregister so us a shift for %a0 to produce
  // the GR containing %a0 and %a1.

  // ear <reg>, %a0
  BuildMI(*MBB, MI, MI->getDebugLoc(), get(SystemZ::EAR), Reg32)
    .addReg(SystemZ::A0)
    .addReg(Reg64, RegState::ImplicitDefine);

  // sllg <reg>, <reg>, 32
  BuildMI(*MBB, MI, MI->getDebugLoc(), get(SystemZ::SLLG), Reg64)
    .addReg(Reg64)
    .addReg(0)
    .addImm(32);

  // ear <reg>, %a1
  BuildMI(*MBB, MI, MI->getDebugLoc(), get(SystemZ::EAR), Reg32)
    .addReg(SystemZ::A1);

  // lg <reg>, 40(<reg>)
  MI->setDesc(get(SystemZ::LG));
  MachineInstrBuilder(MF, MI).addReg(Reg64).addImm(40).addReg(0);
}

// Emit a zero-extending move from 32-bit GPR SrcReg to 32-bit GPR
// DestReg before MBBI in MBB.  Use LowLowOpcode when both DestReg and SrcReg
// are low registers, otherwise use RISB[LH]G.  Size is the number of bits
// taken from the low end of SrcReg (8 for LLCR, 16 for LLHR and 32 for LR).
// KillSrc is true if this move is the last use of SrcReg.
MachineInstrBuilder
SystemZInstrInfo::emitGRX32Move(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                const DebugLoc &DL, unsigned DestReg,
                                unsigned SrcReg, unsigned LowLowOpcode,
                                unsigned Size, bool KillSrc,
                                bool UndefSrc) const {
  unsigned Opcode;
  bool DestIsHigh = SystemZ::isHighReg(DestReg);
  bool SrcIsHigh = SystemZ::isHighReg(SrcReg);
  if (DestIsHigh && SrcIsHigh)
    Opcode = SystemZ::RISBHH;
  else if (DestIsHigh && !SrcIsHigh)
    Opcode = SystemZ::RISBHL;
  else if (!DestIsHigh && SrcIsHigh)
    Opcode = SystemZ::RISBLH;
  else {
    return BuildMI(MBB, MBBI, DL, get(LowLowOpcode), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc) | getUndefRegState(UndefSrc));
  }
  unsigned Rotate = (DestIsHigh != SrcIsHigh ? 32 : 0);
  return BuildMI(MBB, MBBI, DL, get(Opcode), DestReg)
    .addReg(DestReg, RegState::Undef)
    .addReg(SrcReg, getKillRegState(KillSrc) | getUndefRegState(UndefSrc))
    .addImm(32 - Size).addImm(128 + 31).addImm(Rotate);
}

MachineInstr *SystemZInstrInfo::commuteInstructionImpl(MachineInstr &MI,
                                                       bool NewMI,
                                                       unsigned OpIdx1,
                                                       unsigned OpIdx2) const {
  auto cloneIfNew = [NewMI](MachineInstr &MI) -> MachineInstr & {
    if (NewMI)
      return *MI.getParent()->getParent()->CloneMachineInstr(&MI);
    return MI;
  };

  switch (MI.getOpcode()) {
  case SystemZ::SELRMux:
  case SystemZ::SELFHR:
  case SystemZ::SELR:
  case SystemZ::SELGR:
  case SystemZ::LOCRMux:
  case SystemZ::LOCFHR:
  case SystemZ::LOCR:
  case SystemZ::LOCGR: {
    auto &WorkingMI = cloneIfNew(MI);
    // Invert condition.
    unsigned CCValid = WorkingMI.getOperand(3).getImm();
    unsigned CCMask = WorkingMI.getOperand(4).getImm();
    WorkingMI.getOperand(4).setImm(CCMask ^ CCValid);
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  default:
    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
  }
}

// If MI is a simple load or store for a frame object, return the register
// it loads or stores and set FrameIndex to the index of the frame object.
// Return 0 otherwise.
//
// Flag is SimpleBDXLoad for loads and SimpleBDXStore for stores.
static int isSimpleMove(const MachineInstr &MI, int &FrameIndex,
                        unsigned Flag) {
  const MCInstrDesc &MCID = MI.getDesc();
  if ((MCID.TSFlags & Flag) && MI.getOperand(1).isFI() &&
      MI.getOperand(2).getImm() == 0 && MI.getOperand(3).getReg() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }
  return 0;
}

Register SystemZInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                               int &FrameIndex) const {
  return isSimpleMove(MI, FrameIndex, SystemZII::SimpleBDXLoad);
}

Register SystemZInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                              int &FrameIndex) const {
  return isSimpleMove(MI, FrameIndex, SystemZII::SimpleBDXStore);
}

bool SystemZInstrInfo::isStackSlotCopy(const MachineInstr &MI,
                                       int &DestFrameIndex,
                                       int &SrcFrameIndex) const {
  // Check for MVC 0(Length,FI1),0(FI2)
  const MachineFrameInfo &MFI = MI.getParent()->getParent()->getFrameInfo();
  if (MI.getOpcode() != SystemZ::MVC || !MI.getOperand(0).isFI() ||
      MI.getOperand(1).getImm() != 0 || !MI.getOperand(3).isFI() ||
      MI.getOperand(4).getImm() != 0)
    return false;

  // Check that Length covers the full slots.
  int64_t Length = MI.getOperand(2).getImm();
  unsigned FI1 = MI.getOperand(0).getIndex();
  unsigned FI2 = MI.getOperand(3).getIndex();
  if (MFI.getObjectSize(FI1) != Length ||
      MFI.getObjectSize(FI2) != Length)
    return false;

  DestFrameIndex = FI1;
  SrcFrameIndex = FI2;
  return true;
}

bool SystemZInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock *&TBB,
                                     MachineBasicBlock *&FBB,
                                     SmallVectorImpl<MachineOperand> &Cond,
                                     bool AllowModify) const {
  // Most of the code and comments here are boilerplate.

  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr())
      continue;

    // Working from the bottom, when we see a non-terminator instruction, we're
    // done.
    if (!isUnpredicatedTerminator(*I))
      break;

    // A terminator that isn't a branch can't easily be handled by this
    // analysis.
    if (!I->isBranch())
      return true;

    // Can't handle indirect branches.
    SystemZII::Branch Branch(getBranchInfo(*I));
    if (!Branch.hasMBBTarget())
      return true;

    // Punt on compound branches.
    if (Branch.Type != SystemZII::BranchNormal)
      return true;

    if (Branch.CCMask == SystemZ::CCMASK_ANY) {
      // Handle unconditional branches.
      if (!AllowModify) {
        TBB = Branch.getMBBTarget();
        continue;
      }

      // If the block has any instructions after a JMP, delete them.
      MBB.erase(std::next(I), MBB.end());

      Cond.clear();
      FBB = nullptr;

      // Delete the JMP if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(Branch.getMBBTarget())) {
        TBB = nullptr;
        I->eraseFromParent();
        I = MBB.end();
        continue;
      }

      // TBB is used to indicate the unconditinal destination.
      TBB = Branch.getMBBTarget();
      continue;
    }

    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      // FIXME: add X86-style branch swap
      FBB = TBB;
      TBB = Branch.getMBBTarget();
      Cond.push_back(MachineOperand::CreateImm(Branch.CCValid));
      Cond.push_back(MachineOperand::CreateImm(Branch.CCMask));
      continue;
    }

    // Handle subsequent conditional branches.
    assert(Cond.size() == 2 && TBB && "Should have seen a conditional branch");

    // Only handle the case where all conditional branches branch to the same
    // destination.
    if (TBB != Branch.getMBBTarget())
      return true;

    // If the conditions are the same, we can leave them alone.
    unsigned OldCCValid = Cond[0].getImm();
    unsigned OldCCMask = Cond[1].getImm();
    if (OldCCValid == Branch.CCValid && OldCCMask == Branch.CCMask)
      continue;

    // FIXME: Try combining conditions like X86 does.  Should be easy on Z!
    return false;
  }

  return false;
}

unsigned SystemZInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                        int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");

  // Most of the code and comments here are boilerplate.
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr())
      continue;
    if (!I->isBranch())
      break;
    if (!getBranchInfo(*I).hasMBBTarget())
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
}

bool SystemZInstrInfo::
reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid condition");
  Cond[1].setImm(Cond[1].getImm() ^ Cond[0].getImm());
  return false;
}

unsigned SystemZInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                        MachineBasicBlock *TBB,
                                        MachineBasicBlock *FBB,
                                        ArrayRef<MachineOperand> Cond,
                                        const DebugLoc &DL,
                                        int *BytesAdded) const {
  // In this function we output 32-bit branches, which should always
  // have enough range.  They can be shortened and relaxed by later code
  // in the pipeline, if desired.

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "SystemZ branch conditions have one component!");
  assert(!BytesAdded && "code size not handled");

  if (Cond.empty()) {
    // Unconditional branch?
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(SystemZ::J)).addMBB(TBB);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  unsigned CCValid = Cond[0].getImm();
  unsigned CCMask = Cond[1].getImm();
  BuildMI(&MBB, DL, get(SystemZ::BRC))
    .addImm(CCValid).addImm(CCMask).addMBB(TBB);
  ++Count;

  if (FBB) {
    // Two-way Conditional branch. Insert the second branch.
    BuildMI(&MBB, DL, get(SystemZ::J)).addMBB(FBB);
    ++Count;
  }
  return Count;
}

bool SystemZInstrInfo::analyzeCompare(const MachineInstr &MI, Register &SrcReg,
                                      Register &SrcReg2, int64_t &Mask,
                                      int64_t &Value) const {
  assert(MI.isCompare() && "Caller should have checked for a comparison");

  if (MI.getNumExplicitOperands() == 2 && MI.getOperand(0).isReg() &&
      MI.getOperand(1).isImm()) {
    SrcReg = MI.getOperand(0).getReg();
    SrcReg2 = 0;
    Value = MI.getOperand(1).getImm();
    Mask = ~0;
    return true;
  }

  return false;
}

bool SystemZInstrInfo::canInsertSelect(const MachineBasicBlock &MBB,
                                       ArrayRef<MachineOperand> Pred,
                                       Register DstReg, Register TrueReg,
                                       Register FalseReg, int &CondCycles,
                                       int &TrueCycles,
                                       int &FalseCycles) const {
  // Not all subtargets have LOCR instructions.
  if (!STI.hasLoadStoreOnCond())
    return false;
  if (Pred.size() != 2)
    return false;

  // Check register classes.
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC =
    RI.getCommonSubClass(MRI.getRegClass(TrueReg), MRI.getRegClass(FalseReg));
  if (!RC)
    return false;

  // We have LOCR instructions for 32 and 64 bit general purpose registers.
  if ((STI.hasLoadStoreOnCond2() &&
       SystemZ::GRX32BitRegClass.hasSubClassEq(RC)) ||
      SystemZ::GR32BitRegClass.hasSubClassEq(RC) ||
      SystemZ::GR64BitRegClass.hasSubClassEq(RC)) {
    CondCycles = 2;
    TrueCycles = 2;
    FalseCycles = 2;
    return true;
  }

  // Can't do anything else.
  return false;
}

void SystemZInstrInfo::insertSelect(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    const DebugLoc &DL, Register DstReg,
                                    ArrayRef<MachineOperand> Pred,
                                    Register TrueReg,
                                    Register FalseReg) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC = MRI.getRegClass(DstReg);

  assert(Pred.size() == 2 && "Invalid condition");
  unsigned CCValid = Pred[0].getImm();
  unsigned CCMask = Pred[1].getImm();

  unsigned Opc;
  if (SystemZ::GRX32BitRegClass.hasSubClassEq(RC)) {
    if (STI.hasMiscellaneousExtensions3())
      Opc = SystemZ::SELRMux;
    else if (STI.hasLoadStoreOnCond2())
      Opc = SystemZ::LOCRMux;
    else {
      Opc = SystemZ::LOCR;
      MRI.constrainRegClass(DstReg, &SystemZ::GR32BitRegClass);
      Register TReg = MRI.createVirtualRegister(&SystemZ::GR32BitRegClass);
      Register FReg = MRI.createVirtualRegister(&SystemZ::GR32BitRegClass);
      BuildMI(MBB, I, DL, get(TargetOpcode::COPY), TReg).addReg(TrueReg);
      BuildMI(MBB, I, DL, get(TargetOpcode::COPY), FReg).addReg(FalseReg);
      TrueReg = TReg;
      FalseReg = FReg;
    }
  } else if (SystemZ::GR64BitRegClass.hasSubClassEq(RC)) {
    if (STI.hasMiscellaneousExtensions3())
      Opc = SystemZ::SELGR;
    else
      Opc = SystemZ::LOCGR;
  } else
    llvm_unreachable("Invalid register class");

  BuildMI(MBB, I, DL, get(Opc), DstReg)
    .addReg(FalseReg).addReg(TrueReg)
    .addImm(CCValid).addImm(CCMask);
}

static void transferDeadCC(MachineInstr *OldMI, MachineInstr *NewMI) {
  if (OldMI->registerDefIsDead(SystemZ::CC)) {
    MachineOperand *CCDef = NewMI->findRegisterDefOperand(SystemZ::CC);
    if (CCDef != nullptr)
      CCDef->setIsDead(true);
  }
}

void SystemZInstrInfo::transferMIFlag(MachineInstr *OldMI, MachineInstr *NewMI,
                                      MachineInstr::MIFlag Flag) const {
  if (OldMI->getFlag(Flag))
    NewMI->setFlag(Flag);
}

static cl::opt<bool> DISABLE_FOLDING("disable-folding", cl::init(false));
static cl::opt<bool> FOLD_LDY("fold-ldy", cl::init(false));

MachineInstr *SystemZInstrInfo::optimizeLoadInstr(MachineInstr &MI,
                                                  MachineRegisterInfo *MRI,
                                                  Register &FoldAsLoadDefReg,
                                                  MachineInstr *&DefMI) const {
  if (DISABLE_FOLDING)
    return nullptr;

  const TargetRegisterInfo *TRI = MRI->getTargetRegisterInfo();

  // Check whether we can move the DefMI load, and that it only has one use.
  DefMI = MRI->getVRegDef(FoldAsLoadDefReg);
  assert(DefMI);
  bool SawStore = false;
  if (!DefMI->isSafeToMove(nullptr, SawStore) ||
      !MRI->hasOneNonDBGUse(FoldAsLoadDefReg))
    return nullptr;

  // For reassociable FP operations, any loads have been purposefully left
  // unfolded so that MachineCombiner can do its work on reg/reg
  // opcodes. After that, as many loads as possible are now folded.
  unsigned LoadOpcD12 = 0;
  unsigned LoadOpcD20 = 0;
  unsigned RegMemOpcode = 0;
  const TargetRegisterClass *FPRC = nullptr;
  RegMemOpcode = MI.getOpcode() == SystemZ::WFADB_CCPseudo    ? SystemZ::ADB
                 : MI.getOpcode() == SystemZ::WFSDB_CCPseudo  ? SystemZ::SDB
                 : MI.getOpcode() == SystemZ::WFMDB           ? SystemZ::MDB
                 : MI.getOpcode() == SystemZ::WFMADB_CCPseudo ? SystemZ::MADB
                                                              : 0;
  if (RegMemOpcode) {
    LoadOpcD12 = SystemZ::VL64;
    LoadOpcD20 = SystemZ::LDY;
    FPRC = &SystemZ::FP64BitRegClass;
  } else {
    RegMemOpcode = MI.getOpcode() == SystemZ::WFASB_CCPseudo    ? SystemZ::AEB
                   : MI.getOpcode() == SystemZ::WFSSB_CCPseudo  ? SystemZ::SEB
                   : MI.getOpcode() == SystemZ::WFMSB           ? SystemZ::MEEB
                   : MI.getOpcode() == SystemZ::WFMASB_CCPseudo ? SystemZ::MAEB
                                                                : 0;
    if (RegMemOpcode) {
      LoadOpcD12 = SystemZ::VL32;
      LoadOpcD20 = SystemZ::LEY;
      FPRC = &SystemZ::FP32BitRegClass;
    }
  }

  if (!RegMemOpcode ||
      (DefMI->getOpcode() != LoadOpcD12 && DefMI->getOpcode() != LoadOpcD20))
    return nullptr;

  if (DefMI->getOpcode() == LoadOpcD20 && !FOLD_LDY)
    return nullptr;

  DebugLoc DL = MI.getDebugLoc();
  Register DstReg = MI.getOperand(0).getReg();
  MachineOperand LHS = MI.getOperand(1);
  MachineOperand RHS = MI.getOperand(2);
  bool IsTernary =
    (RegMemOpcode == SystemZ::MADB || RegMemOpcode == SystemZ::MAEB);
  MachineOperand &RegMO = RHS.getReg() == FoldAsLoadDefReg ? LHS : RHS;
  MachineOperand *AccMO = IsTernary ? &MI.getOperand(3) : nullptr;
  if ((RegMemOpcode == SystemZ::SDB || RegMemOpcode == SystemZ::SEB) &&
      FoldAsLoadDefReg != RHS.getReg())
    return nullptr;
  if (IsTernary && FoldAsLoadDefReg == AccMO->getReg())
    return nullptr;

  MachineInstrBuilder MIB =
    BuildMI(*MI.getParent(), MI, DL, get(RegMemOpcode), DstReg);
  MRI->setRegClass(DstReg, FPRC);
  if (IsTernary) {
    MIB.add(*AccMO);
    MRI->setRegClass(AccMO->getReg(), FPRC);
  }
  MIB.add(RegMO);
  MRI->setRegClass(RegMO.getReg(), FPRC);

  MachineOperand &Base = DefMI->getOperand(1);
  MachineOperand &Disp = DefMI->getOperand(2);
  MachineOperand &Indx = DefMI->getOperand(3);
  if (Base.isReg())  // Could be a FrameIndex.
    Base.setIsKill(false);
  Indx.setIsKill(false);
  if (DefMI->getOpcode() == LoadOpcD12) {
    MIB.add(Base).add(Disp).add(Indx);
  } else {
    Register AddrReg = MRI->createVirtualRegister(&SystemZ::ADDR64BitRegClass);
    BuildMI(*MI.getParent(), *MIB, DL, get(SystemZ::LAY), AddrReg)
      .add(Base).add(Disp).add(Indx);
    MIB.addReg(AddrReg).addImm(0).addReg(SystemZ::NoRegister);
  }
  MIB.addMemOperand(*DefMI->memoperands_begin());
  transferMIFlag(&MI, MIB, MachineInstr::NoFPExcept);
  MIB->addRegisterDead(SystemZ::CC, TRI);

  return MIB;
}

bool SystemZInstrInfo::foldImmediate(MachineInstr &UseMI, MachineInstr &DefMI,
                                     Register Reg,
                                     MachineRegisterInfo *MRI) const {
  unsigned DefOpc = DefMI.getOpcode();
  if (DefOpc != SystemZ::LHIMux && DefOpc != SystemZ::LHI &&
      DefOpc != SystemZ::LGHI)
    return false;
  if (DefMI.getOperand(0).getReg() != Reg)
    return false;
  int32_t ImmVal = (int32_t)DefMI.getOperand(1).getImm();

  unsigned UseOpc = UseMI.getOpcode();
  unsigned NewUseOpc;
  unsigned UseIdx;
  int CommuteIdx = -1;
  bool TieOps = false;
  switch (UseOpc) {
  case SystemZ::SELRMux:
    TieOps = true;
    [[fallthrough]];
  case SystemZ::LOCRMux:
    if (!STI.hasLoadStoreOnCond2())
      return false;
    NewUseOpc = SystemZ::LOCHIMux;
    if (UseMI.getOperand(2).getReg() == Reg)
      UseIdx = 2;
    else if (UseMI.getOperand(1).getReg() == Reg)
      UseIdx = 2, CommuteIdx = 1;
    else
      return false;
    break;
  case SystemZ::SELGR:
    TieOps = true;
    [[fallthrough]];
  case SystemZ::LOCGR:
    if (!STI.hasLoadStoreOnCond2())
      return false;
    NewUseOpc = SystemZ::LOCGHI;
    if (UseMI.getOperand(2).getReg() == Reg)
      UseIdx = 2;
    else if (UseMI.getOperand(1).getReg() == Reg)
      UseIdx = 2, CommuteIdx = 1;
    else
      return false;
    break;
  default:
    return false;
  }

  if (CommuteIdx != -1)
    if (!commuteInstruction(UseMI, false, CommuteIdx, UseIdx))
      return false;

  bool DeleteDef = MRI->hasOneNonDBGUse(Reg);
  UseMI.setDesc(get(NewUseOpc));
  if (TieOps)
    UseMI.tieOperands(0, 1);
  UseMI.getOperand(UseIdx).ChangeToImmediate(ImmVal);
  if (DeleteDef)
    DefMI.eraseFromParent();

  return true;
}

bool SystemZInstrInfo::isPredicable(const MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();
  if (Opcode == SystemZ::Return ||
      Opcode == SystemZ::Return_XPLINK ||
      Opcode == SystemZ::Trap ||
      Opcode == SystemZ::CallJG ||
      Opcode == SystemZ::CallBR)
    return true;
  return false;
}

bool SystemZInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &MBB,
                    unsigned NumCycles, unsigned ExtraPredCycles,
                    BranchProbability Probability) const {
  // Avoid using conditional returns at the end of a loop (since then
  // we'd need to emit an unconditional branch to the beginning anyway,
  // making the loop body longer).  This doesn't apply for low-probability
  // loops (eg. compare-and-swap retry), so just decide based on branch
  // probability instead of looping structure.
  // However, since Compare and Trap instructions cost the same as a regular
  // Compare instruction, we should allow the if conversion to convert this
  // into a Conditional Compare regardless of the branch probability.
  if (MBB.getLastNonDebugInstr()->getOpcode() != SystemZ::Trap &&
      MBB.succ_empty() && Probability < BranchProbability(1, 8))
    return false;
  // For now only convert single instructions.
  return NumCycles == 1;
}

bool SystemZInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &TMBB,
                    unsigned NumCyclesT, unsigned ExtraPredCyclesT,
                    MachineBasicBlock &FMBB,
                    unsigned NumCyclesF, unsigned ExtraPredCyclesF,
                    BranchProbability Probability) const {
  // For now avoid converting mutually-exclusive cases.
  return false;
}

bool SystemZInstrInfo::
isProfitableToDupForIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                          BranchProbability Probability) const {
  // For now only duplicate single instructions.
  return NumCycles == 1;
}

bool SystemZInstrInfo::PredicateInstruction(
    MachineInstr &MI, ArrayRef<MachineOperand> Pred) const {
  assert(Pred.size() == 2 && "Invalid condition");
  unsigned CCValid = Pred[0].getImm();
  unsigned CCMask = Pred[1].getImm();
  assert(CCMask > 0 && CCMask < 15 && "Invalid predicate");
  unsigned Opcode = MI.getOpcode();
  if (Opcode == SystemZ::Trap) {
    MI.setDesc(get(SystemZ::CondTrap));
    MachineInstrBuilder(*MI.getParent()->getParent(), MI)
      .addImm(CCValid).addImm(CCMask)
      .addReg(SystemZ::CC, RegState::Implicit);
    return true;
  }
  if (Opcode == SystemZ::Return || Opcode == SystemZ::Return_XPLINK) {
    MI.setDesc(get(Opcode == SystemZ::Return ? SystemZ::CondReturn
                                             : SystemZ::CondReturn_XPLINK));
    MachineInstrBuilder(*MI.getParent()->getParent(), MI)
        .addImm(CCValid)
        .addImm(CCMask)
        .addReg(SystemZ::CC, RegState::Implicit);
    return true;
  }
  if (Opcode == SystemZ::CallJG) {
    MachineOperand FirstOp = MI.getOperand(0);
    const uint32_t *RegMask = MI.getOperand(1).getRegMask();
    MI.removeOperand(1);
    MI.removeOperand(0);
    MI.setDesc(get(SystemZ::CallBRCL));
    MachineInstrBuilder(*MI.getParent()->getParent(), MI)
        .addImm(CCValid)
        .addImm(CCMask)
        .add(FirstOp)
        .addRegMask(RegMask)
        .addReg(SystemZ::CC, RegState::Implicit);
    return true;
  }
  if (Opcode == SystemZ::CallBR) {
    MachineOperand Target = MI.getOperand(0);
    const uint32_t *RegMask = MI.getOperand(1).getRegMask();
    MI.removeOperand(1);
    MI.removeOperand(0);
    MI.setDesc(get(SystemZ::CallBCR));
    MachineInstrBuilder(*MI.getParent()->getParent(), MI)
      .addImm(CCValid).addImm(CCMask)
      .add(Target)
      .addRegMask(RegMask)
      .addReg(SystemZ::CC, RegState::Implicit);
    return true;
  }
  return false;
}

void SystemZInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL, MCRegister DestReg,
                                   MCRegister SrcReg, bool KillSrc) const {
  // Split 128-bit GPR moves into two 64-bit moves. Add implicit uses of the
  // super register in case one of the subregs is undefined.
  // This handles ADDR128 too.
  if (SystemZ::GR128BitRegClass.contains(DestReg, SrcReg)) {
    copyPhysReg(MBB, MBBI, DL, RI.getSubReg(DestReg, SystemZ::subreg_h64),
                RI.getSubReg(SrcReg, SystemZ::subreg_h64), KillSrc);
    MachineInstrBuilder(*MBB.getParent(), std::prev(MBBI))
      .addReg(SrcReg, RegState::Implicit);
    copyPhysReg(MBB, MBBI, DL, RI.getSubReg(DestReg, SystemZ::subreg_l64),
                RI.getSubReg(SrcReg, SystemZ::subreg_l64), KillSrc);
    MachineInstrBuilder(*MBB.getParent(), std::prev(MBBI))
      .addReg(SrcReg, (getKillRegState(KillSrc) | RegState::Implicit));
    return;
  }

  if (SystemZ::GRX32BitRegClass.contains(DestReg, SrcReg)) {
    emitGRX32Move(MBB, MBBI, DL, DestReg, SrcReg, SystemZ::LR, 32, KillSrc,
                  false);
    return;
  }

  // Move 128-bit floating-point values between VR128 and FP128.
  if (SystemZ::VR128BitRegClass.contains(DestReg) &&
      SystemZ::FP128BitRegClass.contains(SrcReg)) {
    MCRegister SrcRegHi =
        RI.getMatchingSuperReg(RI.getSubReg(SrcReg, SystemZ::subreg_h64),
                               SystemZ::subreg_h64, &SystemZ::VR128BitRegClass);
    MCRegister SrcRegLo =
        RI.getMatchingSuperReg(RI.getSubReg(SrcReg, SystemZ::subreg_l64),
                               SystemZ::subreg_h64, &SystemZ::VR128BitRegClass);

    BuildMI(MBB, MBBI, DL, get(SystemZ::VMRHG), DestReg)
      .addReg(SrcRegHi, getKillRegState(KillSrc))
      .addReg(SrcRegLo, getKillRegState(KillSrc));
    return;
  }
  if (SystemZ::FP128BitRegClass.contains(DestReg) &&
      SystemZ::VR128BitRegClass.contains(SrcReg)) {
    MCRegister DestRegHi =
        RI.getMatchingSuperReg(RI.getSubReg(DestReg, SystemZ::subreg_h64),
                               SystemZ::subreg_h64, &SystemZ::VR128BitRegClass);
    MCRegister DestRegLo =
        RI.getMatchingSuperReg(RI.getSubReg(DestReg, SystemZ::subreg_l64),
                               SystemZ::subreg_h64, &SystemZ::VR128BitRegClass);

    if (DestRegHi != SrcReg)
      copyPhysReg(MBB, MBBI, DL, DestRegHi, SrcReg, false);
    BuildMI(MBB, MBBI, DL, get(SystemZ::VREPG), DestRegLo)
      .addReg(SrcReg, getKillRegState(KillSrc)).addImm(1);
    return;
  }

  // Move CC value from a GR32.
  if (DestReg == SystemZ::CC) {
    unsigned Opcode =
      SystemZ::GR32BitRegClass.contains(SrcReg) ? SystemZ::TMLH : SystemZ::TMHH;
    BuildMI(MBB, MBBI, DL, get(Opcode))
      .addReg(SrcReg, getKillRegState(KillSrc))
      .addImm(3 << (SystemZ::IPM_CC - 16));
    return;
  }

  // Everything else needs only one instruction.
  unsigned Opcode;
  if (SystemZ::GR64BitRegClass.contains(DestReg, SrcReg))
    Opcode = SystemZ::LGR;
  else if (SystemZ::FP32BitRegClass.contains(DestReg, SrcReg))
    // For z13 we prefer LDR over LER to avoid partial register dependencies.
    Opcode = STI.hasVector() ? SystemZ::LDR32 : SystemZ::LER;
  else if (SystemZ::FP64BitRegClass.contains(DestReg, SrcReg))
    Opcode = SystemZ::LDR;
  else if (SystemZ::FP128BitRegClass.contains(DestReg, SrcReg))
    Opcode = SystemZ::LXR;
  else if (SystemZ::VR32BitRegClass.contains(DestReg, SrcReg))
    Opcode = SystemZ::VLR32;
  else if (SystemZ::VR64BitRegClass.contains(DestReg, SrcReg))
    Opcode = SystemZ::VLR64;
  else if (SystemZ::VR128BitRegClass.contains(DestReg, SrcReg))
    Opcode = SystemZ::VLR;
  else if (SystemZ::AR32BitRegClass.contains(DestReg, SrcReg))
    Opcode = SystemZ::CPYA;
  else
    llvm_unreachable("Impossible reg-to-reg copy");

  BuildMI(MBB, MBBI, DL, get(Opcode), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc));
}

void SystemZInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, Register SrcReg,
    bool isKill, int FrameIdx, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI, Register VReg) const {
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Callers may expect a single instruction, so keep 128-bit moves
  // together for now and lower them after register allocation.
  unsigned LoadOpcode, StoreOpcode;
  getLoadStoreOpcodes(RC, LoadOpcode, StoreOpcode);
  addFrameReference(BuildMI(MBB, MBBI, DL, get(StoreOpcode))
                        .addReg(SrcReg, getKillRegState(isKill)),
                    FrameIdx);
}

void SystemZInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator MBBI,
                                            Register DestReg, int FrameIdx,
                                            const TargetRegisterClass *RC,
                                            const TargetRegisterInfo *TRI,
                                            Register VReg) const {
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Callers may expect a single instruction, so keep 128-bit moves
  // together for now and lower them after register allocation.
  unsigned LoadOpcode, StoreOpcode;
  getLoadStoreOpcodes(RC, LoadOpcode, StoreOpcode);
  addFrameReference(BuildMI(MBB, MBBI, DL, get(LoadOpcode), DestReg),
                    FrameIdx);
}

// Return true if MI is a simple load or store with a 12-bit displacement
// and no index.  Flag is SimpleBDXLoad for loads and SimpleBDXStore for stores.
static bool isSimpleBD12Move(const MachineInstr *MI, unsigned Flag) {
  const MCInstrDesc &MCID = MI->getDesc();
  return ((MCID.TSFlags & Flag) &&
          isUInt<12>(MI->getOperand(2).getImm()) &&
          MI->getOperand(3).getReg() == 0);
}

namespace {

struct LogicOp {
  LogicOp() = default;
  LogicOp(unsigned regSize, unsigned immLSB, unsigned immSize)
    : RegSize(regSize), ImmLSB(immLSB), ImmSize(immSize) {}

  explicit operator bool() const { return RegSize; }

  unsigned RegSize = 0;
  unsigned ImmLSB = 0;
  unsigned ImmSize = 0;
};

} // end anonymous namespace

static LogicOp interpretAndImmediate(unsigned Opcode) {
  switch (Opcode) {
  case SystemZ::NILMux: return LogicOp(32,  0, 16);
  case SystemZ::NIHMux: return LogicOp(32, 16, 16);
  case SystemZ::NILL64: return LogicOp(64,  0, 16);
  case SystemZ::NILH64: return LogicOp(64, 16, 16);
  case SystemZ::NIHL64: return LogicOp(64, 32, 16);
  case SystemZ::NIHH64: return LogicOp(64, 48, 16);
  case SystemZ::NIFMux: return LogicOp(32,  0, 32);
  case SystemZ::NILF64: return LogicOp(64,  0, 32);
  case SystemZ::NIHF64: return LogicOp(64, 32, 32);
  default:              return LogicOp();
  }
}


MachineInstr *
SystemZInstrInfo::convertToThreeAddress(MachineInstr &MI, LiveVariables *LV,
                                        LiveIntervals *LIS) const {
  MachineBasicBlock *MBB = MI.getParent();

  // Try to convert an AND into an RISBG-type instruction.
  // TODO: It might be beneficial to select RISBG and shorten to AND instead.
  if (LogicOp And = interpretAndImmediate(MI.getOpcode())) {
    uint64_t Imm = MI.getOperand(2).getImm() << And.ImmLSB;
    // AND IMMEDIATE leaves the other bits of the register unchanged.
    Imm |= allOnes(And.RegSize) & ~(allOnes(And.ImmSize) << And.ImmLSB);
    unsigned Start, End;
    if (isRxSBGMask(Imm, And.RegSize, Start, End)) {
      unsigned NewOpcode;
      if (And.RegSize == 64) {
        NewOpcode = SystemZ::RISBG;
        // Prefer RISBGN if available, since it does not clobber CC.
        if (STI.hasMiscellaneousExtensions())
          NewOpcode = SystemZ::RISBGN;
      } else {
        NewOpcode = SystemZ::RISBMux;
        Start &= 31;
        End &= 31;
      }
      MachineOperand &Dest = MI.getOperand(0);
      MachineOperand &Src = MI.getOperand(1);
      MachineInstrBuilder MIB =
          BuildMI(*MBB, MI, MI.getDebugLoc(), get(NewOpcode))
              .add(Dest)
              .addReg(0)
              .addReg(Src.getReg(), getKillRegState(Src.isKill()),
                      Src.getSubReg())
              .addImm(Start)
              .addImm(End + 128)
              .addImm(0);
      if (LV) {
        unsigned NumOps = MI.getNumOperands();
        for (unsigned I = 1; I < NumOps; ++I) {
          MachineOperand &Op = MI.getOperand(I);
          if (Op.isReg() && Op.isKill())
            LV->replaceKillInstruction(Op.getReg(), MI, *MIB);
        }
      }
      if (LIS)
        LIS->ReplaceMachineInstrInMaps(MI, *MIB);
      transferDeadCC(&MI, MIB);
      return MIB;
    }
  }
  return nullptr;
}

static bool hasReassocFlags(const MachineInstr *MI) {
  return (MI->getFlag(MachineInstr::MIFlag::FmReassoc) &&
          MI->getFlag(MachineInstr::MIFlag::FmNsz));
}

bool SystemZInstrInfo::IsReassociableFMA(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  case SystemZ::VFMADB:
  case SystemZ::VFMASB:
  case SystemZ::WFMAXB:
  case SystemZ::WFMADB_CCPseudo:
  case SystemZ::WFMASB_CCPseudo:
    return hasReassocFlags(MI);
  default:
    break;
  }
  return false;
}

bool SystemZInstrInfo::IsReassociableAdd(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  case SystemZ::VFADB:
  case SystemZ::VFASB:
  case SystemZ::WFAXB:
    return hasReassocFlags(MI);
  case SystemZ::WFADB_CCPseudo:
  case SystemZ::WFASB_CCPseudo:
    return hasReassocFlags(MI) &&
           MI->findRegisterDefOperandIdx(SystemZ::CC, true/*isDead*/) != -1;
  default:
    break;
  }
  return false;
}

// EXPERIMENTAL
static cl::opt<bool> Z_FMA("z-fma", cl::init(false));
static cl::opt<bool> PPC_FMA("ppc-fma", cl::init(false));

bool SystemZInstrInfo::getFMAPatterns(
    MachineInstr &Root, SmallVectorImpl<MachineCombinerPattern> &Patterns,
    bool DoRegPressureReduce) const {
  assert(Patterns.empty());
  MachineBasicBlock *MBB = Root.getParent();
  const MachineRegisterInfo *MRI = &MBB->getParent()->getRegInfo();

  if (!IsReassociableFMA(&Root))
    return false;

  const TargetRegisterClass *RC = MRI->getRegClass(Root.getOperand(0).getReg());

  // This is more or less always true.
  auto AllOpsOK = [&MRI, &RC](const MachineInstr &Instr) {
    for (const auto &MO : Instr.explicit_operands())
      if (!(MO.isReg() && MO.getReg().isVirtual() && !MO.getSubReg()))
        return false;
    const TargetRegisterClass *DefRC = MRI->getRegClass(Instr.getOperand(0).getReg());
    if (!DefRC->hasSubClassEq(RC) && !DefRC->hasSuperClassEq(RC))
      return false;
    return true;
  };
  if (!AllOpsOK(Root))
    return false;

  // XXX Rewrite this for the patterns we want to actually use.
  MachineInstr *TopAdd = nullptr;
  std::vector<MachineInstr *> FMAChain;
  FMAChain.push_back(&Root);
  Register Acc = Root.getOperand(3).getReg();
  while (MachineInstr *Prev = MRI->getUniqueVRegDef(Acc)) {
    if (Prev->getParent() != MBB || !MRI->hasOneNonDBGUse(Acc) ||
        !AllOpsOK(*Prev))
      break;
    if (IsReassociableFMA(Prev)) {
      FMAChain.push_back(Prev);
      Acc = Prev->getOperand(3).getReg();
      continue;
    }
    if (IsReassociableAdd(Prev))
      TopAdd = Prev;
    break;
  }

  if (Z_FMA) {
    if (FMAChain.size() >= 2) {
      Patterns.push_back(MachineCombinerPattern::FMA2_P1P0);
      LLVM_DEBUG(dbgs() << "add pattern FMA2_P1P0\n");
      Patterns.push_back(MachineCombinerPattern::FMA2_P0P1);
      LLVM_DEBUG(dbgs() << "add pattern FMA2_P0P1\n");
      Patterns.push_back(MachineCombinerPattern::FMA2);
      LLVM_DEBUG(dbgs() << "add pattern FMA2\n");
    }
    if (FMAChain.size() == 1 && TopAdd) {
      // The latency of the FMA could potentially be hidden above the add:
      // Try both sides of the add and let MachineCombiner decide on
      // profitability.
      Patterns.push_back(MachineCombinerPattern::FMA1_Add_L);
      LLVM_DEBUG(dbgs() << "add pattern FMA1_Add_L\n");
      Patterns.push_back(MachineCombinerPattern::FMA1_Add_R);
      LLVM_DEBUG(dbgs() << "add pattern FMA1_Add_R\n");
    }
  } else if (PPC_FMA) {
    if (FMAChain.size() >= 3) {
      Patterns.push_back(MachineCombinerPattern::FMA3);
      LLVM_DEBUG(dbgs() << "add pattern FMA3\n");
    }
    if (FMAChain.size() == 2 && TopAdd) {
      Patterns.push_back(MachineCombinerPattern::FMA2_Add);
      LLVM_DEBUG(dbgs() << "add pattern FMA2_Add\n");
    }
  }

  return Patterns.size() > 0;
}

bool SystemZInstrInfo::getMachineCombinerPatterns(
    MachineInstr &Root, SmallVectorImpl<MachineCombinerPattern> &Patterns,
    bool DoRegPressureReduce) const {

  if (getFMAPatterns(Root, Patterns, DoRegPressureReduce))
    return true;

  return TargetInstrInfo::getMachineCombinerPatterns(Root, Patterns,
                                                     DoRegPressureReduce);
}

void SystemZInstrInfo::finalizeInsInstrs(
    MachineInstr &Root, MachineCombinerPattern &P,
    SmallVectorImpl<MachineInstr *> &InsInstrs) const {
  const TargetRegisterInfo *TRI =
    Root.getParent()->getParent()->getSubtarget().getRegisterInfo();
  for (auto *Inst : InsInstrs) {
    switch (Inst->getOpcode()) {
    case SystemZ::WFADB_CCPseudo:
    case SystemZ::WFASB_CCPseudo:
    case SystemZ::WFSDB_CCPseudo:
    case SystemZ::WFSSB_CCPseudo:
    case SystemZ::WFMADB_CCPseudo:
    case SystemZ::WFMASB_CCPseudo:
      Inst->addRegisterDead(SystemZ::CC, TRI);
      break;
    default: break;
    }
  }
}

bool SystemZInstrInfo::isAssociativeAndCommutative(const MachineInstr &Inst,
                                                   bool Invert) const {
  unsigned Opc = Inst.getOpcode();
  if (Invert) {
    auto InverseOpcode = getInverseOpcode(Opc);
    if (!InverseOpcode)
      return false;
    Opc = *InverseOpcode;
  }

  switch (Opc) {
  default:
    break;
  // Adds and multiplications.
  case SystemZ::VFADB:
  case SystemZ::VFASB:
  case SystemZ::WFAXB:
  case SystemZ::WFADB_CCPseudo:
  case SystemZ::WFASB_CCPseudo:
  case SystemZ::VFMDB:
  case SystemZ::VFMSB:
  case SystemZ::WFMXB:
  case SystemZ::WFMDB:
  case SystemZ::WFMSB:
    return hasReassocFlags(&Inst);
  }

  return false;
}

std::optional<unsigned>
SystemZInstrInfo::getInverseOpcode(unsigned Opcode) const {
  // fadd <=> fsub in various forms.
  switch (Opcode) {
  case SystemZ::VFADB:           return SystemZ::VFSDB;
  case SystemZ::VFASB:           return SystemZ::VFSSB;
  case SystemZ::WFAXB:           return SystemZ::WFSXB;
  case SystemZ::WFADB_CCPseudo:  return SystemZ::WFSDB_CCPseudo;
  case SystemZ::WFASB_CCPseudo:  return SystemZ::WFSSB_CCPseudo;
  case SystemZ::VFSDB:           return SystemZ::VFADB;
  case SystemZ::VFSSB:           return SystemZ::VFASB;
  case SystemZ::WFSXB:           return SystemZ::WFAXB;
  case SystemZ::WFSDB_CCPseudo:  return SystemZ::WFADB_CCPseudo;
  case SystemZ::WFSSB_CCPseudo:  return SystemZ::WFASB_CCPseudo;
  default:                       return std::nullopt;
  }
}

void SystemZInstrInfo::genAlternativeCodeSequence(
    MachineInstr &Root, MachineCombinerPattern Pattern,
    SmallVectorImpl<MachineInstr *> &InsInstrs,
    SmallVectorImpl<MachineInstr *> &DelInstrs,
    DenseMap<unsigned, unsigned> &InstrIdxForVirtReg) const {
  switch (Pattern) {
  case MachineCombinerPattern::FMA2_P1P0:
  case MachineCombinerPattern::FMA2_P0P1:
  case MachineCombinerPattern::FMA2:
  case MachineCombinerPattern::FMA1_Add_L:
  case MachineCombinerPattern::FMA1_Add_R:
  case MachineCombinerPattern::FMA3:
  case MachineCombinerPattern::FMA2_Add:
    reassociateFMA(Root, Pattern, InsInstrs, DelInstrs, InstrIdxForVirtReg);
    break;
  default:
    // Reassociate default patterns.
    TargetInstrInfo::genAlternativeCodeSequence(Root, Pattern, InsInstrs,
                                                DelInstrs, InstrIdxForVirtReg);
    break;
  }
}

static void getSplitFMAOpcodes(unsigned FMAOpc, unsigned &AddOpc,
                               unsigned &MulOpc) {
  switch (FMAOpc) {
  case SystemZ::VFMADB: AddOpc = SystemZ::VFADB; MulOpc = SystemZ::VFMDB; break;
  case SystemZ::VFMASB: AddOpc = SystemZ::VFASB; MulOpc = SystemZ::VFMSB; break;
  case SystemZ::WFMAXB: AddOpc = SystemZ::WFAXB; MulOpc = SystemZ::WFMXB; break;
  case SystemZ::WFMADB_CCPseudo:
      AddOpc = SystemZ::WFADB_CCPseudo; MulOpc = SystemZ::WFMDB; break;
  case SystemZ::WFMASB_CCPseudo:
      AddOpc = SystemZ::WFASB_CCPseudo; MulOpc = SystemZ::WFMSB; break;
  default:
    llvm_unreachable("Expected FMA opcode.");
  }
}

void SystemZInstrInfo::reassociateFMA(
    MachineInstr &Root, MachineCombinerPattern Pattern,
    SmallVectorImpl<MachineInstr *> &InsInstrs,
    SmallVectorImpl<MachineInstr *> &DelInstrs,
    DenseMap<unsigned, unsigned> &InstrIdxForVirtReg) const {
  MachineFunction *MF = Root.getMF();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterInfo *TRI = &getRegisterInfo();

  const TargetRegisterClass *RC = Root.getRegClassConstraint(0, this, TRI);
  Register DstReg = Root.getOperand(0).getReg();
  std::vector<MachineInstr *> Chain; // XXX Rework this method for final patterns used.
  Chain.push_back(&Root);

  uint16_t IntersectedFlags = Root.getFlags();
  auto getIntersectedFlags = [&]() {
    for (auto *MI : Chain)
      IntersectedFlags &= MI->getFlags();
  };

  auto createNewVReg = [&](unsigned NewInsIdx) -> Register {
    Register NewReg = MRI.createVirtualRegister(RC);
    InstrIdxForVirtReg.insert(std::make_pair(NewReg, NewInsIdx));
    return NewReg;
  };

  auto finalizeNewMIs = [&](ArrayRef<MachineInstr *> NewMIs) {
    for (auto *MI : NewMIs) {
      setSpecialOperandAttr(*MI, IntersectedFlags);
      MI->addRegisterDead(SystemZ::CC, TRI);
      InsInstrs.push_back(MI);
    }
  };

  auto deleteOld = [&InsInstrs, &DelInstrs, &Chain]() {
    assert(!InsInstrs.empty() &&
           "Insertion instructions set should not be empty!");
    // Record old instructions for deletion.
    for (auto *MI : make_range(Chain.rbegin(), Chain.rend()))
      DelInstrs.push_back(MI);
  };

  assert(IsReassociableFMA(&Root));
  unsigned FMAOpc = Root.getOpcode();
  unsigned AddOpc, MulOpc;
  getSplitFMAOpcodes(FMAOpc, AddOpc, MulOpc);

#ifndef NDEBUG
  auto IsAllFMA = [&Chain, &FMAOpc]() {
    for (auto *MI : Chain)
      if (MI->getOpcode() != FMAOpc)
        return false;
    return true;
  };
#endif

  switch (Pattern) {
  case MachineCombinerPattern::FMA2_P1P0:
  case MachineCombinerPattern::FMA2_P0P1: {
    if (Pattern == MachineCombinerPattern::FMA2_P1P0)
      LLVM_DEBUG(dbgs() << "reassociating using pattern FMA_P1P0\n");
    else
      LLVM_DEBUG(dbgs() << "reassociating using pattern FMA_P0P1\n");
    Chain.push_back(MRI.getUniqueVRegDef(Chain.back()->getOperand(3).getReg()));
    assert(IsAllFMA());
    getIntersectedFlags(); // XXX Refactor (here and below)
    Register NewVRA = createNewVReg(0);
    Register NewVRB = createNewVReg(1);
    unsigned FirstMulIdx =
      Pattern == MachineCombinerPattern::FMA2_P1P0 ? 1 : 0;
    unsigned SecondMulIdx = FirstMulIdx == 0 ? 1 : 0;
    MachineInstr *MINewA =
        BuildMI(*MF, Chain[FirstMulIdx]->getDebugLoc(), get(MulOpc), NewVRA)
            .add(Chain[FirstMulIdx]->getOperand(1))
            .add(Chain[FirstMulIdx]->getOperand(2));
    MachineInstr *MINewB =
        BuildMI(*MF, Chain[SecondMulIdx]->getDebugLoc(), get(FMAOpc), NewVRB)
           .add(Chain[SecondMulIdx]->getOperand(1))
            .add(Chain[SecondMulIdx]->getOperand(2))
            .addReg(NewVRA);
    MachineInstr *MINewC =
        BuildMI(*MF, Chain[1]->getDebugLoc(), get(AddOpc), DstReg)
            .add(Chain[1]->getOperand(3))
            .addReg(NewVRB);
    finalizeNewMIs({MINewA, MINewB, MINewC});
    break;
  }
  case MachineCombinerPattern::FMA2: {
    LLVM_DEBUG(dbgs() << "reassociating using pattern FMA2\n");
    Chain.push_back(MRI.getUniqueVRegDef(Chain.back()->getOperand(3).getReg()));
    assert(IsAllFMA());
    getIntersectedFlags();
    Register NewVRA = createNewVReg(0);
    MachineInstr *MINewA =
        BuildMI(*MF, Chain[0]->getDebugLoc(), get(FMAOpc), NewVRA)
            .add(Chain[0]->getOperand(1))
            .add(Chain[0]->getOperand(2))
            .add(Chain[1]->getOperand(3));
    MachineInstr *MINewB =
        BuildMI(*MF, Chain[1]->getDebugLoc(), get(FMAOpc), DstReg)
            .add(Chain[1]->getOperand(1))
            .add(Chain[1]->getOperand(2))
            .addReg(NewVRA);
    finalizeNewMIs({MINewA, MINewB});
    break;
  }
  case MachineCombinerPattern::FMA1_Add_L:
  case MachineCombinerPattern::FMA1_Add_R: {
    if (Pattern == MachineCombinerPattern::FMA1_Add_L)
      LLVM_DEBUG(dbgs() << "reassociating using pattern FMA1_Add_L\n");
    else
      LLVM_DEBUG(dbgs() << "reassociating using pattern FMA1_Add_R\n");
    assert(IsAllFMA());
    Chain.push_back(MRI.getUniqueVRegDef(Chain.back()->getOperand(3).getReg()));
    assert(Chain.back()->getOpcode() == AddOpc && "Expected matching Add");
    getIntersectedFlags();
    unsigned Op = Pattern == MachineCombinerPattern::FMA1_Add_L ? 1 : 2;
    unsigned OtherOp = Op == 1 ? 2 : 1;
    Register NewVRA = createNewVReg(0);
    MachineInstr *MINewA =
        BuildMI(*MF, Chain[0]->getDebugLoc(), get(FMAOpc), NewVRA)
            .add(Chain[0]->getOperand(1))
            .add(Chain[0]->getOperand(2))
            .add(Chain[1]->getOperand(Op));
    MachineInstr *MINewB =
        BuildMI(*MF, Chain[1]->getDebugLoc(), get(AddOpc), DstReg)
            .addReg(NewVRA)
            .add(Chain[1]->getOperand(OtherOp));
    finalizeNewMIs({MINewA, MINewB});
    break;
  }
  case MachineCombinerPattern::FMA3: {
    LLVM_DEBUG(dbgs() << "reassociating using pattern FMA3\n");
    Chain.push_back(MRI.getUniqueVRegDef(Chain.back()->getOperand(3).getReg()));
    Chain.push_back(MRI.getUniqueVRegDef(Chain.back()->getOperand(3).getReg()));
    assert(IsAllFMA());
    getIntersectedFlags();
    Register NewVRA = createNewVReg(0);
    Register NewVRB = createNewVReg(1);
    Register NewVRC = createNewVReg(2);
    MachineInstr *MINewA =
        BuildMI(*MF, Chain[2]->getDebugLoc(), get(MulOpc), NewVRA)
            .add(Chain[2]->getOperand(1))
            .add(Chain[2]->getOperand(2));
    MachineInstr *MINewB =
        BuildMI(*MF, Chain[1]->getDebugLoc(), get(FMAOpc), NewVRB)
            .add(Chain[1]->getOperand(1))
            .add(Chain[1]->getOperand(2))
            .add(Chain[2]->getOperand(3));
    MachineInstr *MINewC =
        BuildMI(*MF, Chain[0]->getDebugLoc(), get(FMAOpc), NewVRC)
            .add(Chain[0]->getOperand(1))
            .add(Chain[0]->getOperand(2))
            .addReg(NewVRA);
    MachineInstr *MINewD =
        BuildMI(*MF, Chain[0]->getDebugLoc(), get(AddOpc), DstReg)
            .addReg(NewVRB)
            .addReg(NewVRC);
    finalizeNewMIs({MINewA, MINewB, MINewC, MINewD});
    break;
  }
  case MachineCombinerPattern::FMA2_Add: {
    LLVM_DEBUG(dbgs() << "reassociating using pattern FMA2_Add\n");
    Chain.push_back(MRI.getUniqueVRegDef(Chain.back()->getOperand(3).getReg()));
    assert(IsAllFMA());
    Chain.push_back(MRI.getUniqueVRegDef(Chain.back()->getOperand(3).getReg()));
    assert(Chain.back()->getOpcode() == AddOpc && "Expected matching Add");
    getIntersectedFlags();
    Register NewVRA = createNewVReg(0);
    Register NewVRB = createNewVReg(1);
    MachineInstr *MINewA =
        BuildMI(*MF, Chain[1]->getDebugLoc(), get(FMAOpc), NewVRA)
            .add(Chain[1]->getOperand(1))
            .add(Chain[1]->getOperand(2))
            .add(Chain[2]->getOperand(1));
    MachineInstr *MINewB =
        BuildMI(*MF, Chain[0]->getDebugLoc(), get(FMAOpc), NewVRB)
            .add(Chain[0]->getOperand(1))
            .add(Chain[0]->getOperand(2))
            .add(Chain[2]->getOperand(2));
    MachineInstr *MINewC =
        BuildMI(*MF, Chain[0]->getDebugLoc(), get(AddOpc), DstReg)
            .addReg(NewVRA)
            .addReg(NewVRB);
    finalizeNewMIs({MINewA, MINewB, MINewC});
    break;
  }
  default:
    llvm_unreachable("not recognized pattern!");
  }

  deleteOld();
}

bool
SystemZInstrInfo::accumulateInstrSeqToRootLatency(MachineInstr &Root) const {
  // This doesn't make much sense for FMA patterns as they typically use an
  // extra Add to do things in parallell.
  if (IsReassociableFMA(&Root))    // XXX Fine tune this a bit depending on
                                   // used patterns.
    return false;

  return true;
}

void SystemZInstrInfo::setSpecialOperandAttr(MachineInstr &MI,
                                             uint32_t Flags) const {
  MI.setFlags(Flags);
  MI.clearFlag(MachineInstr::MIFlag::NoSWrap);
  MI.clearFlag(MachineInstr::MIFlag::NoUWrap);
  MI.clearFlag(MachineInstr::MIFlag::IsExact);
}

MachineInstr *SystemZInstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr &MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, int FrameIndex,
    LiveIntervals *LIS, VirtRegMap *VRM) const {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned Size = MFI.getObjectSize(FrameIndex);
  unsigned Opcode = MI.getOpcode();

  // Check CC liveness if new instruction introduces a dead def of CC.
  SlotIndex MISlot = SlotIndex();
  LiveRange *CCLiveRange = nullptr;
  bool CCLiveAtMI = true;
  if (LIS) {
    MISlot = LIS->getSlotIndexes()->getInstructionIndex(MI).getRegSlot();
    auto CCUnits = TRI->regunits(MCRegister::from(SystemZ::CC));
    assert(range_size(CCUnits) == 1 && "CC only has one reg unit.");
    CCLiveRange = &LIS->getRegUnit(*CCUnits.begin());
    CCLiveAtMI = CCLiveRange->liveAt(MISlot);
  }

  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    if (!CCLiveAtMI && (Opcode == SystemZ::LA || Opcode == SystemZ::LAY) &&
        isInt<8>(MI.getOperand(2).getImm()) && !MI.getOperand(3).getReg()) {
      // LA(Y) %reg, CONST(%reg) -> AGSI %mem, CONST
      MachineInstr *BuiltMI = BuildMI(*InsertPt->getParent(), InsertPt,
                                      MI.getDebugLoc(), get(SystemZ::AGSI))
        .addFrameIndex(FrameIndex)
        .addImm(0)
        .addImm(MI.getOperand(2).getImm());
      BuiltMI->findRegisterDefOperand(SystemZ::CC, /*TRI=*/nullptr)
          ->setIsDead(true);
      CCLiveRange->createDeadDef(MISlot, LIS->getVNInfoAllocator());
      return BuiltMI;
    }
    return nullptr;
  }

  // All other cases require a single operand.
  if (Ops.size() != 1)
    return nullptr;

  unsigned OpNum = Ops[0];
  assert(Size * 8 ==
           TRI->getRegSizeInBits(*MF.getRegInfo()
                               .getRegClass(MI.getOperand(OpNum).getReg())) &&
         "Invalid size combination");

  if ((Opcode == SystemZ::AHI || Opcode == SystemZ::AGHI) && OpNum == 0 &&
      isInt<8>(MI.getOperand(2).getImm())) {
    // A(G)HI %reg, CONST -> A(G)SI %mem, CONST
    Opcode = (Opcode == SystemZ::AHI ? SystemZ::ASI : SystemZ::AGSI);
    MachineInstr *BuiltMI =
        BuildMI(*InsertPt->getParent(), InsertPt, MI.getDebugLoc(), get(Opcode))
            .addFrameIndex(FrameIndex)
            .addImm(0)
            .addImm(MI.getOperand(2).getImm());
    transferDeadCC(&MI, BuiltMI);
    transferMIFlag(&MI, BuiltMI, MachineInstr::NoSWrap);
    return BuiltMI;
  }

  if ((Opcode == SystemZ::ALFI && OpNum == 0 &&
       isInt<8>((int32_t)MI.getOperand(2).getImm())) ||
      (Opcode == SystemZ::ALGFI && OpNum == 0 &&
       isInt<8>((int64_t)MI.getOperand(2).getImm()))) {
    // AL(G)FI %reg, CONST -> AL(G)SI %mem, CONST
    Opcode = (Opcode == SystemZ::ALFI ? SystemZ::ALSI : SystemZ::ALGSI);
    MachineInstr *BuiltMI =
        BuildMI(*InsertPt->getParent(), InsertPt, MI.getDebugLoc(), get(Opcode))
            .addFrameIndex(FrameIndex)
            .addImm(0)
            .addImm((int8_t)MI.getOperand(2).getImm());
    transferDeadCC(&MI, BuiltMI);
    return BuiltMI;
  }

  if ((Opcode == SystemZ::SLFI && OpNum == 0 &&
       isInt<8>((int32_t)-MI.getOperand(2).getImm())) ||
      (Opcode == SystemZ::SLGFI && OpNum == 0 &&
       isInt<8>((int64_t)-MI.getOperand(2).getImm()))) {
    // SL(G)FI %reg, CONST -> AL(G)SI %mem, -CONST
    Opcode = (Opcode == SystemZ::SLFI ? SystemZ::ALSI : SystemZ::ALGSI);
    MachineInstr *BuiltMI =
        BuildMI(*InsertPt->getParent(), InsertPt, MI.getDebugLoc(), get(Opcode))
            .addFrameIndex(FrameIndex)
            .addImm(0)
            .addImm((int8_t)-MI.getOperand(2).getImm());
    transferDeadCC(&MI, BuiltMI);
    return BuiltMI;
  }

  unsigned MemImmOpc = 0;
  switch (Opcode) {
  case SystemZ::LHIMux:
  case SystemZ::LHI:    MemImmOpc = SystemZ::MVHI;  break;
  case SystemZ::LGHI:   MemImmOpc = SystemZ::MVGHI; break;
  case SystemZ::CHIMux:
  case SystemZ::CHI:    MemImmOpc = SystemZ::CHSI;  break;
  case SystemZ::CGHI:   MemImmOpc = SystemZ::CGHSI; break;
  case SystemZ::CLFIMux:
  case SystemZ::CLFI:
    if (isUInt<16>(MI.getOperand(1).getImm()))
      MemImmOpc = SystemZ::CLFHSI;
    break;
  case SystemZ::CLGFI:
    if (isUInt<16>(MI.getOperand(1).getImm()))
      MemImmOpc = SystemZ::CLGHSI;
    break;
  default: break;
  }
  if (MemImmOpc)
    return BuildMI(*InsertPt->getParent(), InsertPt, MI.getDebugLoc(),
                   get(MemImmOpc))
               .addFrameIndex(FrameIndex)
               .addImm(0)
               .addImm(MI.getOperand(1).getImm());

  if (Opcode == SystemZ::LGDR || Opcode == SystemZ::LDGR) {
    bool Op0IsGPR = (Opcode == SystemZ::LGDR);
    bool Op1IsGPR = (Opcode == SystemZ::LDGR);
    // If we're spilling the destination of an LDGR or LGDR, store the
    // source register instead.
    if (OpNum == 0) {
      unsigned StoreOpcode = Op1IsGPR ? SystemZ::STG : SystemZ::STD;
      return BuildMI(*InsertPt->getParent(), InsertPt, MI.getDebugLoc(),
                     get(StoreOpcode))
          .add(MI.getOperand(1))
          .addFrameIndex(FrameIndex)
          .addImm(0)
          .addReg(0);
    }
    // If we're spilling the source of an LDGR or LGDR, load the
    // destination register instead.
    if (OpNum == 1) {
      unsigned LoadOpcode = Op0IsGPR ? SystemZ::LG : SystemZ::LD;
      return BuildMI(*InsertPt->getParent(), InsertPt, MI.getDebugLoc(),
                     get(LoadOpcode))
        .add(MI.getOperand(0))
        .addFrameIndex(FrameIndex)
        .addImm(0)
        .addReg(0);
    }
  }

  // Look for cases where the source of a simple store or the destination
  // of a simple load is being spilled.  Try to use MVC instead.
  //
  // Although MVC is in practice a fast choice in these cases, it is still
  // logically a bytewise copy.  This means that we cannot use it if the
  // load or store is volatile.  We also wouldn't be able to use MVC if
  // the two memories partially overlap, but that case cannot occur here,
  // because we know that one of the memories is a full frame index.
  //
  // For performance reasons, we also want to avoid using MVC if the addresses
  // might be equal.  We don't worry about that case here, because spill slot
  // coloring happens later, and because we have special code to remove
  // MVCs that turn out to be redundant.
  if (OpNum == 0 && MI.hasOneMemOperand()) {
    MachineMemOperand *MMO = *MI.memoperands_begin();
    if (MMO->getSize() == Size && !MMO->isVolatile() && !MMO->isAtomic()) {
      // Handle conversion of loads.
      if (isSimpleBD12Move(&MI, SystemZII::SimpleBDXLoad)) {
        return BuildMI(*InsertPt->getParent(), InsertPt, MI.getDebugLoc(),
                       get(SystemZ::MVC))
            .addFrameIndex(FrameIndex)
            .addImm(0)
            .addImm(Size)
            .add(MI.getOperand(1))
            .addImm(MI.getOperand(2).getImm())
            .addMemOperand(MMO);
      }
      // Handle conversion of stores.
      if (isSimpleBD12Move(&MI, SystemZII::SimpleBDXStore)) {
        return BuildMI(*InsertPt->getParent(), InsertPt, MI.getDebugLoc(),
                       get(SystemZ::MVC))
            .add(MI.getOperand(1))
            .addImm(MI.getOperand(2).getImm())
            .addImm(Size)
            .addFrameIndex(FrameIndex)
            .addImm(0)
            .addMemOperand(MMO);
      }
    }
  }

  // If the spilled operand is the final one or the instruction is
  // commutable, try to change <INSN>R into <INSN>.  Don't introduce a def of
  // CC if it is live and MI does not define it.
  unsigned NumOps = MI.getNumExplicitOperands();
  int MemOpcode = SystemZ::getMemOpcode(Opcode);
  if (MemOpcode == -1 ||
      (CCLiveAtMI && !MI.definesRegister(SystemZ::CC, /*TRI=*/nullptr) &&
       get(MemOpcode).hasImplicitDefOfPhysReg(SystemZ::CC)))
    return nullptr;

  // Check if all other vregs have a usable allocation in the case of vector
  // to FP conversion.
  const MCInstrDesc &MCID = MI.getDesc();
  for (unsigned I = 0, E = MCID.getNumOperands(); I != E; ++I) {
    const MCOperandInfo &MCOI = MCID.operands()[I];
    if (MCOI.OperandType != MCOI::OPERAND_REGISTER || I == OpNum)
      continue;
    const TargetRegisterClass *RC = TRI->getRegClass(MCOI.RegClass);
    if (RC == &SystemZ::VR32BitRegClass || RC == &SystemZ::VR64BitRegClass) {
      Register Reg = MI.getOperand(I).getReg();
      Register PhysReg = Reg.isVirtual()
                             ? (VRM ? Register(VRM->getPhys(Reg)) : Register())
                             : Reg;
      if (!PhysReg ||
          !(SystemZ::FP32BitRegClass.contains(PhysReg) ||
            SystemZ::FP64BitRegClass.contains(PhysReg) ||
            SystemZ::VF128BitRegClass.contains(PhysReg)))
        return nullptr;
    }
  }
  // Fused multiply and add/sub need to have the same dst and accumulator reg.
  bool FusedFPOp = (Opcode == SystemZ::WFMADB || Opcode == SystemZ::WFMASB ||
                    Opcode == SystemZ::WFMSDB || Opcode == SystemZ::WFMSSB);
  if (FusedFPOp) {
    Register DstReg = VRM->getPhys(MI.getOperand(0).getReg());
    Register AccReg = VRM->getPhys(MI.getOperand(3).getReg());
    if (OpNum == 0 || OpNum == 3 || DstReg != AccReg)
      return nullptr;
  }

  // Try to swap compare operands if possible.
  bool NeedsCommute = false;
  if ((MI.getOpcode() == SystemZ::CR || MI.getOpcode() == SystemZ::CGR ||
       MI.getOpcode() == SystemZ::CLR || MI.getOpcode() == SystemZ::CLGR ||
       MI.getOpcode() == SystemZ::WFCDB || MI.getOpcode() == SystemZ::WFCSB ||
       MI.getOpcode() == SystemZ::WFKDB || MI.getOpcode() == SystemZ::WFKSB) &&
      OpNum == 0 && prepareCompareSwapOperands(MI))
    NeedsCommute = true;

  bool CCOperands = false;
  if (MI.getOpcode() == SystemZ::LOCRMux || MI.getOpcode() == SystemZ::LOCGR ||
      MI.getOpcode() == SystemZ::SELRMux || MI.getOpcode() == SystemZ::SELGR) {
    assert(MI.getNumOperands() == 6 && NumOps == 5 &&
           "LOCR/SELR instruction operands corrupt?");
    NumOps -= 2;
    CCOperands = true;
  }

  // See if this is a 3-address instruction that is convertible to 2-address
  // and suitable for folding below.  Only try this with virtual registers
  // and a provided VRM (during regalloc).
  if (NumOps == 3 && SystemZ::getTargetMemOpcode(MemOpcode) != -1) {
    if (VRM == nullptr)
      return nullptr;
    else {
      Register DstReg = MI.getOperand(0).getReg();
      Register DstPhys =
          (DstReg.isVirtual() ? Register(VRM->getPhys(DstReg)) : DstReg);
      Register SrcReg = (OpNum == 2 ? MI.getOperand(1).getReg()
                                    : ((OpNum == 1 && MI.isCommutable())
                                           ? MI.getOperand(2).getReg()
                                           : Register()));
      if (DstPhys && !SystemZ::GRH32BitRegClass.contains(DstPhys) && SrcReg &&
          SrcReg.isVirtual() && DstPhys == VRM->getPhys(SrcReg))
        NeedsCommute = (OpNum == 1);
      else
        return nullptr;
    }
  }

  if ((OpNum == NumOps - 1) || NeedsCommute || FusedFPOp) {
    const MCInstrDesc &MemDesc = get(MemOpcode);
    uint64_t AccessBytes = SystemZII::getAccessSize(MemDesc.TSFlags);
    assert(AccessBytes != 0 && "Size of access should be known");
    assert(AccessBytes <= Size && "Access outside the frame index");
    uint64_t Offset = Size - AccessBytes;
    MachineInstrBuilder MIB = BuildMI(*InsertPt->getParent(), InsertPt,
                                      MI.getDebugLoc(), get(MemOpcode));
    if (MI.isCompare()) {
      assert(NumOps == 2 && "Expected 2 register operands for a compare.");
      MIB.add(MI.getOperand(NeedsCommute ? 1 : 0));
    }
    else if (FusedFPOp) {
      MIB.add(MI.getOperand(0));
      MIB.add(MI.getOperand(3));
      MIB.add(MI.getOperand(OpNum == 1 ? 2 : 1));
    }
    else {
      MIB.add(MI.getOperand(0));
      if (NeedsCommute)
        MIB.add(MI.getOperand(2));
      else
        for (unsigned I = 1; I < OpNum; ++I)
          MIB.add(MI.getOperand(I));
    }
    MIB.addFrameIndex(FrameIndex).addImm(Offset);
    if (MemDesc.TSFlags & SystemZII::HasIndex)
      MIB.addReg(0);
    if (CCOperands) {
      unsigned CCValid = MI.getOperand(NumOps).getImm();
      unsigned CCMask = MI.getOperand(NumOps + 1).getImm();
      MIB.addImm(CCValid);
      MIB.addImm(NeedsCommute ? CCMask ^ CCValid : CCMask);
    }
    if (MIB->definesRegister(SystemZ::CC, /*TRI=*/nullptr) &&
        (!MI.definesRegister(SystemZ::CC, /*TRI=*/nullptr) ||
         MI.registerDefIsDead(SystemZ::CC, /*TRI=*/nullptr))) {
      MIB->addRegisterDead(SystemZ::CC, TRI);
      if (CCLiveRange)
        CCLiveRange->createDeadDef(MISlot, LIS->getVNInfoAllocator());
    }
    // Constrain the register classes if converted from a vector opcode. The
    // allocated regs are in an FP reg-class per previous check above.
    for (const MachineOperand &MO : MIB->operands())
      if (MO.isReg() && MO.getReg().isVirtual()) {
        Register Reg = MO.getReg();
        if (MRI.getRegClass(Reg) == &SystemZ::VR32BitRegClass)
          MRI.setRegClass(Reg, &SystemZ::FP32BitRegClass);
        else if (MRI.getRegClass(Reg) == &SystemZ::VR64BitRegClass)
          MRI.setRegClass(Reg, &SystemZ::FP64BitRegClass);
        else if (MRI.getRegClass(Reg) == &SystemZ::VR128BitRegClass)
          MRI.setRegClass(Reg, &SystemZ::VF128BitRegClass);
      }

    transferDeadCC(&MI, MIB);
    transferMIFlag(&MI, MIB, MachineInstr::NoSWrap);
    transferMIFlag(&MI, MIB, MachineInstr::NoFPExcept);
    return MIB;
  }

  return nullptr;
}

MachineInstr *SystemZInstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr &MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, MachineInstr &LoadMI,
    LiveIntervals *LIS) const {
  return nullptr;
}

bool SystemZInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case SystemZ::L128:
    splitMove(MI, SystemZ::LG);
    return true;

  case SystemZ::ST128:
    splitMove(MI, SystemZ::STG);
    return true;

  case SystemZ::LX:
    splitMove(MI, SystemZ::LD);
    return true;

  case SystemZ::STX:
    splitMove(MI, SystemZ::STD);
    return true;

  case SystemZ::LBMux:
    expandRXYPseudo(MI, SystemZ::LB, SystemZ::LBH);
    return true;

  case SystemZ::LHMux:
    expandRXYPseudo(MI, SystemZ::LH, SystemZ::LHH);
    return true;

  case SystemZ::LLCRMux:
    expandZExtPseudo(MI, SystemZ::LLCR, 8);
    return true;

  case SystemZ::LLHRMux:
    expandZExtPseudo(MI, SystemZ::LLHR, 16);
    return true;

  case SystemZ::LLCMux:
    expandRXYPseudo(MI, SystemZ::LLC, SystemZ::LLCH);
    return true;

  case SystemZ::LLHMux:
    expandRXYPseudo(MI, SystemZ::LLH, SystemZ::LLHH);
    return true;

  case SystemZ::LMux:
    expandRXYPseudo(MI, SystemZ::L, SystemZ::LFH);
    return true;

  case SystemZ::LOCMux:
    expandLOCPseudo(MI, SystemZ::LOC, SystemZ::LOCFH);
    return true;

  case SystemZ::LOCHIMux:
    expandLOCPseudo(MI, SystemZ::LOCHI, SystemZ::LOCHHI);
    return true;

  case SystemZ::STCMux:
    expandRXYPseudo(MI, SystemZ::STC, SystemZ::STCH);
    return true;

  case SystemZ::STHMux:
    expandRXYPseudo(MI, SystemZ::STH, SystemZ::STHH);
    return true;

  case SystemZ::STMux:
    expandRXYPseudo(MI, SystemZ::ST, SystemZ::STFH);
    return true;

  case SystemZ::STOCMux:
    expandLOCPseudo(MI, SystemZ::STOC, SystemZ::STOCFH);
    return true;

  case SystemZ::LHIMux:
    expandRIPseudo(MI, SystemZ::LHI, SystemZ::IIHF, true);
    return true;

  case SystemZ::IIFMux:
    expandRIPseudo(MI, SystemZ::IILF, SystemZ::IIHF, false);
    return true;

  case SystemZ::IILMux:
    expandRIPseudo(MI, SystemZ::IILL, SystemZ::IIHL, false);
    return true;

  case SystemZ::IIHMux:
    expandRIPseudo(MI, SystemZ::IILH, SystemZ::IIHH, false);
    return true;

  case SystemZ::NIFMux:
    expandRIPseudo(MI, SystemZ::NILF, SystemZ::NIHF, false);
    return true;

  case SystemZ::NILMux:
    expandRIPseudo(MI, SystemZ::NILL, SystemZ::NIHL, false);
    return true;

  case SystemZ::NIHMux:
    expandRIPseudo(MI, SystemZ::NILH, SystemZ::NIHH, false);
    return true;

  case SystemZ::OIFMux:
    expandRIPseudo(MI, SystemZ::OILF, SystemZ::OIHF, false);
    return true;

  case SystemZ::OILMux:
    expandRIPseudo(MI, SystemZ::OILL, SystemZ::OIHL, false);
    return true;

  case SystemZ::OIHMux:
    expandRIPseudo(MI, SystemZ::OILH, SystemZ::OIHH, false);
    return true;

  case SystemZ::XIFMux:
    expandRIPseudo(MI, SystemZ::XILF, SystemZ::XIHF, false);
    return true;

  case SystemZ::TMLMux:
    expandRIPseudo(MI, SystemZ::TMLL, SystemZ::TMHL, false);
    return true;

  case SystemZ::TMHMux:
    expandRIPseudo(MI, SystemZ::TMLH, SystemZ::TMHH, false);
    return true;

  case SystemZ::AHIMux:
    expandRIPseudo(MI, SystemZ::AHI, SystemZ::AIH, false);
    return true;

  case SystemZ::AHIMuxK:
    expandRIEPseudo(MI, SystemZ::AHI, SystemZ::AHIK, SystemZ::AIH);
    return true;

  case SystemZ::AFIMux:
    expandRIPseudo(MI, SystemZ::AFI, SystemZ::AIH, false);
    return true;

  case SystemZ::CHIMux:
    expandRIPseudo(MI, SystemZ::CHI, SystemZ::CIH, false);
    return true;

  case SystemZ::CFIMux:
    expandRIPseudo(MI, SystemZ::CFI, SystemZ::CIH, false);
    return true;

  case SystemZ::CLFIMux:
    expandRIPseudo(MI, SystemZ::CLFI, SystemZ::CLIH, false);
    return true;

  case SystemZ::CMux:
    expandRXYPseudo(MI, SystemZ::C, SystemZ::CHF);
    return true;

  case SystemZ::CLMux:
    expandRXYPseudo(MI, SystemZ::CL, SystemZ::CLHF);
    return true;

  case SystemZ::RISBMux: {
    bool DestIsHigh = SystemZ::isHighReg(MI.getOperand(0).getReg());
    bool SrcIsHigh = SystemZ::isHighReg(MI.getOperand(2).getReg());
    if (SrcIsHigh == DestIsHigh)
      MI.setDesc(get(DestIsHigh ? SystemZ::RISBHH : SystemZ::RISBLL));
    else {
      MI.setDesc(get(DestIsHigh ? SystemZ::RISBHL : SystemZ::RISBLH));
      MI.getOperand(5).setImm(MI.getOperand(5).getImm() ^ 32);
    }
    return true;
  }

  case SystemZ::ADJDYNALLOC:
    splitAdjDynAlloc(MI);
    return true;

  case TargetOpcode::LOAD_STACK_GUARD:
    expandLoadStackGuard(&MI);
    return true;

  default:
    return false;
  }
}

unsigned SystemZInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  if (MI.isInlineAsm()) {
    const MachineFunction *MF = MI.getParent()->getParent();
    const char *AsmStr = MI.getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  }
  else if (MI.getOpcode() == SystemZ::PATCHPOINT)
    return PatchPointOpers(&MI).getNumPatchBytes();
  else if (MI.getOpcode() == SystemZ::STACKMAP)
    return MI.getOperand(1).getImm();
  else if (MI.getOpcode() == SystemZ::FENTRY_CALL)
    return 6;

  return MI.getDesc().getSize();
}

SystemZII::Branch
SystemZInstrInfo::getBranchInfo(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case SystemZ::BR:
  case SystemZ::BI:
  case SystemZ::J:
  case SystemZ::JG:
    return SystemZII::Branch(SystemZII::BranchNormal, SystemZ::CCMASK_ANY,
                             SystemZ::CCMASK_ANY, &MI.getOperand(0));

  case SystemZ::BRC:
  case SystemZ::BRCL:
    return SystemZII::Branch(SystemZII::BranchNormal, MI.getOperand(0).getImm(),
                             MI.getOperand(1).getImm(), &MI.getOperand(2));

  case SystemZ::BRCT:
  case SystemZ::BRCTH:
    return SystemZII::Branch(SystemZII::BranchCT, SystemZ::CCMASK_ICMP,
                             SystemZ::CCMASK_CMP_NE, &MI.getOperand(2));

  case SystemZ::BRCTG:
    return SystemZII::Branch(SystemZII::BranchCTG, SystemZ::CCMASK_ICMP,
                             SystemZ::CCMASK_CMP_NE, &MI.getOperand(2));

  case SystemZ::CIJ:
  case SystemZ::CRJ:
    return SystemZII::Branch(SystemZII::BranchC, SystemZ::CCMASK_ICMP,
                             MI.getOperand(2).getImm(), &MI.getOperand(3));

  case SystemZ::CLIJ:
  case SystemZ::CLRJ:
    return SystemZII::Branch(SystemZII::BranchCL, SystemZ::CCMASK_ICMP,
                             MI.getOperand(2).getImm(), &MI.getOperand(3));

  case SystemZ::CGIJ:
  case SystemZ::CGRJ:
    return SystemZII::Branch(SystemZII::BranchCG, SystemZ::CCMASK_ICMP,
                             MI.getOperand(2).getImm(), &MI.getOperand(3));

  case SystemZ::CLGIJ:
  case SystemZ::CLGRJ:
    return SystemZII::Branch(SystemZII::BranchCLG, SystemZ::CCMASK_ICMP,
                             MI.getOperand(2).getImm(), &MI.getOperand(3));

  case SystemZ::INLINEASM_BR:
    // Don't try to analyze asm goto, so pass nullptr as branch target argument.
    return SystemZII::Branch(SystemZII::AsmGoto, 0, 0, nullptr);

  default:
    llvm_unreachable("Unrecognized branch opcode");
  }
}

void SystemZInstrInfo::getLoadStoreOpcodes(const TargetRegisterClass *RC,
                                           unsigned &LoadOpcode,
                                           unsigned &StoreOpcode) const {
  if (RC == &SystemZ::GR32BitRegClass || RC == &SystemZ::ADDR32BitRegClass) {
    LoadOpcode = SystemZ::L;
    StoreOpcode = SystemZ::ST;
  } else if (RC == &SystemZ::GRH32BitRegClass) {
    LoadOpcode = SystemZ::LFH;
    StoreOpcode = SystemZ::STFH;
  } else if (RC == &SystemZ::GRX32BitRegClass) {
    LoadOpcode = SystemZ::LMux;
    StoreOpcode = SystemZ::STMux;
  } else if (RC == &SystemZ::GR64BitRegClass ||
             RC == &SystemZ::ADDR64BitRegClass) {
    LoadOpcode = SystemZ::LG;
    StoreOpcode = SystemZ::STG;
  } else if (RC == &SystemZ::GR128BitRegClass ||
             RC == &SystemZ::ADDR128BitRegClass) {
    LoadOpcode = SystemZ::L128;
    StoreOpcode = SystemZ::ST128;
  } else if (RC == &SystemZ::FP32BitRegClass) {
    LoadOpcode = SystemZ::LE;
    StoreOpcode = SystemZ::STE;
  } else if (RC == &SystemZ::FP64BitRegClass) {
    LoadOpcode = SystemZ::LD;
    StoreOpcode = SystemZ::STD;
  } else if (RC == &SystemZ::FP128BitRegClass) {
    LoadOpcode = SystemZ::LX;
    StoreOpcode = SystemZ::STX;
  } else if (RC == &SystemZ::VR32BitRegClass) {
    LoadOpcode = SystemZ::VL32;
    StoreOpcode = SystemZ::VST32;
  } else if (RC == &SystemZ::VR64BitRegClass) {
    LoadOpcode = SystemZ::VL64;
    StoreOpcode = SystemZ::VST64;
  } else if (RC == &SystemZ::VF128BitRegClass ||
             RC == &SystemZ::VR128BitRegClass) {
    LoadOpcode = SystemZ::VL;
    StoreOpcode = SystemZ::VST;
  } else
    llvm_unreachable("Unsupported regclass to load or store");
}

unsigned SystemZInstrInfo::getOpcodeForOffset(unsigned Opcode,
                                              int64_t Offset,
                                              const MachineInstr *MI) const {
  const MCInstrDesc &MCID = get(Opcode);
  int64_t Offset2 = (MCID.TSFlags & SystemZII::Is128Bit ? Offset + 8 : Offset);
  if (isUInt<12>(Offset) && isUInt<12>(Offset2)) {
    // Get the instruction to use for unsigned 12-bit displacements.
    int Disp12Opcode = SystemZ::getDisp12Opcode(Opcode);
    if (Disp12Opcode >= 0)
      return Disp12Opcode;

    // All address-related instructions can use unsigned 12-bit
    // displacements.
    return Opcode;
  }
  if (isInt<20>(Offset) && isInt<20>(Offset2)) {
    // Get the instruction to use for signed 20-bit displacements.
    int Disp20Opcode = SystemZ::getDisp20Opcode(Opcode);
    if (Disp20Opcode >= 0)
      return Disp20Opcode;

    // Check whether Opcode allows signed 20-bit displacements.
    if (MCID.TSFlags & SystemZII::Has20BitOffset)
      return Opcode;

    // If a VR32/VR64 reg ended up in an FP register, use the FP opcode.
    if (MI && MI->getOperand(0).isReg()) {
      Register Reg = MI->getOperand(0).getReg();
      if (Reg.isPhysical() && SystemZMC::getFirstReg(Reg) < 16) {
        switch (Opcode) {
        case SystemZ::VL32:
          return SystemZ::LEY;
        case SystemZ::VST32:
          return SystemZ::STEY;
        case SystemZ::VL64:
          return SystemZ::LDY;
        case SystemZ::VST64:
          return SystemZ::STDY;
        default: break;
        }
      }
    }
  }
  return 0;
}

bool SystemZInstrInfo::hasDisplacementPairInsn(unsigned Opcode) const {
  const MCInstrDesc &MCID = get(Opcode);
  if (MCID.TSFlags & SystemZII::Has20BitOffset)
    return SystemZ::getDisp12Opcode(Opcode) >= 0;
  return SystemZ::getDisp20Opcode(Opcode) >= 0;
}

unsigned SystemZInstrInfo::getLoadAndTest(unsigned Opcode) const {
  switch (Opcode) {
  case SystemZ::L:      return SystemZ::LT;
  case SystemZ::LY:     return SystemZ::LT;
  case SystemZ::LG:     return SystemZ::LTG;
  case SystemZ::LGF:    return SystemZ::LTGF;
  case SystemZ::LR:     return SystemZ::LTR;
  case SystemZ::LGFR:   return SystemZ::LTGFR;
  case SystemZ::LGR:    return SystemZ::LTGR;
  case SystemZ::LCDFR:  return SystemZ::LCDBR;
  case SystemZ::LPDFR:  return SystemZ::LPDBR;
  case SystemZ::LNDFR:  return SystemZ::LNDBR;
  case SystemZ::LCDFR_32:  return SystemZ::LCEBR;
  case SystemZ::LPDFR_32:  return SystemZ::LPEBR;
  case SystemZ::LNDFR_32:  return SystemZ::LNEBR;
  // On zEC12 we prefer to use RISBGN.  But if there is a chance to
  // actually use the condition code, we may turn it back into RISGB.
  // Note that RISBG is not really a "load-and-test" instruction,
  // but sets the same condition code values, so is OK to use here.
  case SystemZ::RISBGN: return SystemZ::RISBG;
  default:              return 0;
  }
}

bool SystemZInstrInfo::isRxSBGMask(uint64_t Mask, unsigned BitSize,
                                   unsigned &Start, unsigned &End) const {
  // Reject trivial all-zero masks.
  Mask &= allOnes(BitSize);
  if (Mask == 0)
    return false;

  // Handle the 1+0+ or 0+1+0* cases.  Start then specifies the index of
  // the msb and End specifies the index of the lsb.
  unsigned LSB, Length;
  if (isShiftedMask_64(Mask, LSB, Length)) {
    Start = 63 - (LSB + Length - 1);
    End = 63 - LSB;
    return true;
  }

  // Handle the wrap-around 1+0+1+ cases.  Start then specifies the msb
  // of the low 1s and End specifies the lsb of the high 1s.
  if (isShiftedMask_64(Mask ^ allOnes(BitSize), LSB, Length)) {
    assert(LSB > 0 && "Bottom bit must be set");
    assert(LSB + Length < BitSize && "Top bit must be set");
    Start = 63 - (LSB - 1);
    End = 63 - (LSB + Length);
    return true;
  }

  return false;
}

unsigned SystemZInstrInfo::getFusedCompare(unsigned Opcode,
                                           SystemZII::FusedCompareType Type,
                                           const MachineInstr *MI) const {
  switch (Opcode) {
  case SystemZ::CHI:
  case SystemZ::CGHI:
    if (!(MI && isInt<8>(MI->getOperand(1).getImm())))
      return 0;
    break;
  case SystemZ::CLFI:
  case SystemZ::CLGFI:
    if (!(MI && isUInt<8>(MI->getOperand(1).getImm())))
      return 0;
    break;
  case SystemZ::CL:
  case SystemZ::CLG:
    if (!STI.hasMiscellaneousExtensions())
      return 0;
    if (!(MI && MI->getOperand(3).getReg() == 0))
      return 0;
    break;
  }
  switch (Type) {
  case SystemZII::CompareAndBranch:
    switch (Opcode) {
    case SystemZ::CR:
      return SystemZ::CRJ;
    case SystemZ::CGR:
      return SystemZ::CGRJ;
    case SystemZ::CHI:
      return SystemZ::CIJ;
    case SystemZ::CGHI:
      return SystemZ::CGIJ;
    case SystemZ::CLR:
      return SystemZ::CLRJ;
    case SystemZ::CLGR:
      return SystemZ::CLGRJ;
    case SystemZ::CLFI:
      return SystemZ::CLIJ;
    case SystemZ::CLGFI:
      return SystemZ::CLGIJ;
    default:
      return 0;
    }
  case SystemZII::CompareAndReturn:
    switch (Opcode) {
    case SystemZ::CR:
      return SystemZ::CRBReturn;
    case SystemZ::CGR:
      return SystemZ::CGRBReturn;
    case SystemZ::CHI:
      return SystemZ::CIBReturn;
    case SystemZ::CGHI:
      return SystemZ::CGIBReturn;
    case SystemZ::CLR:
      return SystemZ::CLRBReturn;
    case SystemZ::CLGR:
      return SystemZ::CLGRBReturn;
    case SystemZ::CLFI:
      return SystemZ::CLIBReturn;
    case SystemZ::CLGFI:
      return SystemZ::CLGIBReturn;
    default:
      return 0;
    }
  case SystemZII::CompareAndSibcall:
    switch (Opcode) {
    case SystemZ::CR:
      return SystemZ::CRBCall;
    case SystemZ::CGR:
      return SystemZ::CGRBCall;
    case SystemZ::CHI:
      return SystemZ::CIBCall;
    case SystemZ::CGHI:
      return SystemZ::CGIBCall;
    case SystemZ::CLR:
      return SystemZ::CLRBCall;
    case SystemZ::CLGR:
      return SystemZ::CLGRBCall;
    case SystemZ::CLFI:
      return SystemZ::CLIBCall;
    case SystemZ::CLGFI:
      return SystemZ::CLGIBCall;
    default:
      return 0;
    }
  case SystemZII::CompareAndTrap:
    switch (Opcode) {
    case SystemZ::CR:
      return SystemZ::CRT;
    case SystemZ::CGR:
      return SystemZ::CGRT;
    case SystemZ::CHI:
      return SystemZ::CIT;
    case SystemZ::CGHI:
      return SystemZ::CGIT;
    case SystemZ::CLR:
      return SystemZ::CLRT;
    case SystemZ::CLGR:
      return SystemZ::CLGRT;
    case SystemZ::CLFI:
      return SystemZ::CLFIT;
    case SystemZ::CLGFI:
      return SystemZ::CLGIT;
    case SystemZ::CL:
      return SystemZ::CLT;
    case SystemZ::CLG:
      return SystemZ::CLGT;
    default:
      return 0;
    }
  }
  return 0;
}

bool SystemZInstrInfo::
prepareCompareSwapOperands(MachineBasicBlock::iterator const MBBI) const {
  assert(MBBI->isCompare() && MBBI->getOperand(0).isReg() &&
         MBBI->getOperand(1).isReg() && !MBBI->mayLoad() &&
         "Not a compare reg/reg.");

  MachineBasicBlock *MBB = MBBI->getParent();
  bool CCLive = true;
  SmallVector<MachineInstr *, 4> CCUsers;
  for (MachineInstr &MI : llvm::make_range(std::next(MBBI), MBB->end())) {
    if (MI.readsRegister(SystemZ::CC, /*TRI=*/nullptr)) {
      unsigned Flags = MI.getDesc().TSFlags;
      if ((Flags & SystemZII::CCMaskFirst) || (Flags & SystemZII::CCMaskLast))
        CCUsers.push_back(&MI);
      else
        return false;
    }
    if (MI.definesRegister(SystemZ::CC, /*TRI=*/nullptr)) {
      CCLive = false;
      break;
    }
  }
  if (CCLive) {
    LiveRegUnits LiveRegs(*MBB->getParent()->getSubtarget().getRegisterInfo());
    LiveRegs.addLiveOuts(*MBB);
    if (!LiveRegs.available(SystemZ::CC))
      return false;
  }

  // Update all CC users.
  for (unsigned Idx = 0; Idx < CCUsers.size(); ++Idx) {
    unsigned Flags = CCUsers[Idx]->getDesc().TSFlags;
    unsigned FirstOpNum = ((Flags & SystemZII::CCMaskFirst) ?
                           0 : CCUsers[Idx]->getNumExplicitOperands() - 2);
    MachineOperand &CCMaskMO = CCUsers[Idx]->getOperand(FirstOpNum + 1);
    unsigned NewCCMask = SystemZ::reverseCCMask(CCMaskMO.getImm());
    CCMaskMO.setImm(NewCCMask);
  }

  return true;
}

unsigned SystemZ::reverseCCMask(unsigned CCMask) {
  return ((CCMask & SystemZ::CCMASK_CMP_EQ) |
          (CCMask & SystemZ::CCMASK_CMP_GT ? SystemZ::CCMASK_CMP_LT : 0) |
          (CCMask & SystemZ::CCMASK_CMP_LT ? SystemZ::CCMASK_CMP_GT : 0) |
          (CCMask & SystemZ::CCMASK_CMP_UO));
}

MachineBasicBlock *SystemZ::emitBlockAfter(MachineBasicBlock *MBB) {
  MachineFunction &MF = *MBB->getParent();
  MachineBasicBlock *NewMBB = MF.CreateMachineBasicBlock(MBB->getBasicBlock());
  MF.insert(std::next(MachineFunction::iterator(MBB)), NewMBB);
  return NewMBB;
}

MachineBasicBlock *SystemZ::splitBlockAfter(MachineBasicBlock::iterator MI,
                                            MachineBasicBlock *MBB) {
  MachineBasicBlock *NewMBB = emitBlockAfter(MBB);
  NewMBB->splice(NewMBB->begin(), MBB,
                 std::next(MachineBasicBlock::iterator(MI)), MBB->end());
  NewMBB->transferSuccessorsAndUpdatePHIs(MBB);
  return NewMBB;
}

MachineBasicBlock *SystemZ::splitBlockBefore(MachineBasicBlock::iterator MI,
                                             MachineBasicBlock *MBB) {
  MachineBasicBlock *NewMBB = emitBlockAfter(MBB);
  NewMBB->splice(NewMBB->begin(), MBB, MI, MBB->end());
  NewMBB->transferSuccessorsAndUpdatePHIs(MBB);
  return NewMBB;
}

unsigned SystemZInstrInfo::getLoadAndTrap(unsigned Opcode) const {
  if (!STI.hasLoadAndTrap())
    return 0;
  switch (Opcode) {
  case SystemZ::L:
  case SystemZ::LY:
    return SystemZ::LAT;
  case SystemZ::LG:
    return SystemZ::LGAT;
  case SystemZ::LFH:
    return SystemZ::LFHAT;
  case SystemZ::LLGF:
    return SystemZ::LLGFAT;
  case SystemZ::LLGT:
    return SystemZ::LLGTAT;
  }
  return 0;
}

void SystemZInstrInfo::loadImmediate(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     unsigned Reg, uint64_t Value) const {
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned Opcode = 0;
  if (isInt<16>(Value))
    Opcode = SystemZ::LGHI;
  else if (SystemZ::isImmLL(Value))
    Opcode = SystemZ::LLILL;
  else if (SystemZ::isImmLH(Value)) {
    Opcode = SystemZ::LLILH;
    Value >>= 16;
  }
  else if (isInt<32>(Value))
    Opcode = SystemZ::LGFI;
  if (Opcode) {
    BuildMI(MBB, MBBI, DL, get(Opcode), Reg).addImm(Value);
    return;
  }

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  assert (MRI.isSSA() &&  "Huge values only handled before reg-alloc .");
  Register Reg0 = MRI.createVirtualRegister(&SystemZ::GR64BitRegClass);
  Register Reg1 = MRI.createVirtualRegister(&SystemZ::GR64BitRegClass);
  BuildMI(MBB, MBBI, DL, get(SystemZ::IMPLICIT_DEF), Reg0);
  BuildMI(MBB, MBBI, DL, get(SystemZ::IIHF64), Reg1)
    .addReg(Reg0).addImm(Value >> 32);
  BuildMI(MBB, MBBI, DL, get(SystemZ::IILF64), Reg)
    .addReg(Reg1).addImm(Value & ((uint64_t(1) << 32) - 1));
}

bool SystemZInstrInfo::verifyInstruction(const MachineInstr &MI,
                                         StringRef &ErrInfo) const {
  const MCInstrDesc &MCID = MI.getDesc();
  for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
    if (I >= MCID.getNumOperands())
      break;
    const MachineOperand &Op = MI.getOperand(I);
    const MCOperandInfo &MCOI = MCID.operands()[I];
    // Addressing modes have register and immediate operands. Op should be a
    // register (or frame index) operand if MCOI.RegClass contains a valid
    // register class, or an immediate otherwise.
    if (MCOI.OperandType == MCOI::OPERAND_MEMORY &&
        ((MCOI.RegClass != -1 && !Op.isReg() && !Op.isFI()) ||
         (MCOI.RegClass == -1 && !Op.isImm()))) {
      ErrInfo = "Addressing mode operands corrupt!";
      return false;
    }
  }

  return true;
}

bool SystemZInstrInfo::
areMemAccessesTriviallyDisjoint(const MachineInstr &MIa,
                                const MachineInstr &MIb) const {

  if (!MIa.hasOneMemOperand() || !MIb.hasOneMemOperand())
    return false;

  // If mem-operands show that the same address Value is used by both
  // instructions, check for non-overlapping offsets and widths. Not
  // sure if a register based analysis would be an improvement...

  MachineMemOperand *MMOa = *MIa.memoperands_begin();
  MachineMemOperand *MMOb = *MIb.memoperands_begin();
  const Value *VALa = MMOa->getValue();
  const Value *VALb = MMOb->getValue();
  bool SameVal = (VALa && VALb && (VALa == VALb));
  if (!SameVal) {
    const PseudoSourceValue *PSVa = MMOa->getPseudoValue();
    const PseudoSourceValue *PSVb = MMOb->getPseudoValue();
    if (PSVa && PSVb && (PSVa == PSVb))
      SameVal = true;
  }
  if (SameVal) {
    int OffsetA = MMOa->getOffset(), OffsetB = MMOb->getOffset();
    LocationSize WidthA = MMOa->getSize(), WidthB = MMOb->getSize();
    int LowOffset = OffsetA < OffsetB ? OffsetA : OffsetB;
    int HighOffset = OffsetA < OffsetB ? OffsetB : OffsetA;
    LocationSize LowWidth = (LowOffset == OffsetA) ? WidthA : WidthB;
    if (LowWidth.hasValue() &&
        LowOffset + (int)LowWidth.getValue() <= HighOffset)
      return true;
  }

  return false;
}
