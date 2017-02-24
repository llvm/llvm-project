//===-- MipsExpandPseudoInsts.cpp - Expand pseudo instructions ------------===//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions to allow proper scheduling, if-conversion, and other late
// optimizations. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsInstrInfo.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "mips-pseudo"

namespace {
  class MipsExpandPseudo : public MachineFunctionPass {
  public:
    static char ID;
    MipsExpandPseudo() : MachineFunctionPass(ID) {}

    const MipsInstrInfo *TII;
    const MipsSubtarget *STI;

    bool runOnMachineFunction(MachineFunction &Fn) override;

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

    StringRef getPassName() const override {
      return "Mips pseudo instruction expansion pass";
    }

  private:
    bool expandAtomicCmpSwap(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MBBI,
                             MachineBasicBlock::iterator &NextMBBI);
    bool expandAtomicCmpSwapSubword(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    MachineBasicBlock::iterator &NextMBBI);
    bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NMBB);
    bool expandMBB(MachineBasicBlock &MBB);
   };
  char MipsExpandPseudo::ID = 0;
}

static void addPostLoopLiveIns(MachineBasicBlock *MBB, LivePhysRegs &LiveRegs) {
  for (auto I = LiveRegs.begin(); I != LiveRegs.end(); ++I)
    MBB->addLiveIn(*I);
}

bool MipsExpandPseudo::expandAtomicCmpSwapSubword(
    MachineBasicBlock &BB, MachineBasicBlock::iterator I,
    MachineBasicBlock::iterator &NMBBI) {

  MachineFunction *MF = BB.getParent();

  const bool ArePtrs64bit = STI->getABI().ArePtrs64bit();
  DebugLoc DL = I->getDebugLoc();
  unsigned LL, SC;

  unsigned ZERO = Mips::ZERO;
  unsigned BNE = Mips::BNE;
  unsigned BEQ = Mips::BEQ;
  unsigned SEOp =
      I->getOpcode() == Mips::ATOMIC_CMP_SWAP_I8_FRAG ? Mips::SEB : Mips::SEH;

  if (STI->inMicroMipsMode()) {
    LL = Mips::LL_MM;
    SC = Mips::SC_MM;
  } else {
    LL = STI->hasMips32r6() ? (ArePtrs64bit ? Mips::LL64_R6 : Mips::LL_R6)
                            : (ArePtrs64bit ? Mips::LL64 : Mips::LL);
    SC = STI->hasMips32r6() ? (ArePtrs64bit ? Mips::SC64_R6 : Mips::SC_R6)
                            : (ArePtrs64bit ? Mips::SC64 : Mips::SC);
  }

  unsigned Dest = I->getOperand(0).getReg();
  unsigned Ptr = I->getOperand(1).getReg();
  unsigned Mask = I->getOperand(2).getReg();
  unsigned ShiftCmpVal = I->getOperand(3).getReg();
  unsigned Mask2 = I->getOperand(4).getReg();
  unsigned ShiftNewVal = I->getOperand(5).getReg();
  unsigned ShiftAmnt = I->getOperand(6).getReg();

  LivePhysRegs LiveRegs(&TII->getRegisterInfo());
  for (auto MBBI = std::prev(BB.end()); MBBI != I; --MBBI)
    LiveRegs.stepBackward(*MBBI);

  // insert new blocks after the current block
  const BasicBlock *LLVM_BB = BB.getBasicBlock();
  MachineBasicBlock *loop1MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *loop2MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = ++BB.getIterator();
  MF->insert(It, loop1MBB);
  MF->insert(It, loop2MBB);
  MF->insert(It, sinkMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), &BB,
                  std::next(MachineBasicBlock::iterator(I)), BB.end());
  exitMBB->transferSuccessorsAndUpdatePHIs(&BB);

  //  thisMBB:
  //    ...
  //    fallthrough --> loop1MBB
  BB.addSuccessor(loop1MBB, BranchProbability::getOne());
  loop1MBB->addSuccessor(sinkMBB);
  loop1MBB->addSuccessor(loop2MBB);
  loop2MBB->addSuccessor(loop1MBB);
  loop2MBB->addSuccessor(sinkMBB);
  sinkMBB->addSuccessor(exitMBB, BranchProbability::getOne());

  // loop1MBB:
  //   ll dest, 0(ptr)
  //   and Mask', dest, Mask
  //   bne Mask', ShiftCmpVal, exitMBB
  BuildMI(loop1MBB, DL, TII->get(LL), Dest).addReg(Ptr).addImm(0);
  BuildMI(loop1MBB, DL, TII->get(Mips::AND), Mask)
      .addReg(Dest)
      .addReg(Mask);
  BuildMI(loop1MBB, DL, TII->get(BNE))
    .addReg(Mask).addReg(ShiftCmpVal).addMBB(sinkMBB);
  loop1MBB->addLiveIn(Ptr);
  loop1MBB->addLiveIn(Mask);
  loop1MBB->addLiveIn(ShiftCmpVal);

  // loop2MBB:
  //   and dest, dest, mask2
  //   or dest, dest, ShiftNewVal
  //   sc dest, dest, 0(ptr)
  //   beq dest, $0, loop1MBB
  BuildMI(loop2MBB, DL, TII->get(Mips::AND), Dest)
      .addReg(Dest, RegState::Kill)
      .addReg(Mask2);
  BuildMI(loop2MBB, DL, TII->get(Mips::OR), Dest)
      .addReg(Dest, RegState::Kill)
      .addReg(ShiftNewVal);
  BuildMI(loop2MBB, DL, TII->get(SC), Dest)
      .addReg(Dest, RegState::Kill)
      .addReg(Ptr)
      .addImm(0);
  BuildMI(loop2MBB, DL, TII->get(BEQ))
      .addReg(Dest, RegState::Kill)
      .addReg(ZERO)
      .addMBB(loop1MBB);
  loop2MBB->addLiveIn(Ptr);
  loop2MBB->addLiveIn(Mask2);
  loop2MBB->addLiveIn(Dest);
  loop2MBB->addLiveIn(ShiftNewVal);

  //  sinkMBB:
  //    srl     srlres, Mask', shiftamt
  //    sign_extend dest,srlres
  BuildMI(sinkMBB, DL, TII->get(Mips::SRLV), Dest)
      .addReg(Mask)
      .addReg(ShiftAmnt);
  if (STI->hasMips32r2()) {
    BuildMI(sinkMBB, DL, TII->get(SEOp), Dest).addReg(Dest);
  } else {
    const unsigned ShiftImm =
        I->getOpcode() == Mips::ATOMIC_CMP_SWAP_I16_FRAG ? 16 : 24;
    BuildMI(sinkMBB, DL, TII->get(Mips::SLL), Dest)
        .addReg(Dest, RegState::Kill)
        .addImm(ShiftImm);
    BuildMI(sinkMBB, DL, TII->get(Mips::SRA), Dest)
        .addReg(Dest, RegState::Kill)
        .addImm(ShiftImm);
  }
  sinkMBB->addLiveIn(Mask);
  sinkMBB->addLiveIn(ShiftAmnt);

  addPostLoopLiveIns(exitMBB, LiveRegs);
  exitMBB->addLiveIn(Dest);

  NMBBI = BB.end();
  I->eraseFromParent();
  return true;
}

bool MipsExpandPseudo::expandAtomicCmpSwap(MachineBasicBlock &BB,
                                           MachineBasicBlock::iterator I,
                                           MachineBasicBlock::iterator &NMBBI) {

  const unsigned Size = I->getOpcode() == Mips::ATOMIC_CMP_SWAP_I32 ? 4 : 8;
  MachineFunction *MF = BB.getParent();

  const bool ArePtrs64bit = STI->getABI().ArePtrs64bit();
  DebugLoc DL = I->getDebugLoc();

  LivePhysRegs LiveRegs(&TII->getRegisterInfo());
  LiveRegs.addLiveOuts(BB);
  for (auto MBBI = std::prev(BB.end()); MBBI != I; --MBBI)
    LiveRegs.stepBackward(*MBBI);

  unsigned LL, SC, ZERO, BNE, BEQ;

  if (Size == 4) {
    if (STI->inMicroMipsMode()) {
      LL = Mips::LL_MM;
      SC = Mips::SC_MM;
    } else {
      LL = STI->hasMips32r6()
               ? (ArePtrs64bit ? Mips::LL64_R6 : Mips::LL_R6)
               : (ArePtrs64bit ? Mips::LL64 : Mips::LL);
      SC = STI->hasMips32r6()
               ? (ArePtrs64bit ? Mips::SC64_R6 : Mips::SC_R6)
               : (ArePtrs64bit ? Mips::SC64 : Mips::SC);
    }

    ZERO = Mips::ZERO;
    BNE = Mips::BNE;
    BEQ = Mips::BEQ;
  } else {
    LL = STI->hasMips64r6() ? Mips::LLD_R6 : Mips::LLD;
    SC = STI->hasMips64r6() ? Mips::SCD_R6 : Mips::SCD;
    ZERO = Mips::ZERO_64;
    BNE = Mips::BNE64;
    BEQ = Mips::BEQ64;
  }

  unsigned Dest = I->getOperand(0).getReg();
  unsigned Ptr = I->getOperand(1).getReg();
  unsigned OldVal = I->getOperand(2).getReg();
  unsigned NewVal = I->getOperand(3).getReg();

  // insert new blocks after the current block
  const BasicBlock *LLVM_BB = BB.getBasicBlock();
  MachineBasicBlock *loop1MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *loop2MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = ++BB.getIterator();
  MF->insert(It, loop1MBB);
  MF->insert(It, loop2MBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), &BB,
                  std::next(MachineBasicBlock::iterator(I)), BB.end());
  exitMBB->transferSuccessorsAndUpdatePHIs(&BB);

  //  thisMBB:
  //    ...
  //    fallthrough --> loop1MBB
  BB.addSuccessor(loop1MBB, BranchProbability::getOne());
  loop1MBB->addSuccessor(exitMBB);
  loop1MBB->addSuccessor(loop2MBB);
  loop2MBB->addSuccessor(loop1MBB);
  loop2MBB->addSuccessor(exitMBB);

  // loop1MBB:
  //   ll dest, 0(ptr)
  //   bne dest, oldval, exitMBB
  BuildMI(loop1MBB, DL, TII->get(LL), Dest).addReg(Ptr).addImm(0);
  BuildMI(loop1MBB, DL, TII->get(BNE))
    .addReg(Dest).addReg(OldVal).addMBB(exitMBB);
  loop1MBB->addLiveIn(Ptr);
  loop1MBB->addLiveIn(OldVal);

  // loop2MBB:
  //   sc success, newval, 0(ptr)
  //   beq success, $0, loop1MBB
  BuildMI(loop2MBB, DL, TII->get(SC), NewVal)
    .addReg(NewVal).addReg(Ptr).addImm(0);
  BuildMI(loop2MBB, DL, TII->get(BEQ))
    .addReg(NewVal, RegState::Kill).addReg(ZERO).addMBB(loop1MBB);
  loop2MBB->addLiveIn(Ptr);
  loop2MBB->addLiveIn(NewVal);

  addPostLoopLiveIns(exitMBB, LiveRegs);

  NMBBI = BB.end();
  I->eraseFromParent();
  return true;
}

bool MipsExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                MachineBasicBlock::iterator &NMBB) {

  bool Modified = false;
  switch (MBBI->getOpcode()) {
    case Mips::ATOMIC_CMP_SWAP_I32:
    case Mips::ATOMIC_CMP_SWAP_I64:
      return expandAtomicCmpSwap(MBB, MBBI, NMBB);
    case Mips::ATOMIC_CMP_SWAP_I8_FRAG:
    case Mips::ATOMIC_CMP_SWAP_I16_FRAG:
      return expandAtomicCmpSwapSubword(MBB, MBBI, NMBB);
    default:
      return Modified;
  }
}

bool MipsExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool MipsExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &static_cast<const MipsSubtarget &>(MF.getSubtarget());
  TII = STI->getInstrInfo();

  bool Modified = false;
  for (MachineFunction::iterator MFI = MF.begin(), E = MF.end(); MFI != E;
       ++MFI)
    Modified |= expandMBB(*MFI);

  return Modified;
}

/// createMipsExpandPseudoPass - returns an instance of the pseudo instruction
/// expansion pass.
FunctionPass *llvm::createMipsExpandPseudoPass() {
  return new MipsExpandPseudo();
}
