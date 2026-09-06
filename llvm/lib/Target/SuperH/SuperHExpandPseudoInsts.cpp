//===-- SuperHExpandPseudoInsts.cpp - Expand pseudo instructions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "SuperH.h"
#include "SuperHInstrInfo.h"
#include "SuperHTargetMachine.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "sh-expand-pseudo"
#define SH_EXPAND_PSEUDO_NAME "SH pseudo instruction expansion pass"

namespace {
class SuperHExpandPseudo : public MachineFunctionPass {
public:
  static char ID;

  SuperHExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return SH_EXPAND_PSEUDO_NAME; }

private:
  typedef MachineBasicBlock Block;
  typedef Block::iterator BlockIt;

  const SuperHRegisterInfo *TRI;
  const TargetInstrInfo *TII;

  bool expandMBB(Block &MBB);
  bool expandMI(Block &MBB, BlockIt MBBI);
  template <unsigned OP> bool expand(Block &MBB, BlockIt MBBI);
};




//===----------------------------------------------------------------------===//
//                                Helpers
//===----------------------------------------------------------------------===//

// getOffsetForStackOffset - Calculates how much to offset the stack pointer for
// the indirect load/store to be in range.
static int64_t getOffsetForStackOffset(const MachineFrameInfo &MFI, int64_t StackOffset, 
                                       uint8_t Bits, uint8_t Scale = 1) {
  
  // The base value range for the instruction.
  int64_t BM = ((1<<Bits)-1)*Scale;
  int64_t StackSize = MFI.getStackSize();
  int64_t Slot = StackSize/BM;
  return alignTo((BM*Slot)+(StackSize%BM), Scale);
}



//===----------------------------------------------------------------------===//
//                              Frame Stores
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandPseudo::expand<SH::MOVBSFR>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  auto SrcReg = MI.getOperand(0).getReg();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();
  int64_t SpOffset = getOffsetForStackOffset(MFI, Offset, 4, 1);

  LLVM_DEBUG(dbgs() << "Expanding MOVBSFR to MOVBS4...\n");

  // Expand sequence to
  // mov      <base reg>,r1
  // add      #-ROff,r1
  // mov      <src reg>, r0
  // mov.b    r0,@(ROff+Offset,r1)
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
    .addReg(FrameReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::ADDI), SH::R1)
    .addReg(SH::R1)
    .addImm(-SpOffset);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R0)
    .addReg(SrcReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOVBS4))
    .addReg(SH::R1)
    .addImm(SpOffset-Offset);
  MI.eraseFromParent();
  return true;
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVWSFR>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  auto SrcReg = MI.getOperand(0).getReg();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();
  int64_t SpOffset = getOffsetForStackOffset(MFI, Offset, 4, 2);

  LLVM_DEBUG(dbgs() << "Expanding MOVWSFR to MOVWS4...\n");

  // Expand sequence to
  // mov      <base reg>,r1
  // add      #-ROff,r1
  // mov      <src reg>, r0
  // mov.w    r0,@(ROff+Offset,r1)
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
    .addReg(FrameReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::ADDI), SH::R1)
    .addReg(SH::R1)
    .addImm(-SpOffset);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R0)
    .addReg(SrcReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOVWS4))
    .addReg(SH::R1)
    .addImm(SpOffset-Offset);
  MI.eraseFromParent();
  return true;
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVLSFR>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  auto SrcReg = MI.getOperand(0).getReg();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();
  int64_t SpOffset = getOffsetForStackOffset(MFI, Offset, 4, 4);

  LLVM_DEBUG(dbgs() << "Expanding MOVLSFR to MOVLS4...\n");
    
  // Expand sequence to
  // mov      <base reg>,r1
  // add      #-ROff,r1
  // mov.l    <src reg>,@(ROff+Offset,r1)
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
    .addReg(FrameReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::ADDI), SH::R1)
    .addReg(SH::R1)
    .addImm(-SpOffset);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLS4))
    .addReg(SrcReg)
    .addReg(SH::R1)
    .addImm(SpOffset-Offset);
  MI.eraseFromParent();
  return true;
}




//===----------------------------------------------------------------------===//
//                               Frame Loads
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandPseudo::expand<SH::MOVBLFR>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  LLVM_DEBUG(dbgs() << "Expanding MOVBLFR to MOVBL4...\n");

  auto DstReg = MI.getOperand(0).getReg();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();
  int64_t SpOffset = getOffsetForStackOffset(MFI, Offset, 4, 1);

  // HACK:  Need to find a better way of handling this, it's likely to break.
  MBBI = std::next(MBBI);
  MBBI = std::next(MBBI);
    
  // Expand sequence to
  // mov      <base reg>,r1
  // add      #-ROff,r1
  // mov.b    @(ROff+Offset,r1),r0
  // mov      r0, <dst reg>
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
    .addReg(FrameReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::ADDI), SH::R1)
    .addReg(SH::R1)
    .addImm(-SpOffset);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOVBL4))
    .addReg(SH::R1)
    .addImm(SpOffset-Offset);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), DstReg)
    .addReg(SH::R0);
  MI.eraseFromParent();
  return true;
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVWLFR>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  LLVM_DEBUG(dbgs() << "Expanding MOVWLFR to MOVWL4...\n");

  auto DstReg = MI.getOperand(0).getReg();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();
  int64_t SpOffset = getOffsetForStackOffset(MFI, Offset, 4, 2);

  // HACK:  Need to find a better way of handling this, it's likely to break.
  MBBI = std::next(MBBI);
  MBBI = std::next(MBBI);
    
  // Expand sequence to
  // mov      <base reg>,r1
  // add      #-ROff,r1
  // mov.w    @(ROff+Offset,r1),r0
  // mov      r0, <dst reg>
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
    .addReg(FrameReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::ADDI), SH::R1)
    .addReg(SH::R1)
    .addImm(-SpOffset);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOVWL4))
    .addReg(SH::R1)
    .addImm(SpOffset-Offset);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), DstReg)
    .addReg(SH::R0);
  MI.eraseFromParent();
  return true;
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVLLFR>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  auto DstReg = MI.getOperand(0).getReg();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();
  int64_t SpOffset = getOffsetForStackOffset(MFI, Offset, 4, 4);

  LLVM_DEBUG(dbgs() << "Expanding MOVLLFR to MOVLL4...\n");
    
  // Expand sequence to
  // mov      <base reg>,r1
  // add      #-ROff,r1
  // mov.l    @(ROff+Offset,r1),<dest reg>
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
    .addReg(FrameReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::ADDI), SH::R1)
    .addReg(SH::R1)
    .addImm(-SpOffset);
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLL4), DstReg)
    .addReg(SH::R1)
    .addImm(SpOffset-Offset);
  MI.eraseFromParent();
  return true;
}



//===----------------------------------------------------------------------===//
//                            General Interface
//===----------------------------------------------------------------------===//

bool SuperHExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  BlockIt MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    BlockIt NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool SuperHExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  bool Modified = false;

  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  TRI = STI.getRegisterInfo();
  TII = STI.getInstrInfo();

  for (Block &MBB : MF) {
    bool ContinueExpanding = true;
    unsigned ExpandCount = 0;

    // Continue expanding the block until all pseudos are expanded.
    do {
      assert(ExpandCount < 10 && "pseudo expand limit reached");
      (void)ExpandCount;

      bool BlockModified = expandMBB(MBB);
      Modified |= BlockModified;
      ExpandCount++;

      ContinueExpanding = BlockModified;
    } while (ContinueExpanding);
  }

  return Modified;
}

bool SuperHExpandPseudo::expandMI(Block &MBB, BlockIt MBBI) {
  MachineInstr &MI = *MBBI;
  int Opcode = MBBI->getOpcode();

#define EXPAND(Op)                                                             \
  case Op:                                                                     \
    return expand<Op>(MBB, MI)

  switch(Opcode) {
    EXPAND(SH::MOVBSFR);
    EXPAND(SH::MOVWSFR);
    EXPAND(SH::MOVLSFR);
    EXPAND(SH::MOVBLFR);
    EXPAND(SH::MOVWLFR);
    EXPAND(SH::MOVLLFR);
  }
#undef EXPAND
  return false;
}

char SuperHExpandPseudo::ID = 0;

} // namespace


INITIALIZE_PASS(SuperHExpandPseudo, "sh-expand-pseudo", SH_EXPAND_PSEUDO_NAME,
                false, false)

FunctionPass *llvm::createSuperHExpandPseudoPass() {
  return new SuperHExpandPseudo();
}