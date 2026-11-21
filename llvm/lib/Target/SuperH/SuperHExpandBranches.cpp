//===-- SuperHExpandPseudoBRs.cpp - Expand pseudo branch instrs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo branch instructions in a
// secondary pass after main instruction expansion.
//
// Branch instructions require knowing the "final" size of the basic blocks
// they reside within to accurately be able to ascertain whether the destination
// is within range.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "SuperH.h"
#include "SuperHBasicBlockInfo.h"
#include "SuperHConstantPoolValue.h"
#include "SuperHInstrInfo.h"
#include "SuperHMachineFunctionInfo.h"
#include "SuperHSubtarget.h"

#include "llvm/ADT/APInt.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace SH;

#define DEBUG_TYPE "sh-expand-brs"
#define SH_EXPAND_BR_NAME "SH pseudo branch expansion pass"

namespace {
class SuperHExpandBR : public MachineFunctionPass {
public:
  static char ID;

  SuperHExpandBR() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return SH_EXPAND_BR_NAME; }

private:
  using Block = MachineBasicBlock;
  using BlockIt = Block::iterator;
  std::unique_ptr<SuperHBasicBlockUtils> BBUtils = nullptr;

  const SuperHSubtarget *STI;
  const SuperHInstrInfo *TII;
  const SuperHMachineFunctionInfo *MFI;

  bool expandMBB(Block &MBB);
  bool expandMI(Block &MBB, BlockIt MBBI);

  // ensureTargetSym - Ensures that a branch target has a symbol
  // generated for it. This is neccesary because we generate 
  // relocations for the indirect branches when a BB is used
  // directly.
  MachineBasicBlock *ensureTargetSym(MachineBasicBlock *BB);

  template <unsigned OP> bool expand(Block &MBB, BlockIt MBBI);
};




//===----------------------------------------------------------------------===//
//                                 Helpers
//===----------------------------------------------------------------------===//
MachineBasicBlock *
SuperHExpandBR::ensureTargetSym(MachineBasicBlock *BB) {
  if (!BB->hasLabelMustBeEmitted())
    BB->setLabelMustBeEmitted();

  return BB;
}




//===----------------------------------------------------------------------===//
//                                Branching
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandBR::expand<SH::BRCC>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;


  bool HasDelaySlot = MI.getOpcode() == SH::BRSCC;
  auto *DstBB = MI.getOperand(0).getMBB();
  auto CC = (SHCC::CondCode)MI.getOperand(1).getImm();

  // If the basic block is out-of-range, we insert a indirect jump and 
  // skip over it with a flipped branch condition.
  if (!BBUtils->isBBInRange(&MI, DstBB, 254)) {
    LLVM_DEBUG(dbgs() << "BB not in range!\n");

    if (auto *Sym = MFI->getOrCreate(DstBB, *MBB.getParent())) {

      // Skip 4 instructions ahead. with flipped condition
      CC = TII->getOppositeCondCode(CC);
      BuildMI(MBB, MBBI, DL, TII->getBrCond(CC, HasDelaySlot))
        .addImm(0);

      // Load branch destination.
      BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLI), SH::R1)
        .addConstantPoolIndex(Sym->getLabelId());

      BuildMI(MBB, MBBI, DL, TII->get(SH::JMP))
        .addReg(SH::R1);

      // Pad with NOP.
      BuildMI(MBB, MBBI, DL, TII->get(SH::NOP));

      LLVM_DEBUG(dbgs() << "Expanded BRCC\n");
      MI.eraseFromParent();
      return true;
    }

    // Failed.
    llvm_unreachable("Failed to create symbol CPI!");
  }

  // Otherwise just do a basic direct conditional jump.
  BuildMI(MBB, MBBI, DL, TII->getBrCond(CC, HasDelaySlot))
    .addMBB(ensureTargetSym(DstBB));

  MI.eraseFromParent();
  return true;
}

template <>
bool SuperHExpandBR::expand<SH::BR>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  auto *DstBB = MI.getOperand(0).getMBB();

  BuildMI(MBB, MBBI, DL, TII->get(SH::BRA))
    .addMBB(ensureTargetSym(DstBB));

  MI.eraseFromParent();
  return true;
}




//===----------------------------------------------------------------------===//
//                            General Interface
//===----------------------------------------------------------------------===//

bool SuperHExpandBR::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  BlockIt MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    BlockIt NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI);
    MBBI = NMBBI;

    // Recalculate block size if the MBB changed.
    if (Modified) {
      BBUtils->computeBlockSize(&MBB);
      BBUtils->adjustBBOffsetsAfter(&MBB);
    }
  }

  return Modified;
}

bool SuperHExpandBR::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\n************ SuperHExpandBR ************\n");
  bool Modified = false;

  STI = &MF.getSubtarget<SuperHSubtarget>();
  TII = STI->getInstrInfo();
  MFI = MF.getInfo<SuperHMachineFunctionInfo>();
  BBUtils = std::make_unique<SuperHBasicBlockUtils>(MF);
  BBUtils->computeAllBlockSizes();

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

bool SuperHExpandBR::expandMI(Block &MBB, BlockIt MBBI) {
  MachineInstr &MI = *MBBI;
  int Opcode = MBBI->getOpcode();

  switch (Opcode) {
  default:
    break;
  case SH::BR: 
    return expand<SH::BR>(MBB, MI);
  case SH::BRSCC: 
  case SH::BRCC: 
    return expand<SH::BRCC>(MBB, MI);
  }
#undef EXPAND
  return false;
}

char SuperHExpandBR::ID = 0;

} // namespace

INITIALIZE_PASS(SuperHExpandBR, DEBUG_TYPE, SH_EXPAND_BR_NAME,
                false, false)

FunctionPass *llvm::createSuperHExpandBRPass() {
  return new SuperHExpandBR();
}