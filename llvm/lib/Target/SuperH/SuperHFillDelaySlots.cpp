//===-- SuperHFillDelaySlots.cpp - Reordering pass to fill delay slots ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that fills delay slots of branching instructions.
//
//===----------------------------------------------------------------------===//

#include "SuperH.h"
#include "SuperHInstrInfo.h"
#include "SuperHTargetMachine.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/DebugLog.h"

using namespace llvm;

#define DEBUG_TYPE "sh-fill-delay-slots"
#define SUPERH_FILL_DELAY_SLOTS_NAME "SuperH delay slot filling pass"

namespace {
class SuperHFillDelaySlots : public MachineFunctionPass {
public:
  static char ID;

  SuperHFillDelaySlots() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return SUPERH_FILL_DELAY_SLOTS_NAME; }

private:
  typedef MachineBasicBlock Block;
  typedef Block::iterator BlockIt;

  const SuperHRegisterInfo *TRI;
  const SuperHInstrInfo *TII;

  bool hasDelaySlot(MachineInstr &I);
  bool expandMBB(Block &MBB);
  bool expandMI(Block &MBB, BlockIt MBBI);

  // Expansion functions
  MachineInstr *findSlotCandidate(Block &MBB, BlockIt MBBI);
  bool fillDelaySlot(Block &MBB, BlockIt MBBI);
};

} // end namespace




//===----------------------------------------------------------------------===//
//                                Expansions
//===----------------------------------------------------------------------===//

// Walks backwards through the basic block to find a candidate that is eligible
// for filling delay slots.
MachineInstr *SuperHFillDelaySlots::findSlotCandidate(Block &MBB, BlockIt MBBI) {
  MachineInstr &MI = *MBBI;


  // TODO:  Make this a while loop that keeps a list of "used" registers
  //        by instructions.
  if (auto *Prev = MBBI->getPrevNode()) {
    unsigned Opcode = Prev->getOpcode();

    // If we encounter a branch instruction, then it's no longer safe to
    // move the instruction down.
    if (Prev->isBranch() || Prev->isCall() || Prev->isReturn()) 
      return nullptr;

    // NOTE:  RTS has an extra constraint that it cannot have
    //        lds @r15+,PR or equivalent in its delay slot.
    if (MI.isReturn()) {
      
      // Skip the LDS instruction.
      if (Opcode == SH::LDSLRminciPR || Opcode == SH::LDSRmPR)
        return nullptr;
    }

    // NOTE:  Conditional branches can't have their condition code set
    //        in the delay slot. As such, if the previous instruction
    //        implicitly defines the status register, assume that the 
    //        T bit was set.
    if (MI.isConditionalBranch()) {
      if (Prev->definesRegister(SH::SR, TRI))
        return nullptr;
    }

    // Otherwise, select this instruction if can fill a delay slot,
    // has no prior node, or the prior node is not a delay slot.
    if (TII->canFillDelaySlot(Opcode)) {
      if (!Prev->getPrevNode() || !Prev->getPrevNode()->hasDelaySlot())
        return Prev;
    }
  }

  return nullptr;
}

// Finds and fills delay slots of instructions in a basic block.
bool SuperHFillDelaySlots::fillDelaySlot(Block &MBB, BlockIt MBBI) {
  MachineFunction &MF = *MBB.getParent();
  MachineInstr &MI = *MBBI;

  if (auto *Candidate = SuperHFillDelaySlots::findSlotCandidate(MBB, MBBI)) {
      LLVM_DEBUG(dbgs() << "Swapping " << TII->getName(MI.getOpcode()) 
                        << " and " << TII->getName(Candidate->getOpcode()) 
                        << " @ " << MBB.getParent()->getName() << "\n");

      MBB.insertAfter(MBBI, Candidate->removeFromParent());
      return true;
  }

  LLVM_DEBUG(dbgs() << "Inserting NOP after " << TII->getName(MI.getOpcode())
                    << " @ " << MBB.getParent()->getName() << "\n");

  // Otherwise just insert a NOP.
  MBB.insertAfter(MBBI, MF.CreateMachineInstr(TII->get(SH::NOP), MI.getDebugLoc())); 
  return true;
}




//===----------------------------------------------------------------------===//
//                                Helpers
//===----------------------------------------------------------------------===//

bool SuperHFillDelaySlots::expandMI(Block &MBB, BlockIt MBBI) {
  MachineInstr &MI = *MBBI;

  if (MI.hasDelaySlot()) {
    return fillDelaySlot(MBB, MBBI);
  }
  return false;
}

bool SuperHFillDelaySlots::expandMBB(Block &MBB) {
  bool Modified = false;

  BlockIt MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    BlockIt NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool SuperHFillDelaySlots::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\n********** SuperHFillDelaySlots **********\n");

  bool Modified = false;

  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  TRI = STI.getRegisterInfo();
  TII = STI.getInstrInfo();

  for (Block &MBB : MF) {
    Modified |= expandMBB(MBB);
  }

  return Modified;
}


char SuperHFillDelaySlots::ID = 0;

INITIALIZE_PASS(SuperHFillDelaySlots, "sh-fill-delay-slots", SUPERH_FILL_DELAY_SLOTS_NAME,
                false, false)

FunctionPass *llvm::createSuperHFillDelaySlotsPass() {
  return new SuperHFillDelaySlots();
}
