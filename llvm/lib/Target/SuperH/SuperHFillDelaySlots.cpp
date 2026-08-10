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
  const TargetInstrInfo *TII;

  bool hasDelaySlot(MachineInstr &I);
  bool expandMBB(Block &MBB);
  bool expandMI(Block &MBB, BlockIt MBBI);

  // Expansion functions
  bool fillDelaySlot(Block &MBB, BlockIt MBBI);
};

} // end namespace


bool SuperHFillDelaySlots::fillDelaySlot(Block &MBB, BlockIt MBBI) {
  MachineInstr &MI = *MBBI;
  if (auto *Prev = MBBI->getPrevNode()) {

    // If the prior instruction does not have a delay slot
    // we swap the instructions.
    //
    // NOTE:  SuperH does not allow branch instructions
    //        of any kind to be situated in a delay slot.
    //        as such we fall through to the NOP in that
    //        instance.
    if (!Prev->isBranch() && !Prev->hasDelaySlot()) {
      LDBG() << "Swapping " << TII->getName(MI.getOpcode()) 
             << " and " << TII->getName(Prev->getOpcode()) 
             << " @ " << MBB.getParent()->getName();
      MBB.insertAfter(MBBI, Prev->removeFromParent());
      return true;
    }
  }

  LDBG() << "Inserting NOP after " << TII->getName(MI.getOpcode())
         << " @ " << MBB.getParent()->getName();

  // Otherwise just insert a NOP.
  BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(SH::NOP));
  return true;
}



//===----------------------------------------------------------------------===//
//                                HELPERS
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
