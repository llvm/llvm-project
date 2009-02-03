//===-- DelaySlotFiller.cpp - SPARC delay slot filler ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a simple local pass that fills delay slots with NOPs.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "delayslotfiller"
#include "Sparc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");

namespace {
  struct Filler : public MachineFunctionPass {
    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
    Filler(TargetMachine &tm) 
      : MachineFunctionPass(&ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "SPARC Delay Slot Filler";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

  };
  char Filler::ID = 0;
} // end of anonymous namespace

/// createSparcDelaySlotFillerPass - Returns a pass that fills in delay
/// slots in Sparc MachineFunctions
///
FunctionPass *llvm::createSparcDelaySlotFillerPass(TargetMachine &tm) {
  return new Filler(tm);
}

/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// Currently, we fill delay slots with NOPs. We assume there is only one
/// delay slot per delayed instruction.
///
bool Filler::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I)
    if (I->getDesc().hasDelaySlot()) {
      MachineBasicBlock::iterator J = I;
      ++J;
      BuildMI(MBB, J, TII->get(SP::NOP));
      ++FilledSlots;
      Changed = true;
    }
  return Changed;
}
