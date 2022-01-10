//===-- M88kDelaySlotFiller.cpp - Delay Slot Filler for M88k --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple pass to fill delay slots with useful instructions.
//
//===----------------------------------------------------------------------===//

#include "M88kTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "m88k-delay-slot-filler"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");

static cl::opt<bool>
    DisableDelaySlotFiller("disable-m88k-delay-slot-filler", cl::init(false),
                           cl::desc("Do not fill delay slots."), cl::Hidden);

namespace {
class M88kDelaySlotFiller : public MachineFunctionPass {
public:
  static char ID;

  M88kDelaySlotFiller();

  StringRef getPassName() const override { return "M88kDelaySlotFiller"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // end anonymous namespace

// TODO This should not be needed here.
namespace llvm {
void initializeM88kDelaySlotFillerPass(PassRegistry &Registry);
}

M88kDelaySlotFiller::M88kDelaySlotFiller() : MachineFunctionPass(ID) {
  initializeM88kDelaySlotFillerPass(*PassRegistry::getPassRegistry());
}

bool M88kDelaySlotFiller::runOnMachineFunction(MachineFunction &MF) {
  if (DisableDelaySlotFiller)
    return false;
  return false;
}

char M88kDelaySlotFiller::ID = 0;
INITIALIZE_PASS_BEGIN(M88kDelaySlotFiller, DEBUG_TYPE, "Fill M88k delay slots",
                      false, false)
INITIALIZE_PASS_END(M88kDelaySlotFiller, DEBUG_TYPE, "Fill M88k delay slots",
                    false, false)

namespace llvm {
FunctionPass *createM88kDelaySlotFiller() { return new M88kDelaySlotFiller(); }
} // end namespace llvm
