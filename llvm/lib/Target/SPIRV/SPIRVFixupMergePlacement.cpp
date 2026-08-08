//===-- SPIRVFixupMergePlacement.cpp - keep merges before branches -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVInstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-fixup-merge-placement"

namespace {
class SPIRVFixupMergePlacement : public MachineFunctionPass {
public:
  static char ID;
  SPIRVFixupMergePlacement() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SPIRV fixup merge placement";
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};
} // namespace

static bool isStructuredMerge(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == SPIRV::OpLoopMerge || Opc == SPIRV::OpSelectionMerge;
}

bool SPIRVFixupMergePlacement::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // A structured block carries at most one merge instruction (the
    // structurizer guarantees this). Find it, if present.
    MachineInstr *Merge = nullptr;
    for (MachineInstr &MI : MBB) {
      if (isStructuredMerge(MI)) {
        Merge = &MI;
        break;
      }
    }
    if (!Merge)
      continue;

    // The merge must sit immediately before the block's branch terminator.
    MachineBasicBlock::iterator Term = MBB.getFirstTerminator();
    if (Term == MBB.end())
      continue; // No explicit terminator (e.g. unreachable), nothing to fix.

    // Already second-to-last? Then the invariant holds.
    if (&*Term == Merge->getNextNode())
      continue;

    // Otherwise a later pass (MachineCSE) slipped instructions between the
    // merge and the terminator. Sink the merge down to just before the
    // terminator.
    LLVM_DEBUG(dbgs() << "SPIRVFixupMergePlacement: re-seating merge in "
                      << MBB.getName() << "\n");
    Merge->removeFromParent();
    MBB.insert(Term, Merge);
    Changed = true;
  }
  return Changed;
}

INITIALIZE_PASS(SPIRVFixupMergePlacement, DEBUG_TYPE,
                "SPIRV fixup merge placement", false, false)

char SPIRVFixupMergePlacement::ID = 0;

FunctionPass *llvm::createSPIRVFixupMergePlacementPass() {
  return new SPIRVFixupMergePlacement();
}
