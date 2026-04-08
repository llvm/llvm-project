//=- AArch64RedundantCondBranch.cpp - Remove redundant conditional branches -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Late in the pipeline, especially with zero phi operands propagated after tail
// duplications, we can end up with CBZ/CBNZ/TBZ/TBNZ with a zero register. This
// simple pass looks at the terminators to a block, removing the redundant
// instructions where necessary.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-redundantcondbranch"

namespace {

class AArch64RedundantCondBranchImpl {
public:
  bool run(MachineFunction &MF);
};
class AArch64RedundantCondBranchLegacy : public MachineFunctionPass {
public:
  static char ID;
  AArch64RedundantCondBranchLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
  StringRef getPassName() const override {
    return "AArch64 Redundant Conditional Branch Elimination";
  }
};
char AArch64RedundantCondBranchLegacy::ID = 0;
} // namespace

INITIALIZE_PASS(AArch64RedundantCondBranchLegacy, "aarch64-redundantcondbranch",
                "AArch64 Redundant Conditional Branch Elimination pass", false,
                false)

bool AArch64RedundantCondBranchImpl::run(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeTerminators(&MBB, TII);
  return Changed;
}

bool AArch64RedundantCondBranchLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  return AArch64RedundantCondBranchImpl().run(MF);
}

PreservedAnalyses
AArch64RedundantCondBranchPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &) {
  if (AArch64RedundantCondBranchImpl().run(MF)) {
    return getMachineFunctionPassPreservedAnalyses();
  }
  return PreservedAnalyses::all();
}

FunctionPass *llvm::createAArch64RedundantCondBranchPass() {
  return new AArch64RedundantCondBranchLegacy();
}
