//===------ RISCVIndirectBranchTracking.cpp - Enables lpad mechanism ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass adds LPAD (AUIPC with rd = X0) machine instructions at the
// beginning of each basic block or function that is referenced by an indirect
// jump/call instruction.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "riscv-indirect-branch-tracking"
#define PASS_NAME "RISC-V Indirect Branch Tracking"

using namespace llvm;

namespace {
class RISCVIndirectBranchTrackingLegacy : public MachineFunctionPass {
public:
  static char ID;
  RISCVIndirectBranchTrackingLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

INITIALIZE_PASS(RISCVIndirectBranchTrackingLegacy, DEBUG_TYPE, PASS_NAME, false,
                false)

char RISCVIndirectBranchTrackingLegacy::ID = 0;

FunctionPass *llvm::createRISCVIndirectBranchTrackingLegacyPass() {
  return new RISCVIndirectBranchTrackingLegacy();
}

static void
emitLpad(MachineBasicBlock &MBB, const RISCVInstrInfo *TII, uint32_t Label,
         MachineBasicBlock::iterator I = MachineBasicBlock::iterator{}) {
  if (!I.isValid())
    I = MBB.begin();
  BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(RISCV::AUIPC), RISCV::X0)
      .addImm(Label);
}

static bool runIndirectBranchTracking(MachineFunction &MF) {
  const auto &Subtarget = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo *TII = Subtarget.getInstrInfo();
  const Align LpadAlign = Align(4);

  if (!MF.getInfo<RISCVMachineFunctionInfo>()->hasCFProtectionBranch())
    return false;

  uint32_t FixedLabel = 0;
  if (std::optional<uint32_t> Label = Subtarget.getCLOpts().landing_pad_label) {
    if (!isUInt<20>(*Label))
      report_fatal_error("riscv-landing-pad-label=<val>, <val> needs to fit in "
                         "unsigned 20-bits");
    FixedLabel = *Label;
  }

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    if (&MBB == &MF.front()) {
      Function &F = MF.getFunction();
      // When trap is taken, landing pad is not needed.
      if (F.hasFnAttribute("interrupt"))
        continue;

      if (F.hasAddressTaken() || !F.hasLocalLinkage()) {
        emitLpad(MBB, TII, FixedLabel);
        if (MF.getAlignment() < LpadAlign)
          MF.setAlignment(LpadAlign);
        Changed = true;
      }
      continue;
    }

    if (MBB.hasAddressTaken()) {
      emitLpad(MBB, TII, FixedLabel);
      if (MBB.getAlignment() < LpadAlign)
        MBB.setAlignment(LpadAlign);
      Changed = true;
    }
  }

  return Changed;
}

bool RISCVIndirectBranchTrackingLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return runIndirectBranchTracking(MF);
}

PreservedAnalyses
RISCVIndirectBranchTrackingPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  return runIndirectBranchTracking(MF)
             ? getMachineFunctionPassPreservedAnalyses()
                   .preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}
