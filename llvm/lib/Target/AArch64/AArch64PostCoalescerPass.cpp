//===- AArch64PostCoalescerPass.cpp - AArch64 Post Coalescer pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64MachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-post-coalescer"

namespace {

bool runAArch64PostCoalescer(MachineFunction &MF, LiveIntervals &LIS) {
  AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
  if (!FuncInfo->hasStreamingModeChanges())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      switch (MI.getOpcode()) {
      default:
        break;
      case AArch64::COALESCER_BARRIER_FPR16:
      case AArch64::COALESCER_BARRIER_FPR32:
      case AArch64::COALESCER_BARRIER_FPR64:
      case AArch64::COALESCER_BARRIER_FPR128: {
        Register Src = MI.getOperand(1).getReg();
        Register Dst = MI.getOperand(0).getReg();
        if (Src != Dst)
          MRI.replaceRegWith(Dst, Src);

        if (MI.getOperand(1).isUndef())
          for (MachineOperand &MO : MRI.use_operands(Dst))
            MO.setIsUndef();

        // MI must be erased from the basic block before recalculating the live
        // interval.
        LIS.RemoveMachineInstrFromMaps(MI);
        MI.eraseFromParent();

        LIS.removeInterval(Src);
        LIS.createAndComputeVirtRegInterval(Src);

        Changed = true;
        break;
      }
      }
    }
  }

  return Changed;
}

struct AArch64PostCoalescerLegacy : public MachineFunctionPass {
  static char ID;

  AArch64PostCoalescerLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 Post Coalescer pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char AArch64PostCoalescerLegacy::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(AArch64PostCoalescerLegacy, "aarch64-post-coalescer",
                      "AArch64 Post Coalescer Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AArch64PostCoalescerLegacy, "aarch64-post-coalescer",
                    "AArch64 Post Coalescer Pass", false, false)

bool AArch64PostCoalescerLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return runAArch64PostCoalescer(MF, LIS);
}

PreservedAnalyses
AArch64PostCoalescerPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  auto &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const bool Changed = runAArch64PostCoalescer(MF, LIS);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<LiveIntervalsAnalysis>();
  PA.preserve<SlotIndexesAnalysis>();
  return PA;
}

FunctionPass *llvm::createAArch64PostCoalescerPass() {
  return new AArch64PostCoalescerLegacy();
}
