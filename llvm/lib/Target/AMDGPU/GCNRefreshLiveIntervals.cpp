//===-- GCNRefreshLiveIntervals.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNRefreshLiveIntervals.h"
#include "AMDGPU.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "gcn-refresh-live-intervals"

static void refreshLiveIntervals(MachineFunction &MF, SlotIndexes &SI,
                                 LiveIntervals &LIS) {
  SI.reanalyze(MF);
  LIS.recompute(MF);
}

class GCNRefreshLiveIntervalsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNRefreshLiveIntervalsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    refreshLiveIntervals(MF, getAnalysis<SlotIndexesWrapperPass>().getSI(),
                         getAnalysis<LiveIntervalsWrapperPass>().getLIS());
    return false;
  }

  StringRef getPassName() const override {
    return "GCN Refresh Live Intervals";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char GCNRefreshLiveIntervalsLegacy::ID = 0;

char &llvm::GCNRefreshLiveIntervalsID = GCNRefreshLiveIntervalsLegacy::ID;

INITIALIZE_PASS_BEGIN(GCNRefreshLiveIntervalsLegacy, DEBUG_TYPE,
                      "GCN Refresh Live Intervals", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(GCNRefreshLiveIntervalsLegacy, DEBUG_TYPE,
                    "GCN Refresh Live Intervals", false, false)

FunctionPass *llvm::createGCNRefreshLiveIntervalsLegacyPass() {
  return new GCNRefreshLiveIntervalsLegacy();
}

PreservedAnalyses
GCNRefreshLiveIntervalsPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  refreshLiveIntervals(MF, MFAM.getResult<SlotIndexesAnalysis>(MF),
                       MFAM.getResult<LiveIntervalsAnalysis>(MF));

  PreservedAnalyses PA;
  PA.preserve<SlotIndexesAnalysis>();
  PA.preserve<LiveIntervalsAnalysis>();
  PA.preserve<MachineDominatorTreeAnalysis>();
  return PA;
}
