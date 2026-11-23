//===- SIInsertAsyncBarrierArriveWorkaround.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Add a counted replay pseudo after affected async barrier arrivals. MC
/// emission saves EXEC, clears it for the dummy load, and restores it.
/// Run after scheduling to keep the pseudo after its arrival, and before
/// wait-count analysis so it accounts for the added async load.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "si-insert-async-barrier-arrive-workaround"

static bool insertAsyncBarrierArriveWorkaround(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.needsAsyncBarrierArriveReplayWorkaround())
    return false;

  // Collect arrivals before inserting new definitions of ASYNCcnt.
  SmallVector<MachineInstr *, 4> AsyncBarrierArrivals;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineInstr &MI : MRI.def_instructions(AMDGPU::ASYNCcnt)) {
    if (MI.getOpcode() == AMDGPU::DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64)
      AsyncBarrierArrivals.push_back(&MI);
  }
  if (AsyncBarrierArrivals.empty())
    return false;

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const Register SavedExec = MFI->getSGPRForEXECCopy();
  if (!SavedExec) {
    reportFatalUsageError(
        "missing EXEC copy SGPR for async barrier workaround");
  }

  const SIInstrInfo &TII = *ST.getInstrInfo();
  for (MachineInstr *MI : AsyncBarrierArrivals) {
    MachineBasicBlock &MBB = *MI->getParent();
    const DebugLoc &DL = MI->getDebugLoc();
    auto InsertPt = std::next(MI->getIterator());
    // Leave explicit S_WAIT_ASYNCCNT thresholds unchanged.
    BuildMI(MBB, InsertPt, DL,
            TII.get(AMDGPU::GLOBAL_LOAD_ASYNC_TO_LDS_B8_replay))
        .addReg(AMDGPU::VGPR0, RegState::Undef)
        .addReg(AMDGPU::VGPR0_VGPR1, RegState::Undef)
        .addImm(0)
        .addImm(0)
        .addReg(SavedExec, RegState::ImplicitDefine);
  }
  return true;
}

namespace {
class SIInsertAsyncBarrierArriveWorkaroundLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIInsertAsyncBarrierArriveWorkaroundLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return insertAsyncBarrierArriveWorkaround(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace

INITIALIZE_PASS(SIInsertAsyncBarrierArriveWorkaroundLegacy, DEBUG_TYPE,
                "SI insert async barrier arrival workaround", false, false)

char SIInsertAsyncBarrierArriveWorkaroundLegacy::ID = 0;

FunctionPass *llvm::createSIInsertAsyncBarrierArriveWorkaroundPass() {
  return new SIInsertAsyncBarrierArriveWorkaroundLegacy();
}

PreservedAnalyses SIInsertAsyncBarrierArriveWorkaroundPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  if (!insertAsyncBarrierArriveWorkaround(MF))
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
