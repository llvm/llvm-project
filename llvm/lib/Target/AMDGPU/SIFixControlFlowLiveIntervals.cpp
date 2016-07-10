//===-- SIFixControlFlowLiveIntervals.cpp - Fix CF live intervals ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Spilling of EXEC masks used for control flow messes up control flow
/// lowering, so mark all live intervals associated with CF instructions as
/// non-spillable.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-cf-live-intervals"

namespace {

class SIFixControlFlowLiveIntervals : public MachineFunctionPass {
public:
  static char ID;

public:
  SIFixControlFlowLiveIntervals() : MachineFunctionPass(ID) {
    initializeSIFixControlFlowLiveIntervalsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Fix CF Live Intervals";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIFixControlFlowLiveIntervals, DEBUG_TYPE,
                      "SI Fix CF Live Intervals", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(SIFixControlFlowLiveIntervals, DEBUG_TYPE,
                    "SI Fix CF Live Intervals", false, false)

char SIFixControlFlowLiveIntervals::ID = 0;

char &llvm::SIFixControlFlowLiveIntervalsID = SIFixControlFlowLiveIntervals::ID;

FunctionPass *llvm::createSIFixControlFlowLiveIntervalsPass() {
  return new SIFixControlFlowLiveIntervals();
}

bool SIFixControlFlowLiveIntervals::runOnMachineFunction(MachineFunction &MF) {
  LiveIntervals *LIS = &getAnalysis<LiveIntervals>();

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
        case AMDGPU::SI_IF:
        case AMDGPU::SI_ELSE:
        case AMDGPU::SI_BREAK:
        case AMDGPU::SI_IF_BREAK:
        case AMDGPU::SI_ELSE_BREAK:
        case AMDGPU::SI_END_CF: {
          unsigned Reg = MI.getOperand(0).getReg();
          LIS->getInterval(Reg).markNotSpillable();
          break;
        }
        default:
          break;
      }
    }
  }

  return false;
}
