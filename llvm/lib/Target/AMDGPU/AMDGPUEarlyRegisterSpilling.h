//===------------------- AMDGPUEarlyRegisterSpilling.h --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Early Register Spilling.
//
// This is based on ideas from the paper:
// "Register Spilling and Live-Range Splitting for SSA-Form Programs"
// Matthias Braun and Sebastian Hack, CC'09
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUEARLYREGISTERSPILLING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUEARLYREGISTERSPILLING_H

#include "AMDGPUNextUseAnalysis.h"
#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

using SetVectorType = SmallSetVector<MachineInstr *, 32>;

struct RegisterSpillCandidate {
  Register Reg;
  NextUseDistance Dist;
  LaneBitmask Mask;
};

class AMDGPUEarlyRegisterSpilling : public MachineFunctionPass {
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const SIMachineFunctionInfo *MFI = nullptr;
  MachineFrameInfo *FrameInfo = nullptr;
  LiveIntervals *LIS = nullptr;
  SlotIndexes *Indexes = nullptr;
  MachineDominatorTree *DT = nullptr;
  AMDGPUNextUseAnalysis *NUA = nullptr;
  bool TgSplit = false;
  // Spilled registers are kept here to avoid respilling.
  // TODO: Support spilling of a register more than once.
  DenseSet<Register> SpilledRegs;
  // We do not spill the registers that are returned by restore instructions.
  DenseSet<Register> RestoredRegs;

  unsigned MaxVGPRs = 0;
  unsigned MaxSGPRs = 0;

  /// Check if it is legal to spill \p CandidateReg e.g. is not a physical
  /// register.
  bool isLegalCandidate(Register CandidateReg);

  /// Return the registers with the longest next-use distance that we need to
  /// spill. \p CurMI is the high-register-pressure point.
  SmallVector<RegisterSpillCandidate>
  getCandidates(MachineInstr *CurMI, GCNDownwardRPTracker &RPTracker);

  MachineInstr *emitRestore(Register CandidateReg, MachineInstr *DefRegUseInstr,
                            int FI);

  MachineInstr *emitRestore(Register CandidateReg, MachineBasicBlock &InsertBB,
                            int FI);

  /// Main spill function that emits the spill and restore code.
  void spill(MachineInstr *CurMI, GCNDownwardRPTracker &RPTracker,
             unsigned NumOfSpills);

  bool hasPHIUseInSameBB(Register Reg, MachineBasicBlock *MBB);

  /// Calculate the initial maximum register pressure per basic block (before
  /// any spilling) because it gives us the maximum number of VGPRs and SGPRs.
  GCNRegPressure getMaxPressure(const MachineFunction &MF);

  bool isSpilledReg(Register Reg) { return SpilledRegs.contains(Reg); }

  bool isRestoredReg(Register Reg) { return RestoredRegs.contains(Reg); }

  void clearTables() {
    SpilledRegs.clear();
    RestoredRegs.clear();
  }

public:
  static char ID;

  AMDGPUEarlyRegisterSpilling() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &) override;

  StringRef getPassName() const override {
    return "AMDGPU Early Register Spilling";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.addRequired<AMDGPUNextUseAnalysisLegacyPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineBlockFrequencyInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUEARLYREGISTERSPILLING_H
