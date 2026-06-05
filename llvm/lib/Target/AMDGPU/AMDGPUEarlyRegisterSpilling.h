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
  // Spilled registers are kept here to avoid respilling.
  // TODO: Support spilling of a register more than once.
  DenseSet<Register> SpilledRegs;
  // We avoid spilling restored registers.
  DenseSet<Register> RestoredRegs;

  unsigned MaxVGPRs = 0;
  unsigned MaxSGPRs = 0;

  /// Check if it is legal to spill \p CandidateReg e.g. is not a physical
  /// register.
  bool isLegalToSpill(Register CandidateReg);

  /// Return the registers with the longest next-use distance that we need to
  /// spill. \p CurMI is the high-register-pressure point.
  SmallVector<std::pair<Register, LaneBitmask>>
  getRegistersToSpill(MachineInstr *CurMI, GCNDownwardRPTracker &RPTracker);

  /// Return where we have to spill \p DefRegToSpill. It can be one of:
  /// (i) the high register pressure point,
  /// (ii) the definition block of \p DefRegToSpill,
  /// (iii) the common dominator of \p CurMI and related uses.
  /// \p CurMI is the high-register-pressure point.
  std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
  getWhereToSpill(MachineInstr *CurMI, Register DefRegToSpill);

  /// Return where we have to spill if the definition of the spilled register is
  /// inside a loop. \p CurMI is the high-register-pressure point.
  std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
  getWhereToSpillIfDefintionInLoop(MachineInstr *CurMI,
                                   MachineBasicBlock *DefRegMBB);

  /// Main spill function that emits the spill and restore code.
  void spill(MachineInstr *CurMI, GCNDownwardRPTracker &RPTracker,
             unsigned NumOfSpills);

  /// Emit restore instruction where it is needed
  MachineInstr *emitRestore(Register SpillReg, MachineInstr *UseMI, int FI);
  /// Emit restore instruction at the end of a basic block.
  MachineInstr *emitRestore(Register SpillReg, MachineBasicBlock &InsertBB,
                            int FI);

  /// Emit restore instructions for each group that contains the uses that are
  /// dominated by the head of the group.
  void emitRestoreInstrsForDominatedUses(
      Register DefRegToSpill, MachineInstr *SpillInstruction,
      MachineInstr *CurMI, SetVectorType &DominatedUses,
      SmallVector<MachineInstr *> &RestoreInstrs,
      SmallVector<MachineInstr *> &RestoreUses, int FI);

  /// Check if it is legal or profitable to emit a restore in the common
  /// dominator.
  bool shouldEmitRestoreInCommonDominator(
      MachineBasicBlock *SpillBlock, MachineBasicBlock *CurMBB,
      MachineBasicBlock *CommonDominatorToRestore);

  /// Find the common dominator of the reachable uses and the block that we
  /// intend to spill.
  MachineBasicBlock *
  findCommonDominatorToSpill(MachineBasicBlock *SpillBlock,
                             Register DefRegToSpill,
                             const SetVectorType &NonDominatedReachableUses);

  /// Collect Non Dominated Reachable and Unreachable uses.
  void classifyUses(MachineBasicBlock *SpillBlock, Register DefRegToSpill,
                    MachineInstr *CurMI, SetVectorType &DominatedUses,
                    SetVectorType &NonDominatedReachableUses,
                    SetVectorType &UnreachableUses);

  /// Helper functions to update the live interval analysis which is used by
  /// the Register Pressure Tracker.
  void updateIndexes(MachineInstr *MI);
  void updateLiveness(Register Reg);
  void updateLiveness(MachineInstr *MI);

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
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<AMDGPUNextUseAnalysisLegacyPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineBlockFrequencyInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUEARLYREGISTERSPILLING_H
