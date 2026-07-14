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

/// Helper data structure for grouping together uses where the head of the group
/// dominates all the other uses in the group.
class DomGroup {
public:
  enum class RestorePlacement {
    BeforeHead,
    LoopPreheader,
    IncomingBlockOfPhi,
  };

private:
  SmallVector<MachineInstr *> Uses;
  SmallVector<MachineBasicBlock *> UseBlocks;
  SmallDenseMap<MachineInstr *, MachineBasicBlock *> PHIInstrToRestoreBlock;
  MachineInstr *Restore = nullptr;
  MachineBasicBlock *CommonDominator = nullptr;
  bool Deleted = false;
  RestorePlacement WhereToRestore;

public:
  DomGroup(MachineInstr *MI, MachineBasicBlock *RestoreBlock,
           RestorePlacement WhereToRestore)
      : WhereToRestore(WhereToRestore) {
    Uses.push_back(MI);
    UseBlocks.push_back(RestoreBlock);
    if (MI->isPHI())
      PHIInstrToRestoreBlock[MI] = RestoreBlock;
  }
  DomGroup() = default;
  MachineInstr *getHead() const { return Uses.front(); }
  bool isDeleted() const { return Deleted; }
  void merge(DomGroup &Other) {
    for (auto *MI : Other.Uses)
      Uses.push_back(MI);

    for (auto *UseMBB : Other.UseBlocks)
      UseBlocks.push_back(UseMBB);

    PHIInstrToRestoreBlock.insert(Other.PHIInstrToRestoreBlock.begin(),
                                  Other.PHIInstrToRestoreBlock.end());

    Other.Deleted = true;
  }
  const auto &getUses() const { return Uses; }
  const auto &getUseBlocks() const { return UseBlocks; }
  size_t size() const { return Uses.size(); }
  void setCommonDominator(MachineBasicBlock *CD) { CommonDominator = CD; }
  MachineBasicBlock *getCommonDominator() const { return CommonDominator; }
  void setRestore(MachineInstr *R) { Restore = R; }
  MachineInstr *getRestore() const { return Restore; }
  bool hasCommonDominator() const { return CommonDominator != nullptr; }
  MachineBasicBlock *getRestoreBlock() const { return UseBlocks.front(); }
  void setRestoreBlock(MachineBasicBlock *NewRestoreBlock) {
    *UseBlocks.begin() = NewRestoreBlock;
  }
  MachineBasicBlock *getRestoreBlockForPHI(MachineInstr *PHI) const {
    assert(PHI->isPHI() && "The instruction is not a PHI node.");
    auto It = PHIInstrToRestoreBlock.find(PHI);
    assert(It != PHIInstrToRestoreBlock.end() &&
           "The PHI node does not exist in the map.");
    return It->second;
  }
  RestorePlacement getWhereToRestore() const { return WhereToRestore; }
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
  // Spilled registers are kept here to avoid respilling.
  // TODO: Support spilling of a register more than once.
  DenseSet<Register> SpilledRegs;
  // We do not spill the registers that are returned by restore instructions.
  DenseMap<Register, DomGroup> RestoreRegToDomGroup;

  unsigned MaxVGPRs = 0;
  unsigned MaxSGPRs = 0;

  /// Check if it is legal to spill \p CandidateReg e.g. is not a physical
  /// register.
  bool isLegalCandidate(Register CandidateReg);

  /// Return the registers with the longest next-use distance that we need to
  /// spill. \p CurMI is the high-register-pressure point.
  SmallVector<RegisterSpillCandidate>
  getCandidates(MachineInstr *CurMI, GCNDownwardRPTracker &RPTracker);

  /// Return where we have to spill \p RegToSpill. It can be one of:
  /// (i) the high register pressure point,
  /// (ii) the definition block of \p RegToSpill,
  /// (iii) the common dominator of \p CurMI and related uses.
  /// \p CurMI is the high-register-pressure point.
  std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
  getWhereToSpill(MachineInstr *CurMI, Register RegToSpill);

  /// Return where we have to spill if the definition of the spilled register is
  /// inside a loop. \p CurMI is the high-register-pressure point.
  std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
  getWhereToSpillIfDefintionInLoop(MachineInstr *CurMI,
                                   MachineBasicBlock *DefRegMBB);

  /// Main spill function that emits the spill and restore code.
  void spill(MachineInstr *CurMI, GCNDownwardRPTracker &RPTracker,
             unsigned NumOfSpills);

  /// Emit restore instructions for each group that contains the uses that are
  /// dominated by the head of the group.
  void groupUses(Register RegToSpill, MachineBasicBlock *SpillBlock,
                 MachineInstr *CurMI, SetVectorType &DominatedUses,
                 SmallVector<DomGroup> &GroupOfUses);

  /// Check if it is legal or profitable to emit a restore in the common
  /// dominator.
  bool shouldEmitRestoreInCommonDominator(
      MachineBasicBlock *SpillBlock, MachineBasicBlock *CurMBB,
      MachineBasicBlock *CommonDominatorToRestore);

  /// Find the common dominator of the reachable uses and the block that we
  /// intend to spill.
  MachineBasicBlock *
  findCommonDominatorToSpill(MachineBasicBlock *SpillBlock, Register RegToSpill,
                             const SetVectorType &NonDominatedReachableUses);

  /// Collect Non Dominated Reachable and Unreachable uses.
  void classifyUses(MachineBasicBlock *SpillBlock, Register RegToSpill,
                    MachineInstr *CurMI, SetVectorType &DominatedUses,
                    SetVectorType &NonDominatedReachableUses,
                    SetVectorType &UnreachableUses);

  bool hasPHIUseInSameBB(Register Reg, MachineBasicBlock *MBB);

  /// Calculate the initial maximum register pressure per basic block (before
  /// any spilling) because it gives us the maximum number of VGPRs and SGPRs.
  GCNRegPressure getMaxPressure(const MachineFunction &MF);

  bool isSpilledReg(Register Reg) { return SpilledRegs.contains(Reg); }

  bool isRestoredReg(Register Reg) {
    return RestoreRegToDomGroup.contains(Reg);
  }

  void clearTables() {
    SpilledRegs.clear();
    RestoreRegToDomGroup.clear();
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
