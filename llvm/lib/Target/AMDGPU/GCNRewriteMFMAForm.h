//===-- GCNRewriteMFMAForm.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The pre-RA scheduling stage that moves MFMA accumulator chains into AGPRs.
///
/// A region whose ArchVGPR pressure is over the addressable limit is going to
/// spill. Moving whole MFMA chains to the AGPR bank takes that pressure off the
/// ArchVGPRs at the price of a bridge copy at each end of a chain, which the
/// stage weighs against the spills it expects to avoid.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNREWRITEMFMAFORM_H
#define LLVM_LIB_TARGET_AMDGPU_GCNREWRITEMFMAFORM_H

#include "GCNSchedStrategy.h"

namespace llvm {

class RewriteMFMAFormStage : public GCNSchedStage {
private:
  // Record regions with excess archvgpr register pressure over the physical
  // register limit. Register pressure in these regions usually will result in
  // spilling.
  BitVector RegionsWithExcessArchVGPR;

  const SIInstrInfo *TII;
  const SIRegisterInfo *SRI;

  /// Shared MFMA->AGPR analysis, with the candidate set as its active set.
  std::optional<AMDGPU::MFMAToAGPR> MFMAToAGPR;

  /// What buildPlan() decided for one MFMA. apply() replays these decisions; it
  /// repeats none of the analysis, so the cost model and the rewrite cannot
  /// disagree.
  struct MFMARewriteEntry {
    MachineInstr *MFMA;
    /// Reaching defs of src2 that need a bridge copy after them.
    SmallVector<MachineInstr *, 4> Src2Bridges;
    /// Reaching uses of dst that must read a bridge copy instead of dst.
    SmallVector<MachineOperand *, 4> DstUseBridges;
    /// Reaching defs of those uses that need a bridge copy after them.
    SmallVector<MachineInstr *, 4> DstUseDefBridges;
    /// True if the AGPR-form result has to be copied back into the original
    /// dst register right after the MFMA, because the reaching uses cannot be
    /// redirected to a bridge copy of their own.
    bool DstNeedsRestore;
    /// True if dst only has to change class, with no bridge anywhere.
    bool DstRecolorSafe;
  };

  /// An AGPR->VGPR copy or restore that apply() emits.
  struct BridgeCopy {
    MachineBasicBlock *MBB;
    Register Reg;
    /// The uses redirected to the copy. Empty for a restore.
    SmallVector<MachineOperand *, 4> Uses;
  };

  /// A complete, not-yet-applied rewrite of the function.
  struct MFMARewritePlan {
    /// The MFMAs whose opcode flips, in program order.
    SmallVector<MachineInstr *, 32> MFMAs;
    /// MFMAs as a set, used as the shared analysis' active set.
    SmallPtrSet<MachineInstr *, 16> MFMASet;
    /// Per-MFMA decisions, parallel to MFMAs.
    SmallVector<MFMARewriteEntry, 32> Entries;
    /// Registers that move from their VGPR class to the AGPR equivalent.
    SmallSetVector<Register, 16> RecolorRegs;
    /// (reaching def, register) pairs that get one bridge copy each.
    SmallSetVector<std::pair<MachineInstr *, Register>, 16> DefBridges;
    /// Registers whose MFMA result is copied back into the original VGPR after
    /// the MFMA.
    SmallSetVector<Register, 16> RestoredRegs;
    /// One entry per copy or restore apply() emits.
    SmallVector<BridgeCopy, 16> UseBridges;
  };

  /// Plan the rewrite of \p Cands, which must be in program order, into
  /// \p Plan. Pure analysis: it mutates nothing.
  ///
  /// \returns false if there is nothing to rewrite.
  bool buildPlan(MFMARewritePlan &Plan, ArrayRef<MachineInstr *> Cands);

  /// \returns the MFMAs of the first \p N chains, in program order.
  SmallVector<MachineInstr *, 32>
  selectChainPrefix(ArrayRef<SmallVector<MachineInstr *, 8>> Chains,
                    unsigned N) const;

  /// \returns how many of the (longest first) \p Chains are worth converting,
  /// or 0 if converting none of them is.
  unsigned findBestChainCount(ArrayRef<SmallVector<MachineInstr *, 8>> Chains);

  /// \returns the pressure of every region once \p Plan is applied. The AGPR
  /// form is reverted before returning. Restored registers and bridged src2s
  /// stay VGPR, with their AGPR counted where it is live.
  SmallVector<GCNRegPressure, 8> measure(const MFMARewritePlan &Plan);

  /// \returns the pressure \p Reg would have as an AGPR.
  GCNRegPressure agprPressureOf(Register Reg);

  /// What cost() thinks of a plan. Compared by findBestChainCount(); see
  /// isBetterThan() for the ordering.
  struct MFMARewriteCost {
    /// Spill saving plus bridge copies. Negative means profitable.
    int64_t Cost = std::numeric_limits<int64_t>::max();
    /// The spill saving plus the copies in blocks hotter than entry.
    int64_t HotCost = std::numeric_limits<int64_t>::max();
    /// The most ArchVGPRs any region needs after the plan; the maximum past the
    /// soft AGPR limit.
    unsigned ArchPeak = std::numeric_limits<unsigned>::max();

    /// Cheapest in the loops wins, then the lowest ArchVGPR peak, then the
    /// cheapest overall.
    bool isBetterThan(const MFMARewriteCost &O) const;
  };

  /// \returns the cost of \p Plan given the pressure measure() reported in
  /// \p PressuresAfter. Negative cost means the rewrite is profitable.
  MFMARewriteCost cost(const MFMARewritePlan &Plan,
                       ArrayRef<GCNRegPressure> PressuresAfter);

  /// Realize \p Plan: flip the opcodes, insert the bridge copies, and
  /// reclassify the registers.
  void apply(const MFMARewritePlan &Plan);

  /// Insert "MappedReg = COPY Reg" after the reaching def \p RD, unless \p
  /// AlreadyBridged says an earlier call did. Keeps \p LastMIToRegion, which
  /// maps a region's last MI to the region, current.
  void bridgeAfterDef(MachineInstr *RD, Register Reg, Register MappedReg,
                      SmallPtrSetImpl<MachineInstr *> &AlreadyBridged,
                      DenseMap<MachineInstr *, unsigned> &LastMIToRegion);

  /// Appends every instruction whose def of \p Reg reaches \p UseMI to \p
  /// Defs, without duplicates.
  ///
  /// This is the last write on each path, not the producer of each lane. One
  /// whole-register copy after it holds every lane.
  void findReachingDefs(Register Reg, const MachineInstr &UseMI,
                        const LiveIntervals *LIS,
                        SmallVectorImpl<MachineInstr *> &Defs);

  /// Finds all the reaching uses of \p DefMI and stores the use operands in \p
  /// ReachingUses
  void findReachingUses(const MachineInstr *DefMI, const LiveIntervals *LIS,
                        SmallVectorImpl<MachineOperand *> &ReachingUses);

public:
  bool initGCNSchedStage() override;

  RewriteMFMAFormStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNREWRITEMFMAFORM_H
