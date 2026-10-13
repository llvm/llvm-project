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
    /// (block, register) pairs that get one bridge copy each.
    SmallSetVector<std::pair<MachineBasicBlock *, Register>, 16> UseBridges;
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

  /// \returns the pressure of every region over the ArchVGPR limit, as it would
  /// be once \p Plan is applied. The AGPR form is reverted before returning.
  SmallVector<GCNRegPressure, 8> measure(const MFMARewritePlan &Plan);

  /// What cost() thinks of a plan. Compared by findBestChainCount(); see
  /// isBetterThan() for the ordering.
  struct MFMARewriteCost {
    /// Spill saving plus bridge copies. Negative means profitable.
    int64_t Cost = std::numeric_limits<int64_t>::max();
    /// AGPRs the plan leaves in use, or 0 once that is past the soft limit.
    unsigned AGPRUtil = 0;

    /// Cheapest wins; ties go to the plan holding the most in AGPRs.
    bool isBetterThan(const MFMARewriteCost &O) const;
  };

  /// \returns the cost of \p Plan given the pressure measure() reported in
  /// \p PressuresAfter. Negative cost means the rewrite is profitable.
  MFMARewriteCost cost(const MFMARewritePlan &Plan,
                       ArrayRef<GCNRegPressure> PressuresAfter);

  /// Realize \p Plan: flip the opcodes, insert the bridge copies, and
  /// reclassify the registers.
  void apply(const MFMARewritePlan &Plan);

  /// Appends every instruction whose def of \p UseMO's register reaches
  /// \p UseMO to \p Defs, without duplicates.
  void findReachingDefs(const MachineOperand &UseMO, LiveIntervals *LIS,
                        SmallVectorImpl<MachineInstr *> &Defs);

  /// Finds all the reaching uses of \p DefMI and stores the use operands in \p
  /// ReachingUses
  void findReachingUses(const MachineInstr *DefMI, LiveIntervals *LIS,
                        SmallVectorImpl<MachineOperand *> &ReachingUses);

public:
  /// \returns false if the stage is switched off on the command line.
  static bool isEnabled();

  bool initGCNSchedStage() override;

  RewriteMFMAFormStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNREWRITEMFMAFORM_H
