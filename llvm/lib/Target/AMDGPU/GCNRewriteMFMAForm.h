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

  /// Per-candidate cache of the src2 "needs VGPR" decision, computed once
  /// and reused on-demand.
  DenseMap<const MachineInstr *, bool> Src2NeedsVGPRCache;

  /// Do a speculative rewrite and collect copy locations. The speculative
  /// rewrite allows us to calculate the RP of the code after the rewrite, and
  /// the copy locations allow us to calculate the total cost of copies required
  /// for the rewrite. Stores the rewritten instructions in \p RewriteCands ,
  /// the copy locations for uses (of the MFMA result) in \p CopyForUse and the
  /// copy locations for defs (of the MFMA operands) in \p CopyForDef
  bool
  initHeuristics(std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands,
                 DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
                 SmallPtrSetImpl<MachineInstr *> &CopyForDef);

  /// Calculate the rewrite cost and undo the state change (e.g. rewriting) done
  /// in initHeuristics. Uses \p CopyForUse and \p CopyForDef to calculate copy
  /// costs, and \p RewriteCands to undo rewriting.
  int64_t getRewriteCost(
      ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands,
      const DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
      const SmallPtrSetImpl<MachineInstr *> &CopyForDef);

  /// Do the final rewrite on \p RewriteCands and insert any needed copies.
  bool rewrite(ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands);

  /// \returns true if this MI is a rewrite candidate.
  bool isRewriteCandidate(MachineInstr *MI) const;

  /// Resets all candidates in \p RewriteCands back to VGPR form.
  void resetRewriteCandsToVGPR(
      ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands);

  /// Finds all the reaching defs of \p UseMO and stores the SlotIndexes into \p
  /// DefIdxs
  void findReachingDefs(MachineOperand &UseMO, LiveIntervals *LIS,
                        SmallVectorImpl<SlotIndex> &DefIdxs);

  /// Finds all the reaching uses of \p DefMI and stores the use operands in \p
  /// ReachingUses
  void findReachingUses(const MachineInstr *DefMI, LiveIntervals *LIS,
                        SmallVectorImpl<MachineOperand *> &ReachingUses);

  /// Returns true if the src2 register with reaching defs \p Src2ReachingDefs
  /// has a use other than a group MFMA (in \p RewriteSet) or a copy, which
  /// would keep it in VGPR form rather than let it be reclassified to AGPR.
  bool hasUseRequiringVGPR(ArrayRef<SlotIndex> Src2ReachingDefs,
                           const SmallPtrSetImpl<MachineInstr *> &RewriteSet);

public:
  bool initGCNSchedStage() override;

  RewriteMFMAFormStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNREWRITEMFMAFORM_H
