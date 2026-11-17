//===- llvm/CodeGen/MachineBlockPlacement.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEBLOCKPLACEMENT_H
#define LLVM_CODEGEN_MACHINEBLOCKPLACEMENT_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class BranchProbability;
class MachineBasicBlock;
class MachineBranchProbabilityInfo;
class MachineLoopInfo;
class MBFIWrapper;
class ProfileSummaryInfo;

/// Interface providing profile queries for `alignMachineBlocks`, abstracting
/// over `MachineBlockFrequencyInfo`/`MachineBranchProbabilityInfo` and
/// post-layout basic block section profiles.
struct LLVM_ABI BlockAlignmentProfile {
  virtual ~BlockAlignmentProfile() = default;
  virtual void anchor();

  /// Returns true if \p BB's execution frequency is strictly less than
  /// \p ColdRatio of \p RefBB's frequency, using `BranchProbability` as a
  /// [0, 1] scaling factor: `Freq(BB) < Freq(RefBB) * ColdRatio`.
  virtual bool isColdRelativeTo(const MachineBasicBlock &BB,
                                const MachineBasicBlock &RefBB,
                                BranchProbability ColdRatio) const = 0;

  /// Returns true if the edge `\p Src -> \p Dst` accounts for at most
  /// \p ColdRatio of \p Dst's execution frequency
  /// (`Freq(Src -> Dst) <= Freq(Dst) * ColdRatio`).
  virtual bool isEdgeColdRelativeToDest(const MachineBasicBlock &Src,
                                        const MachineBasicBlock &Dst,
                                        BranchProbability ColdRatio) const = 0;

  /// Returns true if \p BB should be optimized for code size.
  virtual bool shouldOptForSize(const MachineBasicBlock &BB) const = 0;
};

/// `BlockAlignmentProfile` implementation backed by `MBFIWrapper`,
/// `MachineBranchProbabilityInfo`, and `ProfileSummaryInfo`.
struct LLVM_ABI MBFIBlockAlignmentProfile final : BlockAlignmentProfile {
  MBFIWrapper &MBFI;
  const MachineBranchProbabilityInfo &MBPI;
  ProfileSummaryInfo *PSI;

  MBFIBlockAlignmentProfile(MBFIWrapper &MBFI,
                            const MachineBranchProbabilityInfo &MBPI,
                            ProfileSummaryInfo *PSI)
      : MBFI(MBFI), MBPI(MBPI), PSI(PSI) {}

  bool isColdRelativeTo(const MachineBasicBlock &BB,
                        const MachineBasicBlock &RefBB,
                        BranchProbability ColdRatio) const override;

  bool isEdgeColdRelativeToDest(const MachineBasicBlock &Src,
                                const MachineBasicBlock &Dst,
                                BranchProbability ColdRatio) const override;

  bool shouldOptForSize(const MachineBasicBlock &BB) const override;
};

/// Assigns loop and forced alignments to basic blocks in \p MF after block
/// layout is finalized. When \p Prof and \p MLI are non-null, hot loop headers
/// (and rotated backedge targets) dominated by jump entries are aligned; flag
/// overrides (`-align-all-blocks`, `-align-all-nofallthru-blocks`) are applied
/// unconditionally.
LLVM_ABI void alignMachineBlocks(MachineFunction &MF,
                                 const MachineLoopInfo *MLI,
                                 const BlockAlignmentProfile *Prof);

class MachineBlockPlacementPass
    : public OptionalPassInfoMixin<MachineBlockPlacementPass> {

  bool AllowTailMerge = true;

public:
  MachineBlockPlacementPass(bool AllowTailMerge)
      : AllowTailMerge(AllowTailMerge) {}
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);

  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName) const;
};

class MachineBlockPlacementStatsPass
    : public RequiredPassInfoMixin<MachineBlockPlacementStatsPass> {

public:
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINEBLOCKPLACEMENT_H
