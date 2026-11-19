//===- MachineBlockAlignment.h - Basic block alignment ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assigns basic block alignments after the block layout is finalized.
// Shared by MachineBlockPlacement and BasicBlockSections (which may align
// blocks after reordering them, see -align-blocks-in-bbsections).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MACHINEBLOCKALIGNMENT_H
#define LLVM_LIB_CODEGEN_MACHINEBLOCKALIGNMENT_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstdint>

namespace llvm {

class MachineBasicBlock;
class MachineBranchProbabilityInfo;
class MachineFunction;
class MachineLoopInfo;
class MBFIWrapper;
class ProfileSummaryInfo;

/// Profile queries for `alignMachineBlocks` backed by `MBFIWrapper`,
/// `MachineBranchProbabilityInfo`, and `ProfileSummaryInfo`.
class MBFIBlockAlignmentProfile {
public:
  MBFIBlockAlignmentProfile(MBFIWrapper &MBFI,
                            const MachineBranchProbabilityInfo &MBPI,
                            ProfileSummaryInfo *PSI)
      : MBFI(MBFI), MBPI(MBPI), PSI(PSI) {}

  /// Returns true if `Freq(BB) < Freq(RefBB) * ColdProb` (`ColdProb` = 20%).
  bool isColdRelativeTo(const MachineBasicBlock &BB,
                        const MachineBasicBlock &RefBB) const;
  /// Returns true if `Freq(Src -> Dst) <= Freq(Dst) * ColdProb`.
  bool isEdgeColdRelativeToDest(const MachineBasicBlock &Src,
                                const MachineBasicBlock &Dst) const;
  /// Returns true if \p BB should be optimized for code size.
  bool shouldOptForSize(const MachineBasicBlock &BB) const;

private:
  MBFIWrapper &MBFI;
  const MachineBranchProbabilityInfo &MBPI;
  ProfileSummaryInfo *PSI;
};

/// Profile queries for `alignMachineBlocks` backed by the basic block sections
/// profile's block and edge counts (or the inferred weights from
/// `BasicBlockMatchingAndInference`).
class BBSectionsBlockAlignmentProfile {
public:
  using BlockCountFn = function_ref<uint64_t(const MachineBasicBlock &)>;
  using EdgeCountFn = function_ref<uint64_t(const MachineBasicBlock &,
                                            const MachineBasicBlock &)>;

  BBSectionsBlockAlignmentProfile(BlockCountFn BlockCount,
                                  EdgeCountFn EdgeCount)
      : BlockCount(BlockCount), EdgeCount(EdgeCount) {}

  bool isColdRelativeTo(const MachineBasicBlock &BB,
                        const MachineBasicBlock &RefBB) const;
  bool isEdgeColdRelativeToDest(const MachineBasicBlock &Src,
                                const MachineBasicBlock &Dst) const;
  /// The profile summary thresholds used by profile-guided size optimization
  /// are not comparable with the basic block sections profile counts. Cold
  /// blocks are already excluded by `isColdRelativeTo`.
  bool shouldOptForSize(const MachineBasicBlock &) const { return false; }

private:
  BlockCountFn BlockCount;
  EdgeCountFn EdgeCount;
};

/// Assigns loop alignments to the blocks of \p MF after block layout is
/// finalized, using \p Prof for profile queries, and applies the
/// `-align-all-blocks` and `-align-all-nofallthru-blocks` overrides.
void alignMachineBlocks(MachineFunction &MF, const MachineLoopInfo &MLI,
                        const MBFIBlockAlignmentProfile &Prof);
void alignMachineBlocks(MachineFunction &MF, const MachineLoopInfo &MLI,
                        const BBSectionsBlockAlignmentProfile &Prof);

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_MACHINEBLOCKALIGNMENT_H
