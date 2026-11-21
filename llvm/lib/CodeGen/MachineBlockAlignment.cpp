//===- MachineBlockAlignment.cpp - Basic block alignment ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assigns basic block alignments after the block layout is finalized.
// The alignment logic is written once, templated on the profile type, and
// statically dispatched through the `alignMachineBlocks` overloads.
//
//===----------------------------------------------------------------------===//

#include "MachineBlockAlignment.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MBFIWrapper.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineSizeOpts.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<unsigned> AlignAllBlock(
    "align-all-blocks",
    cl::desc("Force the alignment of all blocks in the function in log2 format "
             "(e.g 4 means align on 16B boundaries)."),
    cl::init(0), cl::Hidden);

static cl::opt<unsigned> AlignAllNonFallThruBlocks(
    "align-all-nofallthru-blocks",
    cl::desc("Force the alignment of all blocks that have no fall-through "
             "predecessors (i.e. don't add nops that are executed). In log2 "
             "format (e.g 4 means align on 16B boundaries)."),
    cl::init(0), cl::Hidden);

static cl::opt<unsigned> MaxBytesForAlignmentOverride(
    "max-bytes-for-alignment",
    cl::desc("Forces the maximum bytes allowed to be emitted when padding for "
             "alignment"),
    cl::init(0), cl::Hidden);

/// A block (or edge) is cold relative to a reference block if its frequency is
/// less than this fraction of the reference block's frequency.
static constexpr BranchProbability ColdProb(1, 5); // 20%

bool MBFIBlockAlignmentProfile::isColdRelativeTo(
    const MachineBasicBlock &BB, const MachineBasicBlock &RefBB) const {
  return MBFI.getBlockFreq(&BB) < MBFI.getBlockFreq(&RefBB) * ColdProb;
}

bool MBFIBlockAlignmentProfile::isEdgeColdRelativeToDest(
    const MachineBasicBlock &Src, const MachineBasicBlock &Dst) const {
  BranchProbability EdgeProb = MBPI.getEdgeProbability(&Src, &Dst);
  BlockFrequency EdgeFreq = MBFI.getBlockFreq(&Src) * EdgeProb;
  return EdgeFreq <= MBFI.getBlockFreq(&Dst) * ColdProb;
}

bool MBFIBlockAlignmentProfile::shouldOptForSize(
    const MachineBasicBlock &BB) const {
  return llvm::shouldOptimizeForSize(&BB, PSI, &MBFI);
}

bool BBSectionsBlockAlignmentProfile::isColdRelativeTo(
    const MachineBasicBlock &BB, const MachineBasicBlock &RefBB) const {
  uint64_t Count = BlockCount(BB);
  uint64_t RefCount = BlockCount(RefBB);
  // A block without any samples is cold. Otherwise, compare the ratio of the
  // raw counts against `ColdProb` instead of `Count < RefCount * ColdProb`,
  // whose right-hand side truncates to zero for small reference counts.
  if (Count == 0)
    return true;
  if (Count >= RefCount)
    return false;
  return BranchProbability::getBranchProbability(Count, RefCount) < ColdProb;
}

bool BBSectionsBlockAlignmentProfile::isEdgeColdRelativeToDest(
    const MachineBasicBlock &Src, const MachineBasicBlock &Dst) const {
  uint64_t Count = EdgeCount(Src, Dst);
  uint64_t DstCount = BlockCount(Dst);
  if (Count == 0)
    return true;
  if (Count >= DstCount)
    return false;
  return BranchProbability::getBranchProbability(Count, DstCount) <= ColdProb;
}

/// Assigns loop alignments to the blocks of \p MF after block layout is
/// finalized, using \p Prof for profile queries, and applies the
/// `-align-all-blocks` and `-align-all-nofallthru-blocks` overrides.
template <typename ProfileT>
static void alignMachineBlocksImpl(MachineFunction &MF,
                                   const MachineLoopInfo &MLI,
                                   const ProfileT &Prof) {
  // Walk through the backedges of the function now that we have fully laid out
  // the basic blocks and align the destination of each backedge. We don't rely
  // exclusively on the loop info here so that we can align backedges in
  // unnatural CFGs and backedges that were introduced purely because of the
  // loop rotations done during layout.
  const TargetLoweringBase *TLI = MF.getSubtarget().getTargetLowering();
  if (!AlignAllBlock && !AlignAllNonFallThruBlocks) {
    if (MF.getFunction().hasMinSize() ||
        (MF.getFunction().hasOptSize() && !TLI->alignLoopsWithOptSize()))
      return;
  }

  if (MF.empty())
    return;

  for (MachineBasicBlock &BB : MF) {
    // The entry block is aligned via the function alignment.
    if (&BB == &MF.front())
      continue;

    // Don't align non-looping basic blocks. These are unlikely to execute
    // enough times to matter in practice. Note that we'll still handle
    // unnatural CFGs inside of a natural outer loop (the common case) and
    // rotated loops.
    MachineLoop *L = MLI.getLoopFor(&BB);
    if (!L)
      continue;

    // Query the block being aligned rather than only the LoopInfo header.
    // After loop rotation, BB can be a different backedge destination.
    const Align TLIAlign = TLI->getPrefLoopAlignment(L, &BB);
    unsigned MDAlign = 1;
    MDNode *LoopID = L->getLoopID();
    if (LoopID) {
      for (const MDOperand &MDO : llvm::drop_begin(LoopID->operands())) {
        MDNode *MD = dyn_cast<MDNode>(MDO);
        if (MD == nullptr)
          continue;
        MDString *S = dyn_cast<MDString>(MD->getOperand(0));
        if (S == nullptr)
          continue;
        if (S->getString() == "llvm.loop.align") {
          MDAlign =
              mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue();
        }
      }
    }

    // Use max of the TLIAlign and MDAlign
    const Align LoopAlign = std::max(TLIAlign, Align(MDAlign));
    // Don't lower alignments already set by other passes (e.g., when alignment
    // is deferred to BasicBlockSections).
    if (LoopAlign <= BB.getAlignment())
      continue;

    // If the block is cold relative to the function entry don't waste space
    // aligning it.
    if (Prof.isColdRelativeTo(BB, MF.front()))
      continue;

    // If the block is cold relative to its loop header, don't align it
    // regardless of what edges into the block exist.
    if (Prof.isColdRelativeTo(BB, *L->getHeader()))
      continue;

    // If the global profiles indicates so, don't align it.
    if (Prof.shouldOptForSize(BB) && !TLI->alignLoopsWithOptSize())
      continue;

    // Check for the existence of a non-layout predecessor which would benefit
    // from aligning this block.
    MachineBasicBlock *LayoutPred = &*std::prev(MachineFunction::iterator(&BB));

    auto DetermineMaxAlignmentPadding = [&]() {
      // Set the maximum bytes allowed to be emitted for alignment.
      unsigned MaxBytes;
      if (MaxBytesForAlignmentOverride.getNumOccurrences() > 0)
        MaxBytes = MaxBytesForAlignmentOverride;
      else
        MaxBytes = TLI->getMaxPermittedBytesForAlignment(&BB);
      BB.setMaxBytesForAlignment(MaxBytes);
    };

    // Force alignment if all the predecessors are jumps. We already checked
    // that the block isn't cold above. A block that begins a section never
    // has a fall-through predecessor, as the linker may reorder sections.
    if (BB.isBeginSection() || !LayoutPred->isSuccessor(&BB)) {
      BB.setAlignment(LoopAlign);
      DetermineMaxAlignmentPadding();
      continue;
    }

    // Align this block if the layout predecessor's edge into this block is
    // cold relative to the block. When this is true, other predecessors make up
    // all of the hot entries into the block and thus alignment is likely to be
    // important.
    if (Prof.isEdgeColdRelativeToDest(*LayoutPred, BB)) {
      BB.setAlignment(LoopAlign);
      DetermineMaxAlignmentPadding();
    }
  }

  const bool HasMaxBytesOverride =
      MaxBytesForAlignmentOverride.getNumOccurrences() > 0;

  if (AlignAllBlock)
    // Align all of the blocks in the function to a specific alignment.
    for (MachineBasicBlock &MBB : MF) {
      if (HasMaxBytesOverride)
        MBB.setAlignment(Align::fromLog2(AlignAllBlock),
                         MaxBytesForAlignmentOverride);
      else
        MBB.setAlignment(Align::fromLog2(AlignAllBlock));
    }
  else if (AlignAllNonFallThruBlocks) {
    // Align all of the blocks that have no fall-through predecessors to a
    // specific alignment. A block that begins a section never has a
    // fall-through predecessor, as the linker may reorder sections.
    for (auto MBI = std::next(MF.begin()), MBE = MF.end(); MBI != MBE; ++MBI) {
      auto LayoutPred = std::prev(MBI);
      if (MBI->isBeginSection() || !LayoutPred->isSuccessor(&*MBI)) {
        if (HasMaxBytesOverride)
          MBI->setAlignment(Align::fromLog2(AlignAllNonFallThruBlocks),
                            MaxBytesForAlignmentOverride);
        else
          MBI->setAlignment(Align::fromLog2(AlignAllNonFallThruBlocks));
      }
    }
  }
}

void llvm::alignMachineBlocks(MachineFunction &MF, const MachineLoopInfo &MLI,
                              const MBFIBlockAlignmentProfile &Prof) {
  alignMachineBlocksImpl(MF, MLI, Prof);
}

void llvm::alignMachineBlocks(MachineFunction &MF, const MachineLoopInfo &MLI,
                              const BBSectionsBlockAlignmentProfile &Prof) {
  alignMachineBlocksImpl(MF, MLI, Prof);
}
