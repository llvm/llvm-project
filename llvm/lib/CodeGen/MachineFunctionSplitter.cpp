//===-- MachineFunctionSplitter.cpp - Split machine functions //-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Uses profile information to split out cold blocks.
//
// This pass splits out cold machine basic blocks from the parent function. This
// implementation leverages the basic block section framework. Blocks marked
// cold by this pass are grouped together in a separate section prefixed with
// ".text.unlikely.*". The linker can then group these together as a cold
// section. The split part of the function is a contiguous region identified by
// the symbol "foo.cold". Grouping all cold blocks across functions together
// decreases fragmentation and improves icache and itlb utilization. Note that
// the overall changes to the binary size are negligible; only a small number of
// additional jump instructions may be introduced.
//
// For the original RFC of this pass please see
// https://groups.google.com/d/msg/llvm-dev/RUegaMg-iqc/wFAVxa6fCgAJ
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/BasicBlockSectionUtils.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// FIXME: This cutoff value is CPU dependent and should be moved to
// TargetTransformInfo once we consider enabling this on other platforms.
// The value is expressed as a ProfileSummaryInfo integer percentile cutoff.
// Defaults to 999950, i.e. all blocks colder than 99.995 percentile are split.
// The default was empirically determined to be optimal when considering cutoff
// values between 99%-ile to 100%-ile with respect to iTLB and icache metrics on
// Intel CPUs.
static cl::opt<unsigned>
    PercentileCutoff("mfs-psi-cutoff",
                     cl::desc("Percentile profile summary cutoff used to "
                              "determine cold blocks. Unused if set to zero."),
                     cl::init(999950), cl::Hidden);

static cl::opt<unsigned> ColdCountThreshold(
    "mfs-count-threshold",
    cl::desc(
        "Minimum number of times a block must be executed to be retained."),
    cl::init(1), cl::Hidden);

namespace {

class MachineFunctionSplitter : public MachineFunctionPass {
public:
  static char ID;
  MachineFunctionSplitter() : MachineFunctionPass(ID) {
    initializeMachineFunctionSplitterPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Machine Function Splitter Transformation";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &F) override;
};
} // end anonymous namespace

static bool isColdBlock(MachineBasicBlock &MBB,
                        const MachineBlockFrequencyInfo *MBFI,
                        ProfileSummaryInfo *PSI) {
  Optional<uint64_t> Count = MBFI->getBlockProfileCount(&MBB);
  if (!Count.hasValue())
    return true;

  if (PercentileCutoff > 0) {
    return PSI->isColdCountNthPercentile(PercentileCutoff, *Count);
  }
  return (*Count < ColdCountThreshold);
}

bool MachineFunctionSplitter::runOnMachineFunction(MachineFunction &MF) {
  // TODO: We only target functions with profile data. Static information may
  // also be considered but we don't see performance improvements yet.
  if (!MF.getFunction().hasProfileData())
    return false;

  // TODO: We don't split functions where a section attribute has been set
  // since the split part may not be placed in a contiguous region. It may also
  // be more beneficial to augment the linker to ensure contiguous layout of
  // split functions within the same section as specified by the attribute.
  if (!MF.getFunction().getSection().empty())
    return false;

  // We don't want to proceed further for cold functions
  // or functions of unknown hotness. Lukewarm functions have no prefix.
  Optional<StringRef> SectionPrefix = MF.getFunction().getSectionPrefix();
  if (SectionPrefix.hasValue() &&
      (SectionPrefix.getValue().equals("unlikely") ||
       SectionPrefix.getValue().equals("unknown"))) {
    return false;
  }

  // Renumbering blocks here preserves the order of the blocks as
  // sortBasicBlocksAndUpdateBranches uses the numeric identifier to sort
  // blocks. Preserving the order of blocks is essential to retaining decisions
  // made by prior passes such as MachineBlockPlacement.
  MF.RenumberBlocks();
  MF.setBBSectionsType(BasicBlockSection::Preset);
  auto *MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  auto *PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

  for (auto &MBB : MF) {
    // FIXME: We retain the entry block and conservatively keep all landing pad
    // blocks as part of the original function. Once D73739 is submitted, we can
    // improve the handling of ehpads.
    if ((MBB.pred_empty() || MBB.isEHPad()))
      continue;
    if (isColdBlock(MBB, MBFI, PSI))
      MBB.setSectionID(MBBSectionID::ColdSectionID);
  }

  auto Comparator = [](const MachineBasicBlock &X, const MachineBasicBlock &Y) {
    return X.getSectionID().Type < Y.getSectionID().Type;
  };
  llvm::sortBasicBlocksAndUpdateBranches(MF, Comparator);

  return true;
}

void MachineFunctionSplitter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineModuleInfoWrapperPass>();
  AU.addRequired<MachineBlockFrequencyInfo>();
  AU.addRequired<ProfileSummaryInfoWrapperPass>();
}

char MachineFunctionSplitter::ID = 0;
INITIALIZE_PASS(MachineFunctionSplitter, "machine-function-splitter",
                "Split machine functions using profile information", false,
                false)

MachineFunctionPass *llvm::createMachineFunctionSplitterPass() {
  return new MachineFunctionSplitter();
}
