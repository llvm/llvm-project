//===- StaticDataSplitter.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass uses profile information to partition static data sections into
// hot and cold ones. It begins to split jump tables based on profile, and
// subsequent patches will handle constant pools and other module internal data.
//
// For the original RFC of this pass please see
// https://discourse.llvm.org/t/rfc-profile-guided-static-data-partitioning/83744.

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/MBFIWrapper.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "static-data-splitter"

STATISTIC(NumHotJumpTables, "Number of hot jump tables seen");
STATISTIC(NumColdJumpTables, "Number of cold jump tables seen");
STATISTIC(NumUnknownJumpTables,
          "Number of jump tables with unknown hotness. Such jump tables will "
          "be placed in the hot-suffixed section by default.");

class StaticDataSplitter : public MachineFunctionPass {
  const MachineBranchProbabilityInfo *MBPI = nullptr;
  const MachineBlockFrequencyInfo *MBFI = nullptr;
  const ProfileSummaryInfo *PSI = nullptr;

  // Returns true iff any jump table is hot-cold categorized.
  bool splitJumpTables(MachineFunction &MF);

  // Same as above but works on functions with profile information.
  bool splitJumpTablesWithProfiles(MachineFunction &MF,
                                   MachineJumpTableInfo &MJTI);

public:
  static char ID;

  StaticDataSplitter() : MachineFunctionPass(ID) {
    initializeStaticDataSplitterPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Static Data Splitter"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

bool StaticDataSplitter::runOnMachineFunction(MachineFunction &MF) {
  MBPI = &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
  MBFI = &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

  // Split jump tables based on profile information. Subsequent patches will
  // handle other data types like constant pools, module-internal data, etc.
  return splitJumpTables(MF);
}

bool StaticDataSplitter::splitJumpTablesWithProfiles(
    MachineFunction &MF, MachineJumpTableInfo &MJTI) {
  int NumChangedJumpTables = 0;
  // Regard a jump table as hot by default. If the source and all of destination
  // blocks are cold, regard the jump table as cold.
  DataHotness Hotness = DataHotness::Hot;
  for (const auto &MBB : MF) {
    // IMPORTANT, `getJumpTableIndex` is a thin wrapper around per-target
    // interface `TargetInstrInfo::getjumpTableIndex`, and only X86 implements
    // it so far.
    const int JTI = MBB.getJumpTableIndex();
    // This is not a source block of jump table.
    if (JTI == -1)
      continue;

    bool AllBlocksCold = true;

    if (!PSI->isColdBlock(&MBB, MBFI))
      AllBlocksCold = false;

    for (const MachineBasicBlock *MBB : MJTI.getJumpTables()[JTI].MBBs)
      if (!PSI->isColdBlock(MBB, MBFI))
        AllBlocksCold = false;

    if (AllBlocksCold) {
      Hotness = DataHotness::Cold;
      ++NumColdJumpTables;
    } else {
      ++NumHotJumpTables;
    }

    MF.getJumpTableInfo()->updateJumpTableHotness(JTI, Hotness);
    ++NumChangedJumpTables;
  }
  return NumChangedJumpTables > 0;
}

bool StaticDataSplitter::splitJumpTables(MachineFunction &MF) {
  MachineJumpTableInfo *MJTI = MF.getJumpTableInfo();
  if (!MJTI || MJTI->getJumpTables().empty())
    return false;

  // Place jump tables according to block hotness if block counters are
  // available. Check function entry count because BFI depends on it to derive
  // block counters.
  if (PSI && PSI->hasProfileSummary() && MBFI &&
      MF.getFunction().getEntryCount())
    return splitJumpTablesWithProfiles(MF, *MJTI);

  // Conservatively place all jump tables in the hot-suffixed section if profile
  // information for the function is not available, or the target doesn't
  // implement `TargetInstrInfo::getJumpTableIndex` yet.
  for (size_t JTI = 0; JTI < MJTI->getJumpTables().size(); JTI++)
    MF.getJumpTableInfo()->updateJumpTableHotness(JTI, DataHotness::Hot);

  NumUnknownJumpTables += MJTI->getJumpTables().size();
  return true;
}

char StaticDataSplitter::ID = 0;

INITIALIZE_PASS_BEGIN(StaticDataSplitter, DEBUG_TYPE, "Split static data",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(StaticDataSplitter, DEBUG_TYPE, "Split static data", false,
                    false)

MachineFunctionPass *llvm::createStaticDataSplitterPass() {
  return new StaticDataSplitter();
}
