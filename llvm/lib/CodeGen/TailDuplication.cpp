//===- TailDuplication.cpp - Duplicate blocks into predecessors' tails ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass duplicates basic blocks ending in unconditional branches
/// into the tails of their predecessors, using the TailDuplicator utility
/// class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TailDuplication.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/LazyMachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MBFIWrapper.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TailDuplicator.h"
#include "llvm/IR/Analysis.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "tailduplication"

namespace {

class TailDuplicateBaseLegacy : public MachineFunctionPass {
  TailDuplicator Duplicator;
  std::unique_ptr<MBFIWrapper> MBFIW;
  bool PreRegAlloc;
public:
  TailDuplicateBaseLegacy(char &PassID, bool PreRegAlloc)
      : MachineFunctionPass(PassID), PreRegAlloc(PreRegAlloc) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
    AU.addRequired<LazyMachineBlockFrequencyInfoPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class TailDuplicateLegacy : public TailDuplicateBaseLegacy {
public:
  static char ID;
  TailDuplicateLegacy() : TailDuplicateBaseLegacy(ID, false) {
    initializeTailDuplicateLegacyPass(*PassRegistry::getPassRegistry());
  }
};

class EarlyTailDuplicateLegacy : public TailDuplicateBaseLegacy {
public:
  static char ID;
  EarlyTailDuplicateLegacy() : TailDuplicateBaseLegacy(ID, true) {
    initializeEarlyTailDuplicateLegacyPass(*PassRegistry::getPassRegistry());
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties()
      .set(MachineFunctionProperties::Property::NoPHIs);
  }
};

} // end anonymous namespace

char TailDuplicateLegacy::ID;
char EarlyTailDuplicateLegacy::ID;

char &llvm::TailDuplicateLegacyID = TailDuplicateLegacy::ID;
char &llvm::EarlyTailDuplicateLegacyID = EarlyTailDuplicateLegacy::ID;

INITIALIZE_PASS(TailDuplicateLegacy, DEBUG_TYPE, "Tail Duplication", false,
                false)
INITIALIZE_PASS(EarlyTailDuplicateLegacy, "early-tailduplication",
                "Early Tail Duplication", false, false)

bool TailDuplicateBaseLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto MBPI = &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
  auto *PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  auto *MBFI = (PSI && PSI->hasProfileSummary()) ?
               &getAnalysis<LazyMachineBlockFrequencyInfoPass>().getBFI() :
               nullptr;
  if (MBFI)
    MBFIW = std::make_unique<MBFIWrapper>(*MBFI);
  Duplicator.initMF(MF, PreRegAlloc, MBPI, MBFI ? MBFIW.get() : nullptr, PSI,
                    /*LayoutMode=*/false);

  bool MadeChange = false;
  while (Duplicator.tailDuplicateBlocks())
    MadeChange = true;

  return MadeChange;
}

template <typename DerivedT, bool PreRegAlloc>
PreservedAnalyses TailDuplicatePassBase<DerivedT, PreRegAlloc>::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(static_cast<DerivedT &>(*this), MF);

  auto *MBPI = &MFAM.getResult<MachineBranchProbabilityAnalysis>(MF);
  auto *PSI = MFAM.getResult<ModuleAnalysisManagerMachineFunctionProxy>(MF)
                  .getCachedResult<ProfileSummaryAnalysis>(
                      *MF.getFunction().getParent());
  auto *MBFI = (PSI && PSI->hasProfileSummary()
                    ? &MFAM.getResult<MachineBlockFrequencyAnalysis>(MF)
                    : nullptr);
  if (MBFI)
    MBFIW = std::make_unique<MBFIWrapper>(*MBFI);

  TailDuplicator Duplicator;
  Duplicator.initMF(MF, PreRegAlloc, MBPI, MBFI ? MBFIW.get() : nullptr, PSI,
                    /*LayoutMode=*/false);
  bool MadeChange = false;
  while (Duplicator.tailDuplicateBlocks())
    MadeChange = true;

  if (!MadeChange)
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses();
}

template class llvm::TailDuplicatePassBase<TailDuplicatePass, false>;
template class llvm::TailDuplicatePassBase<EarlyTailDuplicatePass, true>;
