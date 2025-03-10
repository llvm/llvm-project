//===- StaticDataAnnotator - Annotate static data's section prefix --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// To reason about module-wide data hotness in a module granularity, this file
// implements a module pass StaticDataAnnotator to work coordinately with the
// StaticDataSplitter pass.
//
// The StaticDataSplitter pass is a machine function pass. It analyzes data
// hotness based on code and adds counters in the StaticDataProfileInfo.
// The StaticDataAnnotator pass is a module pass. It iterates global variables
// in the module, looks up counters from StaticDataProfileInfo and sets the
// section prefix based on profiles.
//
// The three-pass structure is implemented for practical reasons, to work around
// the limitation that a module pass based on legacy pass manager cannot make
// use of MachineBlockFrequencyInfo analysis. In the future, we can consider
// porting the StaticDataSplitter pass to a module-pass using the new pass
// manager framework. That way, analysis are lazily computed as opposed to
// eagerly scheduled, and a module pass can use MachineBlockFrequencyInfo.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "static-data-annotator"

using namespace llvm;

class StaticDataAnnotator : public ModulePass {
public:
  static char ID;

  StaticDataProfileInfo *SDPI = nullptr;
  const ProfileSummaryInfo *PSI = nullptr;

  StaticDataAnnotator() : ModulePass(ID) {
    initializeStaticDataAnnotatorPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<StaticDataProfileInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "Static Data Annotator"; }

  bool runOnModule(Module &M) override;
};

// Returns true if the global variable already has a section prefix that is the
// same as `Prefix`.
static bool alreadyHasSectionPrefix(const GlobalVariable &GV,
                                    StringRef Prefix) {
  std::optional<StringRef> SectionPrefix = GV.getSectionPrefix();
  return SectionPrefix && (*SectionPrefix == Prefix);
}

bool StaticDataAnnotator::runOnModule(Module &M) {
  SDPI = &getAnalysis<StaticDataProfileInfoWrapperPass>()
              .getStaticDataProfileInfo();
  PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

  if (!PSI->hasProfileSummary())
    return false;

  bool Changed = false;
  for (auto &GV : M.globals()) {
    if (GV.isDeclarationForLinker())
      continue;

    // Skip global variables without profile counts. The module may not be
    // profiled or instrumented.
    auto Count = SDPI->getConstantProfileCount(&GV);
    if (!Count)
      continue;

    if (PSI->isHotCount(*Count) && !alreadyHasSectionPrefix(GV, "hot")) {
      // The variable counter is hot, set 'hot' section prefix if the section
      // prefix isn't hot already.
      GV.setSectionPrefix("hot");
      Changed = true;
    } else if (PSI->isColdCount(*Count) && !SDPI->hasUnknownCount(&GV) &&
               !alreadyHasSectionPrefix(GV, "unlikely")) {
      // The variable counter is cold, set 'unlikely' section prefix when
      // 1) the section prefix isn't unlikely already, and
      // 2) the variable is not seen without profile counts. The reason is that
      // a variable without profile counts doesn't have all its uses profiled,
      // for example when a function is not instrumented, or not sampled (new
      // code paths).
      GV.setSectionPrefix("unlikely");
      Changed = true;
    }
  }

  return Changed;
}

char StaticDataAnnotator::ID = 0;

INITIALIZE_PASS(StaticDataAnnotator, DEBUG_TYPE, "Static Data Annotator", false,
                false)

ModulePass *llvm::createStaticDataAnnotatorPass() {
  return new StaticDataAnnotator();
}
