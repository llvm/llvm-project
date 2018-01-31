//===- AggressiveInstCombine.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the aggressive expression pattern combiner classes.
// Currently, it handles expression patterns for:
//  * Truncate instruction
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "AggressiveInstCombineInternal.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "aggressive-instcombine"

namespace {
/// Contains expression pattern combiner logic.
/// This class provides both the logic to combine expression patterns and
/// combine them. It differs from InstCombiner class in that each pattern
/// combiner runs only once as opposed to InstCombine's multi-iteration,
/// which allows pattern combiner to have higher complexity than the O(1)
/// required by the instruction combiner.
class AggressiveInstCombinerLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  AggressiveInstCombinerLegacyPass() : FunctionPass(ID) {
    initializeAggressiveInstCombinerLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Run all expression pattern optimizations on the given /p F function.
  ///
  /// \param F function to optimize.
  /// \returns true if the IR is changed.
  bool runOnFunction(Function &F) override;
};
} // namespace

void AggressiveInstCombinerLegacyPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
}

bool AggressiveInstCombinerLegacyPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  auto &DL = F.getParent()->getDataLayout();

  bool MadeIRChange = false;

  // Handle TruncInst patterns
  TruncInstCombine TIC(TLI, DL, DT);
  MadeIRChange |= TIC.run(F);

  // TODO: add more patterns to handle...

  return MadeIRChange;
}

PreservedAnalyses AggressiveInstCombinePass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &DL = F.getParent()->getDataLayout();
  bool MadeIRChange = false;

  // Handle TruncInst patterns
  TruncInstCombine TIC(TLI, DL, DT);
  MadeIRChange |= TIC.run(F);
  if (!MadeIRChange)
    // No changes, all analyses are preserved.
    return PreservedAnalyses::all();

  // Mark all the analyses that instcombine updates as preserved.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<AAManager>();
  PA.preserve<GlobalsAA>();
  return PA;
}

char AggressiveInstCombinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(AggressiveInstCombinerLegacyPass,
                      "aggressive-instcombine",
                      "Combine pattern based expressions", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AggressiveInstCombinerLegacyPass, "aggressive-instcombine",
                    "Combine pattern based expressions", false, false)

FunctionPass *llvm::createAggressiveInstCombinerPass() {
  return new AggressiveInstCombinerLegacyPass();
}
