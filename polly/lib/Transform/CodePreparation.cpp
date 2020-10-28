//===---- CodePreparation.cpp - Code preparation for Scop Detection -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Polly code preparation pass is executed before SCoP detection. Its
// currently only splits the entry block of the SCoP to make room for alloc
// instructions as they are generated during code generation.
//
// XXX: In the future, we should remove the need for this pass entirely and
// instead add this spitting to the code generation pass.
//
//===----------------------------------------------------------------------===//

#include "polly/CodePreparation.h"
#include "polly/LinkAllPasses.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/InitializePasses.h"

using namespace llvm;
using namespace polly;

namespace {

/// Prepare the IR for the scop detection.
///
class CodePreparation : public FunctionPass {
  CodePreparation(const CodePreparation &) = delete;
  const CodePreparation &operator=(const CodePreparation &) = delete;

  LoopInfo *LI;
  ScalarEvolution *SE;

  void clear();

public:
  static char ID;

  explicit CodePreparation() : FunctionPass(ID) {}
  ~CodePreparation();

  /// @name FunctionPass interface.
  //@{
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  bool runOnFunction(Function &F) override;
  void print(raw_ostream &OS, const Module *) const override;
  //@}
};
} // namespace

PreservedAnalyses CodePreparationPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {

  // Find first non-alloca instruction. Every basic block has a non-alloca
  // instruction, as every well formed basic block has a terminator.
  auto &EntryBlock = F.getEntryBlock();
  BasicBlock::iterator I = EntryBlock.begin();
  while (isa<AllocaInst>(I))
    ++I;

  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = FAM.getResult<LoopAnalysis>(F);

  // splitBlock updates DT, LI and RI.
  splitEntryBlockForAlloca(&EntryBlock, &DT, &LI, nullptr);

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}

void CodePreparation::clear() {}

CodePreparation::~CodePreparation() { clear(); }

void CodePreparation::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();

  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addPreserved<RegionInfoPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<DominanceFrontierWrapperPass>();
}

bool CodePreparation::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  splitEntryBlockForAlloca(&F.getEntryBlock(), this);

  return true;
}

void CodePreparation::releaseMemory() { clear(); }

void CodePreparation::print(raw_ostream &OS, const Module *) const {}

char CodePreparation::ID = 0;
char &polly::CodePreparationID = CodePreparation::ID;

Pass *polly::createCodePreparationPass() { return new CodePreparation(); }

INITIALIZE_PASS_BEGIN(CodePreparation, "polly-prepare",
                      "Polly - Prepare code for polly", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(CodePreparation, "polly-prepare",
                    "Polly - Prepare code for polly", false, false)
