//===---- CodePreparation.cpp - Code preparation for Scop Detection -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "polly/LinkAllPasses.h"
#include "polly/ScopDetection.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Local.h"

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
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
  virtual void print(raw_ostream &OS, const Module *) const;
  //@}
};
} // namespace

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
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  splitEntryBlockForAlloca(&F.getEntryBlock(), this);

  return false;
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
