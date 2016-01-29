//===--------- LoopSimplifyCFG.cpp - Loop CFG Simplification Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Loop SimplifyCFG Pass. This pass is responsible for
// basic loop CFG cleanup, primarily to assist other loop passes. If you
// encounter a noncanonical CFG construct that causes another loop pass to
// perform suboptimally, this is the place to fix it up.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "loop-simplifycfg"

namespace {
class LoopSimplifyCFG : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid
  LoopSimplifyCFG() : LoopPass(ID) {
    initializeLoopSimplifyCFGPass(*PassRegistry::getPassRegistry());
  }

  bool runOnLoop(Loop *L, LPPassManager &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();

    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addPreserved<SCEVAAWrapperPass>();
    AU.addPreserved<DependenceAnalysis>();
    AU.addPreservedID(LoopSimplifyID);
    AU.addPreservedID(LCSSAID);
  }
};
}

char LoopSimplifyCFG::ID = 0;
INITIALIZE_PASS_BEGIN(LoopSimplifyCFG, "loop-simplifycfg", "Simplify loop CFG",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopSimplifyCFG, "loop-simplifycfg", "Simplify loop CFG",
                    false, false)

Pass *llvm::createLoopSimplifyCFGPass() { return new LoopSimplifyCFG(); }

static bool simplifyLoopCFG(Loop *L, DominatorTree *DT, LoopInfo *LI) {
  bool Changed = false;
  // Copy blocks into a temporary array to avoid iterator invalidation issues
  // as we remove them.
  SmallVector<WeakVH, 16> Blocks(L->blocks());

  for (auto &Block : Blocks) {
    // Attempt to merge blocks in the trivial case. Don't modify blocks which
    // belong to other loops.
    BasicBlock *Succ = cast_or_null<BasicBlock>(Block);
    if (!Succ)
      continue;

    BasicBlock *Pred = Succ->getSinglePredecessor();
    if (!Pred || !Pred->getSingleSuccessor() || LI->getLoopFor(Pred) != L)
      continue;

    // Pred is going to disappear, so we need to update the loop info.
    if (L->getHeader() == Pred)
      L->moveToHeader(Succ);
    LI->removeBlock(Pred);
    MergeBasicBlockIntoOnlyPred(Succ, DT);
    Changed = true;
  }

  return Changed;
}

/// runOnLoop - Perform basic CFG simplifications to assist other loop passes.
/// For now, this only attempts to merge blocks in the trivial case.
bool LoopSimplifyCFG::runOnLoop(Loop *L, LPPassManager &) {
  if (skipOptnoneFunction(L))
    return false;

  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  return simplifyLoopCFG(L, DT, LI);
}
