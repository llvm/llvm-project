//===-- LCSSA.cpp - Convert loops into loop-closed SSA form ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms loops by placing phi nodes at the end of the loops for
// all values that are live across the loop boundary.  For example, it turns
// the left into the right code:
// 
// for (...)                for (...)
//   if (c)                   if (c)
//     X1 = ...                 X1 = ...
//   else                     else
//     X2 = ...                 X2 = ...
//   X3 = phi(X1, X2)         X3 = phi(X1, X2)
// ... = X3 + 4             X4 = phi(X3)
//                          ... = X4 + 4
//
// This is still valid LLVM; the extra phi nodes are purely redundant, and will
// be trivially eliminated by InstCombine.  The major benefit of this 
// transformation is that it makes many other loop optimizations, such as 
// LoopUnswitching, simpler.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

#define DEBUG_TYPE "lcssa"

STATISTIC(NumLCSSA, "Number of live out of a loop variables");

/// Return true if the specified block is in the list.
static bool isExitBlock(BasicBlock *BB,
                        const SmallVectorImpl<BasicBlock *> &ExitBlocks) {
  return is_contained(ExitBlocks, BB);
}

/// For every instruction from the worklist, check to see if it has any uses
/// that are outside the current loop.  If so, insert LCSSA PHI nodes and
/// rewrite the uses.
bool llvm::formLCSSAForInstructions(SmallVectorImpl<Instruction *> &Worklist,
                                    DominatorTree &DT, LoopInfo &LI) {
  SmallVector<Use *, 16> UsesToRewrite;
  SmallVector<BasicBlock *, 8> ExitBlocks;
  SmallSetVector<PHINode *, 16> PHIsToRemove;
  PredIteratorCache PredCache;
  bool Changed = false;

  while (!Worklist.empty()) {
    UsesToRewrite.clear();
    ExitBlocks.clear();

    Instruction *I = Worklist.pop_back_val();
    BasicBlock *InstBB = I->getParent();
    Loop *L = LI.getLoopFor(InstBB);
    L->getExitBlocks(ExitBlocks);

    if (ExitBlocks.empty())
      continue;

    // Tokens cannot be used in PHI nodes, so we skip over them.
    // We can run into tokens which are live out of a loop with catchswitch
    // instructions in Windows EH if the catchswitch has one catchpad which
    // is inside the loop and another which is not.
    if (I->getType()->isTokenTy())
      continue;

    for (Use &U : I->uses()) {
      Instruction *User = cast<Instruction>(U.getUser());
      BasicBlock *UserBB = User->getParent();
      if (PHINode *PN = dyn_cast<PHINode>(User))
        UserBB = PN->getIncomingBlock(U);

      if (InstBB != UserBB && !L->contains(UserBB))
        UsesToRewrite.push_back(&U);
    }

    // If there are no uses outside the loop, exit with no change.
    if (UsesToRewrite.empty())
      continue;

    ++NumLCSSA; // We are applying the transformation

    // Invoke instructions are special in that their result value is not
    // available along their unwind edge. The code below tests to see whether
    // DomBB dominates the value, so adjust DomBB to the normal destination
    // block, which is effectively where the value is first usable.
    BasicBlock *DomBB = InstBB;
    if (InvokeInst *Inv = dyn_cast<InvokeInst>(I))
      DomBB = Inv->getNormalDest();

    DomTreeNode *DomNode = DT.getNode(DomBB);

    SmallVector<PHINode *, 16> AddedPHIs;
    SmallVector<PHINode *, 8> PostProcessPHIs;

    SmallVector<PHINode *, 4> InsertedPHIs;
    SSAUpdater SSAUpdate(&InsertedPHIs);
    SSAUpdate.Initialize(I->getType(), I->getName());

    // Insert the LCSSA phi's into all of the exit blocks dominated by the
    // value, and add them to the Phi's map.
    for (BasicBlock *ExitBB : ExitBlocks) {
      if (!DT.dominates(DomNode, DT.getNode(ExitBB)))
        continue;

      // If we already inserted something for this BB, don't reprocess it.
      if (SSAUpdate.HasValueForBlock(ExitBB))
        continue;

      PHINode *PN = PHINode::Create(I->getType(), PredCache.size(ExitBB),
                                    I->getName() + ".lcssa", &ExitBB->front());

      // Add inputs from inside the loop for this PHI.
      for (BasicBlock *Pred : PredCache.get(ExitBB)) {
        PN->addIncoming(I, Pred);

        // If the exit block has a predecessor not within the loop, arrange for
        // the incoming value use corresponding to that predecessor to be
        // rewritten in terms of a different LCSSA PHI.
        if (!L->contains(Pred))
          UsesToRewrite.push_back(
              &PN->getOperandUse(PN->getOperandNumForIncomingValue(
                  PN->getNumIncomingValues() - 1)));
      }

      AddedPHIs.push_back(PN);

      // Remember that this phi makes the value alive in this block.
      SSAUpdate.AddAvailableValue(ExitBB, PN);

      // LoopSimplify might fail to simplify some loops (e.g. when indirect
      // branches are involved). In such situations, it might happen that an
      // exit for Loop L1 is the header of a disjoint Loop L2. Thus, when we
      // create PHIs in such an exit block, we are also inserting PHIs into L2's
      // header. This could break LCSSA form for L2 because these inserted PHIs
      // can also have uses outside of L2. Remember all PHIs in such situation
      // as to revisit than later on. FIXME: Remove this if indirectbr support
      // into LoopSimplify gets improved.
      if (auto *OtherLoop = LI.getLoopFor(ExitBB))
        if (!L->contains(OtherLoop))
          PostProcessPHIs.push_back(PN);
    }

    // Rewrite all uses outside the loop in terms of the new PHIs we just
    // inserted.
    for (Use *UseToRewrite : UsesToRewrite) {
      // If this use is in an exit block, rewrite to use the newly inserted PHI.
      // This is required for correctness because SSAUpdate doesn't handle uses
      // in the same block.  It assumes the PHI we inserted is at the end of the
      // block.
      Instruction *User = cast<Instruction>(UseToRewrite->getUser());
      BasicBlock *UserBB = User->getParent();
      if (PHINode *PN = dyn_cast<PHINode>(User))
        UserBB = PN->getIncomingBlock(*UseToRewrite);

      if (isa<PHINode>(UserBB->begin()) && isExitBlock(UserBB, ExitBlocks)) {
        // Tell the VHs that the uses changed. This updates SCEV's caches.
        if (UseToRewrite->get()->hasValueHandle())
          ValueHandleBase::ValueIsRAUWd(*UseToRewrite, &UserBB->front());
        UseToRewrite->set(&UserBB->front());
        continue;
      }

      // Otherwise, do full PHI insertion.
      SSAUpdate.RewriteUse(*UseToRewrite);
    }

    // SSAUpdater might have inserted phi-nodes inside other loops. We'll need
    // to post-process them to keep LCSSA form.
    for (PHINode *InsertedPN : InsertedPHIs) {
      if (auto *OtherLoop = LI.getLoopFor(InsertedPN->getParent()))
        if (!L->contains(OtherLoop))
          PostProcessPHIs.push_back(InsertedPN);
    }

    // Post process PHI instructions that were inserted into another disjoint
    // loop and update their exits properly.
    for (auto *PostProcessPN : PostProcessPHIs) {
      if (PostProcessPN->use_empty())
        continue;

      // Reprocess each PHI instruction.
      Worklist.push_back(PostProcessPN);
    }

    // Keep track of PHI nodes that we want to remove because they did not have
    // any uses rewritten.
    for (PHINode *PN : AddedPHIs)
      if (PN->use_empty())
        PHIsToRemove.insert(PN);

    Changed = true;
  }
  // Remove PHI nodes that did not have any uses rewritten.
  for (PHINode *PN : PHIsToRemove) {
    assert (PN->use_empty() && "Trying to remove a phi with uses.");
    PN->eraseFromParent();
  }
  return Changed;
}

/// Return true if the specified block dominates at least
/// one of the blocks in the specified list.
static bool
blockDominatesAnExit(BasicBlock *BB,
                     DominatorTree &DT,
                     const SmallVectorImpl<BasicBlock *> &ExitBlocks) {
  DomTreeNode *DomNode = DT.getNode(BB);
  return any_of(ExitBlocks, [&](BasicBlock *EB) {
    return DT.dominates(DomNode, DT.getNode(EB));
  });
}

bool llvm::formLCSSA(Loop &L, DominatorTree &DT, LoopInfo *LI,
                     ScalarEvolution *SE) {
  bool Changed = false;

  // Get the set of exiting blocks.
  SmallVector<BasicBlock *, 8> ExitBlocks;
  L.getExitBlocks(ExitBlocks);

  if (ExitBlocks.empty())
    return false;

  SmallVector<Instruction *, 8> Worklist;

  // Look at all the instructions in the loop, checking to see if they have uses
  // outside the loop.  If so, put them into the worklist to rewrite those uses.
  for (BasicBlock *BB : L.blocks()) {
    // For large loops, avoid use-scanning by using dominance information:  In
    // particular, if a block does not dominate any of the loop exits, then none
    // of the values defined in the block could be used outside the loop.
    if (!blockDominatesAnExit(BB, DT, ExitBlocks))
      continue;

    for (Instruction &I : *BB) {
      // Reject two common cases fast: instructions with no uses (like stores)
      // and instructions with one use that is in the same block as this.
      if (I.use_empty() ||
          (I.hasOneUse() && I.user_back()->getParent() == BB &&
           !isa<PHINode>(I.user_back())))
        continue;

      Worklist.push_back(&I);
    }
  }
  Changed = formLCSSAForInstructions(Worklist, DT, *LI);

  // If we modified the code, remove any caches about the loop from SCEV to
  // avoid dangling entries.
  // FIXME: This is a big hammer, can we clear the cache more selectively?
  if (SE && Changed)
    SE->forgetLoop(&L);

  assert(L.isLCSSAForm(DT));

  return Changed;
}

/// Process a loop nest depth first.
bool llvm::formLCSSARecursively(Loop &L, DominatorTree &DT, LoopInfo *LI,
                                ScalarEvolution *SE) {
  bool Changed = false;

  // Recurse depth-first through inner loops.
  for (Loop *SubLoop : L.getSubLoops())
    Changed |= formLCSSARecursively(*SubLoop, DT, LI, SE);

  Changed |= formLCSSA(L, DT, LI, SE);
  return Changed;
}

/// Process all loops in the function, inner-most out.
static bool formLCSSAOnAllLoops(LoopInfo *LI, DominatorTree &DT,
                                ScalarEvolution *SE) {
  bool Changed = false;
  for (auto &L : *LI)
    Changed |= formLCSSARecursively(*L, DT, LI, SE);
  return Changed;
}

namespace {
struct LCSSAWrapperPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  LCSSAWrapperPass() : FunctionPass(ID) {
    initializeLCSSAWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  // Cached analysis information for the current function.
  DominatorTree *DT;
  LoopInfo *LI;
  ScalarEvolution *SE;

  bool runOnFunction(Function &F) override;
  void verifyAnalysis() const override {
    assert(
        all_of(*LI, [&](Loop *L) { return L->isRecursivelyLCSSAForm(*DT); }) &&
        "LCSSA form is broken!");
  };

  /// This transformation requires natural loop information & requires that
  /// loop preheaders be inserted into the CFG.  It maintains both of these,
  /// as well as the CFG.  It also requires dominator information.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();

    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreservedID(LoopSimplifyID);
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addPreserved<SCEVAAWrapperPass>();
  }
};
}

char LCSSAWrapperPass::ID = 0;
INITIALIZE_PASS_BEGIN(LCSSAWrapperPass, "lcssa", "Loop-Closed SSA Form Pass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LCSSAWrapperPass, "lcssa", "Loop-Closed SSA Form Pass",
                    false, false)

Pass *llvm::createLCSSAPass() { return new LCSSAWrapperPass(); }
char &llvm::LCSSAID = LCSSAWrapperPass::ID;

/// Transform \p F into loop-closed SSA form.
bool LCSSAWrapperPass::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto *SEWP = getAnalysisIfAvailable<ScalarEvolutionWrapperPass>();
  SE = SEWP ? &SEWP->getSE() : nullptr;

  return formLCSSAOnAllLoops(LI, *DT, SE);
}

PreservedAnalyses LCSSAPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto *SE = AM.getCachedResult<ScalarEvolutionAnalysis>(F);
  if (!formLCSSAOnAllLoops(&LI, DT, SE))
    return PreservedAnalyses::all();

  // FIXME: This should also 'preserve the CFG'.
  PreservedAnalyses PA;
  PA.preserve<BasicAA>();
  PA.preserve<GlobalsAA>();
  PA.preserve<SCEVAA>();
  PA.preserve<ScalarEvolutionAnalysis>();
  return PA;
}
