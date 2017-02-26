//===- LoopDeletion.cpp - Dead Loop Deletion Pass ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dead Loop Deletion Pass. This pass is responsible
// for eliminating loops with non-infinite computable trip counts that have no
// side effects or volatile instructions, and do not contribute to the
// computation of the function's return value.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
using namespace llvm;

#define DEBUG_TYPE "loop-delete"

STATISTIC(NumDeleted, "Number of loops deleted");

/// Determines if a loop is dead.
///
/// This assumes that we've already checked for unique exit and exiting blocks,
/// and that the code is in LCSSA form.
static bool isLoopDead(Loop *L, ScalarEvolution &SE,
                       SmallVectorImpl<BasicBlock *> &ExitingBlocks,
                       BasicBlock *ExitBlock, bool &Changed,
                       BasicBlock *Preheader) {
  // Make sure that all PHI entries coming from the loop are loop invariant.
  // Because the code is in LCSSA form, any values used outside of the loop
  // must pass through a PHI in the exit block, meaning that this check is
  // sufficient to guarantee that no loop-variant values are used outside
  // of the loop.
  BasicBlock::iterator BI = ExitBlock->begin();
  bool AllEntriesInvariant = true;
  bool AllOutgoingValuesSame = true;
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    Value *incoming = P->getIncomingValueForBlock(ExitingBlocks[0]);

    // Make sure all exiting blocks produce the same incoming value for the exit
    // block.  If there are different incoming values for different exiting
    // blocks, then it is impossible to statically determine which value should
    // be used.
    AllOutgoingValuesSame =
        all_of(makeArrayRef(ExitingBlocks).slice(1), [&](BasicBlock *BB) {
          return incoming == P->getIncomingValueForBlock(BB);
        });

    if (!AllOutgoingValuesSame)
      break;

    if (Instruction *I = dyn_cast<Instruction>(incoming))
      if (!L->makeLoopInvariant(I, Changed, Preheader->getTerminator())) {
        AllEntriesInvariant = false;
        break;
      }

    ++BI;
  }

  if (Changed)
    SE.forgetLoopDispositions(L);

  if (!AllEntriesInvariant || !AllOutgoingValuesSame)
    return false;

  // Make sure that no instructions in the block have potential side-effects.
  // This includes instructions that could write to memory, and loads that are
  // marked volatile.
  for (auto &I : L->blocks())
    if (any_of(*I, [](Instruction &I) { return I.mayHaveSideEffects(); }))
      return false;
  return true;
}

/// Remove a loop if it is dead.
///
/// A loop is considered dead if it does not impact the observable behavior of
/// the program other than finite running time. This never removes a loop that
/// might be infinite, as doing so could change the halting/non-halting nature
/// of a program.
///
/// This entire process relies pretty heavily on LoopSimplify form and LCSSA in
/// order to make various safety checks work.
///
/// \returns true if any changes were made. This may mutate the loop even if it
/// is unable to delete it due to hoisting trivially loop invariant
/// instructions out of the loop.
///
/// This also updates the relevant analysis information in \p DT, \p SE, and \p
/// LI. It also updates the loop PM if an updater struct is provided.
static bool deleteLoopIfDead(Loop *L, DominatorTree &DT, ScalarEvolution &SE,
                             LoopInfo &LI, LPMUpdater *Updater = nullptr) {
  assert(L->isLCSSAForm(DT) && "Expected LCSSA!");

  // We can only remove the loop if there is a preheader that we can
  // branch from after removing it.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader)
    return false;

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->hasDedicatedExits())
    return false;

  // We can't remove loops that contain subloops.  If the subloops were dead,
  // they would already have been removed in earlier executions of this pass.
  if (L->begin() != L->end())
    return false;

  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  // We require that the loop only have a single exit block.  Otherwise, we'd
  // be in the situation of needing to be able to solve statically which exit
  // block will be branched to, or trying to preserve the branching logic in
  // a loop invariant manner.
  BasicBlock *ExitBlock = L->getUniqueExitBlock();
  if (!ExitBlock)
    return false;

  // Finally, we have to check that the loop really is dead.
  bool Changed = false;
  if (!isLoopDead(L, SE, ExitingBlocks, ExitBlock, Changed, Preheader))
    return Changed;

  // Don't remove loops for which we can't solve the trip count.
  // They could be infinite, in which case we'd be changing program behavior.
  const SCEV *S = SE.getMaxBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(S))
    return Changed;

  // Now that we know the removal is safe, remove the loop by changing the
  // branch from the preheader to go to the single exit block.
  //
  // Because we're deleting a large chunk of code at once, the sequence in which
  // we remove things is very important to avoid invalidation issues.

  // If we have an LPM updater, tell it about the loop being removed.
  if (Updater)
    Updater->markLoopAsDeleted(*L);

  // Tell ScalarEvolution that the loop is deleted. Do this before
  // deleting the loop so that ScalarEvolution can look at the loop
  // to determine what it needs to clean up.
  SE.forgetLoop(L);

  // Connect the preheader directly to the exit block.
  TerminatorInst *TI = Preheader->getTerminator();
  TI->replaceUsesOfWith(L->getHeader(), ExitBlock);

  // Rewrite phis in the exit block to get their inputs from
  // the preheader instead of the exiting block.
  BasicBlock *ExitingBlock = ExitingBlocks[0];
  BasicBlock::iterator BI = ExitBlock->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    int j = P->getBasicBlockIndex(ExitingBlock);
    assert(j >= 0 && "Can't find exiting block in exit block's phi node!");
    P->setIncomingBlock(j, Preheader);
    for (unsigned i = 1; i < ExitingBlocks.size(); ++i)
      P->removeIncomingValue(ExitingBlocks[i]);
    ++BI;
  }

  // Update the dominator tree and remove the instructions and blocks that will
  // be deleted from the reference counting scheme.
  SmallVector<DomTreeNode*, 8> ChildNodes;
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    // Move all of the block's children to be children of the Preheader, which
    // allows us to remove the domtree entry for the block.
    ChildNodes.insert(ChildNodes.begin(), DT[*LI]->begin(), DT[*LI]->end());
    for (DomTreeNode *ChildNode : ChildNodes) {
      DT.changeImmediateDominator(ChildNode, DT[Preheader]);
    }

    ChildNodes.clear();
    DT.eraseNode(*LI);

    // Remove the block from the reference counting scheme, so that we can
    // delete it freely later.
    (*LI)->dropAllReferences();
  }

  // Erase the instructions and the blocks without having to worry
  // about ordering because we already dropped the references.
  // NOTE: This iteration is safe because erasing the block does not remove its
  // entry from the loop's block list.  We do that in the next section.
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI)
    (*LI)->eraseFromParent();

  // Finally, the blocks from loopinfo.  This has to happen late because
  // otherwise our loop iterators won't work.

  SmallPtrSet<BasicBlock *, 8> blocks;
  blocks.insert(L->block_begin(), L->block_end());
  for (BasicBlock *BB : blocks)
    LI.removeBlock(BB);

  // The last step is to update LoopInfo now that we've eliminated this loop.
  LI.markAsRemoved(L);
  ++NumDeleted;

  return true;
}

PreservedAnalyses LoopDeletionPass::run(Loop &L, LoopAnalysisManager &AM,
                                        LoopStandardAnalysisResults &AR,
                                        LPMUpdater &Updater) {
  if (!deleteLoopIfDead(&L, AR.DT, AR.SE, AR.LI, &Updater))
    return PreservedAnalyses::all();

  return getLoopPassPreservedAnalyses();
}

namespace {
class LoopDeletionLegacyPass : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid
  LoopDeletionLegacyPass() : LoopPass(ID) {
    initializeLoopDeletionLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // Possibly eliminate loop L if it is dead.
  bool runOnLoop(Loop *L, LPPassManager &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    getLoopAnalysisUsage(AU);
  }
};
}

char LoopDeletionLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LoopDeletionLegacyPass, "loop-deletion",
                      "Delete dead loops", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_END(LoopDeletionLegacyPass, "loop-deletion",
                    "Delete dead loops", false, false)

Pass *llvm::createLoopDeletionPass() { return new LoopDeletionLegacyPass(); }

bool LoopDeletionLegacyPass::runOnLoop(Loop *L, LPPassManager &) {
  if (skipLoop(L))
    return false;

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  return deleteLoopIfDead(L, DT, SE, LI);
}
