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
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
using namespace llvm;

#define DEBUG_TYPE "loop-delete"

STATISTIC(NumDeleted, "Number of loops deleted");

/// This function deletes dead loops. The caller of this function needs to
/// guarantee that the loop is infact dead. Here we handle two kinds of dead
/// loop. The first kind (\p isLoopDead) is where only invariant values from
/// within the loop are used outside of it. The second kind (\p
/// isLoopNeverExecuted) is where the loop is provably never executed. We can
/// always remove never executed loops since they will not cause any difference
/// to program behaviour.
/// 
/// This also updates the relevant analysis information in \p DT, \p SE, and \p
/// LI. It also updates the loop PM if an updater struct is provided.
// TODO: This function will be used by loop-simplifyCFG as well. So, move this
// to LoopUtils.cpp
static void deleteDeadLoop(Loop *L, DominatorTree &DT, ScalarEvolution &SE,
                           LoopInfo &LI, LPMUpdater *Updater = nullptr);
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

/// This function returns true if there is no viable path from the
/// entry block to the header of \p L. Right now, it only does
/// a local search to save compile time.
static bool isLoopNeverExecuted(Loop *L) {
  using namespace PatternMatch;

  auto *Preheader = L->getLoopPreheader();
  // TODO: We can relax this constraint, since we just need a loop
  // predecessor.
  assert(Preheader && "Needs preheader!");

  if (Preheader == &Preheader->getParent()->getEntryBlock())
    return false;
  // All predecessors of the preheader should have a constant conditional
  // branch, with the loop's preheader as not-taken.
  for (auto *Pred: predecessors(Preheader)) {
    BasicBlock *Taken, *NotTaken;
    ConstantInt *Cond;
    if (!match(Pred->getTerminator(),
               m_Br(m_ConstantInt(Cond), Taken, NotTaken)))
      return false;
    if (!Cond->getZExtValue())
      std::swap(Taken, NotTaken);
    if (Taken == Preheader)
      return false;
  }
  assert(!pred_empty(Preheader) &&
         "Preheader should have predecessors at this point!");
  // All the predecessors have the loop preheader as not-taken target.
  return true;
}

/// Remove a loop if it is dead.
///
/// A loop is considered dead if it does not impact the observable behavior of
/// the program other than finite running time. This never removes a loop that
/// might be infinite (unless it is never executed), as doing so could change
/// the halting/non-halting nature of a program.
///
/// This entire process relies pretty heavily on LoopSimplify form and LCSSA in
/// order to make various safety checks work.
///
/// \returns true if any changes were made. This may mutate the loop even if it
/// is unable to delete it due to hoisting trivially loop invariant
/// instructions out of the loop.
static bool deleteLoopIfDead(Loop *L, DominatorTree &DT, ScalarEvolution &SE,
                             LoopInfo &LI, LPMUpdater *Updater = nullptr) {
  assert(L->isLCSSAForm(DT) && "Expected LCSSA!");

  // We can only remove the loop if there is a preheader that we can branch from
  // after removing it. Also, if LoopSimplify form is not available, stay out
  // of trouble.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader || !L->hasDedicatedExits()) {
    DEBUG(dbgs()
          << "Deletion requires Loop with preheader and dedicated exits.\n");
    return false;
  }
  // We can't remove loops that contain subloops.  If the subloops were dead,
  // they would already have been removed in earlier executions of this pass.
  if (L->begin() != L->end()) {
    DEBUG(dbgs() << "Loop contains subloops.\n");
    return false;
  }


  BasicBlock *ExitBlock = L->getUniqueExitBlock();

  if (ExitBlock && isLoopNeverExecuted(L)) {
    DEBUG(dbgs() << "Loop is proven to never execute, delete it!");
    // Set incoming value to undef for phi nodes in the exit block.
    BasicBlock::iterator BI = ExitBlock->begin();
    while (PHINode *P = dyn_cast<PHINode>(BI)) {
      for (unsigned i = 0; i < P->getNumIncomingValues(); i++)
        P->setIncomingValue(i, UndefValue::get(P->getType()));
      BI++;
    }
    deleteDeadLoop(L, DT, SE, LI, Updater);
    ++NumDeleted;
    return true;
  }

  // The remaining checks below are for a loop being dead because all statements
  // in the loop are invariant.
  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  // We require that the loop only have a single exit block.  Otherwise, we'd
  // be in the situation of needing to be able to solve statically which exit
  // block will be branched to, or trying to preserve the branching logic in
  // a loop invariant manner.
  if (!ExitBlock) {
    DEBUG(dbgs() << "Deletion requires single exit block\n");
    return false;
  }
  // Finally, we have to check that the loop really is dead.
  bool Changed = false;
  if (!isLoopDead(L, SE, ExitingBlocks, ExitBlock, Changed, Preheader)) {
    DEBUG(dbgs() << "Loop is not invariant, cannot delete.\n");
    return Changed;
  }

  // Don't remove loops for which we can't solve the trip count.
  // They could be infinite, in which case we'd be changing program behavior.
  const SCEV *S = SE.getMaxBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(S)) {
    DEBUG(dbgs() << "Could not compute SCEV MaxBackedgeTakenCount.\n");
    return Changed;
  }

  DEBUG(dbgs() << "Loop is invariant, delete it!");
  deleteDeadLoop(L, DT, SE, LI, Updater);
  ++NumDeleted;

  return true;
}

static void deleteDeadLoop(Loop *L, DominatorTree &DT, ScalarEvolution &SE,
                           LoopInfo &LI, LPMUpdater *Updater) {
  assert(L->isLCSSAForm(DT) && "Expected LCSSA!");
  auto *Preheader = L->getLoopPreheader();
  assert(Preheader && "Preheader should exist!");

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

  auto *ExitBlock = L->getUniqueExitBlock();
  assert(ExitBlock && "Should have a unique exit block!");

  assert(L->hasDedicatedExits() && "Loop should have dedicated exits!");

  // Connect the preheader directly to the exit block.
  // Even when the loop is never executed, we cannot remove the edge from the
  // source block to the exit block. Consider the case where the unexecuted loop
  // branches back to an outer loop. If we deleted the loop and removed the edge
  // coming to this inner loop, this will break the outer loop structure (by
  // deleting the backedge of the outer loop). If the outer loop is indeed a
  // non-loop, it will be deleted in a future iteration of loop deletion pass.
  Preheader->getTerminator()->replaceUsesOfWith(L->getHeader(), ExitBlock);

  // Rewrite phis in the exit block to get their inputs from the Preheader
  // instead of the exiting block.
  BasicBlock::iterator BI = ExitBlock->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    // Set the zero'th element of Phi to be from the preheader and remove all
    // other incoming values. Given the loop has dedicated exits, all other
    // incoming values must be from the exiting blocks.
    int PredIndex = 0;
    P->setIncomingBlock(PredIndex, Preheader);
    // Removes all incoming values from all other exiting blocks (including
    // duplicate values from an exiting block).
    // Nuke all entries except the zero'th entry which is the preheader entry.
    // NOTE! We need to remove Incoming Values in the reverse order as done
    // below, to keep the indices valid for deletion (removeIncomingValues
    // updates getNumIncomingValues and shifts all values down into the operand
    // being deleted).
    for (unsigned i = 0, e = P->getNumIncomingValues() - 1; i != e; ++i)
      P->removeIncomingValue(e-i, false);

    assert((P->getNumIncomingValues() == 1 &&
            P->getIncomingBlock(PredIndex) == Preheader) &&
           "Should have exactly one value and that's from the preheader!");
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
}

PreservedAnalyses LoopDeletionPass::run(Loop &L, LoopAnalysisManager &AM,
                                        LoopStandardAnalysisResults &AR,
                                        LPMUpdater &Updater) {

  DEBUG(dbgs() << "Analyzing Loop for deletion: ");
  DEBUG(L.dump());
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

  DEBUG(dbgs() << "Analyzing Loop for deletion: ");
  DEBUG(L->dump());
  return deleteLoopIfDead(L, DT, SE, LI);
}
