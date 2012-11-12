//===- LoopInfo.cpp - Natural Loop Calculator -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopInfo class that is used to identify natural loops
// and determine the loop depth of various nodes of the CFG.  Note that the
// loops identified may actually be several natural loops that share the same
// header node... not just a single natural loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfoImpl.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
using namespace llvm;

// Explicitly instantiate methods in LoopInfoImpl.h for IR-level Loops.
template class llvm::LoopBase<BasicBlock, Loop>;
template class llvm::LoopInfoBase<BasicBlock, Loop>;

// Always verify loopinfo if expensive checking is enabled.
#ifdef XDEBUG
static bool VerifyLoopInfo = true;
#else
static bool VerifyLoopInfo = false;
#endif
static cl::opt<bool,true>
VerifyLoopInfoX("verify-loop-info", cl::location(VerifyLoopInfo),
                cl::desc("Verify loop info (time consuming)"));

char LoopInfo::ID = 0;
INITIALIZE_PASS_BEGIN(LoopInfo, "loops", "Natural Loop Information", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(LoopInfo, "loops", "Natural Loop Information", true, true)

//===----------------------------------------------------------------------===//
// Loop implementation
//

/// isLoopInvariant - Return true if the specified value is loop invariant
///
bool Loop::isLoopInvariant(Value *V) const {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return !contains(I);
  return true;  // All non-instructions are loop invariant
}

/// hasLoopInvariantOperands - Return true if all the operands of the
/// specified instruction are loop invariant.
bool Loop::hasLoopInvariantOperands(Instruction *I) const {
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (!isLoopInvariant(I->getOperand(i)))
      return false;

  return true;
}

/// makeLoopInvariant - If the given value is an instruciton inside of the
/// loop and it can be hoisted, do so to make it trivially loop-invariant.
/// Return true if the value after any hoisting is loop invariant. This
/// function can be used as a slightly more aggressive replacement for
/// isLoopInvariant.
///
/// If InsertPt is specified, it is the point to hoist instructions to.
/// If null, the terminator of the loop preheader is used.
///
bool Loop::makeLoopInvariant(Value *V, bool &Changed,
                             Instruction *InsertPt) const {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return makeLoopInvariant(I, Changed, InsertPt);
  return true;  // All non-instructions are loop-invariant.
}

/// makeLoopInvariant - If the given instruction is inside of the
/// loop and it can be hoisted, do so to make it trivially loop-invariant.
/// Return true if the instruction after any hoisting is loop invariant. This
/// function can be used as a slightly more aggressive replacement for
/// isLoopInvariant.
///
/// If InsertPt is specified, it is the point to hoist instructions to.
/// If null, the terminator of the loop preheader is used.
///
bool Loop::makeLoopInvariant(Instruction *I, bool &Changed,
                             Instruction *InsertPt) const {
  // Test if the value is already loop-invariant.
  if (isLoopInvariant(I))
    return true;
  if (!isSafeToSpeculativelyExecute(I))
    return false;
  if (I->mayReadFromMemory())
    return false;
  // The landingpad instruction is immobile.
  if (isa<LandingPadInst>(I))
    return false;
  // Determine the insertion point, unless one was given.
  if (!InsertPt) {
    BasicBlock *Preheader = getLoopPreheader();
    // Without a preheader, hoisting is not feasible.
    if (!Preheader)
      return false;
    InsertPt = Preheader->getTerminator();
  }
  // Don't hoist instructions with loop-variant operands.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (!makeLoopInvariant(I->getOperand(i), Changed, InsertPt))
      return false;

  // Hoist.
  I->moveBefore(InsertPt);
  Changed = true;
  return true;
}

/// getCanonicalInductionVariable - Check to see if the loop has a canonical
/// induction variable: an integer recurrence that starts at 0 and increments
/// by one each time through the loop.  If so, return the phi node that
/// corresponds to it.
///
/// The IndVarSimplify pass transforms loops to have a canonical induction
/// variable.
///
PHINode *Loop::getCanonicalInductionVariable() const {
  BasicBlock *H = getHeader();

  BasicBlock *Incoming = 0, *Backedge = 0;
  pred_iterator PI = pred_begin(H);
  assert(PI != pred_end(H) &&
         "Loop must have at least one backedge!");
  Backedge = *PI++;
  if (PI == pred_end(H)) return 0;  // dead loop
  Incoming = *PI++;
  if (PI != pred_end(H)) return 0;  // multiple backedges?

  if (contains(Incoming)) {
    if (contains(Backedge))
      return 0;
    std::swap(Incoming, Backedge);
  } else if (!contains(Backedge))
    return 0;

  // Loop over all of the PHI nodes, looking for a canonical indvar.
  for (BasicBlock::iterator I = H->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    if (ConstantInt *CI =
        dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Incoming)))
      if (CI->isNullValue())
        if (Instruction *Inc =
            dyn_cast<Instruction>(PN->getIncomingValueForBlock(Backedge)))
          if (Inc->getOpcode() == Instruction::Add &&
                Inc->getOperand(0) == PN)
            if (ConstantInt *CI = dyn_cast<ConstantInt>(Inc->getOperand(1)))
              if (CI->equalsInt(1))
                return PN;
  }
  return 0;
}

/// isLCSSAForm - Return true if the Loop is in LCSSA form
bool Loop::isLCSSAForm(DominatorTree &DT) const {
  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallPtrSet<BasicBlock*, 16> LoopBBs(block_begin(), block_end());

  for (block_iterator BI = block_begin(), E = block_end(); BI != E; ++BI) {
    BasicBlock *BB = *BI;
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;++I)
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI) {
        User *U = *UI;
        BasicBlock *UserBB = cast<Instruction>(U)->getParent();
        if (PHINode *P = dyn_cast<PHINode>(U))
          UserBB = P->getIncomingBlock(UI);

        // Check the current block, as a fast-path, before checking whether
        // the use is anywhere in the loop.  Most values are used in the same
        // block they are defined in.  Also, blocks not reachable from the
        // entry are special; uses in them don't need to go through PHIs.
        if (UserBB != BB &&
            !LoopBBs.count(UserBB) &&
            DT.isReachableFromEntry(UserBB))
          return false;
      }
  }

  return true;
}

/// isLoopSimplifyForm - Return true if the Loop is in the form that
/// the LoopSimplify form transforms loops to, which is sometimes called
/// normal form.
bool Loop::isLoopSimplifyForm() const {
  // Normal-form loops have a preheader, a single backedge, and all of their
  // exits have all their predecessors inside the loop.
  return getLoopPreheader() && getLoopLatch() && hasDedicatedExits();
}

/// isSafeToClone - Return true if the loop body is safe to clone in practice.
/// Routines that reform the loop CFG and split edges often fail on indirectbr.
bool Loop::isSafeToClone() const {
  // Return false if any loop blocks contain indirectbrs.
  for (Loop::block_iterator I = block_begin(), E = block_end(); I != E; ++I) {
    if (isa<IndirectBrInst>((*I)->getTerminator()))
      return false;
  }
  return true;
}

/// hasDedicatedExits - Return true if no exit block for the loop
/// has a predecessor that is outside the loop.
bool Loop::hasDedicatedExits() const {
  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallPtrSet<BasicBlock *, 16> LoopBBs(block_begin(), block_end());
  // Each predecessor of each exit block of a normal loop is contained
  // within the loop.
  SmallVector<BasicBlock *, 4> ExitBlocks;
  getExitBlocks(ExitBlocks);
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    for (pred_iterator PI = pred_begin(ExitBlocks[i]),
         PE = pred_end(ExitBlocks[i]); PI != PE; ++PI)
      if (!LoopBBs.count(*PI))
        return false;
  // All the requirements are met.
  return true;
}

/// getUniqueExitBlocks - Return all unique successor blocks of this loop.
/// These are the blocks _outside of the current loop_ which are branched to.
/// This assumes that loop exits are in canonical form.
///
void
Loop::getUniqueExitBlocks(SmallVectorImpl<BasicBlock *> &ExitBlocks) const {
  assert(hasDedicatedExits() &&
         "getUniqueExitBlocks assumes the loop has canonical form exits!");

  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallVector<BasicBlock *, 128> LoopBBs(block_begin(), block_end());
  std::sort(LoopBBs.begin(), LoopBBs.end());

  SmallVector<BasicBlock *, 32> switchExitBlocks;

  for (block_iterator BI = block_begin(), BE = block_end(); BI != BE; ++BI) {

    BasicBlock *current = *BI;
    switchExitBlocks.clear();

    for (succ_iterator I = succ_begin(*BI), E = succ_end(*BI); I != E; ++I) {
      // If block is inside the loop then it is not a exit block.
      if (std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I))
        continue;

      pred_iterator PI = pred_begin(*I);
      BasicBlock *firstPred = *PI;

      // If current basic block is this exit block's first predecessor
      // then only insert exit block in to the output ExitBlocks vector.
      // This ensures that same exit block is not inserted twice into
      // ExitBlocks vector.
      if (current != firstPred)
        continue;

      // If a terminator has more then two successors, for example SwitchInst,
      // then it is possible that there are multiple edges from current block
      // to one exit block.
      if (std::distance(succ_begin(current), succ_end(current)) <= 2) {
        ExitBlocks.push_back(*I);
        continue;
      }

      // In case of multiple edges from current block to exit block, collect
      // only one edge in ExitBlocks. Use switchExitBlocks to keep track of
      // duplicate edges.
      if (std::find(switchExitBlocks.begin(), switchExitBlocks.end(), *I)
          == switchExitBlocks.end()) {
        switchExitBlocks.push_back(*I);
        ExitBlocks.push_back(*I);
      }
    }
  }
}

/// getUniqueExitBlock - If getUniqueExitBlocks would return exactly one
/// block, return that block. Otherwise return null.
BasicBlock *Loop::getUniqueExitBlock() const {
  SmallVector<BasicBlock *, 8> UniqueExitBlocks;
  getUniqueExitBlocks(UniqueExitBlocks);
  if (UniqueExitBlocks.size() == 1)
    return UniqueExitBlocks[0];
  return 0;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void Loop::dump() const {
  print(dbgs());
}
#endif

//===----------------------------------------------------------------------===//
// UnloopUpdater implementation
//

namespace {
/// Find the new parent loop for all blocks within the "unloop" whose last
/// backedges has just been removed.
class UnloopUpdater {
  Loop *Unloop;
  LoopInfo *LI;

  LoopBlocksDFS DFS;

  // Map unloop's immediate subloops to their nearest reachable parents. Nested
  // loops within these subloops will not change parents. However, an immediate
  // subloop's new parent will be the nearest loop reachable from either its own
  // exits *or* any of its nested loop's exits.
  DenseMap<Loop*, Loop*> SubloopParents;

  // Flag the presence of an irreducible backedge whose destination is a block
  // directly contained by the original unloop.
  bool FoundIB;

public:
  UnloopUpdater(Loop *UL, LoopInfo *LInfo) :
    Unloop(UL), LI(LInfo), DFS(UL), FoundIB(false) {}

  void updateBlockParents();

  void removeBlocksFromAncestors();

  void updateSubloopParents();

protected:
  Loop *getNearestLoop(BasicBlock *BB, Loop *BBLoop);
};
} // end anonymous namespace

/// updateBlockParents - Update the parent loop for all blocks that are directly
/// contained within the original "unloop".
void UnloopUpdater::updateBlockParents() {
  if (Unloop->getNumBlocks()) {
    // Perform a post order CFG traversal of all blocks within this loop,
    // propagating the nearest loop from sucessors to predecessors.
    LoopBlocksTraversal Traversal(DFS, LI);
    for (LoopBlocksTraversal::POTIterator POI = Traversal.begin(),
           POE = Traversal.end(); POI != POE; ++POI) {

      Loop *L = LI->getLoopFor(*POI);
      Loop *NL = getNearestLoop(*POI, L);

      if (NL != L) {
        // For reducible loops, NL is now an ancestor of Unloop.
        assert((NL != Unloop && (!NL || NL->contains(Unloop))) &&
               "uninitialized successor");
        LI->changeLoopFor(*POI, NL);
      }
      else {
        // Or the current block is part of a subloop, in which case its parent
        // is unchanged.
        assert((FoundIB || Unloop->contains(L)) && "uninitialized successor");
      }
    }
  }
  // Each irreducible loop within the unloop induces a round of iteration using
  // the DFS result cached by Traversal.
  bool Changed = FoundIB;
  for (unsigned NIters = 0; Changed; ++NIters) {
    assert(NIters < Unloop->getNumBlocks() && "runaway iterative algorithm");

    // Iterate over the postorder list of blocks, propagating the nearest loop
    // from successors to predecessors as before.
    Changed = false;
    for (LoopBlocksDFS::POIterator POI = DFS.beginPostorder(),
           POE = DFS.endPostorder(); POI != POE; ++POI) {

      Loop *L = LI->getLoopFor(*POI);
      Loop *NL = getNearestLoop(*POI, L);
      if (NL != L) {
        assert(NL != Unloop && (!NL || NL->contains(Unloop)) &&
               "uninitialized successor");
        LI->changeLoopFor(*POI, NL);
        Changed = true;
      }
    }
  }
}

/// removeBlocksFromAncestors - Remove unloop's blocks from all ancestors below
/// their new parents.
void UnloopUpdater::removeBlocksFromAncestors() {
  // Remove all unloop's blocks (including those in nested subloops) from
  // ancestors below the new parent loop.
  for (Loop::block_iterator BI = Unloop->block_begin(),
         BE = Unloop->block_end(); BI != BE; ++BI) {
    Loop *OuterParent = LI->getLoopFor(*BI);
    if (Unloop->contains(OuterParent)) {
      while (OuterParent->getParentLoop() != Unloop)
        OuterParent = OuterParent->getParentLoop();
      OuterParent = SubloopParents[OuterParent];
    }
    // Remove blocks from former Ancestors except Unloop itself which will be
    // deleted.
    for (Loop *OldParent = Unloop->getParentLoop(); OldParent != OuterParent;
         OldParent = OldParent->getParentLoop()) {
      assert(OldParent && "new loop is not an ancestor of the original");
      OldParent->removeBlockFromLoop(*BI);
    }
  }
}

/// updateSubloopParents - Update the parent loop for all subloops directly
/// nested within unloop.
void UnloopUpdater::updateSubloopParents() {
  while (!Unloop->empty()) {
    Loop *Subloop = *llvm::prior(Unloop->end());
    Unloop->removeChildLoop(llvm::prior(Unloop->end()));

    assert(SubloopParents.count(Subloop) && "DFS failed to visit subloop");
    if (Loop *Parent = SubloopParents[Subloop])
      Parent->addChildLoop(Subloop);
    else
      LI->addTopLevelLoop(Subloop);
  }
}

/// getNearestLoop - Return the nearest parent loop among this block's
/// successors. If a successor is a subloop header, consider its parent to be
/// the nearest parent of the subloop's exits.
///
/// For subloop blocks, simply update SubloopParents and return NULL.
Loop *UnloopUpdater::getNearestLoop(BasicBlock *BB, Loop *BBLoop) {

  // Initially for blocks directly contained by Unloop, NearLoop == Unloop and
  // is considered uninitialized.
  Loop *NearLoop = BBLoop;

  Loop *Subloop = 0;
  if (NearLoop != Unloop && Unloop->contains(NearLoop)) {
    Subloop = NearLoop;
    // Find the subloop ancestor that is directly contained within Unloop.
    while (Subloop->getParentLoop() != Unloop) {
      Subloop = Subloop->getParentLoop();
      assert(Subloop && "subloop is not an ancestor of the original loop");
    }
    // Get the current nearest parent of the Subloop exits, initially Unloop.
    NearLoop =
      SubloopParents.insert(std::make_pair(Subloop, Unloop)).first->second;
  }

  succ_iterator I = succ_begin(BB), E = succ_end(BB);
  if (I == E) {
    assert(!Subloop && "subloop blocks must have a successor");
    NearLoop = 0; // unloop blocks may now exit the function.
  }
  for (; I != E; ++I) {
    if (*I == BB)
      continue; // self loops are uninteresting

    Loop *L = LI->getLoopFor(*I);
    if (L == Unloop) {
      // This successor has not been processed. This path must lead to an
      // irreducible backedge.
      assert((FoundIB || !DFS.hasPostorder(*I)) && "should have seen IB");
      FoundIB = true;
    }
    if (L != Unloop && Unloop->contains(L)) {
      // Successor is in a subloop.
      if (Subloop)
        continue; // Branching within subloops. Ignore it.

      // BB branches from the original into a subloop header.
      assert(L->getParentLoop() == Unloop && "cannot skip into nested loops");

      // Get the current nearest parent of the Subloop's exits.
      L = SubloopParents[L];
      // L could be Unloop if the only exit was an irreducible backedge.
    }
    if (L == Unloop) {
      continue;
    }
    // Handle critical edges from Unloop into a sibling loop.
    if (L && !L->contains(Unloop)) {
      L = L->getParentLoop();
    }
    // Remember the nearest parent loop among successors or subloop exits.
    if (NearLoop == Unloop || !NearLoop || NearLoop->contains(L))
      NearLoop = L;
  }
  if (Subloop) {
    SubloopParents[Subloop] = NearLoop;
    return BBLoop;
  }
  return NearLoop;
}

//===----------------------------------------------------------------------===//
// LoopInfo implementation
//
bool LoopInfo::runOnFunction(Function &) {
  releaseMemory();
  LI.Analyze(getAnalysis<DominatorTree>().getBase());
  return false;
}

/// updateUnloop - The last backedge has been removed from a loop--now the
/// "unloop". Find a new parent for the blocks contained within unloop and
/// update the loop tree. We don't necessarily have valid dominators at this
/// point, but LoopInfo is still valid except for the removal of this loop.
///
/// Note that Unloop may now be an empty loop. Calling Loop::getHeader without
/// checking first is illegal.
void LoopInfo::updateUnloop(Loop *Unloop) {

  // First handle the special case of no parent loop to simplify the algorithm.
  if (!Unloop->getParentLoop()) {
    // Since BBLoop had no parent, Unloop blocks are no longer in a loop.
    for (Loop::block_iterator I = Unloop->block_begin(),
         E = Unloop->block_end(); I != E; ++I) {

      // Don't reparent blocks in subloops.
      if (getLoopFor(*I) != Unloop)
        continue;

      // Blocks no longer have a parent but are still referenced by Unloop until
      // the Unloop object is deleted.
      LI.changeLoopFor(*I, 0);
    }

    // Remove the loop from the top-level LoopInfo object.
    for (LoopInfo::iterator I = LI.begin();; ++I) {
      assert(I != LI.end() && "Couldn't find loop");
      if (*I == Unloop) {
        LI.removeLoop(I);
        break;
      }
    }

    // Move all of the subloops to the top-level.
    while (!Unloop->empty())
      LI.addTopLevelLoop(Unloop->removeChildLoop(llvm::prior(Unloop->end())));

    return;
  }

  // Update the parent loop for all blocks within the loop. Blocks within
  // subloops will not change parents.
  UnloopUpdater Updater(Unloop, this);
  Updater.updateBlockParents();

  // Remove blocks from former ancestor loops.
  Updater.removeBlocksFromAncestors();

  // Add direct subloops as children in their new parent loop.
  Updater.updateSubloopParents();

  // Remove unloop from its parent loop.
  Loop *ParentLoop = Unloop->getParentLoop();
  for (Loop::iterator I = ParentLoop->begin();; ++I) {
    assert(I != ParentLoop->end() && "Couldn't find loop");
    if (*I == Unloop) {
      ParentLoop->removeChildLoop(I);
      break;
    }
  }
}

void LoopInfo::verifyAnalysis() const {
  // LoopInfo is a FunctionPass, but verifying every loop in the function
  // each time verifyAnalysis is called is very expensive. The
  // -verify-loop-info option can enable this. In order to perform some
  // checking by default, LoopPass has been taught to call verifyLoop
  // manually during loop pass sequences.

  if (!VerifyLoopInfo) return;

  DenseSet<const Loop*> Loops;
  for (iterator I = begin(), E = end(); I != E; ++I) {
    assert(!(*I)->getParentLoop() && "Top-level loop has a parent!");
    (*I)->verifyLoopNest(&Loops);
  }

  // Verify that blocks are mapped to valid loops.
  for (DenseMap<BasicBlock*, Loop*>::const_iterator I = LI.BBMap.begin(),
         E = LI.BBMap.end(); I != E; ++I) {
    assert(Loops.count(I->second) && "orphaned loop");
    assert(I->second->contains(I->first) && "orphaned block");
  }
}

void LoopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTree>();
}

void LoopInfo::print(raw_ostream &OS, const Module*) const {
  LI.print(OS);
}

//===----------------------------------------------------------------------===//
// LoopBlocksDFS implementation
//

/// Traverse the loop blocks and store the DFS result.
/// Useful for clients that just want the final DFS result and don't need to
/// visit blocks during the initial traversal.
void LoopBlocksDFS::perform(LoopInfo *LI) {
  LoopBlocksTraversal Traversal(*this, LI);
  for (LoopBlocksTraversal::POTIterator POI = Traversal.begin(),
         POE = Traversal.end(); POI != POE; ++POI) ;
}
