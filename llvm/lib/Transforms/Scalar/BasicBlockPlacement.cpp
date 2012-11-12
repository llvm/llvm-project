//===-- BasicBlockPlacement.cpp - Basic Block Code Layout optimization ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a very simple profile guided basic block placement
// algorithm.  The idea is to put frequently executed blocks together at the
// start of the function, and hopefully increase the number of fall-through
// conditional branches.  If there is no profile information for a particular
// function, this pass basically orders blocks in depth-first order
//
// The algorithm implemented here is basically "Algo1" from "Profile Guided Code
// Positioning" by Pettis and Hansen, except that it uses basic block counts
// instead of edge counts.  This should be improved in many ways, but is very
// simple for now.
//
// Basically we "place" the entry block, then loop over all successors in a DFO,
// placing the most frequently executed successor until we run out of blocks.  I
// told you this was _extremely_ simplistic. :) This is also much slower than it
// could be.  When it becomes important, this pass will be rewritten to use a
// better algorithm, and then we can worry about efficiency.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "block-placement"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Transforms/Scalar.h"
#include <set>
using namespace llvm;

STATISTIC(NumMoved, "Number of basic blocks moved");

namespace {
  struct BlockPlacement : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    BlockPlacement() : FunctionPass(ID) {
      initializeBlockPlacementPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<ProfileInfo>();
      //AU.addPreserved<ProfileInfo>();  // Does this work?
    }
  private:
    /// PI - The profile information that is guiding us.
    ///
    ProfileInfo *PI;

    /// NumMovedBlocks - Every time we move a block, increment this counter.
    ///
    unsigned NumMovedBlocks;

    /// PlacedBlocks - Every time we place a block, remember it so we don't get
    /// into infinite loops.
    std::set<BasicBlock*> PlacedBlocks;

    /// InsertPos - This an iterator to the next place we want to insert a
    /// block.
    Function::iterator InsertPos;

    /// PlaceBlocks - Recursively place the specified blocks and any unplaced
    /// successors.
    void PlaceBlocks(BasicBlock *BB);
  };
}

char BlockPlacement::ID = 0;
INITIALIZE_PASS_BEGIN(BlockPlacement, "block-placement",
                "Profile Guided Basic Block Placement", false, false)
INITIALIZE_AG_DEPENDENCY(ProfileInfo)
INITIALIZE_PASS_END(BlockPlacement, "block-placement",
                "Profile Guided Basic Block Placement", false, false)

FunctionPass *llvm::createBlockPlacementPass() { return new BlockPlacement(); }

bool BlockPlacement::runOnFunction(Function &F) {
  PI = &getAnalysis<ProfileInfo>();

  NumMovedBlocks = 0;
  InsertPos = F.begin();

  // Recursively place all blocks.
  PlaceBlocks(F.begin());

  PlacedBlocks.clear();
  NumMoved += NumMovedBlocks;
  return NumMovedBlocks != 0;
}


/// PlaceBlocks - Recursively place the specified blocks and any unplaced
/// successors.
void BlockPlacement::PlaceBlocks(BasicBlock *BB) {
  assert(!PlacedBlocks.count(BB) && "Already placed this block!");
  PlacedBlocks.insert(BB);

  // Place the specified block.
  if (&*InsertPos != BB) {
    // Use splice to move the block into the right place.  This avoids having to
    // remove the block from the function then readd it, which causes a bunch of
    // symbol table traffic that is entirely pointless.
    Function::BasicBlockListType &Blocks = BB->getParent()->getBasicBlockList();
    Blocks.splice(InsertPos, Blocks, BB);

    ++NumMovedBlocks;
  } else {
    // This block is already in the right place, we don't have to do anything.
    ++InsertPos;
  }

  // Keep placing successors until we run out of ones to place.  Note that this
  // loop is very inefficient (N^2) for blocks with many successors, like switch
  // statements.  FIXME!
  while (1) {
    // Okay, now place any unplaced successors.
    succ_iterator SI = succ_begin(BB), E = succ_end(BB);

    // Scan for the first unplaced successor.
    for (; SI != E && PlacedBlocks.count(*SI); ++SI)
      /*empty*/;
    if (SI == E) return;  // No more successors to place.

    double MaxExecutionCount = PI->getExecutionCount(*SI);
    BasicBlock *MaxSuccessor = *SI;

    // Scan for more frequently executed successors
    for (; SI != E; ++SI)
      if (!PlacedBlocks.count(*SI)) {
        double Count = PI->getExecutionCount(*SI);
        if (Count > MaxExecutionCount ||
            // Prefer to not disturb the code.
            (Count == MaxExecutionCount && *SI == &*InsertPos)) {
          MaxExecutionCount = Count;
          MaxSuccessor = *SI;
        }
      }

    // Now that we picked the maximally executed successor, place it.
    PlaceBlocks(MaxSuccessor);
  }
}
