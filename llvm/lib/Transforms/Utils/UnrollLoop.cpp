//===-- UnrollLoop.cpp - Loop unrolling utilities -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements some loop unrolling utilities. It does not define any
// actual pass or policy, but provides a single function to perform loop
// unrolling.
//
// It works best when loops have been canonicalized by the -indvars pass,
// allowing it to determine the trip counts of loops easily.
//
// The process of unrolling can produce extraneous basic blocks linked with
// unconditional branches.  This will be corrected in the future.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-unroll"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/BasicBlock.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cstdio>

using namespace llvm;

// TODO: Should these be here or in LoopUnroll?
STATISTIC(NumCompletelyUnrolled, "Number of loops completely unrolled");
STATISTIC(NumUnrolled,    "Number of loops unrolled (completely or otherwise)");

/// RemapInstruction - Convert the instruction operands from referencing the
/// current values into those specified by ValueMap.
static inline void RemapInstruction(Instruction *I,
                                    DenseMap<const Value *, Value*> &ValueMap) {
  for (unsigned op = 0, E = I->getNumOperands(); op != E; ++op) {
    Value *Op = I->getOperand(op);
    DenseMap<const Value *, Value*>::iterator It = ValueMap.find(Op);
    if (It != ValueMap.end()) Op = It->second;
    I->setOperand(op, Op);
  }
}

/// FoldBlockIntoPredecessor - Folds a basic block into its predecessor if it
/// only has one predecessor, and that predecessor only has one successor.
/// The LoopInfo Analysis that is passed will be kept consistent.
/// Returns the new combined block.
static BasicBlock *FoldBlockIntoPredecessor(BasicBlock *BB, LoopInfo* LI) {
  // Merge basic blocks into their predecessor if there is only one distinct
  // pred, and if there is only one distinct successor of the predecessor, and
  // if there are no PHI nodes.
  BasicBlock *OnlyPred = BB->getSinglePredecessor();
  if (!OnlyPred) return 0;

  if (OnlyPred->getTerminator()->getNumSuccessors() != 1)
    return 0;

  DOUT << "Merging: " << *BB << "into: " << *OnlyPred;

  // Resolve any PHI nodes at the start of the block.  They are all
  // guaranteed to have exactly one entry if they exist, unless there are
  // multiple duplicate (but guaranteed to be equal) entries for the
  // incoming edges.  This occurs when there are multiple edges from
  // OnlyPred to OnlySucc.
  FoldSingleEntryPHINodes(BB);

  // Delete the unconditional branch from the predecessor...
  OnlyPred->getInstList().pop_back();

  // Move all definitions in the successor to the predecessor...
  OnlyPred->getInstList().splice(OnlyPred->end(), BB->getInstList());

  // Make all PHI nodes that referred to BB now refer to Pred as their
  // source...
  BB->replaceAllUsesWith(OnlyPred);

  std::string OldName = BB->getName();

  // Erase basic block from the function...
  LI->removeBlock(BB);
  BB->eraseFromParent();

  // Inherit predecessor's name if it exists...
  if (!OldName.empty() && !OnlyPred->hasName())
    OnlyPred->setName(OldName);

  return OnlyPred;
}

/// Unroll the given loop by Count. The loop must be in LCSSA form. Returns true
/// if unrolling was succesful, or false if the loop was unmodified. Unrolling
/// can only fail when the loop's latch block is not terminated by a conditional
/// branch instruction. However, if the trip count (and multiple) are not known,
/// loop unrolling will mostly produce more code that is no faster.
///
/// The LoopInfo Analysis that is passed will be kept consistent.
///
/// If a LoopPassManager is passed in, and the loop is fully removed, it will be
/// removed from the LoopPassManager as well. LPM can also be NULL.
bool llvm::UnrollLoop(Loop *L, unsigned Count, LoopInfo* LI, LPPassManager* LPM) {
  assert(L->isLCSSAForm());

  BasicBlock *Header = L->getHeader();
  BasicBlock *LatchBlock = L->getLoopLatch();
  BranchInst *BI = dyn_cast<BranchInst>(LatchBlock->getTerminator());
  
  if (!BI || BI->isUnconditional()) {
    // The loop-rotate pass can be helpful to avoid this in many cases.
    DOUT << "  Can't unroll; loop not terminated by a conditional branch.\n";
    return false;
  }

  // Find trip count
  unsigned TripCount = L->getSmallConstantTripCount();
  // Find trip multiple if count is not available
  unsigned TripMultiple = 1;
  if (TripCount == 0)
    TripMultiple = L->getSmallConstantTripMultiple();

  if (TripCount != 0)
    DOUT << "  Trip Count = " << TripCount << "\n";
  if (TripMultiple != 1)
    DOUT << "  Trip Multiple = " << TripMultiple << "\n";

  // Effectively "DCE" unrolled iterations that are beyond the tripcount
  // and will never be executed.
  if (TripCount != 0 && Count > TripCount)
    Count = TripCount;

  assert(Count > 0);
  assert(TripMultiple > 0);
  assert(TripCount == 0 || TripCount % TripMultiple == 0);

  // Are we eliminating the loop control altogether?
  bool CompletelyUnroll = Count == TripCount;

  // If we know the trip count, we know the multiple...
  unsigned BreakoutTrip = 0;
  if (TripCount != 0) {
    BreakoutTrip = TripCount % Count;
    TripMultiple = 0;
  } else {
    // Figure out what multiple to use.
    BreakoutTrip = TripMultiple =
      (unsigned)GreatestCommonDivisor64(Count, TripMultiple);
  }

  if (CompletelyUnroll) {
    DOUT << "COMPLETELY UNROLLING loop %" << Header->getName()
         << " with trip count " << TripCount << "!\n";
  } else {
    DOUT << "UNROLLING loop %" << Header->getName()
         << " by " << Count;
    if (TripMultiple == 0 || BreakoutTrip != TripMultiple) {
      DOUT << " with a breakout at trip " << BreakoutTrip;
    } else if (TripMultiple != 1) {
      DOUT << " with " << TripMultiple << " trips per branch";
    }
    DOUT << "!\n";
  }

  std::vector<BasicBlock*> LoopBlocks = L->getBlocks();

  bool ContinueOnTrue = L->contains(BI->getSuccessor(0));
  BasicBlock *LoopExit = BI->getSuccessor(ContinueOnTrue);

  // For the first iteration of the loop, we should use the precloned values for
  // PHI nodes.  Insert associations now.
  typedef DenseMap<const Value*, Value*> ValueMapTy;
  ValueMapTy LastValueMap;
  std::vector<PHINode*> OrigPHINode;
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    OrigPHINode.push_back(PN);
    if (Instruction *I = 
                dyn_cast<Instruction>(PN->getIncomingValueForBlock(LatchBlock)))
      if (L->contains(I->getParent()))
        LastValueMap[I] = I;
  }

  std::vector<BasicBlock*> Headers;
  std::vector<BasicBlock*> Latches;
  Headers.push_back(Header);
  Latches.push_back(LatchBlock);

  for (unsigned It = 1; It != Count; ++It) {
    char SuffixBuffer[100];
    sprintf(SuffixBuffer, ".%d", It);
    
    std::vector<BasicBlock*> NewBlocks;
    
    for (std::vector<BasicBlock*>::iterator BB = LoopBlocks.begin(),
         E = LoopBlocks.end(); BB != E; ++BB) {
      ValueMapTy ValueMap;
      BasicBlock *New = CloneBasicBlock(*BB, ValueMap, SuffixBuffer);
      Header->getParent()->getBasicBlockList().push_back(New);

      // Loop over all of the PHI nodes in the block, changing them to use the
      // incoming values from the previous block.
      if (*BB == Header)
        for (unsigned i = 0, e = OrigPHINode.size(); i != e; ++i) {
          PHINode *NewPHI = cast<PHINode>(ValueMap[OrigPHINode[i]]);
          Value *InVal = NewPHI->getIncomingValueForBlock(LatchBlock);
          if (Instruction *InValI = dyn_cast<Instruction>(InVal))
            if (It > 1 && L->contains(InValI->getParent()))
              InVal = LastValueMap[InValI];
          ValueMap[OrigPHINode[i]] = InVal;
          New->getInstList().erase(NewPHI);
        }

      // Update our running map of newest clones
      LastValueMap[*BB] = New;
      for (ValueMapTy::iterator VI = ValueMap.begin(), VE = ValueMap.end();
           VI != VE; ++VI)
        LastValueMap[VI->first] = VI->second;

      L->addBasicBlockToLoop(New, LI->getBase());

      // Add phi entries for newly created values to all exit blocks except
      // the successor of the latch block.  The successor of the exit block will
      // be updated specially after unrolling all the way.
      if (*BB != LatchBlock)
        for (Value::use_iterator UI = (*BB)->use_begin(), UE = (*BB)->use_end();
             UI != UE;) {
          Instruction *UseInst = cast<Instruction>(*UI);
          ++UI;
          if (isa<PHINode>(UseInst) && !L->contains(UseInst->getParent())) {
            PHINode *phi = cast<PHINode>(UseInst);
            Value *Incoming = phi->getIncomingValueForBlock(*BB);
            phi->addIncoming(Incoming, New);
          }
        }

      // Keep track of new headers and latches as we create them, so that
      // we can insert the proper branches later.
      if (*BB == Header)
        Headers.push_back(New);
      if (*BB == LatchBlock) {
        Latches.push_back(New);

        // Also, clear out the new latch's back edge so that it doesn't look
        // like a new loop, so that it's amenable to being merged with adjacent
        // blocks later on.
        TerminatorInst *Term = New->getTerminator();
        assert(L->contains(Term->getSuccessor(!ContinueOnTrue)));
        assert(Term->getSuccessor(ContinueOnTrue) == LoopExit);
        Term->setSuccessor(!ContinueOnTrue, NULL);
      }

      NewBlocks.push_back(New);
    }
    
    // Remap all instructions in the most recent iteration
    for (unsigned i = 0; i < NewBlocks.size(); ++i)
      for (BasicBlock::iterator I = NewBlocks[i]->begin(),
           E = NewBlocks[i]->end(); I != E; ++I)
        RemapInstruction(I, LastValueMap);
  }
  
  // The latch block exits the loop.  If there are any PHI nodes in the
  // successor blocks, update them to use the appropriate values computed as the
  // last iteration of the loop.
  if (Count != 1) {
    SmallPtrSet<PHINode*, 8> Users;
    for (Value::use_iterator UI = LatchBlock->use_begin(),
         UE = LatchBlock->use_end(); UI != UE; ++UI)
      if (PHINode *phi = dyn_cast<PHINode>(*UI))
        Users.insert(phi);
    
    BasicBlock *LastIterationBB = cast<BasicBlock>(LastValueMap[LatchBlock]);
    for (SmallPtrSet<PHINode*,8>::iterator SI = Users.begin(), SE = Users.end();
         SI != SE; ++SI) {
      PHINode *PN = *SI;
      Value *InVal = PN->removeIncomingValue(LatchBlock, false);
      // If this value was defined in the loop, take the value defined by the
      // last iteration of the loop.
      if (Instruction *InValI = dyn_cast<Instruction>(InVal)) {
        if (L->contains(InValI->getParent()))
          InVal = LastValueMap[InVal];
      }
      PN->addIncoming(InVal, LastIterationBB);
    }
  }

  // Now, if we're doing complete unrolling, loop over the PHI nodes in the
  // original block, setting them to their incoming values.
  if (CompletelyUnroll) {
    BasicBlock *Preheader = L->getLoopPreheader();
    for (unsigned i = 0, e = OrigPHINode.size(); i != e; ++i) {
      PHINode *PN = OrigPHINode[i];
      PN->replaceAllUsesWith(PN->getIncomingValueForBlock(Preheader));
      Header->getInstList().erase(PN);
    }
  }

  // Now that all the basic blocks for the unrolled iterations are in place,
  // set up the branches to connect them.
  for (unsigned i = 0, e = Latches.size(); i != e; ++i) {
    // The original branch was replicated in each unrolled iteration.
    BranchInst *Term = cast<BranchInst>(Latches[i]->getTerminator());

    // The branch destination.
    unsigned j = (i + 1) % e;
    BasicBlock *Dest = Headers[j];
    bool NeedConditional = true;

    // For a complete unroll, make the last iteration end with a branch
    // to the exit block.
    if (CompletelyUnroll && j == 0) {
      Dest = LoopExit;
      NeedConditional = false;
    }

    // If we know the trip count or a multiple of it, we can safely use an
    // unconditional branch for some iterations.
    if (j != BreakoutTrip && (TripMultiple == 0 || j % TripMultiple != 0)) {
      NeedConditional = false;
    }

    if (NeedConditional) {
      // Update the conditional branch's successor for the following
      // iteration.
      Term->setSuccessor(!ContinueOnTrue, Dest);
    } else {
      Term->setUnconditionalDest(Dest);
      // Merge adjacent basic blocks, if possible.
      if (BasicBlock *Fold = FoldBlockIntoPredecessor(Dest, LI)) {
        std::replace(Latches.begin(), Latches.end(), Dest, Fold);
        std::replace(Headers.begin(), Headers.end(), Dest, Fold);
      }
    }
  }
  
  // At this point, the code is well formed.  We now do a quick sweep over the
  // inserted code, doing constant propagation and dead code elimination as we
  // go.
  const std::vector<BasicBlock*> &NewLoopBlocks = L->getBlocks();
  for (std::vector<BasicBlock*>::const_iterator BB = NewLoopBlocks.begin(),
       BBE = NewLoopBlocks.end(); BB != BBE; ++BB)
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E; ) {
      Instruction *Inst = I++;

      if (isInstructionTriviallyDead(Inst))
        (*BB)->getInstList().erase(Inst);
      else if (Constant *C = ConstantFoldInstruction(Inst)) {
        Inst->replaceAllUsesWith(C);
        (*BB)->getInstList().erase(Inst);
      }
    }

  NumCompletelyUnrolled += CompletelyUnroll;
  ++NumUnrolled;
  // Remove the loop from the LoopPassManager if it's completely removed.
  if (CompletelyUnroll && LPM != NULL)
    LPM->deleteLoopFromQueue(L);

  // If we didn't completely unroll the loop, it should still be in LCSSA form.
  if (!CompletelyUnroll)
    assert(L->isLCSSAForm());

  return true;
}
