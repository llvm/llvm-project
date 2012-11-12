//===-- UnrollLoopRuntime.cpp - Runtime Loop unrolling utilities ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements some loop unrolling utilities for loops with run-time
// trip counts.  See LoopUnroll.cpp for unrolling loops with compile-time
// trip counts.
//
// The functions in this file are used to generate extra code when the
// run-time trip count modulo the unroll factor is not 0.  When this is the
// case, we need to generate code to execute these 'left over' iterations.
//
// The current strategy generates an if-then-else sequence prior to the
// unrolled loop to execute the 'left over' iterations.  Other strategies
// include generate a loop before or after the unrolled loop.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-unroll"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/BasicBlock.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>

using namespace llvm;

STATISTIC(NumRuntimeUnrolled,
          "Number of loops unrolled with run-time trip counts");

/// Connect the unrolling prolog code to the original loop.
/// The unrolling prolog code contains code to execute the
/// 'extra' iterations if the run-time trip count modulo the
/// unroll count is non-zero.
///
/// This function performs the following:
/// - Create PHI nodes at prolog end block to combine values
///   that exit the prolog code and jump around the prolog.
/// - Add a PHI operand to a PHI node at the loop exit block
///   for values that exit the prolog and go around the loop.
/// - Branch around the original loop if the trip count is less
///   than the unroll factor.
///
static void ConnectProlog(Loop *L, Value *TripCount, unsigned Count,
                          BasicBlock *LastPrologBB, BasicBlock *PrologEnd,
                          BasicBlock *OrigPH, BasicBlock *NewPH,
                          ValueToValueMapTy &LVMap, Pass *P) {
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch != 0 && "Loop must have a latch");

  // Create a PHI node for each outgoing value from the original loop
  // (which means it is an outgoing value from the prolog code too).
  // The new PHI node is inserted in the prolog end basic block.
  // The new PHI name is added as an operand of a PHI node in either
  // the loop header or the loop exit block.
  for (succ_iterator SBI = succ_begin(Latch), SBE = succ_end(Latch);
       SBI != SBE; ++SBI) {
    for (BasicBlock::iterator BBI = (*SBI)->begin();
         PHINode *PN = dyn_cast<PHINode>(BBI); ++BBI) {

      // Add a new PHI node to the prolog end block and add the
      // appropriate incoming values.
      PHINode *NewPN = PHINode::Create(PN->getType(), 2, PN->getName()+".unr",
                                       PrologEnd->getTerminator());
      // Adding a value to the new PHI node from the original loop preheader.
      // This is the value that skips all the prolog code.
      if (L->contains(PN)) {
        NewPN->addIncoming(PN->getIncomingValueForBlock(NewPH), OrigPH);
      } else {
        NewPN->addIncoming(Constant::getNullValue(PN->getType()), OrigPH);
      }

      Value *V = PN->getIncomingValueForBlock(Latch);
      if (Instruction *I = dyn_cast<Instruction>(V)) {
        if (L->contains(I)) {
          V = LVMap[I];
        }
      }
      // Adding a value to the new PHI node from the last prolog block
      // that was created.
      NewPN->addIncoming(V, LastPrologBB);

      // Update the existing PHI node operand with the value from the
      // new PHI node.  How this is done depends on if the existing
      // PHI node is in the original loop block, or the exit block.
      if (L->contains(PN)) {
        PN->setIncomingValue(PN->getBasicBlockIndex(NewPH), NewPN);
      } else {
        PN->addIncoming(NewPN, PrologEnd);
      }
    }
  }

  // Create a branch around the orignal loop, which is taken if the
  // trip count is less than the unroll factor.
  Instruction *InsertPt = PrologEnd->getTerminator();
  Instruction *BrLoopExit =
    new ICmpInst(InsertPt, ICmpInst::ICMP_ULT, TripCount,
                 ConstantInt::get(TripCount->getType(), Count));
  BasicBlock *Exit = L->getUniqueExitBlock();
  assert(Exit != 0 && "Loop must have a single exit block only");
  // Split the exit to maintain loop canonicalization guarantees
  SmallVector<BasicBlock*, 4> Preds(pred_begin(Exit), pred_end(Exit));
  if (!Exit->isLandingPad()) {
    SplitBlockPredecessors(Exit, Preds, ".unr-lcssa", P);
  } else {
    SmallVector<BasicBlock*, 2> NewBBs;
    SplitLandingPadPredecessors(Exit, Preds, ".unr1-lcssa", ".unr2-lcssa",
                                P, NewBBs);
  }
  // Add the branch to the exit block (around the unrolled loop)
  BranchInst::Create(Exit, NewPH, BrLoopExit, InsertPt);
  InsertPt->eraseFromParent();
}

/// Create a clone of the blocks in a loop and connect them together.
/// This function doesn't create a clone of the loop structure.
///
/// There are two value maps that are defined and used.  VMap is
/// for the values in the current loop instance.  LVMap contains
/// the values from the last loop instance.  We need the LVMap values
/// to update the initial values for the current loop instance.
///
static void CloneLoopBlocks(Loop *L,
                            bool FirstCopy,
                            BasicBlock *InsertTop,
                            BasicBlock *InsertBot,
                            std::vector<BasicBlock *> &NewBlocks,
                            LoopBlocksDFS &LoopBlocks,
                            ValueToValueMapTy &VMap,
                            ValueToValueMapTy &LVMap,
                            LoopInfo *LI) {

  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  Function *F = Header->getParent();
  LoopBlocksDFS::RPOIterator BlockBegin = LoopBlocks.beginRPO();
  LoopBlocksDFS::RPOIterator BlockEnd = LoopBlocks.endRPO();
  // For each block in the original loop, create a new copy,
  // and update the value map with the newly created values.
  for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB) {
    BasicBlock *NewBB = CloneBasicBlock(*BB, VMap, ".unr", F);
    NewBlocks.push_back(NewBB);

    if (Loop *ParentLoop = L->getParentLoop())
      ParentLoop->addBasicBlockToLoop(NewBB, LI->getBase());

    VMap[*BB] = NewBB;
    if (Header == *BB) {
      // For the first block, add a CFG connection to this newly
      // created block
      InsertTop->getTerminator()->setSuccessor(0, NewBB);

      // Change the incoming values to the ones defined in the
      // previously cloned loop.
      for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
        PHINode *NewPHI = cast<PHINode>(VMap[I]);
        if (FirstCopy) {
          // We replace the first phi node with the value from the preheader
          VMap[I] = NewPHI->getIncomingValueForBlock(Preheader);
          NewBB->getInstList().erase(NewPHI);
        } else {
          // Update VMap with values from the previous block
          unsigned idx = NewPHI->getBasicBlockIndex(Latch);
          Value *InVal = NewPHI->getIncomingValue(idx);
          if (Instruction *I = dyn_cast<Instruction>(InVal))
            if (L->contains(I))
              InVal = LVMap[InVal];
          NewPHI->setIncomingValue(idx, InVal);
          NewPHI->setIncomingBlock(idx, InsertTop);
        }
      }
    }

    if (Latch == *BB) {
      VMap.erase((*BB)->getTerminator());
      NewBB->getTerminator()->eraseFromParent();
      BranchInst::Create(InsertBot, NewBB);
    }
  }
  // LastValueMap is updated with the values for the current loop
  // which are used the next time this function is called.
  for (ValueToValueMapTy::iterator VI = VMap.begin(), VE = VMap.end();
       VI != VE; ++VI) {
    LVMap[VI->first] = VI->second;
  }
}

/// Insert code in the prolog code when unrolling a loop with a
/// run-time trip-count.
///
/// This method assumes that the loop unroll factor is total number
/// of loop bodes in the loop after unrolling. (Some folks refer
/// to the unroll factor as the number of *extra* copies added).
/// We assume also that the loop unroll factor is a power-of-two. So, after
/// unrolling the loop, the number of loop bodies executed is 2,
/// 4, 8, etc.  Note - LLVM converts the if-then-sequence to a switch
/// instruction in SimplifyCFG.cpp.  Then, the backend decides how code for
/// the switch instruction is generated.
///
///    extraiters = tripcount % loopfactor
///    if (extraiters == 0) jump Loop:
///    if (extraiters == loopfactor) jump L1
///    if (extraiters == loopfactor-1) jump L2
///    ...
///    L1:  LoopBody;
///    L2:  LoopBody;
///    ...
///    if tripcount < loopfactor jump End
///    Loop:
///    ...
///    End:
///
bool llvm::UnrollRuntimeLoopProlog(Loop *L, unsigned Count, LoopInfo *LI,
                                   LPPassManager *LPM) {
  // for now, only unroll loops that contain a single exit
  if (!L->getExitingBlock())
    return false;

  // Make sure the loop is in canonical form, and there is a single
  // exit block only.
  if (!L->isLoopSimplifyForm() || L->getUniqueExitBlock() == 0)
    return false;

  // Use Scalar Evolution to compute the trip count.  This allows more
  // loops to be unrolled than relying on induction var simplification
  if (!LPM)
    return false;
  ScalarEvolution *SE = LPM->getAnalysisIfAvailable<ScalarEvolution>();
  if (SE == 0)
    return false;

  // Only unroll loops with a computable trip count and the trip count needs
  // to be an int value (allowing a pointer type is a TODO item)
  const SCEV *BECount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BECount) || !BECount->getType()->isIntegerTy())
    return false;

  // Add 1 since the backedge count doesn't include the first loop iteration
  const SCEV *TripCountSC =
    SE->getAddExpr(BECount, SE->getConstant(BECount->getType(), 1));
  if (isa<SCEVCouldNotCompute>(TripCountSC))
    return false;

  // We only handle cases when the unroll factor is a power of 2.
  // Count is the loop unroll factor, the number of extra copies added + 1.
  if ((Count & (Count-1)) != 0)
    return false;

  // If this loop is nested, then the loop unroller changes the code in
  // parent loop, so the Scalar Evolution pass needs to be run again
  if (Loop *ParentLoop = L->getParentLoop())
    SE->forgetLoop(ParentLoop);

  BasicBlock *PH = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  // It helps to splits the original preheader twice, one for the end of the
  // prolog code and one for a new loop preheader
  BasicBlock *PEnd = SplitEdge(PH, Header, LPM->getAsPass());
  BasicBlock *NewPH = SplitBlock(PEnd, PEnd->getTerminator(), LPM->getAsPass());
  BranchInst *PreHeaderBR = cast<BranchInst>(PH->getTerminator());

  // Compute the number of extra iterations required, which is:
  //  extra iterations = run-time trip count % (loop unroll factor + 1)
  SCEVExpander Expander(*SE, "loop-unroll");
  Value *TripCount = Expander.expandCodeFor(TripCountSC, TripCountSC->getType(),
                                            PreHeaderBR);
  Type *CountTy = TripCount->getType();
  BinaryOperator *ModVal =
    BinaryOperator::CreateURem(TripCount,
                               ConstantInt::get(CountTy, Count),
                               "xtraiter");
  ModVal->insertBefore(PreHeaderBR);

  // Check if for no extra iterations, then jump to unrolled loop
  Value *BranchVal = new ICmpInst(PreHeaderBR,
                                  ICmpInst::ICMP_NE, ModVal,
                                  ConstantInt::get(CountTy, 0), "lcmp");
  // Branch to either the extra iterations or the unrolled loop
  // We will fix up the true branch label when adding loop body copies
  BranchInst::Create(PEnd, PEnd, BranchVal, PreHeaderBR);
  assert(PreHeaderBR->isUnconditional() &&
         PreHeaderBR->getSuccessor(0) == PEnd &&
         "CFG edges in Preheader are not correct");
  PreHeaderBR->eraseFromParent();

  ValueToValueMapTy LVMap;
  Function *F = Header->getParent();
  // These variables are used to update the CFG links in each iteration
  BasicBlock *CompareBB = 0;
  BasicBlock *LastLoopBB = PH;
  // Get an ordered list of blocks in the loop to help with the ordering of the
  // cloned blocks in the prolog code
  LoopBlocksDFS LoopBlocks(L);
  LoopBlocks.perform(LI);

  //
  // For each extra loop iteration, create a copy of the loop's basic blocks
  // and generate a condition that branches to the copy depending on the
  // number of 'left over' iterations.
  //
  for (unsigned leftOverIters = Count-1; leftOverIters > 0; --leftOverIters) {
    std::vector<BasicBlock*> NewBlocks;
    ValueToValueMapTy VMap;

    // Clone all the basic blocks in the loop, but we don't clone the loop
    // This function adds the appropriate CFG connections.
    CloneLoopBlocks(L, (leftOverIters == Count-1), LastLoopBB, PEnd, NewBlocks,
                    LoopBlocks, VMap, LVMap, LI);
    LastLoopBB = cast<BasicBlock>(VMap[Latch]);

    // Insert the cloned blocks into function just before the original loop
    F->getBasicBlockList().splice(PEnd, F->getBasicBlockList(),
                                  NewBlocks[0], F->end());

    // Generate the code for the comparison which determines if the loop
    // prolog code needs to be executed.
    if (leftOverIters == Count-1) {
      // There is no compare block for the fall-thru case when for the last
      // left over iteration
      CompareBB = NewBlocks[0];
    } else {
      // Create a new block for the comparison
      BasicBlock *NewBB = BasicBlock::Create(CompareBB->getContext(), "unr.cmp",
                                             F, CompareBB);
      if (Loop *ParentLoop = L->getParentLoop()) {
        // Add the new block to the parent loop, if needed
        ParentLoop->addBasicBlockToLoop(NewBB, LI->getBase());
      }

      // The comparison w/ the extra iteration value and branch
      Value *BranchVal = new ICmpInst(*NewBB, ICmpInst::ICMP_EQ, ModVal,
                                      ConstantInt::get(CountTy, leftOverIters),
                                      "un.tmp");
      // Branch to either the extra iterations or the unrolled loop
      BranchInst::Create(NewBlocks[0], CompareBB,
                         BranchVal, NewBB);
      CompareBB = NewBB;
      PH->getTerminator()->setSuccessor(0, NewBB);
      VMap[NewPH] = CompareBB;
    }

    // Rewrite the cloned instruction operands to use the values
    // created when the clone is created.
    for (unsigned i = 0, e = NewBlocks.size(); i != e; ++i) {
      for (BasicBlock::iterator I = NewBlocks[i]->begin(),
             E = NewBlocks[i]->end(); I != E; ++I) {
        RemapInstruction(I, VMap,
                         RF_NoModuleLevelChanges|RF_IgnoreMissingEntries);
      }
    }
  }

  // Connect the prolog code to the original loop and update the
  // PHI functions.
  ConnectProlog(L, TripCount, Count, LastLoopBB, PEnd, PH, NewPH, LVMap,
                LPM->getAsPass());
  NumRuntimeUnrolled++;
  return true;
}
