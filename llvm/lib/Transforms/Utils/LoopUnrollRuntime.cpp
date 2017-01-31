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
// unrolled loop to execute the 'left over' iterations before or after the
// unrolled loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "loop-unroll"

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
static void ConnectProlog(Loop *L, Value *BECount, unsigned Count,
                          BasicBlock *PrologExit, BasicBlock *PreHeader,
                          BasicBlock *NewPreHeader, ValueToValueMapTy &VMap,
                          DominatorTree *DT, LoopInfo *LI, bool PreserveLCSSA) {
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch && "Loop must have a latch");
  BasicBlock *PrologLatch = cast<BasicBlock>(VMap[Latch]);

  // Create a PHI node for each outgoing value from the original loop
  // (which means it is an outgoing value from the prolog code too).
  // The new PHI node is inserted in the prolog end basic block.
  // The new PHI node value is added as an operand of a PHI node in either
  // the loop header or the loop exit block.
  for (BasicBlock *Succ : successors(Latch)) {
    for (Instruction &BBI : *Succ) {
      PHINode *PN = dyn_cast<PHINode>(&BBI);
      // Exit when we passed all PHI nodes.
      if (!PN)
        break;
      // Add a new PHI node to the prolog end block and add the
      // appropriate incoming values.
      PHINode *NewPN = PHINode::Create(PN->getType(), 2, PN->getName() + ".unr",
                                       PrologExit->getFirstNonPHI());
      // Adding a value to the new PHI node from the original loop preheader.
      // This is the value that skips all the prolog code.
      if (L->contains(PN)) {
        NewPN->addIncoming(PN->getIncomingValueForBlock(NewPreHeader),
                           PreHeader);
      } else {
        NewPN->addIncoming(UndefValue::get(PN->getType()), PreHeader);
      }

      Value *V = PN->getIncomingValueForBlock(Latch);
      if (Instruction *I = dyn_cast<Instruction>(V)) {
        if (L->contains(I)) {
          V = VMap.lookup(I);
        }
      }
      // Adding a value to the new PHI node from the last prolog block
      // that was created.
      NewPN->addIncoming(V, PrologLatch);

      // Update the existing PHI node operand with the value from the
      // new PHI node.  How this is done depends on if the existing
      // PHI node is in the original loop block, or the exit block.
      if (L->contains(PN)) {
        PN->setIncomingValue(PN->getBasicBlockIndex(NewPreHeader), NewPN);
      } else {
        PN->addIncoming(NewPN, PrologExit);
      }
    }
  }

  // Make sure that created prolog loop is in simplified form
  SmallVector<BasicBlock *, 4> PrologExitPreds;
  Loop *PrologLoop = LI->getLoopFor(PrologLatch);
  if (PrologLoop) {
    for (BasicBlock *PredBB : predecessors(PrologExit))
      if (PrologLoop->contains(PredBB))
        PrologExitPreds.push_back(PredBB);

    SplitBlockPredecessors(PrologExit, PrologExitPreds, ".unr-lcssa", DT, LI,
                           PreserveLCSSA);
  }

  // Create a branch around the original loop, which is taken if there are no
  // iterations remaining to be executed after running the prologue.
  Instruction *InsertPt = PrologExit->getTerminator();
  IRBuilder<> B(InsertPt);

  assert(Count != 0 && "nonsensical Count!");

  // If BECount <u (Count - 1) then (BECount + 1) % Count == (BECount + 1)
  // This means %xtraiter is (BECount + 1) and all of the iterations of this
  // loop were executed by the prologue.  Note that if BECount <u (Count - 1)
  // then (BECount + 1) cannot unsigned-overflow.
  Value *BrLoopExit =
      B.CreateICmpULT(BECount, ConstantInt::get(BECount->getType(), Count - 1));
  BasicBlock *Exit = L->getUniqueExitBlock();
  assert(Exit && "Loop must have a single exit block only");
  // Split the exit to maintain loop canonicalization guarantees
  SmallVector<BasicBlock*, 4> Preds(predecessors(Exit));
  SplitBlockPredecessors(Exit, Preds, ".unr-lcssa", DT, LI,
                         PreserveLCSSA);
  // Add the branch to the exit block (around the unrolled loop)
  B.CreateCondBr(BrLoopExit, Exit, NewPreHeader);
  InsertPt->eraseFromParent();
  if (DT)
    DT->changeImmediateDominator(Exit, PrologExit);
}

/// Connect the unrolling epilog code to the original loop.
/// The unrolling epilog code contains code to execute the
/// 'extra' iterations if the run-time trip count modulo the
/// unroll count is non-zero.
///
/// This function performs the following:
/// - Update PHI nodes at the unrolling loop exit and epilog loop exit
/// - Create PHI nodes at the unrolling loop exit to combine
///   values that exit the unrolling loop code and jump around it.
/// - Update PHI operands in the epilog loop by the new PHI nodes
/// - Branch around the epilog loop if extra iters (ModVal) is zero.
///
static void ConnectEpilog(Loop *L, Value *ModVal, BasicBlock *NewExit,
                          BasicBlock *Exit, BasicBlock *PreHeader,
                          BasicBlock *EpilogPreHeader, BasicBlock *NewPreHeader,
                          ValueToValueMapTy &VMap, DominatorTree *DT,
                          LoopInfo *LI, bool PreserveLCSSA)  {
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch && "Loop must have a latch");
  BasicBlock *EpilogLatch = cast<BasicBlock>(VMap[Latch]);

  // Loop structure should be the following:
  //
  // PreHeader
  // NewPreHeader
  //   Header
  //   ...
  //   Latch
  // NewExit (PN)
  // EpilogPreHeader
  //   EpilogHeader
  //   ...
  //   EpilogLatch
  // Exit (EpilogPN)

  // Update PHI nodes at NewExit and Exit.
  for (Instruction &BBI : *NewExit) {
    PHINode *PN = dyn_cast<PHINode>(&BBI);
    // Exit when we passed all PHI nodes.
    if (!PN)
      break;
    // PN should be used in another PHI located in Exit block as
    // Exit was split by SplitBlockPredecessors into Exit and NewExit
    // Basicaly it should look like:
    // NewExit:
    //   PN = PHI [I, Latch]
    // ...
    // Exit:
    //   EpilogPN = PHI [PN, EpilogPreHeader]
    //
    // There is EpilogPreHeader incoming block instead of NewExit as
    // NewExit was spilt 1 more time to get EpilogPreHeader.
    assert(PN->hasOneUse() && "The phi should have 1 use");
    PHINode *EpilogPN = cast<PHINode> (PN->use_begin()->getUser());
    assert(EpilogPN->getParent() == Exit && "EpilogPN should be in Exit block");

    // Add incoming PreHeader from branch around the Loop
    PN->addIncoming(UndefValue::get(PN->getType()), PreHeader);

    Value *V = PN->getIncomingValueForBlock(Latch);
    Instruction *I = dyn_cast<Instruction>(V);
    if (I && L->contains(I))
      // If value comes from an instruction in the loop add VMap value.
      V = VMap.lookup(I);
    // For the instruction out of the loop, constant or undefined value
    // insert value itself.
    EpilogPN->addIncoming(V, EpilogLatch);

    assert(EpilogPN->getBasicBlockIndex(EpilogPreHeader) >= 0 &&
          "EpilogPN should have EpilogPreHeader incoming block");
    // Change EpilogPreHeader incoming block to NewExit.
    EpilogPN->setIncomingBlock(EpilogPN->getBasicBlockIndex(EpilogPreHeader),
                               NewExit);
    // Now PHIs should look like:
    // NewExit:
    //   PN = PHI [I, Latch], [undef, PreHeader]
    // ...
    // Exit:
    //   EpilogPN = PHI [PN, NewExit], [VMap[I], EpilogLatch]
  }

  // Create PHI nodes at NewExit (from the unrolling loop Latch and PreHeader).
  // Update corresponding PHI nodes in epilog loop.
  for (BasicBlock *Succ : successors(Latch)) {
    // Skip this as we already updated phis in exit blocks.
    if (!L->contains(Succ))
      continue;
    for (Instruction &BBI : *Succ) {
      PHINode *PN = dyn_cast<PHINode>(&BBI);
      // Exit when we passed all PHI nodes.
      if (!PN)
        break;
      // Add new PHI nodes to the loop exit block and update epilog
      // PHIs with the new PHI values.
      PHINode *NewPN = PHINode::Create(PN->getType(), 2, PN->getName() + ".unr",
                                       NewExit->getFirstNonPHI());
      // Adding a value to the new PHI node from the unrolling loop preheader.
      NewPN->addIncoming(PN->getIncomingValueForBlock(NewPreHeader), PreHeader);
      // Adding a value to the new PHI node from the unrolling loop latch.
      NewPN->addIncoming(PN->getIncomingValueForBlock(Latch), Latch);

      // Update the existing PHI node operand with the value from the new PHI
      // node.  Corresponding instruction in epilog loop should be PHI.
      PHINode *VPN = cast<PHINode>(VMap[&BBI]);
      VPN->setIncomingValue(VPN->getBasicBlockIndex(EpilogPreHeader), NewPN);
    }
  }

  Instruction *InsertPt = NewExit->getTerminator();
  IRBuilder<> B(InsertPt);
  Value *BrLoopExit = B.CreateIsNotNull(ModVal, "lcmp.mod");
  assert(Exit && "Loop must have a single exit block only");
  // Split the epilogue exit to maintain loop canonicalization guarantees
  SmallVector<BasicBlock*, 4> Preds(predecessors(Exit));
  SplitBlockPredecessors(Exit, Preds, ".epilog-lcssa", DT, LI,
                         PreserveLCSSA);
  // Add the branch to the exit block (around the unrolling loop)
  B.CreateCondBr(BrLoopExit, EpilogPreHeader, Exit);
  InsertPt->eraseFromParent();
  if (DT)
    DT->changeImmediateDominator(Exit, NewExit);

  // Split the main loop exit to maintain canonicalization guarantees.
  SmallVector<BasicBlock*, 4> NewExitPreds{Latch};
  SplitBlockPredecessors(NewExit, NewExitPreds, ".loopexit", DT, LI,
                         PreserveLCSSA);
}

/// Create a clone of the blocks in a loop and connect them together.
/// If CreateRemainderLoop is false, loop structure will not be cloned,
/// otherwise a new loop will be created including all cloned blocks, and the
/// iterator of it switches to count NewIter down to 0.
/// The cloned blocks should be inserted between InsertTop and InsertBot.
/// If loop structure is cloned InsertTop should be new preheader, InsertBot
/// new loop exit.
///
static void CloneLoopBlocks(Loop *L, Value *NewIter,
                            const bool CreateRemainderLoop,
                            const bool UseEpilogRemainder,
                            BasicBlock *InsertTop, BasicBlock *InsertBot,
                            BasicBlock *Preheader,
                            std::vector<BasicBlock *> &NewBlocks,
                            LoopBlocksDFS &LoopBlocks, ValueToValueMapTy &VMap,
                            DominatorTree *DT, LoopInfo *LI) {
  StringRef suffix = UseEpilogRemainder ? "epil" : "prol";
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  Function *F = Header->getParent();
  LoopBlocksDFS::RPOIterator BlockBegin = LoopBlocks.beginRPO();
  LoopBlocksDFS::RPOIterator BlockEnd = LoopBlocks.endRPO();
  Loop *ParentLoop = L->getParentLoop();
  NewLoopsMap NewLoops;
  NewLoops[ParentLoop] = ParentLoop;
  if (!CreateRemainderLoop)
    NewLoops[L] = ParentLoop;

  // For each block in the original loop, create a new copy,
  // and update the value map with the newly created values.
  for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB) {
    BasicBlock *NewBB = CloneBasicBlock(*BB, VMap, "." + suffix, F);
    NewBlocks.push_back(NewBB);

    // If we're unrolling the outermost loop, there's no remainder loop,
    // and this block isn't in a nested loop, then the new block is not
    // in any loop. Otherwise, add it to loopinfo.
    if (CreateRemainderLoop || LI->getLoopFor(*BB) != L || ParentLoop)
      addClonedBlockToLoopInfo(*BB, NewBB, LI, NewLoops);

    VMap[*BB] = NewBB;
    if (Header == *BB) {
      // For the first block, add a CFG connection to this newly
      // created block.
      InsertTop->getTerminator()->setSuccessor(0, NewBB);
    }

    if (DT) {
      if (Header == *BB) {
        // The header is dominated by the preheader.
        DT->addNewBlock(NewBB, InsertTop);
      } else {
        // Copy information from original loop to unrolled loop.
        BasicBlock *IDomBB = DT->getNode(*BB)->getIDom()->getBlock();
        DT->addNewBlock(NewBB, cast<BasicBlock>(VMap[IDomBB]));
      }
    }

    if (Latch == *BB) {
      // For the last block, if CreateRemainderLoop is false, create a direct
      // jump to InsertBot. If not, create a loop back to cloned head.
      VMap.erase((*BB)->getTerminator());
      BasicBlock *FirstLoopBB = cast<BasicBlock>(VMap[Header]);
      BranchInst *LatchBR = cast<BranchInst>(NewBB->getTerminator());
      IRBuilder<> Builder(LatchBR);
      if (!CreateRemainderLoop) {
        Builder.CreateBr(InsertBot);
      } else {
        PHINode *NewIdx = PHINode::Create(NewIter->getType(), 2,
                                          suffix + ".iter",
                                          FirstLoopBB->getFirstNonPHI());
        Value *IdxSub =
            Builder.CreateSub(NewIdx, ConstantInt::get(NewIdx->getType(), 1),
                              NewIdx->getName() + ".sub");
        Value *IdxCmp =
            Builder.CreateIsNotNull(IdxSub, NewIdx->getName() + ".cmp");
        Builder.CreateCondBr(IdxCmp, FirstLoopBB, InsertBot);
        NewIdx->addIncoming(NewIter, InsertTop);
        NewIdx->addIncoming(IdxSub, NewBB);
      }
      LatchBR->eraseFromParent();
    }
  }

  // Change the incoming values to the ones defined in the preheader or
  // cloned loop.
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *NewPHI = cast<PHINode>(VMap[&*I]);
    if (!CreateRemainderLoop) {
      if (UseEpilogRemainder) {
        unsigned idx = NewPHI->getBasicBlockIndex(Preheader);
        NewPHI->setIncomingBlock(idx, InsertTop);
        NewPHI->removeIncomingValue(Latch, false);
      } else {
        VMap[&*I] = NewPHI->getIncomingValueForBlock(Preheader);
        cast<BasicBlock>(VMap[Header])->getInstList().erase(NewPHI);
      }
    } else {
      unsigned idx = NewPHI->getBasicBlockIndex(Preheader);
      NewPHI->setIncomingBlock(idx, InsertTop);
      BasicBlock *NewLatch = cast<BasicBlock>(VMap[Latch]);
      idx = NewPHI->getBasicBlockIndex(Latch);
      Value *InVal = NewPHI->getIncomingValue(idx);
      NewPHI->setIncomingBlock(idx, NewLatch);
      if (Value *V = VMap.lookup(InVal))
        NewPHI->setIncomingValue(idx, V);
    }
  }
  if (CreateRemainderLoop) {
    Loop *NewLoop = NewLoops[L];
    assert(NewLoop && "L should have been cloned");
    // Add unroll disable metadata to disable future unrolling for this loop.
    SmallVector<Metadata *, 4> MDs;
    // Reserve first location for self reference to the LoopID metadata node.
    MDs.push_back(nullptr);
    MDNode *LoopID = NewLoop->getLoopID();
    if (LoopID) {
      // First remove any existing loop unrolling metadata.
      for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
        bool IsUnrollMetadata = false;
        MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
        if (MD) {
          const MDString *S = dyn_cast<MDString>(MD->getOperand(0));
          IsUnrollMetadata = S && S->getString().startswith("llvm.loop.unroll.");
        }
        if (!IsUnrollMetadata)
          MDs.push_back(LoopID->getOperand(i));
      }
    }

    LLVMContext &Context = NewLoop->getHeader()->getContext();
    SmallVector<Metadata *, 1> DisableOperands;
    DisableOperands.push_back(MDString::get(Context, "llvm.loop.unroll.disable"));
    MDNode *DisableNode = MDNode::get(Context, DisableOperands);
    MDs.push_back(DisableNode);

    MDNode *NewLoopID = MDNode::get(Context, MDs);
    // Set operand 0 to refer to the loop id itself.
    NewLoopID->replaceOperandWith(0, NewLoopID);
    NewLoop->setLoopID(NewLoopID);
  }
}

/// Insert code in the prolog/epilog code when unrolling a loop with a
/// run-time trip-count.
///
/// This method assumes that the loop unroll factor is total number
/// of loop bodies in the loop after unrolling. (Some folks refer
/// to the unroll factor as the number of *extra* copies added).
/// We assume also that the loop unroll factor is a power-of-two. So, after
/// unrolling the loop, the number of loop bodies executed is 2,
/// 4, 8, etc.  Note - LLVM converts the if-then-sequence to a switch
/// instruction in SimplifyCFG.cpp.  Then, the backend decides how code for
/// the switch instruction is generated.
///
/// ***Prolog case***
///        extraiters = tripcount % loopfactor
///        if (extraiters == 0) jump Loop:
///        else jump Prol:
/// Prol:  LoopBody;
///        extraiters -= 1                 // Omitted if unroll factor is 2.
///        if (extraiters != 0) jump Prol: // Omitted if unroll factor is 2.
///        if (tripcount < loopfactor) jump End:
/// Loop:
/// ...
/// End:
///
/// ***Epilog case***
///        extraiters = tripcount % loopfactor
///        if (tripcount < loopfactor) jump LoopExit:
///        unroll_iters = tripcount - extraiters
/// Loop:  LoopBody; (executes unroll_iter times);
///        unroll_iter -= 1
///        if (unroll_iter != 0) jump Loop:
/// LoopExit:
///        if (extraiters == 0) jump EpilExit:
/// Epil:  LoopBody; (executes extraiters times)
///        extraiters -= 1                 // Omitted if unroll factor is 2.
///        if (extraiters != 0) jump Epil: // Omitted if unroll factor is 2.
/// EpilExit:

bool llvm::UnrollRuntimeLoopRemainder(Loop *L, unsigned Count,
                                      bool AllowExpensiveTripCount,
                                      bool UseEpilogRemainder,
                                      LoopInfo *LI, ScalarEvolution *SE,
                                      DominatorTree *DT, bool PreserveLCSSA) {
  // for now, only unroll loops that contain a single exit
  if (!L->getExitingBlock())
    return false;

  // Make sure the loop is in canonical form, and there is a single
  // exit block only.
  if (!L->isLoopSimplifyForm())
    return false;
  BasicBlock *Exit = L->getUniqueExitBlock(); // successor out of loop
  if (!Exit)
    return false;

  // Use Scalar Evolution to compute the trip count. This allows more loops to
  // be unrolled than relying on induction var simplification.
  if (!SE)
    return false;

  // Only unroll loops with a computable trip count, and the trip count needs
  // to be an int value (allowing a pointer type is a TODO item).
  const SCEV *BECountSC = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BECountSC) ||
      !BECountSC->getType()->isIntegerTy())
    return false;

  unsigned BEWidth = cast<IntegerType>(BECountSC->getType())->getBitWidth();

  // Add 1 since the backedge count doesn't include the first loop iteration.
  const SCEV *TripCountSC =
      SE->getAddExpr(BECountSC, SE->getConstant(BECountSC->getType(), 1));
  if (isa<SCEVCouldNotCompute>(TripCountSC))
    return false;

  BasicBlock *Header = L->getHeader();
  BasicBlock *PreHeader = L->getLoopPreheader();
  BranchInst *PreHeaderBR = cast<BranchInst>(PreHeader->getTerminator());
  const DataLayout &DL = Header->getModule()->getDataLayout();
  SCEVExpander Expander(*SE, DL, "loop-unroll");
  if (!AllowExpensiveTripCount &&
      Expander.isHighCostExpansion(TripCountSC, L, PreHeaderBR))
    return false;

  // This constraint lets us deal with an overflowing trip count easily; see the
  // comment on ModVal below.
  if (Log2_32(Count) > BEWidth)
    return false;

  BasicBlock *Latch = L->getLoopLatch();

  // Loop structure is the following:
  //
  // PreHeader
  //   Header
  //   ...
  //   Latch
  // Exit

  BasicBlock *NewPreHeader;
  BasicBlock *NewExit = nullptr;
  BasicBlock *PrologExit = nullptr;
  BasicBlock *EpilogPreHeader = nullptr;
  BasicBlock *PrologPreHeader = nullptr;

  if (UseEpilogRemainder) {
    // If epilog remainder
    // Split PreHeader to insert a branch around loop for unrolling.
    NewPreHeader = SplitBlock(PreHeader, PreHeader->getTerminator(), DT, LI);
    NewPreHeader->setName(PreHeader->getName() + ".new");
    // Split Exit to create phi nodes from branch above.
    SmallVector<BasicBlock*, 4> Preds(predecessors(Exit));
    NewExit = SplitBlockPredecessors(Exit, Preds, ".unr-lcssa",
                                     DT, LI, PreserveLCSSA);
    // Split NewExit to insert epilog remainder loop.
    EpilogPreHeader = SplitBlock(NewExit, NewExit->getTerminator(), DT, LI);
    EpilogPreHeader->setName(Header->getName() + ".epil.preheader");
  } else {
    // If prolog remainder
    // Split the original preheader twice to insert prolog remainder loop
    PrologPreHeader = SplitEdge(PreHeader, Header, DT, LI);
    PrologPreHeader->setName(Header->getName() + ".prol.preheader");
    PrologExit = SplitBlock(PrologPreHeader, PrologPreHeader->getTerminator(),
                            DT, LI);
    PrologExit->setName(Header->getName() + ".prol.loopexit");
    // Split PrologExit to get NewPreHeader.
    NewPreHeader = SplitBlock(PrologExit, PrologExit->getTerminator(), DT, LI);
    NewPreHeader->setName(PreHeader->getName() + ".new");
  }
  // Loop structure should be the following:
  //  Epilog             Prolog
  //
  // PreHeader         PreHeader
  // *NewPreHeader     *PrologPreHeader
  //   Header          *PrologExit
  //   ...             *NewPreHeader
  //   Latch             Header
  // *NewExit            ...
  // *EpilogPreHeader    Latch
  // Exit              Exit

  // Calculate conditions for branch around loop for unrolling
  // in epilog case and around prolog remainder loop in prolog case.
  // Compute the number of extra iterations required, which is:
  //  extra iterations = run-time trip count % loop unroll factor
  PreHeaderBR = cast<BranchInst>(PreHeader->getTerminator());
  Value *TripCount = Expander.expandCodeFor(TripCountSC, TripCountSC->getType(),
                                            PreHeaderBR);
  Value *BECount = Expander.expandCodeFor(BECountSC, BECountSC->getType(),
                                          PreHeaderBR);
  IRBuilder<> B(PreHeaderBR);
  Value *ModVal;
  // Calculate ModVal = (BECount + 1) % Count.
  // Note that TripCount is BECount + 1.
  if (isPowerOf2_32(Count)) {
    // When Count is power of 2 we don't BECount for epilog case, however we'll
    // need it for a branch around unrolling loop for prolog case.
    ModVal = B.CreateAnd(TripCount, Count - 1, "xtraiter");
    //  1. There are no iterations to be run in the prolog/epilog loop.
    // OR
    //  2. The addition computing TripCount overflowed.
    //
    // If (2) is true, we know that TripCount really is (1 << BEWidth) and so
    // the number of iterations that remain to be run in the original loop is a
    // multiple Count == (1 << Log2(Count)) because Log2(Count) <= BEWidth (we
    // explicitly check this above).
  } else {
    // As (BECount + 1) can potentially unsigned overflow we count
    // (BECount % Count) + 1 which is overflow safe as BECount % Count < Count.
    Value *ModValTmp = B.CreateURem(BECount,
                                    ConstantInt::get(BECount->getType(),
                                                     Count));
    Value *ModValAdd = B.CreateAdd(ModValTmp,
                                   ConstantInt::get(ModValTmp->getType(), 1));
    // At that point (BECount % Count) + 1 could be equal to Count.
    // To handle this case we need to take mod by Count one more time.
    ModVal = B.CreateURem(ModValAdd,
                          ConstantInt::get(BECount->getType(), Count),
                          "xtraiter");
  }
  Value *BranchVal =
      UseEpilogRemainder ? B.CreateICmpULT(BECount,
                                           ConstantInt::get(BECount->getType(),
                                                            Count - 1)) :
                           B.CreateIsNotNull(ModVal, "lcmp.mod");
  BasicBlock *RemainderLoop = UseEpilogRemainder ? NewExit : PrologPreHeader;
  BasicBlock *UnrollingLoop = UseEpilogRemainder ? NewPreHeader : PrologExit;
  // Branch to either remainder (extra iterations) loop or unrolling loop.
  B.CreateCondBr(BranchVal, RemainderLoop, UnrollingLoop);
  PreHeaderBR->eraseFromParent();
  if (DT) {
    if (UseEpilogRemainder)
      DT->changeImmediateDominator(NewExit, PreHeader);
    else
      DT->changeImmediateDominator(PrologExit, PreHeader);
  }
  Function *F = Header->getParent();
  // Get an ordered list of blocks in the loop to help with the ordering of the
  // cloned blocks in the prolog/epilog code
  LoopBlocksDFS LoopBlocks(L);
  LoopBlocks.perform(LI);

  //
  // For each extra loop iteration, create a copy of the loop's basic blocks
  // and generate a condition that branches to the copy depending on the
  // number of 'left over' iterations.
  //
  std::vector<BasicBlock *> NewBlocks;
  ValueToValueMapTy VMap;

  // For unroll factor 2 remainder loop will have 1 iterations.
  // Do not create 1 iteration loop.
  bool CreateRemainderLoop = (Count != 2);

  // Clone all the basic blocks in the loop. If Count is 2, we don't clone
  // the loop, otherwise we create a cloned loop to execute the extra
  // iterations. This function adds the appropriate CFG connections.
  BasicBlock *InsertBot = UseEpilogRemainder ? Exit : PrologExit;
  BasicBlock *InsertTop = UseEpilogRemainder ? EpilogPreHeader : PrologPreHeader;
  CloneLoopBlocks(L, ModVal, CreateRemainderLoop, UseEpilogRemainder, InsertTop,
                  InsertBot, NewPreHeader, NewBlocks, LoopBlocks, VMap, DT, LI);

  // Insert the cloned blocks into the function.
  F->getBasicBlockList().splice(InsertBot->getIterator(),
                                F->getBasicBlockList(),
                                NewBlocks[0]->getIterator(),
                                F->end());

  // Loop structure should be the following:
  //  Epilog             Prolog
  //
  // PreHeader         PreHeader
  // NewPreHeader      PrologPreHeader
  //   Header            PrologHeader
  //   ...               ...
  //   Latch             PrologLatch
  // NewExit           PrologExit
  // EpilogPreHeader   NewPreHeader
  //   EpilogHeader      Header
  //   ...               ...
  //   EpilogLatch       Latch
  // Exit              Exit

  // Rewrite the cloned instruction operands to use the values created when the
  // clone is created.
  for (BasicBlock *BB : NewBlocks) {
    for (Instruction &I : *BB) {
      RemapInstruction(&I, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    }
  }

  if (UseEpilogRemainder) {
    // Connect the epilog code to the original loop and update the
    // PHI functions.
    ConnectEpilog(L, ModVal, NewExit, Exit, PreHeader,
                  EpilogPreHeader, NewPreHeader, VMap, DT, LI,
                  PreserveLCSSA);

    // Update counter in loop for unrolling.
    // I should be multiply of Count.
    IRBuilder<> B2(NewPreHeader->getTerminator());
    Value *TestVal = B2.CreateSub(TripCount, ModVal, "unroll_iter");
    BranchInst *LatchBR = cast<BranchInst>(Latch->getTerminator());
    B2.SetInsertPoint(LatchBR);
    PHINode *NewIdx = PHINode::Create(TestVal->getType(), 2, "niter",
                                      Header->getFirstNonPHI());
    Value *IdxSub =
        B2.CreateSub(NewIdx, ConstantInt::get(NewIdx->getType(), 1),
                     NewIdx->getName() + ".nsub");
    Value *IdxCmp;
    if (LatchBR->getSuccessor(0) == Header)
      IdxCmp = B2.CreateIsNotNull(IdxSub, NewIdx->getName() + ".ncmp");
    else
      IdxCmp = B2.CreateIsNull(IdxSub, NewIdx->getName() + ".ncmp");
    NewIdx->addIncoming(TestVal, NewPreHeader);
    NewIdx->addIncoming(IdxSub, Latch);
    LatchBR->setCondition(IdxCmp);
  } else {
    // Connect the prolog code to the original loop and update the
    // PHI functions.
    ConnectProlog(L, BECount, Count, PrologExit, PreHeader, NewPreHeader,
                  VMap, DT, LI, PreserveLCSSA);
  }

  // If this loop is nested, then the loop unroller changes the code in the
  // parent loop, so the Scalar Evolution pass needs to be run again.
  if (Loop *ParentLoop = L->getParentLoop())
    SE->forgetLoop(ParentLoop);

  NumRuntimeUnrolled++;
  return true;
}
