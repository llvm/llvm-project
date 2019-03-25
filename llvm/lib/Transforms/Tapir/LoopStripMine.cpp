//===-- LoopStripMine.cpp - Loop strip-mining utilities -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements some loop strip-mining utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/LoopStripMine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

using namespace llvm;

#define LSM_NAME "loop-stripmine"
#define DEBUG_TYPE LSM_NAME

/// Create an analysis remark that explains why stripmining failed.
///
/// \p RemarkName is the identifier for the remark.  If \p I is passed it is an
/// instruction that prevents vectorization.  Otherwise \p TheLoop is used for
/// the location of the remark.  \return the remark object that can be streamed
/// to.
static OptimizationRemarkAnalysis
createMissedAnalysis(StringRef RemarkName, const Loop *TheLoop,
                     Instruction *I = nullptr) {
  const Value *CodeRegion = TheLoop->getHeader();
  DebugLoc DL = TheLoop->getStartLoc();

  if (I) {
    CodeRegion = I->getParent();
    // If there is no debug location attached to the instruction, revert back to
    // using the loop's.
    if (I->getDebugLoc())
      DL = I->getDebugLoc();
  }

  OptimizationRemarkAnalysis R("loop-stripmine", RemarkName, DL, CodeRegion);
  R << "Tapir loop not transformed: ";
  return R;
}

/// The function chooses which type of stripmine (epilog or prolog) is more
/// profitabale.
/// Epilog stripmine is more profitable when there is PHI that starts from
/// constant.  In this case epilog will leave PHI start from constant,
/// but prolog will convert it to non-constant.
///
/// loop:
///   PN = PHI [I, Latch], [CI, Preheader]
///   I = foo(PN)
///   ...
///
/// Epilog stripmine case.
/// loop:
///   PN = PHI [I2, Latch], [CI, Preheader]
///   I1 = foo(PN)
///   I2 = foo(I1)
///   ...
/// Prolog stripmine case.
///   NewPN = PHI [PrologI, Prolog], [CI, Preheader]
/// loop:
///   PN = PHI [I2, Latch], [NewPN, Preheader]
///   I1 = foo(PN)
///   I2 = foo(I1)
///   ...
///
static bool isEpilogProfitable(const Loop *L) {
  const BasicBlock *Preheader = L->getLoopPreheader();
  const BasicBlock *Header = L->getHeader();
  assert(Preheader && Header);
  for (const PHINode &PN : Header->phis()) {
    if (isa<ConstantInt>(PN.getIncomingValueForBlock(Preheader)))
      return true;
  }
  return false;
}

/// Perform some cleanup and simplifications on loops after stripmining. It is
/// useful to simplify the IV's in the new loop, as well as do a quick
/// simplify/dce pass of the instructions.
void llvm::simplifyLoopAfterStripMine(Loop *L, bool SimplifyIVs, LoopInfo *LI,
                                      ScalarEvolution *SE, DominatorTree *DT,
                                      AssumptionCache *AC) {
  // Simplify any new induction variables in the stripmined loop.
  if (SE && SimplifyIVs) {
    SmallVector<WeakTrackingVH, 16> DeadInsts;
    simplifyLoopIVs(L, SE, DT, LI, DeadInsts);

    // Aggressively clean up dead instructions that simplifyLoopIVs already
    // identified. Any remaining should be cleaned up below.
    while (!DeadInsts.empty())
      if (Instruction *Inst =
              dyn_cast_or_null<Instruction>(&*DeadInsts.pop_back_val()))
        RecursivelyDeleteTriviallyDeadInstructions(Inst);
  }

  // At this point, the code is well formed.  We now do a quick sweep over the
  // inserted code, doing constant propagation and dead code elimination as we
  // go.
  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
  const std::vector<BasicBlock *> &NewLoopBlocks = L->getBlocks();
  for (BasicBlock *BB : NewLoopBlocks) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
      Instruction *Inst = &*I++;

      if (Value *V = SimplifyInstruction(Inst, {DL, nullptr, DT, AC}))
        if (LI->replacementPreservesLCSSAForm(Inst, V))
          Inst->replaceAllUsesWith(V);
      if (isInstructionTriviallyDead(Inst))
        BB->getInstList().erase(Inst);
    }
  }

  // TODO: after stripmining, previously loop variant conditions are likely to
  // fold to constants, eagerly propagating those here will require fewer
  // cleanup passes to be run.  Alternatively, a LoopEarlyCSE might be
  // appropriate.
}

static Task *getTapirLoopForStripMining(const Loop *L, TaskInfo &TI,
                                        OptimizationRemarkEmitter *ORE) {
  LLVM_DEBUG(dbgs() << "Analyzing for stripmining: " << *L);
  // We only handle Tapir loops.
  Task *T = getTaskIfTapirLoop(L, &TI);
  if (!T)
    return nullptr;

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    LLVM_DEBUG(dbgs()
               << "  Can't stripmine; loop preheader-insertion failed.\n");
    if (ORE)
      ORE->emit(createMissedAnalysis("NoPreheader", L)
                << "loop lacks a preheader");
    return nullptr;
  }
  BranchInst *PreheaderBR = dyn_cast<BranchInst>(Preheader->getTerminator());
  assert(PreheaderBR && "Preheader not terminated by a branch");

  BasicBlock *LatchBlock = L->getLoopLatch();
  if (!LatchBlock) {
    LLVM_DEBUG(dbgs()
               << "  Can't stripmine; loop exit-block-insertion failed.\n");
    if (ORE)
      ORE->emit(createMissedAnalysis("NoLatch", L)
                << "loop lacks a latch");
    return nullptr;
  }

  // Loops with indirectbr cannot be cloned.
  if (!L->isSafeToClone()) {
    LLVM_DEBUG(dbgs() << "  Can't stripmine; Loop body cannot be cloned.\n");
    if (ORE)
      ORE->emit(createMissedAnalysis("UnsafeToClone", L)
                << "loop is not safe to clone");
    return nullptr;
  }

  // The current loop-stripmine pass can only stripmine loops with a single
  // latch that's a conditional branch exiting the loop.
  // FIXME: The implementation can be extended to work with more complicated
  // cases, e.g. loops with multiple latches.
  BranchInst *BI = dyn_cast<BranchInst>(LatchBlock->getTerminator());

  if (!BI || BI->isUnconditional()) {
    // The loop-rotate pass can be helpful to avoid this in many cases.
    LLVM_DEBUG(
        dbgs()
        << "  Can't stripmine; loop not terminated by a conditional branch.\n");
    if (ORE)
      ORE->emit(createMissedAnalysis("NoLatchBranch", L)
                << "loop latch is not terminated by a conditional branch");
    return nullptr;
  }

  BasicBlock *Header = L->getHeader();
  auto CheckSuccessors = [&](unsigned S1, unsigned S2) {
    return BI->getSuccessor(S1) == Header && !L->contains(BI->getSuccessor(S2));
  };

  if (!CheckSuccessors(0, 1) && !CheckSuccessors(1, 0)) {
    LLVM_DEBUG(dbgs() << "Can't stripmine; only loops with one conditional"
                         "  latch exiting the loop can be stripmined.\n");
    if (ORE)
      ORE->emit(createMissedAnalysis("ComplexLatchBranch", L)
                << "loop has multiple exiting conditional latches");
    return nullptr;
  }

  if (Header->hasAddressTaken()) {
    // The loop-rotate pass can be helpful to avoid this in many cases.
    LLVM_DEBUG(
        dbgs() << "  Won't stripmine loop: address of header block is taken.\n");
    if (ORE)
      ORE->emit(createMissedAnalysis("HeaderAddressTaken", L)
                << "loop header block has address taken");
    return nullptr;
  }

  // Don't stripmine loops with the convergent attribute.
  for (auto &BB : L->blocks())
    for (auto &I : *BB)
      if (auto CS = CallSite(&I))
        if (CS.isConvergent()) {
          LLVM_DEBUG(
              dbgs() << "  Won't stripmine loop: contains convergent attribute.\n");
          if (ORE)
            ORE->emit(createMissedAnalysis("ConvergentLoop", L)
                      << "loop contains convergent attribute");
          return nullptr;
        }

  // TODO: Generalize this condition to support stripmining with a prolog.
  assert(isEpilogProfitable(L) &&
         "Stripmining loop with unprofitable epilog.");

  // Get the task for this loop.
  return T;
}

/// Connect the stripmining epilog code to the original loop.
/// The stripmining epilog code contains code to execute the
/// 'extra' iterations if the run-time trip count modulo the
/// stripmine count is non-zero.
///
/// This function performs the following:
/// - Update PHI operands in the epilog loop by the new PHI nodes
/// - Branch around the epilog loop if extra iters (ModVal) is zero.
///
static void ConnectEpilog(TapirLoopInfo &TL, Value *EpilStartIter,
                          Value *ModVal, BasicBlock *LoopDet,
                          BasicBlock *LoopEnd, BasicBlock *NewExit,
                          BasicBlock *Exit, BasicBlock *Preheader,
                          BasicBlock *EpilogPreheader, ValueToValueMapTy &VMap,
                          DominatorTree *DT, LoopInfo *LI, ScalarEvolution *SE,
                          const DataLayout &DL, bool PreserveLCSSA) {
  // NewExit should contain no PHI nodes.
#ifndef NDEBUG
  bool ContainsPHIs = false;
  for (PHINode &PN : NewExit->phis()) {
    dbgs() << "NewExit PHI node: " << PN << "\n";
    ContainsPHIs = true;
  }
  assert(!ContainsPHIs && "NewExit should not contain PHI nodes.");
#endif

  // Create PHI nodes at NewExit (from the stripmining loop Latch and
  // Preheader).  Update corresponding PHI nodes in epilog loop.
  IRBuilder<> B(EpilogPreheader->getTerminator());
  for (auto &InductionEntry : *TL.getInductionVars()) {
    // Compute the value of this induction at NewExit.
    const InductionDescriptor &II = InductionEntry.second;
    // Get the new step value for this Phi.
    Value *PhiIter = !II.getStep()->getType()->isIntegerTy()
      ? B.CreateCast(Instruction::SIToFP, EpilStartIter,
                     II.getStep()->getType())
      : B.CreateSExtOrTrunc(EpilStartIter, II.getStep()->getType());
    Value *NewPhiStart = II.transform(B, PhiIter, SE, DL);

    // Update the PHI node in the epilog loop.
    PHINode *PN = cast<PHINode>(VMap[InductionEntry.first]);
    PN->setIncomingValue(PN->getBasicBlockIndex(EpilogPreheader), NewPhiStart);
  }

  Instruction *InsertPt = NewExit->getTerminator();
  B.SetInsertPoint(InsertPt);
  Value *BrLoopExit = B.CreateIsNotNull(ModVal, "lcmp.mod");
  assert(Exit && "Loop must have a single exit block only");
  // Split the epilogue exit to maintain loop canonicalization guarantees
  SmallVector<BasicBlock*, 4> Preds(predecessors(Exit));
  SplitBlockPredecessors(Exit, Preds, ".epilog-lcssa", DT, LI,
                         PreserveLCSSA);
  // Add the branch to the exit block (around the stripmining loop)
  B.CreateCondBr(BrLoopExit, EpilogPreheader, Exit);
  InsertPt->eraseFromParent();
  if (DT)
    DT->changeImmediateDominator(Exit, NewExit);

  // Split the main loop exit to maintain canonicalization guarantees.
  SmallVector<BasicBlock*, 4> NewExitPreds{LoopDet, LoopEnd};
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
/// Return the new cloned loop that is created when CreateRemainderLoop is true.
static Loop *
CloneLoopBlocks(Loop *L, Value *NewIter, const bool CreateRemainderLoop,
                const bool UseEpilogRemainder, const bool UnrollRemainder,
                BasicBlock *InsertTop,
                BasicBlock *InsertBot, BasicBlock *Preheader,
                std::vector<BasicBlock *> &NewBlocks, LoopBlocksDFS &LoopBlocks,
                std::vector<BasicBlock *> &ExtraTaskBlocks,
                ValueToValueMapTy &VMap, DominatorTree *DT, LoopInfo *LI) {
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

    // Add the cloned block to loop info.
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
        // Copy information from original loop to the clone.
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

  // Create new copies of the EH blocks to clone.  We can handle these blocks
  // more simply than the loop blocks.
  for (BasicBlock *BB : ExtraTaskBlocks) {
    BasicBlock *NewBB = CloneBasicBlock(BB, VMap, "." + suffix, F);
    NewBlocks.push_back(NewBB);

    // Add the cloned block to loop info.
    if (LI->getLoopFor(BB))
      addClonedBlockToLoopInfo(BB, NewBB, LI, NewLoops);

    VMap[BB] = NewBB;
    if (DT) {
      // Copy information from original loop to the clone.
      BasicBlock *IDomBB = DT->getNode(BB)->getIDom()->getBlock();
      if (VMap.lookup(IDomBB))
        DT->addNewBlock(NewBB, cast<BasicBlock>(VMap[IDomBB]));
      else
        DT->addNewBlock(NewBB, cast<BasicBlock>(IDomBB));
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

    // Only add loop metadata if the loop is not going to be completely
    // unrolled.
    if (UnrollRemainder)
      return NewLoop;

    // FIXME?
    // // Add unroll disable metadata to disable future unrolling for this loop.
    // NewLoop->setLoopAlreadyUnrolled();
    return NewLoop;
  }
  else
    return nullptr;
}

Loop *llvm::StripMineLoop(
    Loop *L, unsigned Count, bool AllowExpensiveTripCount,
    bool UnrollRemainder, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT,
    AssumptionCache *AC, TaskInfo *TI, OptimizationRemarkEmitter *ORE,
    bool PreserveLCSSA) {
  Task *T = getTapirLoopForStripMining(L, *TI, ORE);
  if (!T)
    return nullptr;

  TapirLoopInfo TL(L, T);

  // TODO: Add support for loop peeling, i.e., using a prolog..

  // Use Scalar Evolution to compute the trip count. This allows more loops to
  // be stripmined than relying on induction var simplification.
  if (!SE)
    return nullptr;
  PredicatedScalarEvolution PSE(*SE, *L);

  TL.collectIVs(PSE, *ORE);

  // If no primary induction was found, just bail.
  PHINode *PrimaryInduction = TL.getPrimaryInduction().first;
  if (!PrimaryInduction) {
    LLVM_DEBUG(dbgs() << "No primary induction variable found in loop.");
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "\tPrimary induction " << *PrimaryInduction << "\n");

  Value *TripCount = TL.getOrCreateTripCount(PSE);

  // Fixup all external uses of the IVs.
  for (auto &InductionEntry : *TL.getInductionVars())
    TL.fixupIVUsers(InductionEntry.first, InductionEntry.second, PSE);

  // High-level algorithm: Generate an epilog for the Tapir loop and insert it
  // between the original latch and its exit.  Then split the entry and reattach
  // block of the loop body to build the serial inner loop.

  BasicBlock *Preheader = L->getLoopPreheader();
  BranchInst *PreheaderBR = cast<BranchInst>(Preheader->getTerminator());
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *Header = L->getHeader();
  BasicBlock *TaskEntry = T->getEntry();
  assert(isa<DetachInst>(Header->getTerminator()) &&
         "Header not terminated by a detach.");
  DetachInst *DI = cast<DetachInst>(Header->getTerminator());
  assert(DI->getDetached() == TaskEntry &&
         "Task entry does not match block detached from header.");
  BasicBlock *ParentEntry = T->getParentTask()->getEntry();
  BranchInst *LatchBR = cast<BranchInst>(Latch->getTerminator());
  unsigned ExitIndex = LatchBR->getSuccessor(0) == Header ? 1 : 0;
  BasicBlock *LatchExit = LatchBR->getSuccessor(ExitIndex);

  // We will use the increment of the primary induction variable to derive
  // wrapping flags.
  Instruction *PrimaryInc =
    cast<Instruction>(PrimaryInduction->getIncomingValueForBlock(Latch));

  // Get all uses of the primary induction variable in the task.
  SmallVector<Use *, 4> PrimaryInductionUsesInTask;
  for (Use &U : PrimaryInduction->uses())
    if (Instruction *User = dyn_cast<Instruction>(U.getUser()))
      if (T->encloses(User->getParent()))
        PrimaryInductionUsesInTask.push_back(&U);

  // Only stripmine loops with a computable trip count, and the trip count needs
  // to be an int value (allowing a pointer type is a TODO item).
  // We calculate the backedge count by using getExitCount on the Latch block,
  // which is proven to be the only exiting block in this loop. This is same as
  // calculating getBackedgeTakenCount on the loop (which computes SCEV for all
  // exiting blocks).
  const SCEV *BECountSC = TL.getBackedgeTakenCount(PSE);
  if (isa<SCEVCouldNotCompute>(BECountSC) ||
      !BECountSC->getType()->isIntegerTy()) {
    LLVM_DEBUG(dbgs() << "Could not compute exit block SCEV\n");
    return nullptr;
  }

  unsigned BEWidth =
    cast<IntegerType>(TL.getWidestInductionType())->getBitWidth();

  // Add 1 since the backedge count doesn't include the first loop iteration.
  const SCEV *TripCountSC = TL.getExitCount(BECountSC, PSE);
  if (isa<SCEVCouldNotCompute>(TripCountSC)) {
    LLVM_DEBUG(dbgs() << "Could not compute trip count SCEV.\n");
    return nullptr;
  }

  const DataLayout &DL = Header->getModule()->getDataLayout();
  SCEVExpander Expander(*SE, DL, "loop-stripmine");
  if (!AllowExpensiveTripCount &&
      Expander.isHighCostExpansion(TripCountSC, L, PreheaderBR)) {
    LLVM_DEBUG(dbgs() << "High cost for expanding trip count scev!\n");
    return nullptr;
  }

  // This constraint lets us deal with an overflowing trip count easily; see the
  // comment on ModVal below.
  if (Log2_32(Count) > BEWidth) {
    LLVM_DEBUG(
        dbgs()
        << "Count failed constraint on overflow trip count calculation.\n");
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "Stripmining loop using grainsize " << Count << "\n");
  using namespace ore;
  ORE->emit([&]() {
              return OptimizationRemark(LSM_NAME, "Stripmined",
                                        L->getStartLoc(), L->getHeader())
                << "stripmined loop using count "
                << NV("StripMineCount", Count);
            });

  // Loop structure is the following:
  //
  // Preheader
  //   Header
  //   ...
  //   Latch
  // LatchExit

  // Insert the epilog remainder.
  BasicBlock *NewPreheader;
  BasicBlock *NewExit = nullptr;
  BasicBlock *EpilogPreheader = nullptr;
  {
    // Split Preheader to insert a branch around loop for stripmining.
    NewPreheader = SplitBlock(Preheader, Preheader->getTerminator(), DT, LI);
    NewPreheader->setName(Preheader->getName() + ".new");
    // Split LatchExit to create phi nodes from branch above.
    SmallVector<BasicBlock*, 4> Preds(predecessors(LatchExit));
    NewExit = SplitBlockPredecessors(LatchExit, Preds, ".strpm-lcssa",
                                     DT, LI, PreserveLCSSA);
    // NewExit gets its DebugLoc from LatchExit, which is not part of the
    // original Loop.
    // Fix this by setting Loop's DebugLoc to NewExit.
    auto *NewExitTerminator = NewExit->getTerminator();
    NewExitTerminator->setDebugLoc(Header->getTerminator()->getDebugLoc());
    // Split NewExit to insert epilog remainder loop.
    EpilogPreheader = SplitBlock(NewExit, NewExitTerminator, DT, LI);
    EpilogPreheader->setName(Header->getName() + ".epil.preheader");
  }

  // Calculate conditions for branch around loop for stripmining
  // in epilog case and around prolog remainder loop in prolog case.
  // Compute the number of extra iterations required, which is:
  //  extra iterations = run-time trip count % loop stripmine factor
  PreheaderBR = cast<BranchInst>(Preheader->getTerminator());
  Value *BECount = Expander.expandCodeFor(BECountSC, BECountSC->getType(),
                                          PreheaderBR);

  // Loop structure should be the following:
  //  Epilog
  //
  // Preheader
  // *NewPreheader
  //   Header
  //   ...
  //   Latch
  // *NewExit
  // *EpilogPreheader
  // LatchExit

  IRBuilder<> B(PreheaderBR);
  Value *ModVal;
  // Calculate ModVal = (BECount + 1) % Count.
  // Note that TripCount is BECount + 1.
  if (isPowerOf2_32(Count)) {
    // When Count is power of 2 we don't BECount for epilog case.  However we'll
    // need it for a branch around stripmined loop for prolog case.
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
  Value *BranchVal = B.CreateICmpULT(BECount,
                                     ConstantInt::get(BECount->getType(),
                                                      Count - 1));
  BasicBlock *RemainderLoop = NewExit;
  BasicBlock *StripminedLoop = NewPreheader;
  // Branch to either remainder (extra iterations) loop or stripmined loop.
  B.CreateCondBr(BranchVal, RemainderLoop, StripminedLoop);
  PreheaderBR->eraseFromParent();
  if (DT) {
    // if (UseEpilogRemainder)
      DT->changeImmediateDominator(NewExit, Preheader);
    // else
    //   DT->changeImmediateDominator(PrologExit, Preheader);
  }
  Function *F = Header->getParent();
  // Get an ordered list of blocks in the loop to help with the ordering of the
  // cloned blocks in the prolog/epilog code
  LoopBlocksDFS LoopBlocks(L);
  LoopBlocks.perform(LI);

  std::vector<BasicBlock *> ExtraTaskBlocks;
  for (Task *SubT : depth_first(T))
    for (Spindle *S : depth_first<InTask<Spindle *>>(SubT->getEntrySpindle()))
      for (BasicBlock *BB : S->blocks())
        // Skip blocks in the loop.
        if (!L->contains(BB))
          ExtraTaskBlocks.push_back(BB);

  SmallVector<Instruction *, 1> Reattaches;
  SmallVector<BasicBlock *, 4> EHBlocksToClone;
  SmallPtrSet<BasicBlock *, 4> EHBlockPreds;
  SmallPtrSet<LandingPadInst *, 1> InlinedLPads;
  SmallVector<Instruction *, 1> DetachedRethrows;
  // Analyze the original task for serialization.
  AnalyzeTaskForSerialization(T, Reattaches, EHBlocksToClone, EHBlockPreds,
                              InlinedLPads, DetachedRethrows);

  // For each extra loop iteration, create a copy of the loop's basic blocks
  // and generate a condition that branches to the copy depending on the
  // number of 'left over' iterations.
  //
  std::vector<BasicBlock *> NewBlocks;
  ValueToValueMapTy VMap;

  // TODO: For stripmine factor 2 remainder loop will have 1 iterations.
  // Do not create 1 iteration loop.
  // bool CreateRemainderLoop = (Count != 2);
  bool CreateRemainderLoop = true;

  // Clone all the basic blocks in the loop. If Count is 2, we don't clone
  // the loop, otherwise we create a cloned loop to execute the extra
  // iterations. This function adds the appropriate CFG connections.
  BasicBlock *InsertBot = LatchExit;
  BasicBlock *InsertTop = EpilogPreheader;
  Loop *remainderLoop = CloneLoopBlocks(
      L, ModVal, CreateRemainderLoop, true, UnrollRemainder,
      InsertTop, InsertBot,
      NewPreheader, NewBlocks, LoopBlocks, ExtraTaskBlocks, VMap, DT, LI);

  // Insert the cloned blocks into the function.
  F->getBasicBlockList().splice(InsertBot->getIterator(),
                                F->getBasicBlockList(),
                                NewBlocks[0]->getIterator(),
                                F->end());

  // Loop structure should be the following:
  //  Epilog
  //
  // Preheader
  // NewPreheader
  //   Header
  //   ...
  //   Latch
  // NewExit
  // EpilogPreheader
  //   EpilogHeader
  //   ...
  //   EpilogLatch
  // LatchExit

  // Rewrite the cloned instruction operands to use the values created when the
  // clone is created.
  for (BasicBlock *BB : NewBlocks)
    for (Instruction &I : *BB)
      RemapInstruction(&I, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

  // Serialize the cloned loop body to render the inner loop serial.
  {
    // Translate all the analysis for the new cloned task.
    SmallVector<Instruction *, 1> ClonedReattaches;
    for (Instruction *I : Reattaches)
      ClonedReattaches.push_back(cast<Instruction>(VMap[I]));
    SmallPtrSet<BasicBlock *, 4> ClonedEHBlockPreds;
    for (BasicBlock *B : EHBlockPreds)
      ClonedEHBlockPreds.insert(cast<BasicBlock>(VMap[B]));
    // Landing pads and detached-rethrow instructions may or may not have been
    // cloned.
    SmallPtrSet<LandingPadInst *, 1> ClonedInlinedLPads;
    for (LandingPadInst *LPad : InlinedLPads) {
      if (VMap[LPad])
        ClonedInlinedLPads.insert(cast<LandingPadInst>(VMap[LPad]));
      else
        ClonedInlinedLPads.insert(LPad);
    }
    SmallVector<Instruction *, 1> ClonedDetachedRethrows;
    for (Instruction *DR : DetachedRethrows) {
      if (VMap[DR])
        ClonedDetachedRethrows.push_back(cast<Instruction>(VMap[DR]));
      else
        ClonedDetachedRethrows.push_back(DR);
    }
    DetachInst *ClonedDI = cast<DetachInst>(VMap[DI]);

    // Serialize the new task.
    SerializeDetach(ClonedDI, ParentEntry, ClonedReattaches,
                    &EHBlocksToClone, &ClonedEHBlockPreds, &ClonedInlinedLPads,
                    &ClonedDetachedRethrows, DT);
  }

  // Detach the stripmined loop.
  Value *SyncReg = DI->getSyncRegion(), *NewSyncReg;
  Module *M = F->getParent();
  BasicBlock *LoopDetach = SplitBlock(NewPreheader,
                                      NewPreheader->getTerminator(), DT, LI);
  LoopDetach->setName(NewPreheader->getName() + ".strpm.detachloop");
  BasicBlock *LoopDetEntry;
  {
    SmallVector<BasicBlock*, 4> HeaderPreds;
    for (BasicBlock *Pred : predecessors(Header))
      if (Pred != Latch)
        HeaderPreds.push_back(Pred);
    LoopDetEntry =
      SplitBlockPredecessors(Header, HeaderPreds, ".strpm.detachloop.entry", DT,
                             LI, PreserveLCSSA);
    NewSyncReg = CallInst::Create(
        Intrinsic::getDeclaration(M, Intrinsic::syncregion_start), {},
        &*LoopDetEntry->getFirstInsertionPt());
    NewSyncReg->setName(SyncReg->getName() + ".strpm.detachloop");
  }
  BasicBlock *LoopReattach = SplitEdge(Latch, NewExit, DT, LI);
  LoopReattach->setName(Header->getName() + ".strpm.detachloop.reattach");

  // Insert new detach instructions
  if (DI->hasUnwindDest()) {
    BasicBlock *UnwindDest = DI->getUnwindDest();
    // Insert a detach instruction to detach the stripmined loop.
    ReplaceInstWithInst(LoopDetach->getTerminator(),
                        DetachInst::Create(LoopDetEntry, NewExit, UnwindDest,
                                           SyncReg));
    for (PHINode &PN : UnwindDest->phis())
      PN.addIncoming(PN.getIncomingValueForBlock(Header), LoopDetach);

    // Split the unwind dest to create a new landing pad for the new detach.
    SmallVector<BasicBlock *, 4> UnwindDestPreds;
    for (BasicBlock *Pred : predecessors(UnwindDest))
      if (Pred != LoopDetach)
        UnwindDestPreds.push_back(Pred);
    SmallVector<BasicBlock *, 2> NewUnwinds;
    SplitLandingPadPredecessors(UnwindDest, UnwindDestPreds, "", ".strpm",
                                NewUnwinds, DT, LI, PreserveLCSSA);
    BasicBlock *OrigUW = NewUnwinds[0], *NewUW = NewUnwinds[1];
    BasicBlock *NewUnreachable =
      SplitBlock(OrigUW, OrigUW->getTerminator(), DT, LI);
    NewUnreachable->setName(OrigUW->getName() + ".unreachable");
    // Add a detached rethrow to the end of OrigUW.
    ReplaceInstWithInst(
        OrigUW->getTerminator(),
        InvokeInst::Create(
            Intrinsic::getDeclaration(
                M, Intrinsic::detached_rethrow,
                { OrigUW->getLandingPadInst()->getType() }),
            NewUnreachable, NewUW, { SyncReg, OrigUW->getLandingPadInst() }));
    // Replace sync regions of existing detached-rethrows.
    for (Instruction *I : DetachedRethrows) {
      InvokeInst *II = cast<InvokeInst>(I);
      II->setArgOperand(0, NewSyncReg);
    }

    // Remove uses of NewUnreachable from PHI nodes.
    for (PHINode &PN : UnwindDest->phis())
      PN.removeIncomingValue(NewUnreachable);
    // Terminate NewUnreachable with an unreachable.
    {
      IRBuilder<> B(NewUnreachable->getTerminator());
      Instruction *UnreachableTerm = cast<Instruction>(B.CreateUnreachable());
      UnreachableTerm->setDebugLoc(
          NewUnreachable->getTerminator()->getDebugLoc());
      NewUnreachable->getTerminator()->eraseFromParent();
    }
    // Update PHI nodes in NewUW to get the same value via OrigUW from the new
    // detach.
    for (PHINode &PN : NewUW->phis())
      PN.addIncoming(PN.getIncomingValueForBlock(LoopDetach), OrigUW);
    // Update the dominator tree
    DT->changeImmediateDominator(UnwindDest, NewUW);
  } else {
    // Insert a detach instruction to detach the stripmined loop.
    ReplaceInstWithInst(LoopDetach->getTerminator(),
                        DetachInst::Create(LoopDetEntry, NewExit, SyncReg));
  }
  // Insert a reattach instruction after the detached stripmined loop.
  ReplaceInstWithInst(LoopReattach->getTerminator(),
                      ReattachInst::Create(NewExit, SyncReg));

  // Get the set of new loop blocks
  SetVector<BasicBlock *> NewLoopBlocks;
  {
    LoopBlocksDFS NewLoopBlocksDFS(L);
    NewLoopBlocksDFS.perform(LI);
    LoopBlocksDFS::RPOIterator BlockBegin = NewLoopBlocksDFS.beginRPO();
    LoopBlocksDFS::RPOIterator BlockEnd = NewLoopBlocksDFS.endRPO();
    for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB)
      NewLoopBlocks.insert(*BB);
  }
  // Create structure in LI for new loop.
  Loop *ParentLoop = L->getParentLoop();
  Loop *NewLoop = LI->AllocateLoop();
  if (ParentLoop)
    ParentLoop->replaceChildLoopWith(L, NewLoop);
  else
    LI->changeTopLevelLoop(L, NewLoop);
  NewLoop->addChildLoop(L);

  // Move the detach/reattach instructions to surround the stripmined loop.
  BasicBlock *NewHeader; {
    SmallVector<BasicBlock*, 4> HeaderPreds;
    for (BasicBlock *Pred : predecessors(Header))
      if (Pred != Latch)
        HeaderPreds.push_back(Pred);
    NewHeader =
      SplitBlockPredecessors(Header, HeaderPreds, ".strpm.outer",
                             DT, LI, PreserveLCSSA);
  }
  BasicBlock *NewEntry =
    SplitBlock(NewHeader, NewHeader->getTerminator(), DT, LI);
  NewEntry->setName(TaskEntry->getName() + ".strpm.outer");
  SmallVector<BasicBlock *, 1> LoopReattachPreds{Latch};
  BasicBlock *NewReattB =
    SplitBlockPredecessors(LoopReattach, LoopReattachPreds, "", DT, LI,
                           PreserveLCSSA);
  NewReattB->setName(Latch->getName() + ".reattach");
  BasicBlock *NewLatch =
    SplitBlock(NewReattB, NewReattB->getTerminator(), DT, LI);
  NewLatch->setName(Latch->getName() + ".strpm.outer");

  // Insert a new detach instruction
  if (DI->hasUnwindDest()) {
    ReplaceInstWithInst(NewHeader->getTerminator(),
                        DetachInst::Create(NewEntry, NewLatch,
                                           DI->getUnwindDest(), NewSyncReg));
    // Update the PHI nodes in the unwind destination of the detach.
    for (PHINode &PN : DI->getUnwindDest()->phis())
      PN.setIncomingBlock(PN.getBasicBlockIndex(Header), NewHeader);

    // Update DT
    DT->changeImmediateDominator(DI->getUnwindDest(), NewHeader);
  } else
    ReplaceInstWithInst(NewHeader->getTerminator(),
                        DetachInst::Create(NewEntry, NewLatch, NewSyncReg));
  // Replace the old detach instruction with a branch
  ReplaceInstWithInst(Header->getTerminator(),
                      BranchInst::Create(DI->getDetached()));

  // Replace the old reattach instructions with branches.  Along the way,
  // determine their common dominator.
  BasicBlock *ReattachDom = nullptr;
  for (Instruction *I : Reattaches) {
    if (!ReattachDom)
      ReattachDom = I->getParent();
    else
      ReattachDom = DT->findNearestCommonDominator(ReattachDom, I->getParent());
    ReplaceInstWithInst(I, BranchInst::Create(Latch));
  }
  // Insert a reattach at the end of NewReattB.
  ReplaceInstWithInst(NewReattB->getTerminator(),
                      ReattachInst::Create(NewLatch, NewSyncReg));
  // Update the dominator tree.
  if (DT->dominates(Header, Latch))
    DT->changeImmediateDominator(Latch, ReattachDom);
  DT->changeImmediateDominator(LoopReattach, NewLatch);

  // The block structure of the stripmined loop should now look like so:
  //
  // LoopDetEntry
  // NewHeader (detach NewEntry, NewLatch)
  // NewEntry
  // Header
  // TaskEntry
  // ...
  // Latch (br Header, NewReattB)
  // NewReattB (reattach NewLatch)
  // NewLatch (br NewHeader, LatchExit)
  // LoopReattach

  // Add check of stripmined loop count.
  IRBuilder<> B2(LoopDetEntry->getTerminator());

  // We compute the loop count of the outer loop using a UDiv by the power-of-2
  // count to ensure that ScalarEvolution can handle this outer loop once we're
  // done.
  //
  // TODO: Generalize to handle non-power-of-2 counts.
  assert(isPowerOf2_32(Count) && "Count is not a power of 2.");
  Value *TestVal = B2.CreateUDiv(TripCount,
                                 ConstantInt::get(TripCount->getType(), Count),
                                 "stripiter");
  // Value *TestVal = B2.CreateSub(TripCount, ModVal, "stripiter", true, true);

  // Value *TestCmp = B2.CreateICmpUGT(TestVal,
  //                                   ConstantInt::get(TestVal->getType(), 0),
  //                                   TestVal->getName() + ".ncmp");
  // ReplaceInstWithInst(NewPreheader->getTerminator(),
  //                     BranchInst::Create(Header, LatchExit, TestCmp));
  // DT->changeImmediateDominator(LatchExit,
  //                              DT->findNearestCommonDominator(LatchExit,
  //                                                             NewPreheader));

  // Add new counter for new outer loop.
  //
  // We introduce a new primary induction variable, NewIdx, into the outer loop,
  // which counts up to the outer-loop trip count from 0, stepping by 1.  In
  // contrast to counting down from the outer-loop trip count, this new variable
  // ensures that future loop passes, including LoopSpawning, can process this
  // outer loop when we're done.
  PHINode *NewIdx = PHINode::Create(TestVal->getType(), 2, "niter",
                                    NewHeader->getFirstNonPHI());
  B2.SetInsertPoint(NewLatch->getTerminator());
  // Instruction *IdxSub = cast<Instruction>(
  //     B2.CreateSub(NewIdx, ConstantInt::get(NewIdx->getType(), 1),
  //                  NewIdx->getName() + ".nsub"));
  // IdxSub->copyIRFlags(PrimaryInc);
  Instruction *IdxAdd = cast<Instruction>(
      B2.CreateAdd(NewIdx, ConstantInt::get(NewIdx->getType(), 1),
                   NewIdx->getName() + ".nadd"));
  IdxAdd->copyIRFlags(PrimaryInc);

  // NewIdx->addIncoming(TestVal, NewPreheader);
  // NewIdx->addIncoming(IdxSub, NewLatch);
  // Value *IdxCmp = B2.CreateIsNull(IdxSub, NewIdx->getName() + ".ncmp");
  NewIdx->addIncoming(ConstantInt::get(TestVal->getType(), 0), LoopDetEntry);
  NewIdx->addIncoming(IdxAdd, NewLatch);
  Value *IdxCmp = B2.CreateICmpEQ(IdxAdd, TestVal,
                                  NewIdx->getName() + ".ncmp");
  ReplaceInstWithInst(NewLatch->getTerminator(),
                      BranchInst::Create(LoopReattach, NewHeader, IdxCmp));
  DT->changeImmediateDominator(NewLatch, NewHeader);

  // Fixup the LoopInfo for the new loop.
  if (!ParentLoop) {
    NewLoop->addBasicBlockToLoop(NewHeader, *LI);
    NewLoop->addBasicBlockToLoop(NewEntry, *LI);
    for (BasicBlock *BB : NewLoopBlocks) {
      NewLoop->addBlockEntry(BB);
    }
    NewLoop->addBasicBlockToLoop(NewReattB, *LI);
    NewLoop->addBasicBlockToLoop(NewLatch, *LI);
  } else {
    LI->changeLoopFor(NewHeader, NewLoop);
    NewLoop->addBlockEntry(NewHeader);
    LI->changeLoopFor(NewEntry, NewLoop);
    NewLoop->addBlockEntry(NewEntry);
    for (BasicBlock *BB : NewLoopBlocks)
      NewLoop->addBlockEntry(BB);
    LI->changeLoopFor(NewReattB, NewLoop);
    NewLoop->addBlockEntry(NewReattB);
    LI->changeLoopFor(NewLatch, NewLoop);
    NewLoop->addBlockEntry(NewLatch);
  }
  // Update loop metadata
  NewLoop->setLoopID(L->getLoopID());
  TapirLoopHints Hints(L);
  Hints.clearHintsMetadata();

  // Update all of the old PHI nodes
  B2.SetInsertPoint(NewEntry->getTerminator());
  Instruction *CountVal = cast<Instruction>(
      B2.CreateMul(ConstantInt::get(PrimaryInduction->getType(), Count),
                   NewIdx));
  CountVal->copyIRFlags(PrimaryInduction);
  for (auto &InductionEntry : *TL.getInductionVars()) {
    PHINode *OrigPhi = InductionEntry.first;
    const InductionDescriptor &II = InductionEntry.second;
    if (II.getStep()->isZero())
      // Nothing to do for this Phi
      continue;
    // Get the new step value for this Phi.
    Value *PhiCount = !II.getStep()->getType()->isIntegerTy()
      ? B2.CreateCast(Instruction::SIToFP, CountVal,
                      II.getStep()->getType())
      : B2.CreateSExtOrTrunc(CountVal, II.getStep()->getType());
    Value *NewStart = II.transform(B2, PhiCount, SE, DL);

    // Get the old increment instruction for this Phi
    int Idx = OrigPhi->getBasicBlockIndex(NewEntry);
    OrigPhi->setIncomingValue(Idx, NewStart);
  }

  // Add new induction variable for inner loop.
  PHINode *InnerIdx = PHINode::Create(PrimaryInduction->getType(), 2,
                                      "inneriter",
                                      Header->getFirstNonPHI());
  Value *InnerTestVal = ConstantInt::get(PrimaryInduction->getType(), Count);
  B2.SetInsertPoint(LatchBR);
  Instruction *InnerSub = cast<Instruction>(
      B2.CreateSub(InnerIdx, ConstantInt::get(InnerIdx->getType(), 1),
                   InnerIdx->getName() + ".nsub"));
  InnerSub->copyIRFlags(PrimaryInc);
  // Instruction *InnerAdd = cast<Instruction>(
  //     B2.CreateAdd(InnerIdx, ConstantInt::get(InnerIdx->getType(), 1),
  //                  InnerIdx->getName() + ".nadd"));
  // InnerAdd->copyIRFlags(PrimaryInc);
  Value *InnerCmp;
  if (LatchBR->getSuccessor(0) == Header)
    InnerCmp = B2.CreateIsNotNull(InnerSub, InnerIdx->getName() + ".ncmp");
  else
    InnerCmp = B2.CreateIsNull(InnerSub, InnerIdx->getName() + ".ncmp");
  InnerIdx->addIncoming(InnerTestVal, NewEntry);
  InnerIdx->addIncoming(InnerSub, Latch);
  // if (LatchBR->getSuccessor(0) == Header)
  //   InnerCmp = B2.CreateICmpNE(InnerAdd, InnerTestVal,
  //                              InnerIdx->getName() + ".ncmp");
  // else
  //   InnerCmp = B2.CreateICmpEQ(InnerAdd, InnerTestVal,
  //                              InnerIdx->getName() + ".ncmp");
  // InnerIdx->addIncoming(ConstantInt::get(InnerIdx->getType(), 0), NewEntry);
  // InnerIdx->addIncoming(InnerAdd, Latch);
  LatchBR->setCondition(InnerCmp);

  // Connect the epilog code to the original loop and update the PHI functions.
  B2.SetInsertPoint(EpilogPreheader->getTerminator());
  Instruction *EpilStartIter = cast<Instruction>(
      B2.CreateSub(TripCount, ModVal));
  EpilStartIter->copyIRFlags(PrimaryInc);
  ConnectEpilog(TL, EpilStartIter, ModVal, LoopDetach, LoopReattach, NewExit,
                LatchExit, Preheader, EpilogPreheader, VMap, DT, LI, SE, DL,
                PreserveLCSSA);

  // If this loop is nested, then the loop stripminer changes the code in the
  // any of its parent loops, so the Scalar Evolution pass needs to be run
  // again.
  SE->forgetTopmostLoop(L);

  // FIXME: Optionally unroll remainder loop
  //
  // if (remainderLoop && UnrollRemainder) {
  //   LLVM_DEBUG(dbgs() << "Unrolling remainder loop\n");
  //   UnrollLoop(remainderLoop, /*Count*/ Count - 1, /*TripCount*/ Count - 1,
  //              /*Force*/ false, /*AllowRuntime*/ false,
  //              /*AllowExpensiveTripCount*/ false, /*PreserveCondBr*/ true,
  //              /*PreserveOnlyFirst*/ false, /*TripMultiple*/ 1,
  //              /*PeelCount*/ 0, /*UnrollRemainder*/ false, LI, SE, DT, AC,
  //              /*TI*/ nullptr, /*ORE*/ nullptr, /*PreserveLCSSA*/ true);
  // }

  // At this point, the code is well formed.  We now simplify the new loops,
  // doing constant propagation and dead code elimination as we go.
  simplifyLoopAfterStripMine(L, /*SimplifyIVs*/true, LI, SE, DT, AC);
  simplifyLoopAfterStripMine(remainderLoop, /*SimplifyIVs*/true, LI, SE, DT,
                             AC);

#ifndef NDEBUG
  DT->verify();
  LI->verify(*DT);
#endif

  // Update TaskInfo manually using the updated DT.
  if (TI)
    // FIXME: Recalculating TaskInfo for the whole function is wasteful.
    // Optimize this routine in the future.
    TI->recalculate(*Header->getParent(), *DT);

  return NewLoop;
}

/// Given an loop id metadata node, returns the loop hint metadata node with the
/// given name (for example, "tapir.loop.stripmine.count"). If no such metadata
/// node exists, then nullptr is returned.
MDNode *llvm::GetStripMineMetadata(MDNode *LoopID, StringRef Name) {
  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (!MD)
      continue;

    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;

    if (Name.equals(S->getString()))
      return MD;
  }
  return nullptr;
}
