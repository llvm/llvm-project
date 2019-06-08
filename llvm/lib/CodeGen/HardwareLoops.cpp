//===-- HardwareLoops.cpp - Target Independent Hardware Loops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Insert hardware loop intrinsics into loops which are deemed profitable by
/// the target, by querying TargetTransformInfo. A hardware loop comprises of
/// two intrinsics: one, outside the loop, to set the loop iteration count and
/// another, in the exit block, to decrement the counter. The decremented value
/// can either be carried through the loop via a phi or handled in some opaque
/// way by the target.
///
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/PassSupport.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "hardware-loops"

#define HW_LOOPS_NAME "Hardware Loop Insertion"

using namespace llvm;

static cl::opt<bool>
ForceHardwareLoops("force-hardware-loops", cl::Hidden, cl::init(false),
                   cl::desc("Force hardware loops intrinsics to be inserted"));

static cl::opt<bool>
ForceHardwareLoopPHI(
  "force-hardware-loop-phi", cl::Hidden, cl::init(false),
  cl::desc("Force hardware loop counter to be updated through a phi"));

static cl::opt<bool>
ForceNestedLoop("force-nested-hardware-loop", cl::Hidden, cl::init(false),
                cl::desc("Force allowance of nested hardware loops"));

static cl::opt<unsigned>
LoopDecrement("hardware-loop-decrement", cl::Hidden, cl::init(1),
            cl::desc("Set the loop decrement value"));

static cl::opt<unsigned>
CounterBitWidth("hardware-loop-counter-bitwidth", cl::Hidden, cl::init(32),
                cl::desc("Set the loop counter bitwidth"));

STATISTIC(NumHWLoops, "Number of loops converted to hardware loops");

namespace {

  using TTI = TargetTransformInfo;

  class HardwareLoops : public FunctionPass {
  public:
    static char ID;

    HardwareLoops() : FunctionPass(ID) {
      initializeHardwareLoopsPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
      AU.addRequired<AssumptionCacheTracker>();
      AU.addRequired<TargetTransformInfoWrapperPass>();
    }

    // Try to convert the given Loop into a hardware loop.
    bool TryConvertLoop(Loop *L);

    // Given that the target believes the loop to be profitable, try to
    // convert it.
    bool TryConvertLoop(TTI::HardwareLoopInfo &HWLoopInfo);

  private:
    ScalarEvolution *SE = nullptr;
    LoopInfo *LI = nullptr;
    const DataLayout *DL = nullptr;
    const TargetTransformInfo *TTI = nullptr;
    DominatorTree *DT = nullptr;
    bool PreserveLCSSA = false;
    AssumptionCache *AC = nullptr;
    TargetLibraryInfo *LibInfo = nullptr;
    Module *M = nullptr;
    bool MadeChange = false;
  };

  class HardwareLoop {
    // Expand the trip count scev into a value that we can use.
    Value *InitLoopCount(BasicBlock *BB);

    // Insert the set_loop_iteration intrinsic.
    void InsertIterationSetup(Value *LoopCountInit, BasicBlock *BB);

    // Insert the loop_decrement intrinsic.
    void InsertLoopDec();

    // Insert the loop_decrement_reg intrinsic.
    Instruction *InsertLoopRegDec(Value *EltsRem);

    // If the target requires the counter value to be updated in the loop,
    // insert a phi to hold the value. The intended purpose is for use by
    // loop_decrement_reg.
    PHINode *InsertPHICounter(Value *NumElts, Value *EltsRem);

    // Create a new cmp, that checks the returned value of loop_decrement*,
    // and update the exit branch to use it.
    void UpdateBranch(Value *EltsRem);

  public:
    HardwareLoop(TTI::HardwareLoopInfo &Info, ScalarEvolution &SE,
                 const DataLayout &DL) :
      SE(SE), DL(DL), L(Info.L), M(L->getHeader()->getModule()),
      ExitCount(Info.ExitCount),
      CountType(Info.CountType),
      ExitBranch(Info.ExitBranch),
      LoopDecrement(Info.LoopDecrement),
      UsePHICounter(Info.CounterInReg) { }

    void Create();

  private:
    ScalarEvolution &SE;
    const DataLayout &DL;
    Loop *L                 = nullptr;
    Module *M               = nullptr;
    const SCEV *ExitCount   = nullptr;
    Type *CountType         = nullptr;
    BranchInst *ExitBranch  = nullptr;
    Value *LoopDecrement      = nullptr;
    bool UsePHICounter      = false;
  };
}

char HardwareLoops::ID = 0;

bool HardwareLoops::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  LLVM_DEBUG(dbgs() << "HWLoops: Running on " << F.getName() << "\n");

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  DL = &F.getParent()->getDataLayout();
  auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
  LibInfo = TLIP ? &TLIP->getTLI() : nullptr;
  PreserveLCSSA = mustPreserveAnalysisID(LCSSAID);
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  M = F.getParent();

  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I) {
    Loop *L = *I;
    if (!L->getParentLoop())
      TryConvertLoop(L);
  }

  return MadeChange;
}

// Return true if the search should stop, which will be when an inner loop is
// converted and the parent loop doesn't support containing a hardware loop.
bool HardwareLoops::TryConvertLoop(Loop *L) {
  // Process nested loops first.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    if (TryConvertLoop(*I))
      return true; // Stop search.

  // Bail out if the loop has irreducible control flow.
  LoopBlocksRPO RPOT(L);
  RPOT.perform(LI);
  if (containsIrreducibleCFG<const BasicBlock *>(RPOT, *LI))
    return false;

  TTI::HardwareLoopInfo HWLoopInfo(L);
  if (TTI->isHardwareLoopProfitable(L, *SE, *AC, LibInfo, HWLoopInfo) ||
      ForceHardwareLoops) {

    // Allow overriding of the counter width and loop decrement value.
    if (CounterBitWidth.getNumOccurrences())
      HWLoopInfo.CountType =
        IntegerType::get(M->getContext(), CounterBitWidth);

    if (LoopDecrement.getNumOccurrences())
      HWLoopInfo.LoopDecrement =
        ConstantInt::get(HWLoopInfo.CountType, LoopDecrement);

    MadeChange |= TryConvertLoop(HWLoopInfo);
    return MadeChange && (!HWLoopInfo.IsNestingLegal && !ForceNestedLoop);
  }

  return false;
}

bool HardwareLoops::TryConvertLoop(TTI::HardwareLoopInfo &HWLoopInfo) {

  Loop *L = HWLoopInfo.L;
  LLVM_DEBUG(dbgs() << "HWLoops: Try to convert profitable loop: " << *L);

  SmallVector<BasicBlock*, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  for (SmallVectorImpl<BasicBlock *>::iterator I = ExitingBlocks.begin(),
       IE = ExitingBlocks.end(); I != IE; ++I) {
    const SCEV *EC = SE->getExitCount(L, *I);
    if (isa<SCEVCouldNotCompute>(EC))
      continue;
    if (const SCEVConstant *ConstEC = dyn_cast<SCEVConstant>(EC)) {
      if (ConstEC->getValue()->isZero())
        continue;
    } else if (!SE->isLoopInvariant(EC, L))
      continue;

    if (SE->getTypeSizeInBits(EC->getType()) >
        HWLoopInfo.CountType->getBitWidth())
      continue;

    // If this exiting block is contained in a nested loop, it is not eligible
    // for insertion of the branch-and-decrement since the inner loop would
    // end up messing up the value in the CTR.
    if (!HWLoopInfo.IsNestingLegal && LI->getLoopFor(*I) != L &&
        !ForceNestedLoop)
      continue;

    // We now have a loop-invariant count of loop iterations (which is not the
    // constant zero) for which we know that this loop will not exit via this
    // existing block.

    // We need to make sure that this block will run on every loop iteration.
    // For this to be true, we must dominate all blocks with backedges. Such
    // blocks are in-loop predecessors to the header block.
    bool NotAlways = false;
    for (pred_iterator PI = pred_begin(L->getHeader()),
         PIE = pred_end(L->getHeader()); PI != PIE; ++PI) {
      if (!L->contains(*PI))
        continue;

      if (!DT->dominates(*I, *PI)) {
        NotAlways = true;
        break;
      }
    }

    if (NotAlways)
      continue;

    // Make sure this blocks ends with a conditional branch.
    Instruction *TI = (*I)->getTerminator();
    if (!TI)
      continue;

    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (!BI->isConditional())
        continue;

      HWLoopInfo.ExitBranch = BI;
    } else
      continue;

    // Note that this block may not be the loop latch block, even if the loop
    // has a latch block.
    HWLoopInfo.ExitBlock = *I;
    HWLoopInfo.ExitCount = EC;
    break;
  }

  if (!HWLoopInfo.ExitBlock)
    return false;

  BasicBlock *Preheader = L->getLoopPreheader();

  // If we don't have a preheader, then insert one.
  if (!Preheader)
    Preheader = InsertPreheaderForLoop(L, DT, LI, nullptr, PreserveLCSSA);
  if (!Preheader)
    return false;

  HardwareLoop HWLoop(HWLoopInfo, *SE, *DL);
  HWLoop.Create();
  ++NumHWLoops;
  return true;
}

void HardwareLoop::Create() {
  LLVM_DEBUG(dbgs() << "HWLoops: Converting loop..\n");
  BasicBlock *BeginBB = L->getLoopPreheader();
  Value *LoopCountInit = InitLoopCount(BeginBB);
  if (!LoopCountInit)
    return;

  InsertIterationSetup(LoopCountInit, BeginBB);

  if (UsePHICounter || ForceHardwareLoopPHI) {
    Instruction *LoopDec = InsertLoopRegDec(LoopCountInit);
    Value *EltsRem = InsertPHICounter(LoopCountInit, LoopDec);
    LoopDec->setOperand(0, EltsRem);
    UpdateBranch(LoopDec);
  } else
    InsertLoopDec();

  // Run through the basic blocks of the loop and see if any of them have dead
  // PHIs that can be removed.
  for (auto I : L->blocks())
    DeleteDeadPHIs(I);
}

Value *HardwareLoop::InitLoopCount(BasicBlock *BB) {
  SCEVExpander SCEVE(SE, DL, "loopcnt");
  if (!ExitCount->getType()->isPointerTy() &&
      ExitCount->getType() != CountType)
    ExitCount = SE.getZeroExtendExpr(ExitCount, CountType);

  ExitCount = SE.getAddExpr(ExitCount, SE.getOne(CountType));

  if (!isSafeToExpandAt(ExitCount, BB->getTerminator(), SE)) {
    LLVM_DEBUG(dbgs() << "HWLoops: Bailing, unsafe to expand ExitCount "
               << *ExitCount << "\n");
    return nullptr;
  }

  Value *Count = SCEVE.expandCodeFor(ExitCount, CountType,
                                     BB->getTerminator());
  LLVM_DEBUG(dbgs() << "HWLoops: Loop Count: " << *Count << "\n");
  return Count;
}

void HardwareLoop::InsertIterationSetup(Value *LoopCountInit,
                                        BasicBlock *BB) {
  IRBuilder<> Builder(BB->getTerminator());
  Type *Ty = LoopCountInit->getType();
  Function *LoopIter =
    Intrinsic::getDeclaration(M, Intrinsic::set_loop_iterations, Ty);
  Builder.CreateCall(LoopIter, LoopCountInit);
}

void HardwareLoop::InsertLoopDec() {
  IRBuilder<> CondBuilder(ExitBranch);

  Function *DecFunc =
    Intrinsic::getDeclaration(M, Intrinsic::loop_decrement,
                              LoopDecrement->getType());
  Value *Ops[] = { LoopDecrement };
  Value *NewCond = CondBuilder.CreateCall(DecFunc, Ops);
  Value *OldCond = ExitBranch->getCondition();
  ExitBranch->setCondition(NewCond);

  // The false branch must exit the loop.
  if (!L->contains(ExitBranch->getSuccessor(0)))
    ExitBranch->swapSuccessors();

  // The old condition may be dead now, and may have even created a dead PHI
  // (the original induction variable).
  RecursivelyDeleteTriviallyDeadInstructions(OldCond);

  LLVM_DEBUG(dbgs() << "HWLoops: Inserted loop dec: " << *NewCond << "\n");
}

Instruction* HardwareLoop::InsertLoopRegDec(Value *EltsRem) {
  IRBuilder<> CondBuilder(ExitBranch);

  Function *DecFunc =
      Intrinsic::getDeclaration(M, Intrinsic::loop_decrement_reg,
                                { EltsRem->getType(), EltsRem->getType(),
                                  LoopDecrement->getType()
                                });
  Value *Ops[] = { EltsRem, LoopDecrement };
  Value *Call = CondBuilder.CreateCall(DecFunc, Ops);

  LLVM_DEBUG(dbgs() << "HWLoops: Inserted loop dec: " << *Call << "\n");
  return cast<Instruction>(Call);
}

PHINode* HardwareLoop::InsertPHICounter(Value *NumElts, Value *EltsRem) {
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = ExitBranch->getParent();
  IRBuilder<> Builder(Header->getFirstNonPHI());
  PHINode *Index = Builder.CreatePHI(NumElts->getType(), 2);
  Index->addIncoming(NumElts, Preheader);
  Index->addIncoming(EltsRem, Latch);
  LLVM_DEBUG(dbgs() << "HWLoops: PHI Counter: " << *Index << "\n");
  return Index;
}

void HardwareLoop::UpdateBranch(Value *EltsRem) {
  IRBuilder<> CondBuilder(ExitBranch);
  Value *NewCond =
    CondBuilder.CreateICmpNE(EltsRem, ConstantInt::get(EltsRem->getType(), 0));
  Value *OldCond = ExitBranch->getCondition();
  ExitBranch->setCondition(NewCond);

  // The false branch must exit the loop.
  if (!L->contains(ExitBranch->getSuccessor(0)))
    ExitBranch->swapSuccessors();

  // The old condition may be dead now, and may have even created a dead PHI
  // (the original induction variable).
  RecursivelyDeleteTriviallyDeadInstructions(OldCond);
}

INITIALIZE_PASS_BEGIN(HardwareLoops, DEBUG_TYPE, HW_LOOPS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(HardwareLoops, DEBUG_TYPE, HW_LOOPS_NAME, false, false)

FunctionPass *llvm::createHardwareLoopsPass() { return new HardwareLoops(); }
