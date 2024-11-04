//===- LoopTermFold.cpp - Eliminate last use of IV in exit branch----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopTermFold.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>
#include <optional>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "loop-term-fold"

STATISTIC(NumTermFold,
          "Number of terminating condition fold recognized and performed");

static std::optional<std::tuple<PHINode *, PHINode *, const SCEV *, bool>>
canFoldTermCondOfLoop(Loop *L, ScalarEvolution &SE, DominatorTree &DT,
                      const LoopInfo &LI, const TargetTransformInfo &TTI) {
  if (!L->isInnermost()) {
    LLVM_DEBUG(dbgs() << "Cannot fold on non-innermost loop\n");
    return std::nullopt;
  }
  // Only inspect on simple loop structure
  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "Cannot fold on non-simple loop\n");
    return std::nullopt;
  }

  if (!SE.hasLoopInvariantBackedgeTakenCount(L)) {
    LLVM_DEBUG(dbgs() << "Cannot fold on backedge that is loop variant\n");
    return std::nullopt;
  }

  BasicBlock *LoopLatch = L->getLoopLatch();
  BranchInst *BI = dyn_cast<BranchInst>(LoopLatch->getTerminator());
  if (!BI || BI->isUnconditional())
    return std::nullopt;
  auto *TermCond = dyn_cast<ICmpInst>(BI->getCondition());
  if (!TermCond) {
    LLVM_DEBUG(
        dbgs() << "Cannot fold on branching condition that is not an ICmpInst");
    return std::nullopt;
  }
  if (!TermCond->hasOneUse()) {
    LLVM_DEBUG(
        dbgs()
        << "Cannot replace terminating condition with more than one use\n");
    return std::nullopt;
  }

  BinaryOperator *LHS = dyn_cast<BinaryOperator>(TermCond->getOperand(0));
  Value *RHS = TermCond->getOperand(1);
  if (!LHS || !L->isLoopInvariant(RHS))
    // We could pattern match the inverse form of the icmp, but that is
    // non-canonical, and this pass is running *very* late in the pipeline.
    return std::nullopt;

  // Find the IV used by the current exit condition.
  PHINode *ToFold;
  Value *ToFoldStart, *ToFoldStep;
  if (!matchSimpleRecurrence(LHS, ToFold, ToFoldStart, ToFoldStep))
    return std::nullopt;

  // Ensure the simple recurrence is a part of the current loop.
  if (ToFold->getParent() != L->getHeader())
    return std::nullopt;

  // If that IV isn't dead after we rewrite the exit condition in terms of
  // another IV, there's no point in doing the transform.
  if (!isAlmostDeadIV(ToFold, LoopLatch, TermCond))
    return std::nullopt;

  // Inserting instructions in the preheader has a runtime cost, scale
  // the allowed cost with the loops trip count as best we can.
  const unsigned ExpansionBudget = [&]() {
    unsigned Budget = 2 * SCEVCheapExpansionBudget;
    if (unsigned SmallTC = SE.getSmallConstantMaxTripCount(L))
      return std::min(Budget, SmallTC);
    if (std::optional<unsigned> SmallTC = getLoopEstimatedTripCount(L))
      return std::min(Budget, *SmallTC);
    // Unknown trip count, assume long running by default.
    return Budget;
  }();

  const SCEV *BECount = SE.getBackedgeTakenCount(L);
  const DataLayout &DL = L->getHeader()->getDataLayout();
  SCEVExpander Expander(SE, DL, "lsr_fold_term_cond");

  PHINode *ToHelpFold = nullptr;
  const SCEV *TermValueS = nullptr;
  bool MustDropPoison = false;
  auto InsertPt = L->getLoopPreheader()->getTerminator();
  for (PHINode &PN : L->getHeader()->phis()) {
    if (ToFold == &PN)
      continue;

    if (!SE.isSCEVable(PN.getType())) {
      LLVM_DEBUG(dbgs() << "IV of phi '" << PN
                        << "' is not SCEV-able, not qualified for the "
                           "terminating condition folding.\n");
      continue;
    }
    const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(&PN));
    // Only speculate on affine AddRec
    if (!AddRec || !AddRec->isAffine()) {
      LLVM_DEBUG(dbgs() << "SCEV of phi '" << PN
                        << "' is not an affine add recursion, not qualified "
                           "for the terminating condition folding.\n");
      continue;
    }

    // Check that we can compute the value of AddRec on the exiting iteration
    // without soundness problems.  evaluateAtIteration internally needs
    // to multiply the stride of the iteration number - which may wrap around.
    // The issue here is subtle because computing the result accounting for
    // wrap is insufficient. In order to use the result in an exit test, we
    // must also know that AddRec doesn't take the same value on any previous
    // iteration. The simplest case to consider is a candidate IV which is
    // narrower than the trip count (and thus original IV), but this can
    // also happen due to non-unit strides on the candidate IVs.
    if (!AddRec->hasNoSelfWrap() ||
        !SE.isKnownNonZero(AddRec->getStepRecurrence(SE)))
      continue;

    const SCEVAddRecExpr *PostInc = AddRec->getPostIncExpr(SE);
    const SCEV *TermValueSLocal = PostInc->evaluateAtIteration(BECount, SE);
    if (!Expander.isSafeToExpand(TermValueSLocal)) {
      LLVM_DEBUG(
          dbgs() << "Is not safe to expand terminating value for phi node" << PN
                 << "\n");
      continue;
    }

    if (Expander.isHighCostExpansion(TermValueSLocal, L, ExpansionBudget, &TTI,
                                     InsertPt)) {
      LLVM_DEBUG(
          dbgs() << "Is too expensive to expand terminating value for phi node"
                 << PN << "\n");
      continue;
    }

    // The candidate IV may have been otherwise dead and poison from the
    // very first iteration.  If we can't disprove that, we can't use the IV.
    if (!mustExecuteUBIfPoisonOnPathTo(&PN, LoopLatch->getTerminator(), &DT)) {
      LLVM_DEBUG(dbgs() << "Can not prove poison safety for IV " << PN << "\n");
      continue;
    }

    // The candidate IV may become poison on the last iteration.  If this
    // value is not branched on, this is a well defined program.  We're
    // about to add a new use to this IV, and we have to ensure we don't
    // insert UB which didn't previously exist.
    bool MustDropPoisonLocal = false;
    Instruction *PostIncV =
        cast<Instruction>(PN.getIncomingValueForBlock(LoopLatch));
    if (!mustExecuteUBIfPoisonOnPathTo(PostIncV, LoopLatch->getTerminator(),
                                       &DT)) {
      LLVM_DEBUG(dbgs() << "Can not prove poison safety to insert use" << PN
                        << "\n");

      // If this is a complex recurrance with multiple instructions computing
      // the backedge value, we might need to strip poison flags from all of
      // them.
      if (PostIncV->getOperand(0) != &PN)
        continue;

      // In order to perform the transform, we need to drop the poison
      // generating flags on this instruction (if any).
      MustDropPoisonLocal = PostIncV->hasPoisonGeneratingFlags();
    }

    // We pick the last legal alternate IV.  We could expore choosing an optimal
    // alternate IV if we had a decent heuristic to do so.
    ToHelpFold = &PN;
    TermValueS = TermValueSLocal;
    MustDropPoison = MustDropPoisonLocal;
  }

  LLVM_DEBUG(if (ToFold && !ToHelpFold) dbgs()
                 << "Cannot find other AddRec IV to help folding\n";);

  LLVM_DEBUG(if (ToFold && ToHelpFold) dbgs()
             << "\nFound loop that can fold terminating condition\n"
             << "  BECount (SCEV): " << *SE.getBackedgeTakenCount(L) << "\n"
             << "  TermCond: " << *TermCond << "\n"
             << "  BrandInst: " << *BI << "\n"
             << "  ToFold: " << *ToFold << "\n"
             << "  ToHelpFold: " << *ToHelpFold << "\n");

  if (!ToFold || !ToHelpFold)
    return std::nullopt;
  return std::make_tuple(ToFold, ToHelpFold, TermValueS, MustDropPoison);
}

static bool RunTermFold(Loop *L, ScalarEvolution &SE, DominatorTree &DT,
                        LoopInfo &LI, const TargetTransformInfo &TTI,
                        TargetLibraryInfo &TLI, MemorySSA *MSSA) {
  std::unique_ptr<MemorySSAUpdater> MSSAU;
  if (MSSA)
    MSSAU = std::make_unique<MemorySSAUpdater>(MSSA);

  auto Opt = canFoldTermCondOfLoop(L, SE, DT, LI, TTI);
  if (!Opt)
    return false;

  auto [ToFold, ToHelpFold, TermValueS, MustDrop] = *Opt;

  NumTermFold++;

  BasicBlock *LoopPreheader = L->getLoopPreheader();
  BasicBlock *LoopLatch = L->getLoopLatch();

  (void)ToFold;
  LLVM_DEBUG(dbgs() << "To fold phi-node:\n"
                    << *ToFold << "\n"
                    << "New term-cond phi-node:\n"
                    << *ToHelpFold << "\n");

  Value *StartValue = ToHelpFold->getIncomingValueForBlock(LoopPreheader);
  (void)StartValue;
  Value *LoopValue = ToHelpFold->getIncomingValueForBlock(LoopLatch);

  // See comment in canFoldTermCondOfLoop on why this is sufficient.
  if (MustDrop)
    cast<Instruction>(LoopValue)->dropPoisonGeneratingFlags();

  // SCEVExpander for both use in preheader and latch
  const DataLayout &DL = L->getHeader()->getDataLayout();
  SCEVExpander Expander(SE, DL, "lsr_fold_term_cond");

  assert(Expander.isSafeToExpand(TermValueS) &&
         "Terminating value was checked safe in canFoldTerminatingCondition");

  // Create new terminating value at loop preheader
  Value *TermValue = Expander.expandCodeFor(TermValueS, ToHelpFold->getType(),
                                            LoopPreheader->getTerminator());

  LLVM_DEBUG(dbgs() << "Start value of new term-cond phi-node:\n"
                    << *StartValue << "\n"
                    << "Terminating value of new term-cond phi-node:\n"
                    << *TermValue << "\n");

  // Create new terminating condition at loop latch
  BranchInst *BI = cast<BranchInst>(LoopLatch->getTerminator());
  ICmpInst *OldTermCond = cast<ICmpInst>(BI->getCondition());
  IRBuilder<> LatchBuilder(LoopLatch->getTerminator());
  Value *NewTermCond =
      LatchBuilder.CreateICmp(CmpInst::ICMP_EQ, LoopValue, TermValue,
                              "lsr_fold_term_cond.replaced_term_cond");
  // Swap successors to exit loop body if IV equals to new TermValue
  if (BI->getSuccessor(0) == L->getHeader())
    BI->swapSuccessors();

  LLVM_DEBUG(dbgs() << "Old term-cond:\n"
                    << *OldTermCond << "\n"
                    << "New term-cond:\n"
                    << *NewTermCond << "\n");

  BI->setCondition(NewTermCond);

  Expander.clear();
  OldTermCond->eraseFromParent();
  DeleteDeadPHIs(L->getHeader(), &TLI, MSSAU.get());
  return true;
}

namespace {

class LoopTermFold : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid

  LoopTermFold();

private:
  bool runOnLoop(Loop *L, LPPassManager &LPM) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // end anonymous namespace

LoopTermFold::LoopTermFold() : LoopPass(ID) {
  initializeLoopTermFoldPass(*PassRegistry::getPassRegistry());
}

void LoopTermFold::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addPreservedID(LoopSimplifyID);
  AU.addRequiredID(LoopSimplifyID);
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  AU.addPreserved<MemorySSAWrapperPass>();
}

bool LoopTermFold::runOnLoop(Loop *L, LPPassManager & /*LPM*/) {
  if (skipLoop(L))
    return false;

  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  const auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(
      *L->getHeader()->getParent());
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(
      *L->getHeader()->getParent());
  auto *MSSAAnalysis = getAnalysisIfAvailable<MemorySSAWrapperPass>();
  MemorySSA *MSSA = nullptr;
  if (MSSAAnalysis)
    MSSA = &MSSAAnalysis->getMSSA();
  return RunTermFold(L, SE, DT, LI, TTI, TLI, MSSA);
}

PreservedAnalyses LoopTermFoldPass::run(Loop &L, LoopAnalysisManager &AM,
                                        LoopStandardAnalysisResults &AR,
                                        LPMUpdater &) {
  if (!RunTermFold(&L, AR.SE, AR.DT, AR.LI, AR.TTI, AR.TLI, AR.MSSA))
    return PreservedAnalyses::all();

  auto PA = getLoopPassPreservedAnalyses();
  if (AR.MSSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}

char LoopTermFold::ID = 0;

INITIALIZE_PASS_BEGIN(LoopTermFold, "loop-term-fold", "Loop Terminator Folding",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(LoopTermFold, "loop-term-fold", "Loop Terminator Folding",
                    false, false)

Pass *llvm::createLoopTermFoldPass() { return new LoopTermFold(); }
