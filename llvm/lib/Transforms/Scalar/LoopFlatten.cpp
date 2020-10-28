//===- LoopFlatten.cpp - Loop flattening pass------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass flattens pairs nested loops into a single loop.
//
// The intention is to optimise loop nests like this, which together access an
// array linearly:
//   for (int i = 0; i < N; ++i)
//     for (int j = 0; j < M; ++j)
//       f(A[i*M+j]);
// into one loop:
//   for (int i = 0; i < (N*M); ++i)
//     f(A[i]);
//
// It can also flatten loops where the induction variables are not used in the
// loop. This is only worth doing if the induction variables are only used in an
// expression like i*M+j. If they had any other uses, we would have to insert a
// div/mod to reconstruct the original values, so this wouldn't be profitable.
//
// We also need to prove that N*M will not overflow.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopFlatten.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "loop-flatten"

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<unsigned> RepeatedInstructionThreshold(
    "loop-flatten-cost-threshold", cl::Hidden, cl::init(2),
    cl::desc("Limit on the cost of instructions that can be repeated due to "
             "loop flattening"));

static cl::opt<bool>
    AssumeNoOverflow("loop-flatten-assume-no-overflow", cl::Hidden,
                     cl::init(false),
                     cl::desc("Assume that the product of the two iteration "
                              "limits will never overflow"));

// Finds the induction variable, increment and limit for a simple loop that we
// can flatten.
static bool findLoopComponents(
    Loop *L, SmallPtrSetImpl<Instruction *> &IterationInstructions,
    PHINode *&InductionPHI, Value *&Limit, BinaryOperator *&Increment,
    BranchInst *&BackBranch, ScalarEvolution *SE) {
  LLVM_DEBUG(dbgs() << "Finding components of loop: " << L->getName() << "\n");

  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "Loop is not in normal form\n");
    return false;
  }

  // There must be exactly one exiting block, and it must be the same at the
  // latch.
  BasicBlock *Latch = L->getLoopLatch();
  if (L->getExitingBlock() != Latch) {
    LLVM_DEBUG(dbgs() << "Exiting and latch block are different\n");
    return false;
  }
  // Latch block must end in a conditional branch.
  BackBranch = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!BackBranch || !BackBranch->isConditional()) {
    LLVM_DEBUG(dbgs() << "Could not find back-branch\n");
    return false;
  }
  IterationInstructions.insert(BackBranch);
  LLVM_DEBUG(dbgs() << "Found back branch: "; BackBranch->dump());
  bool ContinueOnTrue = L->contains(BackBranch->getSuccessor(0));

  // Find the induction PHI. If there is no induction PHI, we can't do the
  // transformation. TODO: could other variables trigger this? Do we have to
  // search for the best one?
  InductionPHI = nullptr;
  for (PHINode &PHI : L->getHeader()->phis()) {
    InductionDescriptor ID;
    if (InductionDescriptor::isInductionPHI(&PHI, L, SE, ID)) {
      InductionPHI = &PHI;
      LLVM_DEBUG(dbgs() << "Found induction PHI: "; InductionPHI->dump());
      break;
    }
  }
  if (!InductionPHI) {
    LLVM_DEBUG(dbgs() << "Could not find induction PHI\n");
    return false;
  }

  auto IsValidPredicate = [&](ICmpInst::Predicate Pred) {
    if (ContinueOnTrue)
      return Pred == CmpInst::ICMP_NE || Pred == CmpInst::ICMP_ULT;
    else
      return Pred == CmpInst::ICMP_EQ;
  };

  // Find Compare and make sure it is valid
  ICmpInst *Compare = dyn_cast<ICmpInst>(BackBranch->getCondition());
  if (!Compare || !IsValidPredicate(Compare->getUnsignedPredicate()) ||
      Compare->hasNUsesOrMore(2)) {
    LLVM_DEBUG(dbgs() << "Could not find valid comparison\n");
    return false;
  }
  IterationInstructions.insert(Compare);
  LLVM_DEBUG(dbgs() << "Found comparison: "; Compare->dump());

  // Find increment and limit from the compare
  Increment = nullptr;
  if (match(Compare->getOperand(0),
            m_c_Add(m_Specific(InductionPHI), m_ConstantInt<1>()))) {
    Increment = dyn_cast<BinaryOperator>(Compare->getOperand(0));
    Limit = Compare->getOperand(1);
  } else if (Compare->getUnsignedPredicate() == CmpInst::ICMP_NE &&
             match(Compare->getOperand(1),
                   m_c_Add(m_Specific(InductionPHI), m_ConstantInt<1>()))) {
    Increment = dyn_cast<BinaryOperator>(Compare->getOperand(1));
    Limit = Compare->getOperand(0);
  }
  if (!Increment || Increment->hasNUsesOrMore(3)) {
    LLVM_DEBUG(dbgs() << "Cound not find valid increment\n");
    return false;
  }
  IterationInstructions.insert(Increment);
  LLVM_DEBUG(dbgs() << "Found increment: "; Increment->dump());
  LLVM_DEBUG(dbgs() << "Found limit: "; Limit->dump());

  assert(InductionPHI->getNumIncomingValues() == 2);
  assert(InductionPHI->getIncomingValueForBlock(Latch) == Increment &&
         "PHI value is not increment inst");

  auto *CI = dyn_cast<ConstantInt>(
      InductionPHI->getIncomingValueForBlock(L->getLoopPreheader()));
  if (!CI || !CI->isZero()) {
    LLVM_DEBUG(dbgs() << "PHI value is not zero: "; CI->dump());
    return false;
  }

  LLVM_DEBUG(dbgs() << "Successfully found all loop components\n");
  return true;
}

static bool checkPHIs(Loop *OuterLoop, Loop *InnerLoop,
                      SmallPtrSetImpl<PHINode *> &InnerPHIsToTransform,
                      PHINode *InnerInductionPHI, PHINode *OuterInductionPHI,
                      TargetTransformInfo *TTI) {
  // All PHIs in the inner and outer headers must either be:
  // - The induction PHI, which we are going to rewrite as one induction in
  //   the new loop. This is already checked by findLoopComponents.
  // - An outer header PHI with all incoming values from outside the loop.
  //   LoopSimplify guarantees we have a pre-header, so we don't need to
  //   worry about that here.
  // - Pairs of PHIs in the inner and outer headers, which implement a
  //   loop-carried dependency that will still be valid in the new loop. To
  //   be valid, this variable must be modified only in the inner loop.

  // The set of PHI nodes in the outer loop header that we know will still be
  // valid after the transformation. These will not need to be modified (with
  // the exception of the induction variable), but we do need to check that
  // there are no unsafe PHI nodes.
  SmallPtrSet<PHINode *, 4> SafeOuterPHIs;
  SafeOuterPHIs.insert(OuterInductionPHI);

  // Check that all PHI nodes in the inner loop header match one of the valid
  // patterns.
  for (PHINode &InnerPHI : InnerLoop->getHeader()->phis()) {
    // The induction PHIs break these rules, and that's OK because we treat
    // them specially when doing the transformation.
    if (&InnerPHI == InnerInductionPHI)
      continue;

    // Each inner loop PHI node must have two incoming values/blocks - one
    // from the pre-header, and one from the latch.
    assert(InnerPHI.getNumIncomingValues() == 2);
    Value *PreHeaderValue =
        InnerPHI.getIncomingValueForBlock(InnerLoop->getLoopPreheader());
    Value *LatchValue =
        InnerPHI.getIncomingValueForBlock(InnerLoop->getLoopLatch());

    // The incoming value from the outer loop must be the PHI node in the
    // outer loop header, with no modifications made in the top of the outer
    // loop.
    PHINode *OuterPHI = dyn_cast<PHINode>(PreHeaderValue);
    if (!OuterPHI || OuterPHI->getParent() != OuterLoop->getHeader()) {
      LLVM_DEBUG(dbgs() << "value modified in top of outer loop\n");
      return false;
    }

    // The other incoming value must come from the inner loop, without any
    // modifications in the tail end of the outer loop. We are in LCSSA form,
    // so this will actually be a PHI in the inner loop's exit block, which
    // only uses values from inside the inner loop.
    PHINode *LCSSAPHI = dyn_cast<PHINode>(
        OuterPHI->getIncomingValueForBlock(OuterLoop->getLoopLatch()));
    if (!LCSSAPHI) {
      LLVM_DEBUG(dbgs() << "could not find LCSSA PHI\n");
      return false;
    }

    // The value used by the LCSSA PHI must be the same one that the inner
    // loop's PHI uses.
    if (LCSSAPHI->hasConstantValue() != LatchValue) {
      LLVM_DEBUG(
          dbgs() << "LCSSA PHI incoming value does not match latch value\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "PHI pair is safe:\n");
    LLVM_DEBUG(dbgs() << "  Inner: "; InnerPHI.dump());
    LLVM_DEBUG(dbgs() << "  Outer: "; OuterPHI->dump());
    SafeOuterPHIs.insert(OuterPHI);
    InnerPHIsToTransform.insert(&InnerPHI);
  }

  for (PHINode &OuterPHI : OuterLoop->getHeader()->phis()) {
    if (!SafeOuterPHIs.count(&OuterPHI)) {
      LLVM_DEBUG(dbgs() << "found unsafe PHI in outer loop: "; OuterPHI.dump());
      return false;
    }
  }

  return true;
}

static bool
checkOuterLoopInsts(Loop *OuterLoop, Loop *InnerLoop,
                    SmallPtrSetImpl<Instruction *> &IterationInstructions,
                    Value *InnerLimit, PHINode *OuterPHI,
                    TargetTransformInfo *TTI) {
  // Check for instructions in the outer but not inner loop. If any of these
  // have side-effects then this transformation is not legal, and if there is
  // a significant amount of code here which can't be optimised out that it's
  // not profitable (as these instructions would get executed for each
  // iteration of the inner loop).
  unsigned RepeatedInstrCost = 0;
  for (auto *B : OuterLoop->getBlocks()) {
    if (InnerLoop->contains(B))
      continue;

    for (auto &I : *B) {
      if (!isa<PHINode>(&I) && !I.isTerminator() &&
          !isSafeToSpeculativelyExecute(&I)) {
        LLVM_DEBUG(dbgs() << "Cannot flatten because instruction may have "
                             "side effects: ";
                   I.dump());
        return false;
      }
      // The execution count of the outer loop's iteration instructions
      // (increment, compare and branch) will be increased, but the
      // equivalent instructions will be removed from the inner loop, so
      // they make a net difference of zero.
      if (IterationInstructions.count(&I))
        continue;
      // The uncoditional branch to the inner loop's header will turn into
      // a fall-through, so adds no cost.
      BranchInst *Br = dyn_cast<BranchInst>(&I);
      if (Br && Br->isUnconditional() &&
          Br->getSuccessor(0) == InnerLoop->getHeader())
        continue;
      // Multiplies of the outer iteration variable and inner iteration
      // count will be optimised out.
      if (match(&I, m_c_Mul(m_Specific(OuterPHI), m_Specific(InnerLimit))))
        continue;
      int Cost = TTI->getUserCost(&I, TargetTransformInfo::TCK_SizeAndLatency);
      LLVM_DEBUG(dbgs() << "Cost " << Cost << ": "; I.dump());
      RepeatedInstrCost += Cost;
    }
  }

  LLVM_DEBUG(dbgs() << "Cost of instructions that will be repeated: "
                    << RepeatedInstrCost << "\n");
  // Bail out if flattening the loops would cause instructions in the outer
  // loop but not in the inner loop to be executed extra times.
  if (RepeatedInstrCost > RepeatedInstructionThreshold)
    return false;

  return true;
}

static bool checkIVUsers(PHINode *InnerPHI, PHINode *OuterPHI,
                         BinaryOperator *InnerIncrement,
                         BinaryOperator *OuterIncrement, Value *InnerLimit,
                         SmallPtrSetImpl<Value *> &LinearIVUses) {
  // We require all uses of both induction variables to match this pattern:
  //
  //   (OuterPHI * InnerLimit) + InnerPHI
  //
  // Any uses of the induction variables not matching that pattern would
  // require a div/mod to reconstruct in the flattened loop, so the
  // transformation wouldn't be profitable.

  // Check that all uses of the inner loop's induction variable match the
  // expected pattern, recording the uses of the outer IV.
  SmallPtrSet<Value *, 4> ValidOuterPHIUses;
  for (User *U : InnerPHI->users()) {
    if (U == InnerIncrement)
      continue;

    LLVM_DEBUG(dbgs() << "Found use of inner induction variable: "; U->dump());

    Value *MatchedMul, *MatchedItCount;
    if (match(U, m_c_Add(m_Specific(InnerPHI), m_Value(MatchedMul))) &&
        match(MatchedMul,
              m_c_Mul(m_Specific(OuterPHI), m_Value(MatchedItCount))) &&
        MatchedItCount == InnerLimit) {
      LLVM_DEBUG(dbgs() << "Use is optimisable\n");
      ValidOuterPHIUses.insert(MatchedMul);
      LinearIVUses.insert(U);
    } else {
      LLVM_DEBUG(dbgs() << "Did not match expected pattern, bailing\n");
      return false;
    }
  }

  // Check that there are no uses of the outer IV other than the ones found
  // as part of the pattern above.
  for (User *U : OuterPHI->users()) {
    if (U == OuterIncrement)
      continue;

    LLVM_DEBUG(dbgs() << "Found use of outer induction variable: "; U->dump());

    if (!ValidOuterPHIUses.count(U)) {
      LLVM_DEBUG(dbgs() << "Did not match expected pattern, bailing\n");
      return false;
    } else {
      LLVM_DEBUG(dbgs() << "Use is optimisable\n");
    }
  }

  LLVM_DEBUG(dbgs() << "Found " << LinearIVUses.size()
                    << " value(s) that can be replaced:\n";
             for (Value *V : LinearIVUses) {
               dbgs() << "  ";
               V->dump();
             });

  return true;
}

// Return an OverflowResult dependant on if overflow of the multiplication of
// InnerLimit and OuterLimit can be assumed not to happen.
static OverflowResult checkOverflow(Loop *OuterLoop, Value *InnerLimit,
                                    Value *OuterLimit,
                                    SmallPtrSetImpl<Value *> &LinearIVUses,
                                    DominatorTree *DT, AssumptionCache *AC) {
  Function *F = OuterLoop->getHeader()->getParent();
  const DataLayout &DL = F->getParent()->getDataLayout();

  // For debugging/testing.
  if (AssumeNoOverflow)
    return OverflowResult::NeverOverflows;

  // Check if the multiply could not overflow due to known ranges of the
  // input values.
  OverflowResult OR = computeOverflowForUnsignedMul(
      InnerLimit, OuterLimit, DL, AC,
      OuterLoop->getLoopPreheader()->getTerminator(), DT);
  if (OR != OverflowResult::MayOverflow)
    return OR;

  for (Value *V : LinearIVUses) {
    for (Value *U : V->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        // The IV is used as the operand of a GEP, and the IV is at least as
        // wide as the address space of the GEP. In this case, the GEP would
        // wrap around the address space before the IV increment wraps, which
        // would be UB.
        if (GEP->isInBounds() &&
            V->getType()->getIntegerBitWidth() >=
                DL.getPointerTypeSizeInBits(GEP->getType())) {
          LLVM_DEBUG(
              dbgs() << "use of linear IV would be UB if overflow occurred: ";
              GEP->dump());
          return OverflowResult::NeverOverflows;
        }
      }
    }
  }

  return OverflowResult::MayOverflow;
}

static bool FlattenLoopPair(Loop *OuterLoop, Loop *InnerLoop, DominatorTree *DT,
                            LoopInfo *LI, ScalarEvolution *SE,
                            AssumptionCache *AC, TargetTransformInfo *TTI,
                            std::function<void(Loop *)> markLoopAsDeleted) {
  Function *F = OuterLoop->getHeader()->getParent();

  LLVM_DEBUG(dbgs() << "Loop flattening running on outer loop "
                    << OuterLoop->getHeader()->getName() << " and inner loop "
                    << InnerLoop->getHeader()->getName() << " in "
                    << F->getName() << "\n");

  SmallPtrSet<Instruction *, 8> IterationInstructions;

  PHINode *InnerInductionPHI, *OuterInductionPHI;
  Value *InnerLimit, *OuterLimit;
  BinaryOperator *InnerIncrement, *OuterIncrement;
  BranchInst *InnerBranch, *OuterBranch;

  if (!findLoopComponents(InnerLoop, IterationInstructions, InnerInductionPHI,
                          InnerLimit, InnerIncrement, InnerBranch, SE))
    return false;
  if (!findLoopComponents(OuterLoop, IterationInstructions, OuterInductionPHI,
                          OuterLimit, OuterIncrement, OuterBranch, SE))
    return false;

  // Both of the loop limit values must be invariant in the outer loop
  // (non-instructions are all inherently invariant).
  if (!OuterLoop->isLoopInvariant(InnerLimit)) {
    LLVM_DEBUG(dbgs() << "inner loop limit not invariant\n");
    return false;
  }
  if (!OuterLoop->isLoopInvariant(OuterLimit)) {
    LLVM_DEBUG(dbgs() << "outer loop limit not invariant\n");
    return false;
  }

  SmallPtrSet<PHINode *, 4> InnerPHIsToTransform;
  if (!checkPHIs(OuterLoop, InnerLoop, InnerPHIsToTransform, InnerInductionPHI,
                 OuterInductionPHI, TTI))
    return false;

  // FIXME: it should be possible to handle different types correctly.
  if (InnerInductionPHI->getType() != OuterInductionPHI->getType())
    return false;

  if (!checkOuterLoopInsts(OuterLoop, InnerLoop, IterationInstructions,
                           InnerLimit, OuterInductionPHI, TTI))
    return false;

  // Find the values in the loop that can be replaced with the linearized
  // induction variable, and check that there are no other uses of the inner
  // or outer induction variable. If there were, we could still do this
  // transformation, but we'd have to insert a div/mod to calculate the
  // original IVs, so it wouldn't be profitable.
  SmallPtrSet<Value *, 4> LinearIVUses;
  if (!checkIVUsers(InnerInductionPHI, OuterInductionPHI, InnerIncrement,
                    OuterIncrement, InnerLimit, LinearIVUses))
    return false;

  // Check if the new iteration variable might overflow. In this case, we
  // need to version the loop, and select the original version at runtime if
  // the iteration space is too large.
  // TODO: We currently don't version the loop.
  // TODO: it might be worth using a wider iteration variable rather than
  // versioning the loop, if a wide enough type is legal.
  bool MustVersionLoop = true;
  OverflowResult OR =
      checkOverflow(OuterLoop, InnerLimit, OuterLimit, LinearIVUses, DT, AC);
  if (OR == OverflowResult::AlwaysOverflowsHigh ||
      OR == OverflowResult::AlwaysOverflowsLow) {
    LLVM_DEBUG(dbgs() << "Multiply would always overflow, so not profitable\n");
    return false;
  } else if (OR == OverflowResult::MayOverflow) {
    LLVM_DEBUG(dbgs() << "Multiply might overflow, not flattening\n");
  } else {
    LLVM_DEBUG(dbgs() << "Multiply cannot overflow, modifying loop in-place\n");
    MustVersionLoop = false;
  }

  // We cannot safely flatten the loop. Exit now.
  if (MustVersionLoop)
    return false;

  // Do the actual transformation.
  LLVM_DEBUG(dbgs() << "Checks all passed, doing the transformation\n");

  {
    using namespace ore;
    OptimizationRemark Remark(DEBUG_TYPE, "Flattened", InnerLoop->getStartLoc(),
                              InnerLoop->getHeader());
    OptimizationRemarkEmitter ORE(F);
    Remark << "Flattened into outer loop";
    ORE.emit(Remark);
  }

  Value *NewTripCount =
      BinaryOperator::CreateMul(InnerLimit, OuterLimit, "flatten.tripcount",
                                OuterLoop->getLoopPreheader()->getTerminator());
  LLVM_DEBUG(dbgs() << "Created new trip count in preheader: ";
             NewTripCount->dump());

  // Fix up PHI nodes that take values from the inner loop back-edge, which
  // we are about to remove.
  InnerInductionPHI->removeIncomingValue(InnerLoop->getLoopLatch());
  for (PHINode *PHI : InnerPHIsToTransform)
    PHI->removeIncomingValue(InnerLoop->getLoopLatch());

  // Modify the trip count of the outer loop to be the product of the two
  // trip counts.
  cast<User>(OuterBranch->getCondition())->setOperand(1, NewTripCount);

  // Replace the inner loop backedge with an unconditional branch to the exit.
  BasicBlock *InnerExitBlock = InnerLoop->getExitBlock();
  BasicBlock *InnerExitingBlock = InnerLoop->getExitingBlock();
  InnerExitingBlock->getTerminator()->eraseFromParent();
  BranchInst::Create(InnerExitBlock, InnerExitingBlock);
  DT->deleteEdge(InnerExitingBlock, InnerLoop->getHeader());

  // Replace all uses of the polynomial calculated from the two induction
  // variables with the one new one.
  for (Value *V : LinearIVUses)
    V->replaceAllUsesWith(OuterInductionPHI);

  // Tell LoopInfo, SCEV and the pass manager that the inner loop has been
  // deleted, and any information that have about the outer loop invalidated.
  markLoopAsDeleted(InnerLoop);
  SE->forgetLoop(OuterLoop);
  SE->forgetLoop(InnerLoop);
  LI->erase(InnerLoop);

  return true;
}

PreservedAnalyses LoopFlattenPass::run(Loop &L, LoopAnalysisManager &AM,
                                       LoopStandardAnalysisResults &AR,
                                       LPMUpdater &Updater) {
  if (L.getSubLoops().size() != 1)
    return PreservedAnalyses::all();

  Loop *InnerLoop = *L.begin();
  std::string LoopName(InnerLoop->getName());
  if (!FlattenLoopPair(
          &L, InnerLoop, &AR.DT, &AR.LI, &AR.SE, &AR.AC, &AR.TTI,
          [&](Loop *L) { Updater.markLoopAsDeleted(*L, LoopName); }))
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}

namespace {
class LoopFlattenLegacyPass : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid
  LoopFlattenLegacyPass() : LoopPass(ID) {
    initializeLoopFlattenLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // Possibly flatten loop L into its child.
  bool runOnLoop(Loop *L, LPPassManager &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    getLoopAnalysisUsage(AU);
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addPreserved<TargetTransformInfoWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addPreserved<AssumptionCacheTracker>();
  }
};
} // namespace

char LoopFlattenLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LoopFlattenLegacyPass, "loop-flatten", "Flattens loops",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(LoopFlattenLegacyPass, "loop-flatten", "Flattens loops",
                    false, false)

Pass *llvm::createLoopFlattenPass() { return new LoopFlattenLegacyPass(); }

bool LoopFlattenLegacyPass::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipLoop(L))
    return false;

  if (L->getSubLoops().size() != 1)
    return false;

  ScalarEvolution *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DominatorTree *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  auto &TTIP = getAnalysis<TargetTransformInfoWrapperPass>();
  TargetTransformInfo *TTI = &TTIP.getTTI(*L->getHeader()->getParent());
  AssumptionCache *AC =
      &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(
          *L->getHeader()->getParent());

  Loop *InnerLoop = *L->begin();
  return FlattenLoopPair(L, InnerLoop, DT, LI, SE, AC, TTI,
                         [&](Loop *L) { LPM.markLoopAsDeleted(*L); });
}
