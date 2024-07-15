//===------ EVLIndVarSimplify.cpp - Optimize vectorized loops w/ EVL IV----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes a vectorized loop with canonical IV to using EVL-based
// IV if it was tail-folded by predicated EVL.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/EVLIndVarSimplify.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#define DEBUG_TYPE "evl-iv-simplify"

using namespace llvm;

STATISTIC(NumEliminatedCanonicalIV, "Number of canonical IVs we eliminated");

static cl::opt<bool> EnableEVLIndVarSimplify(
    "enable-evl-indvar-simplify",
    cl::desc("Enable EVL-based induction variable simplify Pass"), cl::Hidden,
    cl::init(true));

namespace {
struct EVLIndVarSimplifyImpl {
  ScalarEvolution &SE;

  explicit EVLIndVarSimplifyImpl(LoopStandardAnalysisResults &LAR)
      : SE(LAR.SE) {}

  explicit EVLIndVarSimplifyImpl(ScalarEvolution &SE) : SE(SE) {}

  // Returns true if modify the loop.
  bool run(Loop &L);
};

struct EVLIndVarSimplify : public LoopPass {
  static char ID;

  EVLIndVarSimplify() : LoopPass(ID) {
    initializeEVLIndVarSimplifyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // anonymous namespace

static uint32_t getVFFromIndVar(const SCEV *Step, const Function &F) {
  if (!Step)
    return 0U;

  // Looking for loops with IV step value in the form of `(<constant VF> x
  // vscale)`.
  if (auto *Mul = dyn_cast<SCEVMulExpr>(Step)) {
    if (Mul->getNumOperands() == 2) {
      const SCEV *LHS = Mul->getOperand(0);
      const SCEV *RHS = Mul->getOperand(1);
      if (auto *Const = dyn_cast<SCEVConstant>(LHS)) {
        uint64_t V = Const->getAPInt().getLimitedValue();
        if (isa<SCEVVScale>(RHS) && llvm::isUInt<32>(V))
          return static_cast<uint32_t>(V);
      }
    }
  }

  // If not, see if the vscale_range of the parent function is a fixed value,
  // which makes the step value to be replaced by a constant.
  if (F.hasFnAttribute(Attribute::VScaleRange))
    if (auto *ConstStep = dyn_cast<SCEVConstant>(Step)) {
      APInt V = ConstStep->getAPInt().abs();
      ConstantRange CR = llvm::getVScaleRange(&F, 64);
      if (const APInt *Fixed = CR.getSingleElement()) {
        V = V.zextOrTrunc(Fixed->getBitWidth());
        uint64_t VF = V.udiv(*Fixed).getLimitedValue();
        if (VF && llvm::isUInt<32>(VF) &&
            // Make sure step is divisible by vscale.
            V.urem(*Fixed).isZero())
          return static_cast<uint32_t>(VF);
      }
    }

  return 0U;
}

bool EVLIndVarSimplifyImpl::run(Loop &L) {
  if (!EnableEVLIndVarSimplify)
    return false;

  InductionDescriptor IVD;
  PHINode *IndVar = L.getInductionVariable(SE);
  if (!IndVar || !L.getInductionDescriptor(SE, IVD)) {
    LLVM_DEBUG(dbgs() << "Cannot retrieve IV from loop " << L.getName()
                      << "\n");
    return false;
  }

  BasicBlock *InitBlock, *BackEdgeBlock;
  if (!L.getIncomingAndBackEdge(InitBlock, BackEdgeBlock)) {
    LLVM_DEBUG(dbgs() << "Expect unique incoming and backedge in "
                      << L.getName() << "\n");
    return false;
  }

  // Retrieve the loop bounds.
  std::optional<Loop::LoopBounds> Bounds = L.getBounds(SE);
  if (!Bounds) {
    LLVM_DEBUG(dbgs() << "Could not obtain the bounds for loop " << L.getName()
                      << "\n");
    return false;
  }
  Value *CanonicalIVInit = &Bounds->getInitialIVValue();
  Value *CanonicalIVFinal = &Bounds->getFinalIVValue();

  const SCEV *StepV = IVD.getStep();
  uint32_t VF = getVFFromIndVar(StepV, *L.getHeader()->getParent());
  if (!VF) {
    LLVM_DEBUG(dbgs() << "Could not infer VF from IndVar step '" << *StepV
                      << "'\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << "Using VF=" << VF << " for loop " << L.getName()
                    << "\n");

  // Try to find the EVL-based induction variable.
  using namespace PatternMatch;
  BasicBlock *BB = IndVar->getParent();

  Value *EVLIndVar = nullptr;
  Value *RemTC = nullptr;
  auto IntrinsicMatch = m_Intrinsic<Intrinsic::experimental_get_vector_length>(
      m_Value(RemTC), m_SpecificInt(VF),
      /*Scalable=*/m_SpecificInt(1));
  for (auto &PN : BB->phis()) {
    if (&PN == IndVar)
      continue;

    // Check 1: it has to contain both incoming (init) & backedge blocks
    // from IndVar.
    if (PN.getBasicBlockIndex(InitBlock) < 0 ||
        PN.getBasicBlockIndex(BackEdgeBlock) < 0)
      continue;
    // Check 2: EVL index is always increasing, thus its inital value has to be
    // equal to either the initial IV value (when the canonical IV is also
    // increasing) or the last IV value (when canonical IV is decreasing).
    Value *Init = PN.getIncomingValueForBlock(InitBlock);
    using Direction = Loop::LoopBounds::Direction;
    switch (Bounds->getDirection()) {
    case Direction::Increasing:
      if (Init != CanonicalIVInit)
        continue;
      break;
    case Direction::Decreasing:
      if (Init != CanonicalIVFinal)
        continue;
      break;
    case Direction::Unknown:
      // To be more permissive and see if either the initial or final IV value
      // matches PN's init value.
      if (Init != CanonicalIVInit && Init != CanonicalIVFinal)
        continue;
      break;
    }
    Value *RecValue = PN.getIncomingValueForBlock(BackEdgeBlock);
    assert(RecValue);

    LLVM_DEBUG(dbgs() << "Found candidate PN of EVL-based IndVar: " << PN
                      << "\n");

    // Check 3: Pattern match to find the EVL-based index.
    if (match(RecValue,
              m_c_Add(m_ZExtOrSelf(IntrinsicMatch), m_Specific(&PN))) &&
        match(RemTC, m_Sub(m_Value(), m_Specific(&PN)))) {
      EVLIndVar = RecValue;
      break;
    }
  }

  if (!EVLIndVar)
    return false;

  const SCEV *BTC = SE.getBackedgeTakenCount(&L);
  LLVM_DEBUG(dbgs() << "BTC: " << *BTC << "\n");
  if (isa<SCEVCouldNotCompute>(BTC))
    return false;

  const SCEV *VFV = SE.getConstant(BTC->getType(), VF);
  VFV = SE.getMulExpr(VFV, SE.getVScale(VFV->getType()));
  const SCEV *ExitValV = SE.getMulExpr(BTC, VFV);
  LLVM_DEBUG(dbgs() << "ExitVal: " << *ExitValV << "\n");

  // Create an EVL-based comparison and replace the branch to use it as
  // predicate.
  ICmpInst *OrigLatchCmp = L.getLatchCmpInst();
  const DataLayout &DL = L.getHeader()->getDataLayout();
  SCEVExpander Expander(SE, DL, "evl.iv.exitcondition");
  if (!Expander.isSafeToExpandAt(ExitValV, OrigLatchCmp))
    return false;

  LLVM_DEBUG(dbgs() << "Using " << *EVLIndVar << " for EVL-based IndVar\n");

  Value *ExitVal =
      Expander.expandCodeFor(ExitValV, EVLIndVar->getType(), OrigLatchCmp);
  IRBuilder<> Builder(OrigLatchCmp);
  auto *NewPred = Builder.CreateICmp(ICmpInst::ICMP_UGT, EVLIndVar, ExitVal);
  OrigLatchCmp->replaceAllUsesWith(NewPred);

  // llvm::RecursivelyDeleteDeadPHINode only deletes cycles whose values are
  // not used outside the cycles. However, in this case the now-RAUW-ed
  // OrigLatchCmp will be consied a use outside the cycle while in reality it's
  // practically dead. Thus we need to remove it before calling
  // RecursivelyDeleteDeadPHINode.
  (void)RecursivelyDeleteTriviallyDeadInstructions(OrigLatchCmp);
  if (llvm::RecursivelyDeleteDeadPHINode(IndVar))
    LLVM_DEBUG(dbgs() << "Removed original IndVar\n");

  ++NumEliminatedCanonicalIV;

  return true;
}

PreservedAnalyses EVLIndVarSimplifyPass::run(Loop &L, LoopAnalysisManager &LAM,
                                             LoopStandardAnalysisResults &AR,
                                             LPMUpdater &U) {
  if (EVLIndVarSimplifyImpl(AR).run(L))
    return PreservedAnalyses::allInSet<CFGAnalyses>();
  return PreservedAnalyses::all();
}

char EVLIndVarSimplify::ID = 0;

INITIALIZE_PASS_BEGIN(EVLIndVarSimplify, DEBUG_TYPE,
                      "EVL-based Induction Variables Simplify", false, false)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(EVLIndVarSimplify, DEBUG_TYPE,
                    "EVL-based Induction Variables Simplify", false, false)

bool EVLIndVarSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipLoop(L))
    return false;

  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  return EVLIndVarSimplifyImpl(SE).run(*L);
}

Pass *llvm::createEVLIndVarSimplifyPass() { return new EVLIndVarSimplify(); }
