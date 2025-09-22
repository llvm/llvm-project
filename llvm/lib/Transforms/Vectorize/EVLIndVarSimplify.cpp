//===---- EVLIndVarSimplify.cpp - Optimize vectorized loops w/ EVL IV------===//
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

#include "llvm/Transforms/Vectorize/EVLIndVarSimplify.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/Local.h"

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
  OptimizationRemarkEmitter *ORE = nullptr;

  EVLIndVarSimplifyImpl(LoopStandardAnalysisResults &LAR,
                        OptimizationRemarkEmitter *ORE)
      : SE(LAR.SE), ORE(ORE) {}

  /// Returns true if modify the loop.
  bool run(Loop &L);
};
} // anonymous namespace

/// Returns the constant part of vectorization factor from the induction
/// variable's step value SCEV expression.
static uint32_t getVFFromIndVar(const SCEV *Step, const Function &F) {
  if (!Step)
    return 0U;

  // Looking for loops with IV step value in the form of `(<constant VF> x
  // vscale)`.
  if (const auto *Mul = dyn_cast<SCEVMulExpr>(Step)) {
    if (Mul->getNumOperands() == 2) {
      const SCEV *LHS = Mul->getOperand(0);
      const SCEV *RHS = Mul->getOperand(1);
      if (const auto *Const = dyn_cast<SCEVConstant>(LHS);
          Const && isa<SCEVVScale>(RHS)) {
        uint64_t V = Const->getAPInt().getLimitedValue();
        if (llvm::isUInt<32>(V))
          return V;
      }
    }
  }

  // If not, see if the vscale_range of the parent function is a fixed value,
  // which makes the step value to be replaced by a constant.
  if (F.hasFnAttribute(Attribute::VScaleRange))
    if (const auto *ConstStep = dyn_cast<SCEVConstant>(Step)) {
      APInt V = ConstStep->getAPInt().abs();
      ConstantRange CR = llvm::getVScaleRange(&F, 64);
      if (const APInt *Fixed = CR.getSingleElement()) {
        V = V.zextOrTrunc(Fixed->getBitWidth());
        uint64_t VF = V.udiv(*Fixed).getLimitedValue();
        if (VF && llvm::isUInt<32>(VF) &&
            // Make sure step is divisible by vscale.
            V.urem(*Fixed).isZero())
          return VF;
      }
    }

  return 0U;
}

bool EVLIndVarSimplifyImpl::run(Loop &L) {
  if (!EnableEVLIndVarSimplify)
    return false;

  if (!getBooleanLoopAttribute(&L, "llvm.loop.isvectorized"))
    return false;
  const MDOperand *EVLMD =
      findStringMetadataForLoop(&L, "llvm.loop.isvectorized.tailfoldingstyle")
          .value_or(nullptr);
  if (!EVLMD || !EVLMD->equalsStr("evl"))
    return false;

  BasicBlock *LatchBlock = L.getLoopLatch();
  ICmpInst *OrigLatchCmp = L.getLatchCmpInst();
  if (!LatchBlock || !OrigLatchCmp)
    return false;

  InductionDescriptor IVD;
  PHINode *IndVar = L.getInductionVariable(SE);
  if (!IndVar || !L.getInductionDescriptor(SE, IVD)) {
    const char *Reason = (IndVar ? "induction descriptor is not available"
                                 : "cannot recognize induction variable");
    LLVM_DEBUG(dbgs() << "Cannot retrieve IV from loop " << L.getName()
                      << " because" << Reason << "\n");
    if (ORE) {
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnrecognizedIndVar",
                                        L.getStartLoc(), L.getHeader())
               << "Cannot retrieve IV because " << ore::NV("Reason", Reason);
      });
    }
    return false;
  }

  BasicBlock *InitBlock, *BackEdgeBlock;
  if (!L.getIncomingAndBackEdge(InitBlock, BackEdgeBlock)) {
    LLVM_DEBUG(dbgs() << "Expect unique incoming and backedge in "
                      << L.getName() << "\n");
    if (ORE) {
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnrecognizedLoopStructure",
                                        L.getStartLoc(), L.getHeader())
               << "Does not have a unique incoming and backedge";
      });
    }
    return false;
  }

  // Retrieve the loop bounds.
  std::optional<Loop::LoopBounds> Bounds = L.getBounds(SE);
  if (!Bounds) {
    LLVM_DEBUG(dbgs() << "Could not obtain the bounds for loop " << L.getName()
                      << "\n");
    if (ORE) {
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnrecognizedLoopStructure",
                                        L.getStartLoc(), L.getHeader())
               << "Could not obtain the loop bounds";
      });
    }
    return false;
  }
  Value *CanonicalIVInit = &Bounds->getInitialIVValue();
  Value *CanonicalIVFinal = &Bounds->getFinalIVValue();

  const SCEV *StepV = IVD.getStep();
  uint32_t VF = getVFFromIndVar(StepV, *L.getHeader()->getParent());
  if (!VF) {
    LLVM_DEBUG(dbgs() << "Could not infer VF from IndVar step '" << *StepV
                      << "'\n");
    if (ORE) {
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnrecognizedIndVar",
                                        L.getStartLoc(), L.getHeader())
               << "Could not infer VF from IndVar step "
               << ore::NV("Step", StepV);
      });
    }
    return false;
  }
  LLVM_DEBUG(dbgs() << "Using VF=" << VF << " for loop " << L.getName()
                    << "\n");

  // Try to find the EVL-based induction variable.
  using namespace PatternMatch;
  BasicBlock *BB = IndVar->getParent();

  Value *EVLIndVar = nullptr;
  Value *RemTC = nullptr;
  Value *TC = nullptr;
  auto IntrinsicMatch = m_Intrinsic<Intrinsic::experimental_get_vector_length>(
      m_Value(RemTC), m_SpecificInt(VF),
      /*Scalable=*/m_SpecificInt(1));
  for (PHINode &PN : BB->phis()) {
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
    assert(RecValue && "expect recurrent IndVar value");

    LLVM_DEBUG(dbgs() << "Found candidate PN of EVL-based IndVar: " << PN
                      << "\n");

    // Check 3: Pattern match to find the EVL-based index and total trip count
    // (TC).
    if (match(RecValue,
              m_c_Add(m_ZExtOrSelf(IntrinsicMatch), m_Specific(&PN))) &&
        match(RemTC, m_Sub(m_Value(TC), m_Specific(&PN)))) {
      EVLIndVar = RecValue;
      break;
    }
  }

  if (!EVLIndVar || !TC)
    return false;

  LLVM_DEBUG(dbgs() << "Using " << *EVLIndVar << " for EVL-based IndVar\n");
  if (ORE) {
    ORE->emit([&]() {
      DebugLoc DL;
      BasicBlock *Region = nullptr;
      if (auto *I = dyn_cast<Instruction>(EVLIndVar)) {
        DL = I->getDebugLoc();
        Region = I->getParent();
      } else {
        DL = L.getStartLoc();
        Region = L.getHeader();
      }
      return OptimizationRemark(DEBUG_TYPE, "UseEVLIndVar", DL, Region)
             << "Using " << ore::NV("EVLIndVar", EVLIndVar)
             << " for EVL-based IndVar";
    });
  }

  // Create an EVL-based comparison and replace the branch to use it as
  // predicate.

  // Loop::getLatchCmpInst check at the beginning of this function has ensured
  // that latch block ends in a conditional branch.
  auto *LatchBranch = cast<BranchInst>(LatchBlock->getTerminator());
  assert(LatchBranch->isConditional() &&
         "expect the loop latch to be ended with a conditional branch");
  ICmpInst::Predicate Pred;
  if (LatchBranch->getSuccessor(0) == L.getHeader())
    Pred = ICmpInst::ICMP_NE;
  else
    Pred = ICmpInst::ICMP_EQ;

  IRBuilder<> Builder(OrigLatchCmp);
  auto *NewLatchCmp = Builder.CreateICmp(Pred, EVLIndVar, TC);
  OrigLatchCmp->replaceAllUsesWith(NewLatchCmp);

  // llvm::RecursivelyDeleteDeadPHINode only deletes cycles whose values are
  // not used outside the cycles. However, in this case the now-RAUW-ed
  // OrigLatchCmp will be considered a use outside the cycle while in reality
  // it's practically dead. Thus we need to remove it before calling
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
  Function &F = *L.getHeader()->getParent();
  auto &FAMProxy = LAM.getResult<FunctionAnalysisManagerLoopProxy>(L, AR);
  OptimizationRemarkEmitter *ORE =
      FAMProxy.getCachedResult<OptimizationRemarkEmitterAnalysis>(F);

  if (EVLIndVarSimplifyImpl(AR, ORE).run(L))
    return PreservedAnalyses::allInSet<CFGAnalyses>();
  return PreservedAnalyses::all();
}
