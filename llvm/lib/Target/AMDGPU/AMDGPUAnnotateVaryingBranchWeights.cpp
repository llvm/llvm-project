//===- AMDGPUAnnotateVaryingBranchWeights.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Estimate if conditional branches for which SIAnnotateControlFlow introduced
// amdgcn_if or amdgcn_else intrinsics are likely to have different outcomes for
// the threads of each wavefront. If that is the case, BranchWeight metadata is
// added to signal that "then" and "else" blocks are both likely to be executed.
// This may introduce branch weights that would be self-contradictory in a
// non-SIMT setting.
//
// A consequence of this is that SIPreEmitPeephole is more likely to eliminate
// s_cbranch_execz instructions that were introduced to skip these blocks when
// no thread in the wavefront is active for them.
//
// Should only run after SIAnnotateControlFlow.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-annotate-varying-branch-weights"

namespace {

class AMDGPUAnnotateVaryingBranchWeightsImpl {
public:
  AMDGPUAnnotateVaryingBranchWeightsImpl() = delete;
  AMDGPUAnnotateVaryingBranchWeightsImpl(const GCNSubtarget &ST,
                                         const TargetTransformInfo &TTI)
      : ST(ST), TTI(TTI) {
    // Determine weights that signal that a branch is very likely to be
    // predicted correctly, i.e., whose ratio exceeds
    // TTI.getPredictableBranchThreshold().
    auto BranchProbThreshold = TTI.getPredictableBranchThreshold();
    LikelyWeight = BranchProbThreshold.getNumerator();
    UnlikelyWeight = BranchProbThreshold.getDenominator() - LikelyWeight;
    if (UnlikelyWeight > 0)
      --UnlikelyWeight;
  }

  bool run(Function &F);

private:
  const GCNSubtarget &ST;
  const TargetTransformInfo &TTI;
  uint32_t LikelyWeight;
  uint32_t UnlikelyWeight;
  ValueMap<const Value *, bool> LikelyVaryingCache;

  /// Heuristically check if it is likely that a wavefront has dynamically
  /// varying values for V.
  bool isLikelyVarying(const Value *V);

  /// Set branch weights that signal that the "true" successor of Term is the
  /// likely destination, if no prior weights are present.
  /// Return true if weights were set.
  bool setTrueSuccessorLikely(BranchInst *Term);
};

class AMDGPUAnnotateVaryingBranchWeightsLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPUAnnotateVaryingBranchWeightsLegacy() : FunctionPass(ID) {
    initializeAMDGPUAnnotateVaryingBranchWeightsLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "AMDGPU Annotate Varying Branch Weights";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override {
    TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
    const TargetMachine &TM = TPC.getTM<TargetMachine>();
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    const TargetTransformInfo &TTI = TM.getTargetTransformInfo(F);
    return AMDGPUAnnotateVaryingBranchWeightsImpl(ST, TTI).run(F);
  }
};

} // end anonymous namespace

char AMDGPUAnnotateVaryingBranchWeightsLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUAnnotateVaryingBranchWeightsLegacy, DEBUG_TYPE,
                      "Annotate Varying Branch Weights", false, false)
INITIALIZE_PASS_END(AMDGPUAnnotateVaryingBranchWeightsLegacy, DEBUG_TYPE,
                    "Annotate Varying Branch Weights", false, false)

FunctionPass *llvm::createAMDGPUAnnotateVaryingBranchWeightsLegacyPass() {
  return new AMDGPUAnnotateVaryingBranchWeightsLegacy();
}

PreservedAnalyses
AMDGPUAnnotateVaryingBranchWeightsPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
  const TargetTransformInfo &TTI = TM.getTargetTransformInfo(F);
  bool Changed = AMDGPUAnnotateVaryingBranchWeightsImpl(ST, TTI).run(F);

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool AMDGPUAnnotateVaryingBranchWeightsImpl::isLikelyVarying(const Value *V) {
  // Check if V is a source of divergence or if it transitively uses one.
  if (TTI.isSourceOfDivergence(V))
    return true;

  auto *U = dyn_cast<User>(V);
  if (!U)
    return false;

  // Have we already checked V?
  auto CacheEntry = LikelyVaryingCache.find(V);
  if (CacheEntry != LikelyVaryingCache.end())
    return CacheEntry->second;

  // Does it use a likely varying Value?
  bool Result = false;
  for (const auto &Use : U->operands()) {
    Result |= isLikelyVarying(Use);
    if (Result)
      break;
  }

  LikelyVaryingCache.insert({V, Result});
  return Result;
}

bool AMDGPUAnnotateVaryingBranchWeightsImpl::setTrueSuccessorLikely(
    BranchInst *Term) {
  assert(Term->isConditional());

  // Don't overwrite existing branch weights.
  if (hasProfMD(*Term))
    return false;

  llvm::setBranchWeights(*Term, {LikelyWeight, UnlikelyWeight}, false);
  LLVM_DEBUG(dbgs() << "Added branch weights: " << *Term << '\n');
  return true;
}

bool AMDGPUAnnotateVaryingBranchWeightsImpl::run(Function &F) {
  // If the workgroup has only a single thread, the condition cannot vary.
  const auto WGSizes = ST.getFlatWorkGroupSizes(F);
  if (WGSizes.first <= 1)
    return false;

  using namespace PatternMatch;

  bool Changed = false;
  for (auto &BB : F) {
    auto *Term = BB.getTerminator();
    // Look for conditional branches whose condition is an ExtractValueInst
    // that extracts the return value of a call to the amdgcn_if or amdgcn_else
    // intrinsic.
    if (match(Term, m_Br(m_ExtractValue<0>(m_CombineOr(
                             m_Intrinsic<Intrinsic::amdgcn_if>(),
                             m_Intrinsic<Intrinsic::amdgcn_else>())),
                         m_Value(), m_Value()))) {
      // The this condition is an artificial value resulting from the control
      // flow intrinsic, not the actual branch condition. However, the
      // intrinsics connect it via data flow with the actual condition
      // (even for the amdgcn_else intrinsic, via the matching amdgcn_if
      // intrinsic), so isLikelyVarying still produces meaningful results.
      if (isLikelyVarying(cast<BranchInst>(Term)->getCondition()))
        Changed |= setTrueSuccessorLikely(cast<BranchInst>(Term));
    }
  }

  return Changed;
}
