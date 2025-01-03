//===- AMDGPUAnnotateVaryingBranchWeights.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Estimate if conditional branches for which SIAnnotateControlFlow introduced
// amdgcn_if or amdgcn_else intrinsics are likely to have different outcomes for
// the lanes of each wavefront. If that is the case, BranchWeight metadata is
// added to signal that "then" and "else" blocks are both likely to be executed.
// This may introduce branch weights that would be self-contradictory in a
// non-SIMT setting.
//
// A consequence of this is that SIPreEmitPeephole is more likely to eliminate
// s_cbranch_execz instructions that were introduced to skip these blocks when
// no lane in the wavefront is active for them.
//
// Should only run after SIAnnotateControlFlow.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
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
                                         const TargetTransformInfo &GCNTTI,
                                         UniformityInfo &UA)
      : ST(ST), UA(UA) {
    // Determine weights that signal that a branch is very likely to be
    // predicted correctly, i.e., whose ratio exceeds
    // TTI.getPredictableBranchThreshold().
    auto BranchProbThreshold = GCNTTI.getPredictableBranchThreshold();
    LikelyWeight = BranchProbThreshold.getNumerator();
    UnlikelyWeight = BranchProbThreshold.getDenominator() - LikelyWeight;
    if (UnlikelyWeight > 0)
      --UnlikelyWeight;
  }

  bool run(Function &F);

private:
  const GCNSubtarget &ST;
  const UniformityInfo &UA;
  uint32_t LikelyWeight;
  uint32_t UnlikelyWeight;
  ValueMap<const Value *, bool> LikelyVaryingCache;
  unsigned HighestLikelyVaryingDimension = 0;

  bool isRelevantSourceOfDivergence(const Value *V) const;

  /// Heuristically check if it is likely that a wavefront has dynamically
  /// varying values for V.
  bool isLikelyVarying(const Value *V);

  /// Set branch weights that signal that the "true" successor of Term is the
  /// likely destination, if no prior weights are present.
  /// Return true if weights were set.
  bool setTrueSuccessorLikely(BranchInst *Term) const;
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
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();

    AU.setPreservesCFG();
    AU.addPreserved<UniformityInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
    UniformityInfo &UA =
        getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
    const TargetMachine &TM = TPC.getTM<TargetMachine>();
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    const TargetTransformInfo &GCNTTI = TM.getTargetTransformInfo(F);
    return AMDGPUAnnotateVaryingBranchWeightsImpl(ST, GCNTTI, UA).run(F);
  }
};

} // end anonymous namespace

char AMDGPUAnnotateVaryingBranchWeightsLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUAnnotateVaryingBranchWeightsLegacy, DEBUG_TYPE,
                      "Annotate Varying Branch Weights", false, false)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUAnnotateVaryingBranchWeightsLegacy, DEBUG_TYPE,
                    "Annotate Varying Branch Weights", false, false)

FunctionPass *llvm::createAMDGPUAnnotateVaryingBranchWeightsLegacyPass() {
  return new AMDGPUAnnotateVaryingBranchWeightsLegacy();
}

PreservedAnalyses
AMDGPUAnnotateVaryingBranchWeightsPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
  const TargetTransformInfo &GCNTTI = TM.getTargetTransformInfo(F);
  UniformityInfo &UA = AM.getResult<UniformityInfoAnalysis>(F);
  bool Changed = AMDGPUAnnotateVaryingBranchWeightsImpl(ST, GCNTTI, UA).run(F);

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<UniformityInfoAnalysis>();
  return PA;
}

bool AMDGPUAnnotateVaryingBranchWeightsImpl::isRelevantSourceOfDivergence(
    const Value *V) const {
  auto *II = dyn_cast<IntrinsicInst>(V);
  if (!II)
    return false;

  switch (II->getIntrinsicID()) {
  case Intrinsic::amdgcn_workitem_id_z:
  case Intrinsic::r600_read_tidig_z:
    return HighestLikelyVaryingDimension >= 2;
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::r600_read_tidig_y:
    return HighestLikelyVaryingDimension >= 1;
  case Intrinsic::amdgcn_workitem_id_x:
  case Intrinsic::r600_read_tidig_x:
  case Intrinsic::amdgcn_mbcnt_hi:
  case Intrinsic::amdgcn_mbcnt_lo:
    return true;
  }

  return false;
}

bool AMDGPUAnnotateVaryingBranchWeightsImpl::isLikelyVarying(const Value *V) {
  // Check if V is a source of divergence or if it transitively uses one.
  if (isRelevantSourceOfDivergence(V))
    return true;

  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  // ExtractValueInst and IntrinsicInst enable looking through the
  // amdgcn_if/else intrinsics inserted by SIAnnotateControlFlow.
  // This condition excludes PHINodes, which prevents infinite recursion.
  if (!isa<BinaryOperator>(I) && !isa<UnaryOperator>(I) && !isa<CastInst>(I) &&
      !isa<CmpInst>(I) && !isa<ExtractValueInst>(I) && !isa<IntrinsicInst>(I))
    return false;

  // Have we already checked V?
  auto CacheEntry = LikelyVaryingCache.find(V);
  if (CacheEntry != LikelyVaryingCache.end())
    return CacheEntry->second;

  // Does it use a likely varying Value?
  bool Result = false;
  for (const auto &Use : I->operands()) {
    Result |= isLikelyVarying(Use);
    if (Result)
      break;
  }

  LikelyVaryingCache.insert({V, Result});
  return Result;
}

bool AMDGPUAnnotateVaryingBranchWeightsImpl::setTrueSuccessorLikely(
    BranchInst *Term) const {
  assert(Term->isConditional());

  // Don't overwrite existing branch weights.
  if (hasProfMD(*Term))
    return false;

  llvm::setBranchWeights(*Term, {LikelyWeight, UnlikelyWeight}, false);
  LLVM_DEBUG(dbgs() << "Added branch weights: " << *Term << '\n');
  return true;
}

bool AMDGPUAnnotateVaryingBranchWeightsImpl::run(Function &F) {
  unsigned MinWGSize = ST.getFlatWorkGroupSizes(F).first;
  bool MustHaveMoreThanOneThread = MinWGSize > 1;

  // reqd_work_group_size determines the size of the work group in every
  // dimension. If it is present, identify the dimensions where the workitem id
  // differs between the lanes of the same wavefront. Otherwise assume that
  // only dimension 0, i.e., x, varies.
  //
  // TODO can/should we assume that workitems are grouped into waves like that?
  auto *Node = F.getMetadata("reqd_work_group_size");
  if (Node && Node->getNumOperands() == 3) {
    unsigned WavefrontSize = ST.getWavefrontSize();
    unsigned ThreadsSoFar = 1;
    unsigned Dim = 0;
    for (; Dim < 3; ++Dim) {
      ThreadsSoFar *=
          mdconst::extract<ConstantInt>(Node->getOperand(Dim))->getZExtValue();
      if (ThreadsSoFar >= WavefrontSize)
        break;
    }
    HighestLikelyVaryingDimension = Dim;
    LLVM_DEBUG(dbgs() << "Highest Likely Varying Dimension: " << Dim << '\n');
    MustHaveMoreThanOneThread |= ThreadsSoFar > 1;
  }

  // If the workgroup has only a single thread, the condition cannot vary.
  if (!MustHaveMoreThanOneThread)
    return false;

  bool Changed = false;
  for (auto &BB : F) {
    auto *Br = dyn_cast<BranchInst>(BB.getTerminator());
    // Only consider statically non-uniform conditional branches.
    if (!Br || !Br->isConditional() || UA.isUniform(Br))
      continue;

    if (isLikelyVarying(Br->getCondition()))
      Changed |= setTrueSuccessorLikely(Br);
  }

  return Changed;
}
