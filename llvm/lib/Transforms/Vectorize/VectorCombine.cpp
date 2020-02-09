//===------- VectorCombine.cpp - Optimize partial vector operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes scalar/vector interactions using target cost models. The
// transforms implemented here may not fit in traditional loop-based or SLP
// vectorization passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/VectorCombine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "vector-combine"
STATISTIC(NumVecCmp, "Number of vector compares formed");
DEBUG_COUNTER(VecCombineCounter, "vector-combine-transform",
              "Controls transformations in vector-combine pass");

static bool foldExtractCmp(Instruction &I, const TargetTransformInfo &TTI) {
  // Match a cmp with extracted vector operands.
  CmpInst::Predicate Pred;
  Instruction *Ext0, *Ext1;
  if (!match(&I, m_Cmp(Pred, m_Instruction(Ext0), m_Instruction(Ext1))))
    return false;

  Value *V0, *V1;
  ConstantInt *C;
  if (!match(Ext0, m_ExtractElement(m_Value(V0), m_ConstantInt(C))) ||
      !match(Ext1, m_ExtractElement(m_Value(V1), m_Specific(C))) ||
      V0->getType() != V1->getType())
    return false;

  Type *ScalarTy = Ext0->getType();
  Type *VecTy = V0->getType();
  bool IsFP = ScalarTy->isFloatingPointTy();
  unsigned CmpOpcode = IsFP ? Instruction::FCmp : Instruction::ICmp;

  // Check if the existing scalar code or the vector alternative is cheaper.
  // Extra uses of the extracts mean that we include those costs in the
  // vector total because those instructions will not be eliminated.
  // ((2 * extract) + scalar cmp) < (vector cmp + extract) ?
  int ExtractCost = TTI.getVectorInstrCost(Instruction::ExtractElement,
                                           VecTy, C->getZExtValue());
  int ScalarCmpCost = TTI.getOperationCost(CmpOpcode, ScalarTy);
  int VecCmpCost = TTI.getOperationCost(CmpOpcode, VecTy);

  int ScalarCost = 2 * ExtractCost + ScalarCmpCost;
  int VecCost = VecCmpCost + ExtractCost +
                !Ext0->hasOneUse() * ExtractCost +
                !Ext1->hasOneUse() * ExtractCost;
  if (ScalarCost < VecCost)
    return false;

  // cmp Pred (extelt V0, C), (extelt V1, C) --> extelt (cmp Pred V0, V1), C
  ++NumVecCmp;
  IRBuilder<> Builder(&I);
  Value *VecCmp = IsFP ? Builder.CreateFCmp(Pred, V0, V1)
                       : Builder.CreateICmp(Pred, V0, V1);
  Value *Ext = Builder.CreateExtractElement(VecCmp, C);
  I.replaceAllUsesWith(Ext);
  return true;
}

/// This is the entry point for all transforms. Pass manager differences are
/// handled in the callers of this function.
static bool runImpl(Function &F, const TargetTransformInfo &TTI,
                    const DominatorTree &DT) {
  bool MadeChange = false;
  for (BasicBlock &BB : F) {
    // Ignore unreachable basic blocks.
    if (!DT.isReachableFromEntry(&BB))
      continue;
    // Do not delete instructions under here and invalidate the iterator.
    // Walk the block backwards for efficiency. We're matching a chain of
    // use->defs, so we're more likely to succeed by starting from the bottom.
    // TODO: It could be more efficient to remove dead instructions
    //       iteratively in this loop rather than waiting until the end.
    for (Instruction &I : make_range(BB.rbegin(), BB.rend())) {
      MadeChange |= foldExtractCmp(I, TTI);
      // TODO: More transforms go here.
    }
  }

  // We're done with transforms, so remove dead instructions.
  if (MadeChange)
    for (BasicBlock &BB : F)
      SimplifyInstructionsInBlock(&BB);

  return MadeChange;
}

// Pass manager boilerplate below here.

namespace {
class VectorCombineLegacyPass : public FunctionPass {
public:
  static char ID;
  VectorCombineLegacyPass() : FunctionPass(ID) {
    initializeVectorCombineLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    return runImpl(F, TTI, DT);
  }
};
} // namespace

char VectorCombineLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(VectorCombineLegacyPass, "vector-combine",
                      "Optimize scalar/vector ops", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(VectorCombineLegacyPass, "vector-combine",
                    "Optimize scalar/vector ops", false, false)
Pass *llvm::createVectorCombinePass() {
  return new VectorCombineLegacyPass();
}

PreservedAnalyses VectorCombinePass::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  if (!runImpl(F, TTI, DT))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<GlobalsAA>();
  return PA;
}
