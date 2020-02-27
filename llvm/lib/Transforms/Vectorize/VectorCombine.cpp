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
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "vector-combine"
STATISTIC(NumVecCmp, "Number of vector compares formed");
STATISTIC(NumVecBO, "Number of vector binops formed");

static cl::opt<bool> DisableVectorCombine(
    "disable-vector-combine", cl::init(false), cl::Hidden,
    cl::desc("Disable all vector combine transforms"));

/// Compare the relative costs of extracts followed by scalar operation vs.
/// vector operation followed by extract:
/// opcode (extelt V0, C), (extelt V1, C) --> extelt (opcode V0, V1), C
/// Unless the vector op is much more expensive than the scalar op, this
/// eliminates an extract.
static bool isExtractExtractCheap(Instruction *Ext0, Instruction *Ext1,
                                  unsigned Opcode,
                                  const TargetTransformInfo &TTI) {
  assert(Ext0->getOperand(1) == Ext1->getOperand(1) &&
         isa<ConstantInt>(Ext0->getOperand(1)) &&
         "Expected same constant extract index");

  Type *ScalarTy = Ext0->getType();
  Type *VecTy = Ext0->getOperand(0)->getType();
  int ScalarOpCost, VectorOpCost;

  // Get cost estimates for scalar and vector versions of the operation.
  bool IsBinOp = Instruction::isBinaryOp(Opcode);
  if (IsBinOp) {
    ScalarOpCost = TTI.getArithmeticInstrCost(Opcode, ScalarTy);
    VectorOpCost = TTI.getArithmeticInstrCost(Opcode, VecTy);
  } else {
    assert((Opcode == Instruction::ICmp || Opcode == Instruction::FCmp) &&
           "Expected a compare");
    ScalarOpCost = TTI.getCmpSelInstrCost(Opcode, ScalarTy,
                                          CmpInst::makeCmpResultType(ScalarTy));
    VectorOpCost = TTI.getCmpSelInstrCost(Opcode, VecTy,
                                          CmpInst::makeCmpResultType(VecTy));
  }

  // Get cost estimate for the extract element. This cost will factor into
  // both sequences.
  unsigned ExtIndex = cast<ConstantInt>(Ext0->getOperand(1))->getZExtValue();
  int ExtractCost = TTI.getVectorInstrCost(Instruction::ExtractElement,
                                           VecTy, ExtIndex);

  // Extra uses of the extracts mean that we include those costs in the
  // vector total because those instructions will not be eliminated.
  int OldCost, NewCost;
  if (Ext0->getOperand(0) == Ext1->getOperand(0)) {
    // Handle a special case. If the 2 operands are identical, adjust the
    // formulas to account for that. The extra use charge allows for either the
    // CSE'd pattern or an unoptimized form with identical values:
    // opcode (extelt V, C), (extelt V, C) --> extelt (opcode V, V), C
    bool HasUseTax = Ext0 == Ext1 ? !Ext0->hasNUses(2)
                                  : !Ext0->hasOneUse() || !Ext1->hasOneUse();
    OldCost = ExtractCost + ScalarOpCost;
    NewCost = VectorOpCost + ExtractCost + HasUseTax * ExtractCost;
  } else {
    // Handle the general case. Each extract is actually a different value:
    // opcode (extelt V0, C), (extelt V1, C) --> extelt (opcode V0, V1), C
    OldCost = 2 * ExtractCost + ScalarOpCost;
    NewCost = VectorOpCost + ExtractCost + !Ext0->hasOneUse() * ExtractCost +
              !Ext1->hasOneUse() * ExtractCost;
  }
  // Aggressively form a vector op if the cost is equal because the transform
  // may enable further optimization.
  // Codegen can reverse this transform (scalarize) if it was not profitable.
  return OldCost < NewCost;
}

/// Try to reduce extract element costs by converting scalar compares to vector
/// compares followed by extract.
/// cmp (ext0 V0, C), (ext1 V1, C)
static void foldExtExtCmp(Instruction *Ext0, Instruction *Ext1,
                          Instruction &I, const TargetTransformInfo &TTI) {
  assert(isa<CmpInst>(&I) && "Expected a compare");

  // cmp Pred (extelt V0, C), (extelt V1, C) --> extelt (cmp Pred V0, V1), C
  ++NumVecCmp;
  IRBuilder<> Builder(&I);
  CmpInst::Predicate Pred = cast<CmpInst>(&I)->getPredicate();
  Value *V0 = Ext0->getOperand(0), *V1 = Ext1->getOperand(0);
  Value *VecCmp =
      Ext0->getType()->isFloatingPointTy() ? Builder.CreateFCmp(Pred, V0, V1)
                                           : Builder.CreateICmp(Pred, V0, V1);
  Value *Extract = Builder.CreateExtractElement(VecCmp, Ext0->getOperand(1));
  I.replaceAllUsesWith(Extract);
}

/// Try to reduce extract element costs by converting scalar binops to vector
/// binops followed by extract.
/// bo (ext0 V0, C), (ext1 V1, C)
static void foldExtExtBinop(Instruction *Ext0, Instruction *Ext1,
                            Instruction &I, const TargetTransformInfo &TTI) {
  assert(isa<BinaryOperator>(&I) && "Expected a binary operator");

  // bo (extelt V0, C), (extelt V1, C) --> extelt (bo V0, V1), C
  ++NumVecBO;
  IRBuilder<> Builder(&I);
  Value *V0 = Ext0->getOperand(0), *V1 = Ext1->getOperand(0);
  Value *VecBO =
      Builder.CreateBinOp(cast<BinaryOperator>(&I)->getOpcode(), V0, V1);

  // All IR flags are safe to back-propagate because any potential poison
  // created in unused vector elements is discarded by the extract.
  if (auto *VecBOInst = dyn_cast<Instruction>(VecBO))
    VecBOInst->copyIRFlags(&I);

  Value *Extract = Builder.CreateExtractElement(VecBO, Ext0->getOperand(1));
  I.replaceAllUsesWith(Extract);
}

/// Match an instruction with extracted vector operands.
static bool foldExtractExtract(Instruction &I, const TargetTransformInfo &TTI) {
  // It is not safe to transform things like div, urem, etc. because we may
  // create undefined behavior when executing those on unknown vector elements.
  if (!isSafeToSpeculativelyExecute(&I))
    return false;

  Instruction *Ext0, *Ext1;
  CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
  if (!match(&I, m_Cmp(Pred, m_Instruction(Ext0), m_Instruction(Ext1))) &&
      !match(&I, m_BinOp(m_Instruction(Ext0), m_Instruction(Ext1))))
    return false;

  Value *V0, *V1;
  uint64_t C0, C1;
  if (!match(Ext0, m_ExtractElement(m_Value(V0), m_ConstantInt(C0))) ||
      !match(Ext1, m_ExtractElement(m_Value(V1), m_ConstantInt(C1))) ||
      V0->getType() != V1->getType())
    return false;

  // TODO: Handle C0 != C1 by shuffling 1 of the operands.
  if (C0 != C1)
    return false;

  if (isExtractExtractCheap(Ext0, Ext1, I.getOpcode(), TTI))
    return false;

  if (Pred != CmpInst::BAD_ICMP_PREDICATE)
    foldExtExtCmp(Ext0, Ext1, I, TTI);
  else
    foldExtExtBinop(Ext0, Ext1, I, TTI);

  return true;
}

/// This is the entry point for all transforms. Pass manager differences are
/// handled in the callers of this function.
static bool runImpl(Function &F, const TargetTransformInfo &TTI,
                    const DominatorTree &DT) {
  if (DisableVectorCombine)
    return false;

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
    for (Instruction &I : make_range(BB.rbegin(), BB.rend()))
      MadeChange |= foldExtractExtract(I, TTI);
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
