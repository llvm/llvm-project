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

static cl::opt<bool> DisableBinopExtractShuffle(
    "disable-binop-extract-shuffle", cl::init(false), cl::Hidden,
    cl::desc("Disable binop extract to shuffle transforms"));


/// Compare the relative costs of 2 extracts followed by scalar operation vs.
/// vector operation(s) followed by extract. Return true if the existing
/// instructions are cheaper than a vector alternative. Otherwise, return false
/// and if one of the extracts should be transformed to a shufflevector, set
/// \p ConvertToShuffle to that extract instruction.
static bool isExtractExtractCheap(Instruction *Ext0, Instruction *Ext1,
                                  unsigned Opcode,
                                  const TargetTransformInfo &TTI,
                                  Instruction *&ConvertToShuffle) {
  assert(isa<ConstantInt>(Ext0->getOperand(1)) &&
         isa<ConstantInt>(Ext1->getOperand(1)) &&
         "Expected constant extract indexes");
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

  // Get cost estimates for the extract elements. These costs will factor into
  // both sequences.
  unsigned Ext0Index = cast<ConstantInt>(Ext0->getOperand(1))->getZExtValue();
  unsigned Ext1Index = cast<ConstantInt>(Ext1->getOperand(1))->getZExtValue();

  int Extract0Cost = TTI.getVectorInstrCost(Instruction::ExtractElement,
                                            VecTy, Ext0Index);
  int Extract1Cost = TTI.getVectorInstrCost(Instruction::ExtractElement,
                                            VecTy, Ext1Index);

  // A more expensive extract will always be replaced by a splat shuffle.
  // For example, if Ext0 is more expensive:
  // opcode (extelt V0, Ext0), (ext V1, Ext1) -->
  // extelt (opcode (splat V0, Ext0), V1), Ext1
  // TODO: Evaluate whether that always results in lowest cost. Alternatively,
  //       check the cost of creating a broadcast shuffle and shuffling both
  //       operands to element 0.
  int CheapExtractCost = std::min(Extract0Cost, Extract1Cost);

  // Extra uses of the extracts mean that we include those costs in the
  // vector total because those instructions will not be eliminated.
  int OldCost, NewCost;
  if (Ext0->getOperand(0) == Ext1->getOperand(0) && Ext0Index == Ext1Index) {
    // Handle a special case. If the 2 extracts are identical, adjust the
    // formulas to account for that. The extra use charge allows for either the
    // CSE'd pattern or an unoptimized form with identical values:
    // opcode (extelt V, C), (extelt V, C) --> extelt (opcode V, V), C
    bool HasUseTax = Ext0 == Ext1 ? !Ext0->hasNUses(2)
                                  : !Ext0->hasOneUse() || !Ext1->hasOneUse();
    OldCost = CheapExtractCost + ScalarOpCost;
    NewCost = VectorOpCost + CheapExtractCost + HasUseTax * CheapExtractCost;
  } else {
    // Handle the general case. Each extract is actually a different value:
    // opcode (extelt V0, C0), (extelt V1, C1) --> extelt (opcode V0, V1), C
    OldCost = Extract0Cost + Extract1Cost + ScalarOpCost;
    NewCost = VectorOpCost + CheapExtractCost +
              !Ext0->hasOneUse() * Extract0Cost +
              !Ext1->hasOneUse() * Extract1Cost;
  }

  if (Ext0Index == Ext1Index) {
    // If the extract indexes are identical, no shuffle is needed.
    ConvertToShuffle = nullptr;
  } else {
    if (IsBinOp && DisableBinopExtractShuffle)
      return true;

    // If we are extracting from 2 different indexes, then one operand must be
    // shuffled before performing the vector operation. The shuffle mask is
    // undefined except for 1 lane that is being translated to the remaining
    // extraction lane. Therefore, it is a splat shuffle. Ex:
    // ShufMask = { undef, undef, 0, undef }
    // TODO: The cost model has an option for a "broadcast" shuffle
    //       (splat-from-element-0), but no option for a more general splat.
    NewCost +=
        TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, VecTy);

    // The more expensive extract will be replaced by a shuffle. If the extracts
    // have the same cost, replace the extract with the higher index.
    if (Extract0Cost > Extract1Cost)
      ConvertToShuffle = Ext0;
    else if (Extract1Cost > Extract0Cost)
      ConvertToShuffle = Ext1;
    else
      ConvertToShuffle = Ext0Index > Ext1Index ? Ext0 : Ext1;
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

  Instruction *ConvertToShuffle;
  if (isExtractExtractCheap(Ext0, Ext1, I.getOpcode(), TTI, ConvertToShuffle))
    return false;

  if (ConvertToShuffle) {
    // The shuffle mask is undefined except for 1 lane that is being translated
    // to the cheap extraction lane. Example:
    // ShufMask = { 2, undef, undef, undef }
    uint64_t SplatIndex = ConvertToShuffle == Ext0 ? C0 : C1;
    uint64_t CheapExtIndex = ConvertToShuffle == Ext0 ? C1 : C0;
    Type *VecTy = V0->getType();
    Type *I32Ty = IntegerType::getInt32Ty(I.getContext());
    UndefValue *Undef = UndefValue::get(I32Ty);
    SmallVector<Constant *, 32> ShufMask(VecTy->getVectorNumElements(), Undef);
    ShufMask[CheapExtIndex] = ConstantInt::get(I32Ty, SplatIndex);
    IRBuilder<> Builder(ConvertToShuffle);

    // extelt X, C --> extelt (splat X), C'
    Value *Shuf = Builder.CreateShuffleVector(ConvertToShuffle->getOperand(0),
                                              UndefValue::get(VecTy),
                                              ConstantVector::get(ShufMask));
    Value *NewExt = Builder.CreateExtractElement(Shuf, CheapExtIndex);
    if (ConvertToShuffle == Ext0)
      Ext0 = cast<Instruction>(NewExt);
    else
      Ext1 = cast<Instruction>(NewExt);
  }

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
