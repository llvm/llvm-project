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
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "vector-combine"
STATISTIC(NumVecCmp, "Number of vector compares formed");
STATISTIC(NumVecBO, "Number of vector binops formed");
STATISTIC(NumScalarBO, "Number of scalar binops formed");

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
                                  Instruction *&ConvertToShuffle,
                                  unsigned PreferredExtractIndex) {
  assert(isa<ConstantInt>(Ext0->getOperand(1)) &&
         isa<ConstantInt>(Ext1->getOperand(1)) &&
         "Expected constant extract indexes");
  Type *ScalarTy = Ext0->getType();
  auto *VecTy = cast<VectorType>(Ext0->getOperand(0)->getType());
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

    // The more expensive extract will be replaced by a shuffle. If the costs
    // are equal and there is a preferred extract index, shuffle the opposite
    // operand. Otherwise, replace the extract with the higher index.
    if (Extract0Cost > Extract1Cost)
      ConvertToShuffle = Ext0;
    else if (Extract1Cost > Extract0Cost)
      ConvertToShuffle = Ext1;
    else if (PreferredExtractIndex == Ext0Index)
      ConvertToShuffle = Ext1;
    else if (PreferredExtractIndex == Ext1Index)
      ConvertToShuffle = Ext0;
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
                          Instruction &I) {
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
                            Instruction &I) {
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
  if (!match(Ext0, m_ExtractElt(m_Value(V0), m_ConstantInt(C0))) ||
      !match(Ext1, m_ExtractElt(m_Value(V1), m_ConstantInt(C1))) ||
      V0->getType() != V1->getType())
    return false;

  // If the scalar value 'I' is going to be re-inserted into a vector, then try
  // to create an extract to that same element. The extract/insert can be
  // reduced to a "select shuffle".
  // TODO: If we add a larger pattern match that starts from an insert, this
  //       probably becomes unnecessary.
  uint64_t InsertIndex = std::numeric_limits<uint64_t>::max();
  if (I.hasOneUse())
    match(I.user_back(),
          m_InsertElt(m_Value(), m_Value(), m_ConstantInt(InsertIndex)));

  Instruction *ConvertToShuffle;
  if (isExtractExtractCheap(Ext0, Ext1, I.getOpcode(), TTI, ConvertToShuffle,
                            InsertIndex))
    return false;

  if (ConvertToShuffle) {
    // The shuffle mask is undefined except for 1 lane that is being translated
    // to the cheap extraction lane. Example:
    // ShufMask = { 2, undef, undef, undef }
    uint64_t SplatIndex = ConvertToShuffle == Ext0 ? C0 : C1;
    uint64_t CheapExtIndex = ConvertToShuffle == Ext0 ? C1 : C0;
    auto *VecTy = cast<VectorType>(V0->getType());
    SmallVector<int, 32> ShufMask(VecTy->getNumElements(), -1);
    ShufMask[CheapExtIndex] = SplatIndex;
    IRBuilder<> Builder(ConvertToShuffle);

    // extelt X, C --> extelt (splat X), C'
    Value *Shuf = Builder.CreateShuffleVector(ConvertToShuffle->getOperand(0),
                                              UndefValue::get(VecTy), ShufMask);
    Value *NewExt = Builder.CreateExtractElement(Shuf, CheapExtIndex);
    if (ConvertToShuffle == Ext0)
      Ext0 = cast<Instruction>(NewExt);
    else
      Ext1 = cast<Instruction>(NewExt);
  }

  if (Pred != CmpInst::BAD_ICMP_PREDICATE)
    foldExtExtCmp(Ext0, Ext1, I);
  else
    foldExtExtBinop(Ext0, Ext1, I);

  return true;
}

/// If this is a bitcast of a shuffle, try to bitcast the source vector to the
/// destination type followed by shuffle. This can enable further transforms by
/// moving bitcasts or shuffles together.
static bool foldBitcastShuf(Instruction &I, const TargetTransformInfo &TTI) {
  Value *V;
  ArrayRef<int> Mask;
  if (!match(&I, m_BitCast(
                     m_OneUse(m_Shuffle(m_Value(V), m_Undef(), m_Mask(Mask))))))
    return false;

  // Disallow non-vector casts and length-changing shuffles.
  // TODO: We could allow any shuffle.
  auto *DestTy = dyn_cast<VectorType>(I.getType());
  auto *SrcTy = cast<VectorType>(V->getType());
  if (!DestTy || I.getOperand(0)->getType() != SrcTy)
    return false;

  // The new shuffle must not cost more than the old shuffle. The bitcast is
  // moved ahead of the shuffle, so assume that it has the same cost as before.
  if (TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, DestTy) >
      TTI.getShuffleCost(TargetTransformInfo::SK_PermuteSingleSrc, SrcTy))
    return false;

  unsigned DestNumElts = DestTy->getNumElements();
  unsigned SrcNumElts = SrcTy->getNumElements();
  SmallVector<int, 16> NewMask;
  if (SrcNumElts <= DestNumElts) {
    // The bitcast is from wide to narrow/equal elements. The shuffle mask can
    // always be expanded to the equivalent form choosing narrower elements.
    assert(DestNumElts % SrcNumElts == 0 && "Unexpected shuffle mask");
    unsigned ScaleFactor = DestNumElts / SrcNumElts;
    narrowShuffleMaskElts(ScaleFactor, Mask, NewMask);
  } else {
    // The bitcast is from narrow elements to wide elements. The shuffle mask
    // must choose consecutive elements to allow casting first.
    assert(SrcNumElts % DestNumElts == 0 && "Unexpected shuffle mask");
    unsigned ScaleFactor = SrcNumElts / DestNumElts;
    if (!widenShuffleMaskElts(ScaleFactor, Mask, NewMask))
      return false;
  }
  // bitcast (shuf V, MaskC) --> shuf (bitcast V), MaskC'
  IRBuilder<> Builder(&I);
  Value *CastV = Builder.CreateBitCast(V, DestTy);
  Value *Shuf =
      Builder.CreateShuffleVector(CastV, UndefValue::get(DestTy), NewMask);
  I.replaceAllUsesWith(Shuf);
  return true;
}

/// Match a vector binop instruction with inserted scalar operands and convert
/// to scalar binop followed by insertelement.
static bool scalarizeBinop(Instruction &I, const TargetTransformInfo &TTI) {
  Value *Ins0, *Ins1;
  if (!match(&I, m_BinOp(m_Value(Ins0), m_Value(Ins1))))
    return false;

  // Match against one or both scalar values being inserted into constant
  // vectors:
  // vec_bo VecC0, (inselt VecC1, V1, Index)
  // vec_bo (inselt VecC0, V0, Index), VecC1
  // vec_bo (inselt VecC0, V0, Index), (inselt VecC1, V1, Index)
  // TODO: Deal with mismatched index constants and variable indexes?
  Constant *VecC0 = nullptr, *VecC1 = nullptr;
  Value *V0 = nullptr, *V1 = nullptr;
  uint64_t Index0 = 0, Index1 = 0;
  if (!match(Ins0, m_InsertElt(m_Constant(VecC0), m_Value(V0),
                               m_ConstantInt(Index0))) &&
      !match(Ins0, m_Constant(VecC0)))
    return false;
  if (!match(Ins1, m_InsertElt(m_Constant(VecC1), m_Value(V1),
                               m_ConstantInt(Index1))) &&
      !match(Ins1, m_Constant(VecC1)))
    return false;

  bool IsConst0 = !V0;
  bool IsConst1 = !V1;
  if (IsConst0 && IsConst1)
    return false;
  if (!IsConst0 && !IsConst1 && Index0 != Index1)
    return false;

  // Bail for single insertion if it is a load.
  // TODO: Handle this once getVectorInstrCost can cost for load/stores.
  auto *I0 = dyn_cast_or_null<Instruction>(V0);
  auto *I1 = dyn_cast_or_null<Instruction>(V1);
  if ((IsConst0 && I1 && I1->mayReadFromMemory()) ||
      (IsConst1 && I0 && I0->mayReadFromMemory()))
    return false;

  uint64_t Index = IsConst0 ? Index1 : Index0;
  Type *ScalarTy = IsConst0 ? V1->getType() : V0->getType();
  Type *VecTy = I.getType();
  assert(VecTy->isVectorTy() &&
         (IsConst0 || IsConst1 || V0->getType() == V1->getType()) &&
         (ScalarTy->isIntegerTy() || ScalarTy->isFloatingPointTy()) &&
         "Unexpected types for insert into binop");

  Instruction::BinaryOps Opcode = cast<BinaryOperator>(&I)->getOpcode();
  int ScalarOpCost = TTI.getArithmeticInstrCost(Opcode, ScalarTy);
  int VectorOpCost = TTI.getArithmeticInstrCost(Opcode, VecTy);

  // Get cost estimate for the insert element. This cost will factor into
  // both sequences.
  int InsertCost =
      TTI.getVectorInstrCost(Instruction::InsertElement, VecTy, Index);
  int OldCost = (IsConst0 ? 0 : InsertCost) + (IsConst1 ? 0 : InsertCost) +
                VectorOpCost;
  int NewCost = ScalarOpCost + InsertCost +
                (IsConst0 ? 0 : !Ins0->hasOneUse() * InsertCost) +
                (IsConst1 ? 0 : !Ins1->hasOneUse() * InsertCost);

  // We want to scalarize unless the vector variant actually has lower cost.
  if (OldCost < NewCost)
    return false;

  // vec_bo (inselt VecC0, V0, Index), (inselt VecC1, V1, Index) -->
  // inselt NewVecC, (scalar_bo V0, V1), Index
  ++NumScalarBO;
  IRBuilder<> Builder(&I);

  // For constant cases, extract the scalar element, this should constant fold.
  if (IsConst0)
    V0 = ConstantExpr::getExtractElement(VecC0, Builder.getInt64(Index));
  if (IsConst1)
    V1 = ConstantExpr::getExtractElement(VecC1, Builder.getInt64(Index));

  Value *Scalar = Builder.CreateBinOp(Opcode, V0, V1, I.getName() + ".scalar");

  // All IR flags are safe to back-propagate. There is no potential for extra
  // poison to be created by the scalar instruction.
  if (auto *ScalarInst = dyn_cast<Instruction>(Scalar))
    ScalarInst->copyIRFlags(&I);

  // Fold the vector constants in the original vectors into a new base vector.
  Constant *NewVecC = ConstantExpr::get(Opcode, VecC0, VecC1);
  Value *Insert = Builder.CreateInsertElement(NewVecC, Scalar, Index);
  I.replaceAllUsesWith(Insert);
  Insert->takeName(&I);
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
    // Walk the block forwards to enable simple iterative chains of transforms.
    // TODO: It could be more efficient to remove dead instructions
    //       iteratively in this loop rather than waiting until the end.
    for (Instruction &I : BB) {
      if (isa<DbgInfoIntrinsic>(I))
        continue;
      MadeChange |= foldExtractExtract(I, TTI);
      MadeChange |= foldBitcastShuf(I, TTI);
      MadeChange |= scalarizeBinop(I, TTI);
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
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
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
  PA.preserve<AAManager>();
  PA.preserve<BasicAA>();
  return PA;
}
