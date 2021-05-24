//===----- CodeGen/ExpandVectorPredication.cpp - Expand VP intrinsics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR expansion for vector predication intrinsics, allowing
// targets to enable vector predication until just before codegen.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandVectorPredication.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

using VPLegalization = TargetTransformInfo::VPLegalization;
using VPTransform = TargetTransformInfo::VPLegalization::VPTransform;

// Keep this in sync with TargetTransformInfo::VPLegalization.
#define VPINTERNAL_VPLEGAL_CASES                                               \
  VPINTERNAL_CASE(Legal)                                                       \
  VPINTERNAL_CASE(Discard)                                                     \
  VPINTERNAL_CASE(Convert)

#define VPINTERNAL_CASE(X) "|" #X

// Override options.
static cl::opt<std::string> EVLTransformOverride(
    "expandvp-override-evl-transform", cl::init(""), cl::Hidden,
    cl::desc("Options: <empty>" VPINTERNAL_VPLEGAL_CASES
             ". If non-empty, ignore "
             "TargetTransformInfo and "
             "always use this transformation for the %evl parameter (Used in "
             "testing)."));

static cl::opt<std::string> MaskTransformOverride(
    "expandvp-override-mask-transform", cl::init(""), cl::Hidden,
    cl::desc("Options: <empty>" VPINTERNAL_VPLEGAL_CASES
             ". If non-empty, Ignore "
             "TargetTransformInfo and "
             "always use this transformation for the %mask parameter (Used in "
             "testing)."));

#undef VPINTERNAL_CASE
#define VPINTERNAL_CASE(X) .Case(#X, VPLegalization::X)

static VPTransform parseOverrideOption(const std::string &TextOpt) {
  return StringSwitch<VPTransform>(TextOpt) VPINTERNAL_VPLEGAL_CASES;
}

#undef VPINTERNAL_VPLEGAL_CASES

// Whether any override options are set.
static bool anyExpandVPOverridesSet() {
  return !EVLTransformOverride.empty() || !MaskTransformOverride.empty();
}

#define DEBUG_TYPE "expandvp"

STATISTIC(NumFoldedVL, "Number of folded vector length params");
STATISTIC(NumLoweredVPOps, "Number of folded vector predication operations");

///// Helpers {

/// \returns Whether the vector mask \p MaskVal has all lane bits set.
static bool isAllTrueMask(Value *MaskVal) {
  auto *ConstVec = dyn_cast<ConstantVector>(MaskVal);
  return ConstVec && ConstVec->isAllOnesValue();
}

/// \returns A non-excepting divisor constant for this type.
static Constant *getSafeDivisor(Type *DivTy) {
  assert(DivTy->isIntOrIntVectorTy() && "Unsupported divisor type");
  return ConstantInt::get(DivTy, 1u, false);
}

/// Transfer operation properties from \p OldVPI to \p NewVal.
static void transferDecorations(Value &NewVal, VPIntrinsic &VPI) {
  auto *NewInst = dyn_cast<Instruction>(&NewVal);
  if (!NewInst || !isa<FPMathOperator>(NewVal))
    return;

  auto *OldFMOp = dyn_cast<FPMathOperator>(&VPI);
  if (!OldFMOp)
    return;

  NewInst->setFastMathFlags(OldFMOp->getFastMathFlags());
}

/// Transfer all properties from \p OldOp to \p NewOp and replace all uses.
/// OldVP gets erased.
static void replaceOperation(Value &NewOp, VPIntrinsic &OldOp) {
  transferDecorations(NewOp, OldOp);
  OldOp.replaceAllUsesWith(&NewOp);
  OldOp.eraseFromParent();
}

//// } Helpers

namespace {

// Expansion pass state at function scope.
struct CachingVPExpander {
  Function &F;
  const TargetTransformInfo &TTI;

  /// \returns A (fixed length) vector with ascending integer indices
  /// (<0, 1, ..., NumElems-1>).
  /// \p Builder
  ///    Used for instruction creation.
  /// \p LaneTy
  ///    Integer element type of the result vector.
  /// \p NumElems
  ///    Number of vector elements.
  Value *createStepVector(IRBuilder<> &Builder, Type *LaneTy,
                          unsigned NumElems);

  /// \returns A bitmask that is true where the lane position is less-than \p
  /// EVLParam
  ///
  /// \p Builder
  ///    Used for instruction creation.
  /// \p VLParam
  ///    The explicit vector length parameter to test against the lane
  ///    positions.
  /// \p ElemCount
  ///    Static (potentially scalable) number of vector elements.
  Value *convertEVLToMask(IRBuilder<> &Builder, Value *EVLParam,
                          ElementCount ElemCount);

  Value *foldEVLIntoMask(VPIntrinsic &VPI);

  /// "Remove" the %evl parameter of \p PI by setting it to the static vector
  /// length of the operation.
  void discardEVLParameter(VPIntrinsic &PI);

  /// \brief Lower this VP binary operator to a unpredicated binary operator.
  Value *expandPredicationInBinaryOperator(IRBuilder<> &Builder,
                                           VPIntrinsic &PI);

  /// \brief Query TTI and expand the vector predication in \p P accordingly.
  Value *expandPredication(VPIntrinsic &PI);

  /// \brief  Determine how and whether the VPIntrinsic \p VPI shall be
  /// expanded. This overrides TTI with the cl::opts listed at the top of this
  /// file.
  VPLegalization getVPLegalizationStrategy(const VPIntrinsic &VPI) const;
  bool UsingTTIOverrides;

public:
  CachingVPExpander(Function &F, const TargetTransformInfo &TTI)
      : F(F), TTI(TTI), UsingTTIOverrides(anyExpandVPOverridesSet()) {}

  bool expandVectorPredication();
};

//// CachingVPExpander {

Value *CachingVPExpander::createStepVector(IRBuilder<> &Builder, Type *LaneTy,
                                           unsigned NumElems) {
  // TODO add caching
  SmallVector<Constant *, 16> ConstElems;

  for (unsigned Idx = 0; Idx < NumElems; ++Idx)
    ConstElems.push_back(ConstantInt::get(LaneTy, Idx, false));

  return ConstantVector::get(ConstElems);
}

Value *CachingVPExpander::convertEVLToMask(IRBuilder<> &Builder,
                                           Value *EVLParam,
                                           ElementCount ElemCount) {
  // TODO add caching
  // Scalable vector %evl conversion.
  if (ElemCount.isScalable()) {
    auto *M = Builder.GetInsertBlock()->getModule();
    Type *BoolVecTy = VectorType::get(Builder.getInt1Ty(), ElemCount);
    Function *ActiveMaskFunc = Intrinsic::getDeclaration(
        M, Intrinsic::get_active_lane_mask, {BoolVecTy, EVLParam->getType()});
    // `get_active_lane_mask` performs an implicit less-than comparison.
    Value *ConstZero = Builder.getInt32(0);
    return Builder.CreateCall(ActiveMaskFunc, {ConstZero, EVLParam});
  }

  // Fixed vector %evl conversion.
  Type *LaneTy = EVLParam->getType();
  unsigned NumElems = ElemCount.getFixedValue();
  Value *VLSplat = Builder.CreateVectorSplat(NumElems, EVLParam);
  Value *IdxVec = createStepVector(Builder, LaneTy, NumElems);
  return Builder.CreateICmp(CmpInst::ICMP_ULT, IdxVec, VLSplat);
}

Value *
CachingVPExpander::expandPredicationInBinaryOperator(IRBuilder<> &Builder,
                                                     VPIntrinsic &VPI) {
  assert((isSafeToSpeculativelyExecute(&VPI) ||
          VPI.canIgnoreVectorLengthParam()) &&
         "Implicitly dropping %evl in non-speculatable operator!");

  auto OC = static_cast<Instruction::BinaryOps>(*VPI.getFunctionalOpcode());
  assert(Instruction::isBinaryOp(OC));

  Value *Op0 = VPI.getOperand(0);
  Value *Op1 = VPI.getOperand(1);
  Value *Mask = VPI.getMaskParam();

  // Blend in safe operands.
  if (Mask && !isAllTrueMask(Mask)) {
    switch (OC) {
    default:
      // Can safely ignore the predicate.
      break;

    // Division operators need a safe divisor on masked-off lanes (1).
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      // 2nd operand must not be zero.
      Value *SafeDivisor = getSafeDivisor(VPI.getType());
      Op1 = Builder.CreateSelect(Mask, Op1, SafeDivisor);
    }
  }

  Value *NewBinOp = Builder.CreateBinOp(OC, Op0, Op1, VPI.getName());

  replaceOperation(*NewBinOp, VPI);
  return NewBinOp;
}

void CachingVPExpander::discardEVLParameter(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Discard EVL parameter in " << VPI << "\n");

  if (VPI.canIgnoreVectorLengthParam())
    return;

  Value *EVLParam = VPI.getVectorLengthParam();
  if (!EVLParam)
    return;

  ElementCount StaticElemCount = VPI.getStaticVectorLength();
  Value *MaxEVL = nullptr;
  Type *Int32Ty = Type::getInt32Ty(VPI.getContext());
  if (StaticElemCount.isScalable()) {
    // TODO add caching
    auto *M = VPI.getModule();
    Function *VScaleFunc =
        Intrinsic::getDeclaration(M, Intrinsic::vscale, Int32Ty);
    IRBuilder<> Builder(VPI.getParent(), VPI.getIterator());
    Value *FactorConst = Builder.getInt32(StaticElemCount.getKnownMinValue());
    Value *VScale = Builder.CreateCall(VScaleFunc, {}, "vscale");
    MaxEVL = Builder.CreateMul(VScale, FactorConst, "scalable_size",
                               /*NUW*/ true, /*NSW*/ false);
  } else {
    MaxEVL = ConstantInt::get(Int32Ty, StaticElemCount.getFixedValue(), false);
  }
  VPI.setVectorLengthParam(MaxEVL);
}

Value *CachingVPExpander::foldEVLIntoMask(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Folding vlen for " << VPI << '\n');

  IRBuilder<> Builder(&VPI);

  // Ineffective %evl parameter and so nothing to do here.
  if (VPI.canIgnoreVectorLengthParam())
    return &VPI;

  // Only VP intrinsics can have an %evl parameter.
  Value *OldMaskParam = VPI.getMaskParam();
  Value *OldEVLParam = VPI.getVectorLengthParam();
  assert(OldMaskParam && "no mask param to fold the vl param into");
  assert(OldEVLParam && "no EVL param to fold away");

  LLVM_DEBUG(dbgs() << "OLD evl: " << *OldEVLParam << '\n');
  LLVM_DEBUG(dbgs() << "OLD mask: " << *OldMaskParam << '\n');

  // Convert the %evl predication into vector mask predication.
  ElementCount ElemCount = VPI.getStaticVectorLength();
  Value *VLMask = convertEVLToMask(Builder, OldEVLParam, ElemCount);
  Value *NewMaskParam = Builder.CreateAnd(VLMask, OldMaskParam);
  VPI.setMaskParam(NewMaskParam);

  // Drop the %evl parameter.
  discardEVLParameter(VPI);
  assert(VPI.canIgnoreVectorLengthParam() &&
         "transformation did not render the evl param ineffective!");

  // Reassess the modified instruction.
  return &VPI;
}

Value *CachingVPExpander::expandPredication(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Lowering to unpredicated op: " << VPI << '\n');

  IRBuilder<> Builder(&VPI);

  // Try lowering to a LLVM instruction first.
  auto OC = VPI.getFunctionalOpcode();

  if (OC && Instruction::isBinaryOp(*OC))
    return expandPredicationInBinaryOperator(Builder, VPI);

  return &VPI;
}

//// } CachingVPExpander

struct TransformJob {
  VPIntrinsic *PI;
  TargetTransformInfo::VPLegalization Strategy;
  TransformJob(VPIntrinsic *PI, TargetTransformInfo::VPLegalization InitStrat)
      : PI(PI), Strategy(InitStrat) {}

  bool isDone() const { return Strategy.shouldDoNothing(); }
};

void sanitizeStrategy(Instruction &I, VPLegalization &LegalizeStrat) {
  // Speculatable instructions do not strictly need predication.
  if (isSafeToSpeculativelyExecute(&I)) {
    // Converting a speculatable VP intrinsic means dropping %mask and %evl.
    // No need to expand %evl into the %mask only to ignore that code.
    if (LegalizeStrat.OpStrategy == VPLegalization::Convert)
      LegalizeStrat.EVLParamStrategy = VPLegalization::Discard;
    return;
  }

  // We have to preserve the predicating effect of %evl for this
  // non-speculatable VP intrinsic.
  // 1) Never discard %evl.
  // 2) If this VP intrinsic will be expanded to non-VP code, make sure that
  //    %evl gets folded into %mask.
  if ((LegalizeStrat.EVLParamStrategy == VPLegalization::Discard) ||
      (LegalizeStrat.OpStrategy == VPLegalization::Convert)) {
    LegalizeStrat.EVLParamStrategy = VPLegalization::Convert;
  }
}

VPLegalization
CachingVPExpander::getVPLegalizationStrategy(const VPIntrinsic &VPI) const {
  auto VPStrat = TTI.getVPLegalizationStrategy(VPI);
  if (LLVM_LIKELY(!UsingTTIOverrides)) {
    // No overrides - we are in production.
    return VPStrat;
  }

  // Overrides set - we are in testing, the following does not need to be
  // efficient.
  VPStrat.EVLParamStrategy = parseOverrideOption(EVLTransformOverride);
  VPStrat.OpStrategy = parseOverrideOption(MaskTransformOverride);
  return VPStrat;
}

/// \brief Expand llvm.vp.* intrinsics as requested by \p TTI.
bool CachingVPExpander::expandVectorPredication() {
  SmallVector<TransformJob, 16> Worklist;

  // Collect all VPIntrinsics that need expansion and determine their expansion
  // strategy.
  for (auto &I : instructions(F)) {
    auto *VPI = dyn_cast<VPIntrinsic>(&I);
    if (!VPI)
      continue;
    auto VPStrat = getVPLegalizationStrategy(*VPI);
    sanitizeStrategy(I, VPStrat);
    if (!VPStrat.shouldDoNothing())
      Worklist.emplace_back(VPI, VPStrat);
  }
  if (Worklist.empty())
    return false;

  // Transform all VPIntrinsics on the worklist.
  LLVM_DEBUG(dbgs() << "\n:::: Transforming " << Worklist.size()
                    << " instructions ::::\n");
  for (TransformJob Job : Worklist) {
    // Transform the EVL parameter.
    switch (Job.Strategy.EVLParamStrategy) {
    case VPLegalization::Legal:
      break;
    case VPLegalization::Discard:
      discardEVLParameter(*Job.PI);
      break;
    case VPLegalization::Convert:
      if (foldEVLIntoMask(*Job.PI))
        ++NumFoldedVL;
      break;
    }
    Job.Strategy.EVLParamStrategy = VPLegalization::Legal;

    // Replace with a non-predicated operation.
    switch (Job.Strategy.OpStrategy) {
    case VPLegalization::Legal:
      break;
    case VPLegalization::Discard:
      llvm_unreachable("Invalid strategy for operators.");
    case VPLegalization::Convert:
      expandPredication(*Job.PI);
      ++NumLoweredVPOps;
      break;
    }
    Job.Strategy.OpStrategy = VPLegalization::Legal;

    assert(Job.isDone() && "incomplete transformation");
  }

  return true;
}
class ExpandVectorPredication : public FunctionPass {
public:
  static char ID;
  ExpandVectorPredication() : FunctionPass(ID) {
    initializeExpandVectorPredicationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    const auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    CachingVPExpander VPExpander(F, *TTI);
    return VPExpander.expandVectorPredication();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // namespace

char ExpandVectorPredication::ID;
INITIALIZE_PASS_BEGIN(ExpandVectorPredication, "expandvp",
                      "Expand vector predication intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(ExpandVectorPredication, "expandvp",
                    "Expand vector predication intrinsics", false, false)

FunctionPass *llvm::createExpandVectorPredicationPass() {
  return new ExpandVectorPredication();
}

PreservedAnalyses
ExpandVectorPredicationPass::run(Function &F, FunctionAnalysisManager &AM) {
  const auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  CachingVPExpander VPExpander(F, TTI);
  if (!VPExpander.expandVectorPredication())
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
