//===----- CodeGen/ExpandVectorPredication.cpp - Expand VP intrinsics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IR expansion for vector predication intrinsics, allowing
// targets to enable vector predication until just before codegen.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandVectorPredication.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <optional>

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
  if (Value *SplattedVal = getSplatValue(MaskVal))
    if (auto *ConstValue = dyn_cast<Constant>(SplattedVal))
      return ConstValue->isAllOnesValue();

  return false;
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

static bool maySpeculateLanes(VPIntrinsic &VPI) {
  // The result of VP reductions depends on the mask and evl.
  if (isa<VPReductionIntrinsic>(VPI))
    return false;
  // Fallback to whether the intrinsic is speculatable.
  if (auto IntrID = VPI.getFunctionalIntrinsicID())
    return Intrinsic::getFnAttributes(VPI.getContext(), *IntrID)
        .hasAttribute(Attribute::AttrKind::Speculatable);
  if (auto Opc = VPI.getFunctionalOpcode())
    return isSafeToSpeculativelyExecuteWithOpcode(*Opc, &VPI);
  return false;
}

//// } Helpers

namespace {

// Expansion pass state at function scope.
struct CachingVPExpander {
  const TargetTransformInfo &TTI;

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

  /// If needed, folds the EVL in the mask operand and discards the EVL
  /// parameter. Returns a pair of the value of the intrinsic after the change
  /// (if any) and whether the mask was actually folded.
  std::pair<Value *, bool> foldEVLIntoMask(VPIntrinsic &VPI);

  /// "Remove" the %evl parameter of \p PI by setting it to the static vector
  /// length of the operation. Returns true if the %evl (if any) was effectively
  /// changed.
  bool discardEVLParameter(VPIntrinsic &PI);

  /// Lower this VP binary operator to a unpredicated binary operator.
  Value *expandPredicationInBinaryOperator(IRBuilder<> &Builder,
                                           VPIntrinsic &PI);

  /// Lower this VP int call to a unpredicated int call.
  Value *expandPredicationToIntCall(IRBuilder<> &Builder, VPIntrinsic &PI);

  /// Lower this VP fp call to a unpredicated fp call.
  Value *expandPredicationToFPCall(IRBuilder<> &Builder, VPIntrinsic &PI,
                                   unsigned UnpredicatedIntrinsicID);

  /// Lower this VP reduction to a call to an unpredicated reduction intrinsic.
  Value *expandPredicationInReduction(IRBuilder<> &Builder,
                                      VPReductionIntrinsic &PI);

  /// Lower this VP cast operation to a non-VP intrinsic.
  Value *expandPredicationToCastIntrinsic(IRBuilder<> &Builder,
                                          VPIntrinsic &VPI);

  /// Lower this VP memory operation to a non-VP intrinsic.
  Value *expandPredicationInMemoryIntrinsic(IRBuilder<> &Builder,
                                            VPIntrinsic &VPI);

  /// Lower this VP comparison to a call to an unpredicated comparison.
  Value *expandPredicationInComparison(IRBuilder<> &Builder,
                                       VPCmpIntrinsic &PI);

  /// Query TTI and expand the vector predication in \p P accordingly.
  Value *expandPredication(VPIntrinsic &PI);

  /// Determine how and whether the VPIntrinsic \p VPI shall be expanded. This
  /// overrides TTI with the cl::opts listed at the top of this file.
  VPLegalization getVPLegalizationStrategy(const VPIntrinsic &VPI) const;
  bool UsingTTIOverrides;

public:
  CachingVPExpander(const TargetTransformInfo &TTI)
      : TTI(TTI), UsingTTIOverrides(anyExpandVPOverridesSet()) {}

  /// Expand llvm.vp.* intrinsics as requested by \p TTI.
  /// Returns the details of the expansion.
  VPExpansionDetails expandVectorPredication(VPIntrinsic &VPI);
};

//// CachingVPExpander {

Value *CachingVPExpander::convertEVLToMask(IRBuilder<> &Builder,
                                           Value *EVLParam,
                                           ElementCount ElemCount) {
  // TODO add caching
  // Scalable vector %evl conversion.
  if (ElemCount.isScalable()) {
    Type *BoolVecTy = VectorType::get(Builder.getInt1Ty(), ElemCount);
    // `get_active_lane_mask` performs an implicit less-than comparison.
    Value *ConstZero = Builder.getInt32(0);
    return Builder.CreateIntrinsic(Intrinsic::get_active_lane_mask,
                                   {BoolVecTy, EVLParam->getType()},
                                   {ConstZero, EVLParam});
  }

  // Fixed vector %evl conversion.
  Type *LaneTy = EVLParam->getType();
  unsigned NumElems = ElemCount.getFixedValue();
  Value *VLSplat = Builder.CreateVectorSplat(NumElems, EVLParam);
  Value *IdxVec = Builder.CreateStepVector(VectorType::get(LaneTy, ElemCount));
  return Builder.CreateICmp(CmpInst::ICMP_ULT, IdxVec, VLSplat);
}

Value *
CachingVPExpander::expandPredicationInBinaryOperator(IRBuilder<> &Builder,
                                                     VPIntrinsic &VPI) {
  assert((maySpeculateLanes(VPI) || VPI.canIgnoreVectorLengthParam()) &&
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

Value *CachingVPExpander::expandPredicationToIntCall(IRBuilder<> &Builder,
                                                     VPIntrinsic &VPI) {
  std::optional<unsigned> FID = VPI.getFunctionalIntrinsicID();
  if (!FID)
    return nullptr;
  SmallVector<Value *, 2> Argument;
  for (unsigned i = 0; i < VPI.getNumOperands() - 3; i++) {
    Argument.push_back(VPI.getOperand(i));
  }
  Value *NewOp = Builder.CreateIntrinsic(FID.value(), {VPI.getType()}, Argument,
                                         /*FMFSource=*/nullptr, VPI.getName());
  replaceOperation(*NewOp, VPI);
  return NewOp;
}

Value *CachingVPExpander::expandPredicationToFPCall(
    IRBuilder<> &Builder, VPIntrinsic &VPI, unsigned UnpredicatedIntrinsicID) {
  assert((maySpeculateLanes(VPI) || VPI.canIgnoreVectorLengthParam()) &&
         "Implicitly dropping %evl in non-speculatable operator!");

  switch (UnpredicatedIntrinsicID) {
  case Intrinsic::fabs:
  case Intrinsic::sqrt:
  case Intrinsic::maxnum:
  case Intrinsic::minnum: {
    SmallVector<Value *, 2> Argument;
    for (unsigned i = 0; i < VPI.getNumOperands() - 3; i++) {
      Argument.push_back(VPI.getOperand(i));
    }
    Value *NewOp = Builder.CreateIntrinsic(
        UnpredicatedIntrinsicID, {VPI.getType()}, Argument,
        /*FMFSource=*/nullptr, VPI.getName());
    replaceOperation(*NewOp, VPI);
    return NewOp;
  }
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::experimental_constrained_fma:
  case Intrinsic::experimental_constrained_fmuladd: {
    Value *Op0 = VPI.getOperand(0);
    Value *Op1 = VPI.getOperand(1);
    Value *Op2 = VPI.getOperand(2);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        VPI.getModule(), UnpredicatedIntrinsicID, {VPI.getType()});
    Value *NewOp;
    if (Intrinsic::isConstrainedFPIntrinsic(UnpredicatedIntrinsicID))
      NewOp =
          Builder.CreateConstrainedFPCall(Fn, {Op0, Op1, Op2}, VPI.getName());
    else
      NewOp = Builder.CreateCall(Fn, {Op0, Op1, Op2}, VPI.getName());
    replaceOperation(*NewOp, VPI);
    return NewOp;
  }
  }

  return nullptr;
}

static Value *getNeutralReductionElement(const VPReductionIntrinsic &VPI,
                                         Type *EltTy) {
  Intrinsic::ID RdxID = *VPI.getFunctionalIntrinsicID();
  FastMathFlags FMF;
  if (isa<FPMathOperator>(VPI))
    FMF = VPI.getFastMathFlags();
  return getReductionIdentity(RdxID, EltTy, FMF);
}

Value *
CachingVPExpander::expandPredicationInReduction(IRBuilder<> &Builder,
                                                VPReductionIntrinsic &VPI) {
  assert((maySpeculateLanes(VPI) || VPI.canIgnoreVectorLengthParam()) &&
         "Implicitly dropping %evl in non-speculatable operator!");

  Value *Mask = VPI.getMaskParam();
  Value *RedOp = VPI.getOperand(VPI.getVectorParamPos());

  // Insert neutral element in masked-out positions
  if (Mask && !isAllTrueMask(Mask)) {
    auto *NeutralElt = getNeutralReductionElement(VPI, VPI.getType());
    auto *NeutralVector = Builder.CreateVectorSplat(
        cast<VectorType>(RedOp->getType())->getElementCount(), NeutralElt);
    RedOp = Builder.CreateSelect(Mask, RedOp, NeutralVector);
  }

  Value *Reduction;
  Value *Start = VPI.getOperand(VPI.getStartParamPos());

  switch (VPI.getIntrinsicID()) {
  default:
    llvm_unreachable("Impossible reduction kind");
  case Intrinsic::vp_reduce_add:
  case Intrinsic::vp_reduce_mul:
  case Intrinsic::vp_reduce_and:
  case Intrinsic::vp_reduce_or:
  case Intrinsic::vp_reduce_xor: {
    Intrinsic::ID RedID = *VPI.getFunctionalIntrinsicID();
    unsigned Opc = getArithmeticReductionInstruction(RedID);
    assert(Instruction::isBinaryOp(Opc));
    Reduction = Builder.CreateUnaryIntrinsic(RedID, RedOp);
    Reduction =
        Builder.CreateBinOp((Instruction::BinaryOps)Opc, Reduction, Start);
    break;
  }
  case Intrinsic::vp_reduce_smax:
  case Intrinsic::vp_reduce_smin:
  case Intrinsic::vp_reduce_umax:
  case Intrinsic::vp_reduce_umin:
  case Intrinsic::vp_reduce_fmax:
  case Intrinsic::vp_reduce_fmin:
  case Intrinsic::vp_reduce_fmaximum:
  case Intrinsic::vp_reduce_fminimum: {
    Intrinsic::ID RedID = *VPI.getFunctionalIntrinsicID();
    Intrinsic::ID ScalarID = getMinMaxReductionIntrinsicOp(RedID);
    Reduction = Builder.CreateUnaryIntrinsic(RedID, RedOp);
    transferDecorations(*Reduction, VPI);
    Reduction = Builder.CreateBinaryIntrinsic(ScalarID, Reduction, Start);
    break;
  }
  case Intrinsic::vp_reduce_fadd:
    Reduction = Builder.CreateFAddReduce(Start, RedOp);
    break;
  case Intrinsic::vp_reduce_fmul:
    Reduction = Builder.CreateFMulReduce(Start, RedOp);
    break;
  }

  replaceOperation(*Reduction, VPI);
  return Reduction;
}

Value *CachingVPExpander::expandPredicationToCastIntrinsic(IRBuilder<> &Builder,
                                                           VPIntrinsic &VPI) {
  Intrinsic::ID VPID = VPI.getIntrinsicID();
  unsigned CastOpcode = VPIntrinsic::getFunctionalOpcodeForVP(VPID).value();
  assert(Instruction::isCast(CastOpcode));
  Value *CastOp =
      Builder.CreateCast(Instruction::CastOps(CastOpcode), VPI.getOperand(0),
                         VPI.getType(), VPI.getName());

  replaceOperation(*CastOp, VPI);
  return CastOp;
}

Value *
CachingVPExpander::expandPredicationInMemoryIntrinsic(IRBuilder<> &Builder,
                                                      VPIntrinsic &VPI) {
  assert(VPI.canIgnoreVectorLengthParam());

  const auto &DL = VPI.getDataLayout();

  Value *MaskParam = VPI.getMaskParam();
  Value *PtrParam = VPI.getMemoryPointerParam();
  Value *DataParam = VPI.getMemoryDataParam();
  bool IsUnmasked = isAllTrueMask(MaskParam);

  MaybeAlign AlignOpt = VPI.getPointerAlignment();

  Value *NewMemoryInst = nullptr;
  switch (VPI.getIntrinsicID()) {
  default:
    llvm_unreachable("Not a VP memory intrinsic");
  case Intrinsic::vp_store:
    if (IsUnmasked) {
      StoreInst *NewStore =
          Builder.CreateStore(DataParam, PtrParam, /*IsVolatile*/ false);
      if (AlignOpt.has_value())
        NewStore->setAlignment(*AlignOpt);
      NewMemoryInst = NewStore;
    } else
      NewMemoryInst = Builder.CreateMaskedStore(
          DataParam, PtrParam, AlignOpt.valueOrOne(), MaskParam);

    break;
  case Intrinsic::vp_load:
    if (IsUnmasked) {
      LoadInst *NewLoad =
          Builder.CreateLoad(VPI.getType(), PtrParam, /*IsVolatile*/ false);
      if (AlignOpt.has_value())
        NewLoad->setAlignment(*AlignOpt);
      NewMemoryInst = NewLoad;
    } else
      NewMemoryInst = Builder.CreateMaskedLoad(
          VPI.getType(), PtrParam, AlignOpt.valueOrOne(), MaskParam);

    break;
  case Intrinsic::vp_scatter: {
    auto *ElementType =
        cast<VectorType>(DataParam->getType())->getElementType();
    NewMemoryInst = Builder.CreateMaskedScatter(
        DataParam, PtrParam,
        AlignOpt.value_or(DL.getPrefTypeAlign(ElementType)), MaskParam);
    break;
  }
  case Intrinsic::vp_gather: {
    auto *ElementType = cast<VectorType>(VPI.getType())->getElementType();
    NewMemoryInst = Builder.CreateMaskedGather(
        VPI.getType(), PtrParam,
        AlignOpt.value_or(DL.getPrefTypeAlign(ElementType)), MaskParam, nullptr,
        VPI.getName());
    break;
  }
  }

  assert(NewMemoryInst);
  replaceOperation(*NewMemoryInst, VPI);
  return NewMemoryInst;
}

Value *CachingVPExpander::expandPredicationInComparison(IRBuilder<> &Builder,
                                                        VPCmpIntrinsic &VPI) {
  assert((maySpeculateLanes(VPI) || VPI.canIgnoreVectorLengthParam()) &&
         "Implicitly dropping %evl in non-speculatable operator!");

  assert(*VPI.getFunctionalOpcode() == Instruction::ICmp ||
         *VPI.getFunctionalOpcode() == Instruction::FCmp);

  Value *Op0 = VPI.getOperand(0);
  Value *Op1 = VPI.getOperand(1);
  auto Pred = VPI.getPredicate();

  auto *NewCmp = Builder.CreateCmp(Pred, Op0, Op1);

  replaceOperation(*NewCmp, VPI);
  return NewCmp;
}

bool CachingVPExpander::discardEVLParameter(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Discard EVL parameter in " << VPI << "\n");

  if (VPI.canIgnoreVectorLengthParam())
    return false;

  Value *EVLParam = VPI.getVectorLengthParam();
  if (!EVLParam)
    return false;

  ElementCount StaticElemCount = VPI.getStaticVectorLength();
  Value *MaxEVL = nullptr;
  Type *Int32Ty = Type::getInt32Ty(VPI.getContext());
  if (StaticElemCount.isScalable()) {
    // TODO add caching
    IRBuilder<> Builder(VPI.getParent(), VPI.getIterator());
    Value *FactorConst = Builder.getInt32(StaticElemCount.getKnownMinValue());
    Value *VScale = Builder.CreateVScale(Int32Ty, "vscale");
    MaxEVL = Builder.CreateNUWMul(VScale, FactorConst, "scalable_size");
  } else {
    MaxEVL = ConstantInt::get(Int32Ty, StaticElemCount.getFixedValue(), false);
  }
  VPI.setVectorLengthParam(MaxEVL);
  return true;
}

std::pair<Value *, bool> CachingVPExpander::foldEVLIntoMask(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Folding vlen for " << VPI << '\n');

  IRBuilder<> Builder(&VPI);

  // Ineffective %evl parameter and so nothing to do here.
  if (VPI.canIgnoreVectorLengthParam())
    return {&VPI, false};

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
  return {&VPI, true};
}

Value *CachingVPExpander::expandPredication(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Lowering to unpredicated op: " << VPI << '\n');

  IRBuilder<> Builder(&VPI);

  // Try lowering to a LLVM instruction first.
  auto OC = VPI.getFunctionalOpcode();

  if (OC && Instruction::isBinaryOp(*OC))
    return expandPredicationInBinaryOperator(Builder, VPI);

  if (auto *VPRI = dyn_cast<VPReductionIntrinsic>(&VPI))
    return expandPredicationInReduction(Builder, *VPRI);

  if (auto *VPCmp = dyn_cast<VPCmpIntrinsic>(&VPI))
    return expandPredicationInComparison(Builder, *VPCmp);

  if (VPCastIntrinsic::isVPCast(VPI.getIntrinsicID())) {
    return expandPredicationToCastIntrinsic(Builder, VPI);
  }

  switch (VPI.getIntrinsicID()) {
  default:
    break;
  case Intrinsic::vp_fneg: {
    Value *NewNegOp = Builder.CreateFNeg(VPI.getOperand(0), VPI.getName());
    replaceOperation(*NewNegOp, VPI);
    return NewNegOp;
  }
  case Intrinsic::vp_abs:
  case Intrinsic::vp_smax:
  case Intrinsic::vp_smin:
  case Intrinsic::vp_umax:
  case Intrinsic::vp_umin:
  case Intrinsic::vp_bswap:
  case Intrinsic::vp_bitreverse:
  case Intrinsic::vp_ctpop:
  case Intrinsic::vp_ctlz:
  case Intrinsic::vp_cttz:
  case Intrinsic::vp_sadd_sat:
  case Intrinsic::vp_uadd_sat:
  case Intrinsic::vp_ssub_sat:
  case Intrinsic::vp_usub_sat:
  case Intrinsic::vp_fshl:
  case Intrinsic::vp_fshr:
    return expandPredicationToIntCall(Builder, VPI);
  case Intrinsic::vp_fabs:
  case Intrinsic::vp_sqrt:
  case Intrinsic::vp_maxnum:
  case Intrinsic::vp_minnum:
  case Intrinsic::vp_maximum:
  case Intrinsic::vp_minimum:
  case Intrinsic::vp_fma:
  case Intrinsic::vp_fmuladd:
    return expandPredicationToFPCall(Builder, VPI,
                                     VPI.getFunctionalIntrinsicID().value());
  case Intrinsic::vp_load:
  case Intrinsic::vp_store:
  case Intrinsic::vp_gather:
  case Intrinsic::vp_scatter:
    return expandPredicationInMemoryIntrinsic(Builder, VPI);
  }

  if (auto CID = VPI.getConstrainedIntrinsicID())
    if (Value *Call = expandPredicationToFPCall(Builder, VPI, *CID))
      return Call;

  return &VPI;
}

//// } CachingVPExpander

void sanitizeStrategy(VPIntrinsic &VPI, VPLegalization &LegalizeStrat) {
  // Operations with speculatable lanes do not strictly need predication.
  if (maySpeculateLanes(VPI)) {
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

VPExpansionDetails
CachingVPExpander::expandVectorPredication(VPIntrinsic &VPI) {
  auto Strategy = getVPLegalizationStrategy(VPI);
  sanitizeStrategy(VPI, Strategy);

  VPExpansionDetails Changed = VPExpansionDetails::IntrinsicUnchanged;

  // Transform the EVL parameter.
  switch (Strategy.EVLParamStrategy) {
  case VPLegalization::Legal:
    break;
  case VPLegalization::Discard:
    if (discardEVLParameter(VPI))
      Changed = VPExpansionDetails::IntrinsicUpdated;
    break;
  case VPLegalization::Convert:
    if (auto [NewVPI, Folded] = foldEVLIntoMask(VPI); Folded) {
      (void)NewVPI;
      Changed = VPExpansionDetails::IntrinsicUpdated;
      ++NumFoldedVL;
    }
    break;
  }

  // Replace with a non-predicated operation.
  switch (Strategy.OpStrategy) {
  case VPLegalization::Legal:
    break;
  case VPLegalization::Discard:
    llvm_unreachable("Invalid strategy for operators.");
  case VPLegalization::Convert:
    if (Value *V = expandPredication(VPI); V != &VPI) {
      ++NumLoweredVPOps;
      Changed = VPExpansionDetails::IntrinsicReplaced;
    }
    break;
  }

  return Changed;
}
} // namespace

VPExpansionDetails
llvm::expandVectorPredicationIntrinsic(VPIntrinsic &VPI,
                                       const TargetTransformInfo &TTI) {
  return CachingVPExpander(TTI).expandVectorPredication(VPI);
}
