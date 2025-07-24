//===- AMDGPInstCombineIntrinsic.cpp - AMDGPU specific InstCombine pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements a TargetTransformInfo analysis pass specific to the
// AMDGPU target machine. It uses the target's detailed information to provide
// more precise answers to certain TTI queries, while letting the target
// independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUInstrInfo.h"
#include "AMDGPUTargetTransformInfo.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "AMDGPUtti"

namespace {

struct AMDGPUImageDMaskIntrinsic {
  unsigned Intr;
};

#define GET_AMDGPUImageDMaskIntrinsicTable_IMPL
#include "InstCombineTables.inc"

} // end anonymous namespace

// Constant fold llvm.amdgcn.fmed3 intrinsics for standard inputs.
//
// A single NaN input is folded to minnum, so we rely on that folding for
// handling NaNs.
static APFloat fmed3AMDGCN(const APFloat &Src0, const APFloat &Src1,
                           const APFloat &Src2) {
  APFloat Max3 = maxnum(maxnum(Src0, Src1), Src2);

  APFloat::cmpResult Cmp0 = Max3.compare(Src0);
  assert(Cmp0 != APFloat::cmpUnordered && "nans handled separately");
  if (Cmp0 == APFloat::cmpEqual)
    return maxnum(Src1, Src2);

  APFloat::cmpResult Cmp1 = Max3.compare(Src1);
  assert(Cmp1 != APFloat::cmpUnordered && "nans handled separately");
  if (Cmp1 == APFloat::cmpEqual)
    return maxnum(Src0, Src2);

  return maxnum(Src0, Src1);
}

// Check if a value can be converted to a 16-bit value without losing
// precision.
// The value is expected to be either a float (IsFloat = true) or an unsigned
// integer (IsFloat = false).
static bool canSafelyConvertTo16Bit(Value &V, bool IsFloat) {
  Type *VTy = V.getType();
  if (VTy->isHalfTy() || VTy->isIntegerTy(16)) {
    // The value is already 16-bit, so we don't want to convert to 16-bit again!
    return false;
  }
  if (IsFloat) {
    if (ConstantFP *ConstFloat = dyn_cast<ConstantFP>(&V)) {
      // We need to check that if we cast the index down to a half, we do not
      // lose precision.
      APFloat FloatValue(ConstFloat->getValueAPF());
      bool LosesInfo = true;
      FloatValue.convert(APFloat::IEEEhalf(), APFloat::rmTowardZero,
                         &LosesInfo);
      return !LosesInfo;
    }
  } else {
    if (ConstantInt *ConstInt = dyn_cast<ConstantInt>(&V)) {
      // We need to check that if we cast the index down to an i16, we do not
      // lose precision.
      APInt IntValue(ConstInt->getValue());
      return IntValue.getActiveBits() <= 16;
    }
  }

  Value *CastSrc;
  bool IsExt = IsFloat ? match(&V, m_FPExt(PatternMatch::m_Value(CastSrc)))
                       : match(&V, m_ZExt(PatternMatch::m_Value(CastSrc)));
  if (IsExt) {
    Type *CastSrcTy = CastSrc->getType();
    if (CastSrcTy->isHalfTy() || CastSrcTy->isIntegerTy(16))
      return true;
  }

  return false;
}

// Convert a value to 16-bit.
static Value *convertTo16Bit(Value &V, InstCombiner::BuilderTy &Builder) {
  Type *VTy = V.getType();
  if (isa<FPExtInst, SExtInst, ZExtInst>(&V))
    return cast<Instruction>(&V)->getOperand(0);
  if (VTy->isIntegerTy())
    return Builder.CreateIntCast(&V, Type::getInt16Ty(V.getContext()), false);
  if (VTy->isFloatingPointTy())
    return Builder.CreateFPCast(&V, Type::getHalfTy(V.getContext()));

  llvm_unreachable("Should never be called!");
}

/// Applies Func(OldIntr.Args, OldIntr.ArgTys), creates intrinsic call with
/// modified arguments (based on OldIntr) and replaces InstToReplace with
/// this newly created intrinsic call.
static std::optional<Instruction *> modifyIntrinsicCall(
    IntrinsicInst &OldIntr, Instruction &InstToReplace, unsigned NewIntr,
    InstCombiner &IC,
    std::function<void(SmallVectorImpl<Value *> &, SmallVectorImpl<Type *> &)>
        Func) {
  SmallVector<Type *, 4> ArgTys;
  if (!Intrinsic::getIntrinsicSignature(OldIntr.getCalledFunction(), ArgTys))
    return std::nullopt;

  SmallVector<Value *, 8> Args(OldIntr.args());

  // Modify arguments and types
  Func(Args, ArgTys);

  CallInst *NewCall = IC.Builder.CreateIntrinsic(NewIntr, ArgTys, Args);
  NewCall->takeName(&OldIntr);
  NewCall->copyMetadata(OldIntr);
  if (isa<FPMathOperator>(NewCall))
    NewCall->copyFastMathFlags(&OldIntr);

  // Erase and replace uses
  if (!InstToReplace.getType()->isVoidTy())
    IC.replaceInstUsesWith(InstToReplace, NewCall);

  bool RemoveOldIntr = &OldIntr != &InstToReplace;

  auto *RetValue = IC.eraseInstFromFunction(InstToReplace);
  if (RemoveOldIntr)
    IC.eraseInstFromFunction(OldIntr);

  return RetValue;
}

static std::optional<Instruction *>
simplifyAMDGCNImageIntrinsic(const GCNSubtarget *ST,
                             const AMDGPU::ImageDimIntrinsicInfo *ImageDimIntr,
                             IntrinsicInst &II, InstCombiner &IC) {
  // Optimize _L to _LZ when _L is zero
  if (const auto *LZMappingInfo =
          AMDGPU::getMIMGLZMappingInfo(ImageDimIntr->BaseOpcode)) {
    if (auto *ConstantLod =
            dyn_cast<ConstantFP>(II.getOperand(ImageDimIntr->LodIndex))) {
      if (ConstantLod->isZero() || ConstantLod->isNegative()) {
        const AMDGPU::ImageDimIntrinsicInfo *NewImageDimIntr =
            AMDGPU::getImageDimIntrinsicByBaseOpcode(LZMappingInfo->LZ,
                                                     ImageDimIntr->Dim);
        return modifyIntrinsicCall(
            II, II, NewImageDimIntr->Intr, IC, [&](auto &Args, auto &ArgTys) {
              Args.erase(Args.begin() + ImageDimIntr->LodIndex);
            });
      }
    }
  }

  // Optimize _mip away, when 'lod' is zero
  if (const auto *MIPMappingInfo =
          AMDGPU::getMIMGMIPMappingInfo(ImageDimIntr->BaseOpcode)) {
    if (auto *ConstantMip =
            dyn_cast<ConstantInt>(II.getOperand(ImageDimIntr->MipIndex))) {
      if (ConstantMip->isZero()) {
        const AMDGPU::ImageDimIntrinsicInfo *NewImageDimIntr =
            AMDGPU::getImageDimIntrinsicByBaseOpcode(MIPMappingInfo->NONMIP,
                                                     ImageDimIntr->Dim);
        return modifyIntrinsicCall(
            II, II, NewImageDimIntr->Intr, IC, [&](auto &Args, auto &ArgTys) {
              Args.erase(Args.begin() + ImageDimIntr->MipIndex);
            });
      }
    }
  }

  // Optimize _bias away when 'bias' is zero
  if (const auto *BiasMappingInfo =
          AMDGPU::getMIMGBiasMappingInfo(ImageDimIntr->BaseOpcode)) {
    if (auto *ConstantBias =
            dyn_cast<ConstantFP>(II.getOperand(ImageDimIntr->BiasIndex))) {
      if (ConstantBias->isZero()) {
        const AMDGPU::ImageDimIntrinsicInfo *NewImageDimIntr =
            AMDGPU::getImageDimIntrinsicByBaseOpcode(BiasMappingInfo->NoBias,
                                                     ImageDimIntr->Dim);
        return modifyIntrinsicCall(
            II, II, NewImageDimIntr->Intr, IC, [&](auto &Args, auto &ArgTys) {
              Args.erase(Args.begin() + ImageDimIntr->BiasIndex);
              ArgTys.erase(ArgTys.begin() + ImageDimIntr->BiasTyArg);
            });
      }
    }
  }

  // Optimize _offset away when 'offset' is zero
  if (const auto *OffsetMappingInfo =
          AMDGPU::getMIMGOffsetMappingInfo(ImageDimIntr->BaseOpcode)) {
    if (auto *ConstantOffset =
            dyn_cast<ConstantInt>(II.getOperand(ImageDimIntr->OffsetIndex))) {
      if (ConstantOffset->isZero()) {
        const AMDGPU::ImageDimIntrinsicInfo *NewImageDimIntr =
            AMDGPU::getImageDimIntrinsicByBaseOpcode(
                OffsetMappingInfo->NoOffset, ImageDimIntr->Dim);
        return modifyIntrinsicCall(
            II, II, NewImageDimIntr->Intr, IC, [&](auto &Args, auto &ArgTys) {
              Args.erase(Args.begin() + ImageDimIntr->OffsetIndex);
            });
      }
    }
  }

  // Try to use D16
  if (ST->hasD16Images()) {

    const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode =
        AMDGPU::getMIMGBaseOpcodeInfo(ImageDimIntr->BaseOpcode);

    if (BaseOpcode->HasD16) {

      // If the only use of image intrinsic is a fptrunc (with conversion to
      // half) then both fptrunc and image intrinsic will be replaced with image
      // intrinsic with D16 flag.
      if (II.hasOneUse()) {
        Instruction *User = II.user_back();

        if (User->getOpcode() == Instruction::FPTrunc &&
            User->getType()->getScalarType()->isHalfTy()) {

          return modifyIntrinsicCall(II, *User, ImageDimIntr->Intr, IC,
                                     [&](auto &Args, auto &ArgTys) {
                                       // Change return type of image intrinsic.
                                       // Set it to return type of fptrunc.
                                       ArgTys[0] = User->getType();
                                     });
        }
      }

      // Only perform D16 folding if every user of the image sample is
      // an ExtractElementInst immediately followed by an FPTrunc to half.
      SmallVector<std::pair<ExtractElementInst *, FPTruncInst *>, 4>
          ExtractTruncPairs;
      bool AllHalfExtracts = true;

      for (User *U : II.users()) {
        auto *Ext = dyn_cast<ExtractElementInst>(U);
        if (!Ext || !Ext->hasOneUse()) {
          AllHalfExtracts = false;
          break;
        }

        auto *Tr = dyn_cast<FPTruncInst>(*Ext->user_begin());
        if (!Tr || !Tr->getType()->isHalfTy()) {
          AllHalfExtracts = false;
          break;
        }

        ExtractTruncPairs.emplace_back(Ext, Tr);
      }

      if (!ExtractTruncPairs.empty() && AllHalfExtracts) {
        auto *VecTy = cast<VectorType>(II.getType());
        Type *HalfVecTy =
            VecTy->getWithNewType(Type::getHalfTy(II.getContext()));

        // Obtain the original image sample intrinsic's signature
        // and replace its return type with the half-vector for D16 folding
        SmallVector<Type *, 8> SigTys;
        Intrinsic::getIntrinsicSignature(II.getCalledFunction(), SigTys);
        SigTys[0] = HalfVecTy;

        Module *M = II.getModule();
        Function *HalfDecl =
            Intrinsic::getOrInsertDeclaration(M, ImageDimIntr->Intr, SigTys);

        II.mutateType(HalfVecTy);
        II.setCalledFunction(HalfDecl);

        IRBuilder<> Builder(II.getContext());
        for (auto &[Ext, Tr] : ExtractTruncPairs) {
          Value *Idx = Ext->getIndexOperand();

          Builder.SetInsertPoint(Tr);

          Value *HalfExtract = Builder.CreateExtractElement(&II, Idx);
          HalfExtract->takeName(Tr);

          Tr->replaceAllUsesWith(HalfExtract);
        }

        for (auto &[Ext, Tr] : ExtractTruncPairs) {
          IC.eraseInstFromFunction(*Tr);
          IC.eraseInstFromFunction(*Ext);
        }

        return &II;
      }
    }
  }

  // Try to use A16 or G16
  if (!ST->hasA16() && !ST->hasG16())
    return std::nullopt;

  // Address is interpreted as float if the instruction has a sampler or as
  // unsigned int if there is no sampler.
  bool HasSampler =
      AMDGPU::getMIMGBaseOpcodeInfo(ImageDimIntr->BaseOpcode)->Sampler;
  bool FloatCoord = false;
  // true means derivatives can be converted to 16 bit, coordinates not
  bool OnlyDerivatives = false;

  for (unsigned OperandIndex = ImageDimIntr->GradientStart;
       OperandIndex < ImageDimIntr->VAddrEnd; OperandIndex++) {
    Value *Coord = II.getOperand(OperandIndex);
    // If the values are not derived from 16-bit values, we cannot optimize.
    if (!canSafelyConvertTo16Bit(*Coord, HasSampler)) {
      if (OperandIndex < ImageDimIntr->CoordStart ||
          ImageDimIntr->GradientStart == ImageDimIntr->CoordStart) {
        return std::nullopt;
      }
      // All gradients can be converted, so convert only them
      OnlyDerivatives = true;
      break;
    }

    assert(OperandIndex == ImageDimIntr->GradientStart ||
           FloatCoord == Coord->getType()->isFloatingPointTy());
    FloatCoord = Coord->getType()->isFloatingPointTy();
  }

  if (!OnlyDerivatives && !ST->hasA16())
    OnlyDerivatives = true; // Only supports G16

  // Check if there is a bias parameter and if it can be converted to f16
  if (!OnlyDerivatives && ImageDimIntr->NumBiasArgs != 0) {
    Value *Bias = II.getOperand(ImageDimIntr->BiasIndex);
    assert(HasSampler &&
           "Only image instructions with a sampler can have a bias");
    if (!canSafelyConvertTo16Bit(*Bias, HasSampler))
      OnlyDerivatives = true;
  }

  if (OnlyDerivatives && (!ST->hasG16() || ImageDimIntr->GradientStart ==
                                               ImageDimIntr->CoordStart))
    return std::nullopt;

  Type *CoordType = FloatCoord ? Type::getHalfTy(II.getContext())
                               : Type::getInt16Ty(II.getContext());

  return modifyIntrinsicCall(
      II, II, II.getIntrinsicID(), IC, [&](auto &Args, auto &ArgTys) {
        ArgTys[ImageDimIntr->GradientTyArg] = CoordType;
        if (!OnlyDerivatives) {
          ArgTys[ImageDimIntr->CoordTyArg] = CoordType;

          // Change the bias type
          if (ImageDimIntr->NumBiasArgs != 0)
            ArgTys[ImageDimIntr->BiasTyArg] = Type::getHalfTy(II.getContext());
        }

        unsigned EndIndex =
            OnlyDerivatives ? ImageDimIntr->CoordStart : ImageDimIntr->VAddrEnd;
        for (unsigned OperandIndex = ImageDimIntr->GradientStart;
             OperandIndex < EndIndex; OperandIndex++) {
          Args[OperandIndex] =
              convertTo16Bit(*II.getOperand(OperandIndex), IC.Builder);
        }

        // Convert the bias
        if (!OnlyDerivatives && ImageDimIntr->NumBiasArgs != 0) {
          Value *Bias = II.getOperand(ImageDimIntr->BiasIndex);
          Args[ImageDimIntr->BiasIndex] = convertTo16Bit(*Bias, IC.Builder);
        }
      });
}

bool GCNTTIImpl::canSimplifyLegacyMulToMul(const Instruction &I,
                                           const Value *Op0, const Value *Op1,
                                           InstCombiner &IC) const {
  // The legacy behaviour is that multiplying +/-0.0 by anything, even NaN or
  // infinity, gives +0.0. If we can prove we don't have one of the special
  // cases then we can use a normal multiply instead.
  // TODO: Create and use isKnownFiniteNonZero instead of just matching
  // constants here.
  if (match(Op0, PatternMatch::m_FiniteNonZero()) ||
      match(Op1, PatternMatch::m_FiniteNonZero())) {
    // One operand is not zero or infinity or NaN.
    return true;
  }

  SimplifyQuery SQ = IC.getSimplifyQuery().getWithInstruction(&I);
  if (isKnownNeverInfOrNaN(Op0, SQ) && isKnownNeverInfOrNaN(Op1, SQ)) {
    // Neither operand is infinity or NaN.
    return true;
  }
  return false;
}

/// Match an fpext from half to float, or a constant we can convert.
static Value *matchFPExtFromF16(Value *Arg) {
  Value *Src = nullptr;
  ConstantFP *CFP = nullptr;
  if (match(Arg, m_OneUse(m_FPExt(m_Value(Src))))) {
    if (Src->getType()->isHalfTy())
      return Src;
  } else if (match(Arg, m_ConstantFP(CFP))) {
    bool LosesInfo;
    APFloat Val(CFP->getValueAPF());
    Val.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &LosesInfo);
    if (!LosesInfo)
      return ConstantFP::get(Type::getHalfTy(Arg->getContext()), Val);
  }
  return nullptr;
}

// Trim all zero components from the end of the vector \p UseV and return
// an appropriate bitset with known elements.
static APInt trimTrailingZerosInVector(InstCombiner &IC, Value *UseV,
                                       Instruction *I) {
  auto *VTy = cast<FixedVectorType>(UseV->getType());
  unsigned VWidth = VTy->getNumElements();
  APInt DemandedElts = APInt::getAllOnes(VWidth);

  for (int i = VWidth - 1; i > 0; --i) {
    auto *Elt = findScalarElement(UseV, i);
    if (!Elt)
      break;

    if (auto *ConstElt = dyn_cast<Constant>(Elt)) {
      if (!ConstElt->isNullValue() && !isa<UndefValue>(Elt))
        break;
    } else {
      break;
    }

    DemandedElts.clearBit(i);
  }

  return DemandedElts;
}

// Trim elements of the end of the vector \p V, if they are
// equal to the first element of the vector.
static APInt defaultComponentBroadcast(Value *V) {
  auto *VTy = cast<FixedVectorType>(V->getType());
  unsigned VWidth = VTy->getNumElements();
  APInt DemandedElts = APInt::getAllOnes(VWidth);
  Value *FirstComponent = findScalarElement(V, 0);

  SmallVector<int> ShuffleMask;
  if (auto *SVI = dyn_cast<ShuffleVectorInst>(V))
    SVI->getShuffleMask(ShuffleMask);

  for (int I = VWidth - 1; I > 0; --I) {
    if (ShuffleMask.empty()) {
      auto *Elt = findScalarElement(V, I);
      if (!Elt || (Elt != FirstComponent && !isa<UndefValue>(Elt)))
        break;
    } else {
      // Detect identical elements in the shufflevector result, even though
      // findScalarElement cannot tell us what that element is.
      if (ShuffleMask[I] != ShuffleMask[0] && ShuffleMask[I] != PoisonMaskElem)
        break;
    }
    DemandedElts.clearBit(I);
  }

  return DemandedElts;
}

static Value *simplifyAMDGCNMemoryIntrinsicDemanded(InstCombiner &IC,
                                                    IntrinsicInst &II,
                                                    APInt DemandedElts,
                                                    int DMaskIdx = -1,
                                                    bool IsLoad = true);

/// Return true if it's legal to contract llvm.amdgcn.rcp(llvm.sqrt)
static bool canContractSqrtToRsq(const FPMathOperator *SqrtOp) {
  return (SqrtOp->getType()->isFloatTy() &&
          (SqrtOp->hasApproxFunc() || SqrtOp->getFPAccuracy() >= 1.0f)) ||
         SqrtOp->getType()->isHalfTy();
}

/// Return true if we can easily prove that use U is uniform.
static bool isTriviallyUniform(const Use &U) {
  Value *V = U.get();
  if (isa<Constant>(V))
    return true;
  if (const auto *A = dyn_cast<Argument>(V))
    return AMDGPU::isArgPassedInSGPR(A);
  if (const auto *II = dyn_cast<IntrinsicInst>(V)) {
    if (!AMDGPU::isIntrinsicAlwaysUniform(II->getIntrinsicID()))
      return false;
    // If II and U are in different blocks then there is a possibility of
    // temporal divergence.
    return II->getParent() == cast<Instruction>(U.getUser())->getParent();
  }
  return false;
}

/// Simplify a lane index operand (e.g. llvm.amdgcn.readlane src1).
///
/// The instruction only reads the low 5 bits for wave32, and 6 bits for wave64.
bool GCNTTIImpl::simplifyDemandedLaneMaskArg(InstCombiner &IC,
                                             IntrinsicInst &II,
                                             unsigned LaneArgIdx) const {
  unsigned MaskBits = ST->getWavefrontSizeLog2();
  APInt DemandedMask(32, maskTrailingOnes<unsigned>(MaskBits));

  KnownBits Known(32);
  if (IC.SimplifyDemandedBits(&II, LaneArgIdx, DemandedMask, Known))
    return true;

  if (!Known.isConstant())
    return false;

  // Out of bounds indexes may appear in wave64 code compiled for wave32.
  // Unlike the DAG version, SimplifyDemandedBits does not change constants, so
  // manually fix it up.

  Value *LaneArg = II.getArgOperand(LaneArgIdx);
  Constant *MaskedConst =
      ConstantInt::get(LaneArg->getType(), Known.getConstant() & DemandedMask);
  if (MaskedConst != LaneArg) {
    II.getOperandUse(LaneArgIdx).set(MaskedConst);
    return true;
  }

  return false;
}

static CallInst *rewriteCall(IRBuilderBase &B, CallInst &Old,
                             Function &NewCallee, ArrayRef<Value *> Ops) {
  SmallVector<OperandBundleDef, 2> OpBundles;
  Old.getOperandBundlesAsDefs(OpBundles);

  CallInst *NewCall = B.CreateCall(&NewCallee, Ops, OpBundles);
  NewCall->takeName(&Old);
  return NewCall;
}

Instruction *
GCNTTIImpl::hoistLaneIntrinsicThroughOperand(InstCombiner &IC,
                                             IntrinsicInst &II) const {
  const auto IID = II.getIntrinsicID();
  assert(IID == Intrinsic::amdgcn_readlane ||
         IID == Intrinsic::amdgcn_readfirstlane ||
         IID == Intrinsic::amdgcn_permlane64);

  Instruction *OpInst = dyn_cast<Instruction>(II.getOperand(0));

  // Only do this if both instructions are in the same block
  // (so the exec mask won't change) and the readlane is the only user of its
  // operand.
  if (!OpInst || !OpInst->hasOneUser() || OpInst->getParent() != II.getParent())
    return nullptr;

  const bool IsReadLane = (IID == Intrinsic::amdgcn_readlane);

  // If this is a readlane, check that the second operand is a constant, or is
  // defined before OpInst so we know it's safe to move this intrinsic higher.
  Value *LaneID = nullptr;
  if (IsReadLane) {
    LaneID = II.getOperand(1);

    // readlane take an extra operand for the lane ID, so we must check if that
    // LaneID value can be used at the point where we want to move the
    // intrinsic.
    if (auto *LaneIDInst = dyn_cast<Instruction>(LaneID)) {
      if (!IC.getDominatorTree().dominates(LaneIDInst, OpInst))
        return nullptr;
    }
  }

  // Hoist the intrinsic (II) through OpInst.
  //
  // (II (OpInst x)) -> (OpInst (II x))
  const auto DoIt = [&](unsigned OpIdx,
                        Function *NewIntrinsic) -> Instruction * {
    SmallVector<Value *, 2> Ops{OpInst->getOperand(OpIdx)};
    if (IsReadLane)
      Ops.push_back(LaneID);

    // Rewrite the intrinsic call.
    CallInst *NewII = rewriteCall(IC.Builder, II, *NewIntrinsic, Ops);

    // Rewrite OpInst so it takes the result of the intrinsic now.
    Instruction &NewOp = *OpInst->clone();
    NewOp.setOperand(OpIdx, NewII);
    return &NewOp;
  };

  // TODO(?): Should we do more with permlane64?
  if (IID == Intrinsic::amdgcn_permlane64 && !isa<BitCastInst>(OpInst))
    return nullptr;

  if (isa<UnaryOperator>(OpInst))
    return DoIt(0, II.getCalledFunction());

  if (isa<CastInst>(OpInst)) {
    Value *Src = OpInst->getOperand(0);
    Type *SrcTy = Src->getType();
    if (!isTypeLegal(SrcTy))
      return nullptr;

    Function *Remangled =
        Intrinsic::getOrInsertDeclaration(II.getModule(), IID, {SrcTy});
    return DoIt(0, Remangled);
  }

  // We can also hoist through binary operators if the other operand is uniform.
  if (isa<BinaryOperator>(OpInst)) {
    // FIXME: If we had access to UniformityInfo here we could just check
    // if the operand is uniform.
    if (isTriviallyUniform(OpInst->getOperandUse(0)))
      return DoIt(1, II.getCalledFunction());
    if (isTriviallyUniform(OpInst->getOperandUse(1)))
      return DoIt(0, II.getCalledFunction());
  }

  return nullptr;
}

std::optional<Instruction *>
GCNTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  case Intrinsic::amdgcn_rcp: {
    Value *Src = II.getArgOperand(0);
    if (isa<PoisonValue>(Src))
      return IC.replaceInstUsesWith(II, Src);

    // TODO: Move to ConstantFolding/InstSimplify?
    if (isa<UndefValue>(Src)) {
      Type *Ty = II.getType();
      auto *QNaN = ConstantFP::get(Ty, APFloat::getQNaN(Ty->getFltSemantics()));
      return IC.replaceInstUsesWith(II, QNaN);
    }

    if (II.isStrictFP())
      break;

    if (const ConstantFP *C = dyn_cast<ConstantFP>(Src)) {
      const APFloat &ArgVal = C->getValueAPF();
      APFloat Val(ArgVal.getSemantics(), 1);
      Val.divide(ArgVal, APFloat::rmNearestTiesToEven);

      // This is more precise than the instruction may give.
      //
      // TODO: The instruction always flushes denormal results (except for f16),
      // should this also?
      return IC.replaceInstUsesWith(II, ConstantFP::get(II.getContext(), Val));
    }

    FastMathFlags FMF = cast<FPMathOperator>(II).getFastMathFlags();
    if (!FMF.allowContract())
      break;
    auto *SrcCI = dyn_cast<IntrinsicInst>(Src);
    if (!SrcCI)
      break;

    auto IID = SrcCI->getIntrinsicID();
    // llvm.amdgcn.rcp(llvm.amdgcn.sqrt(x)) -> llvm.amdgcn.rsq(x) if contractable
    //
    // llvm.amdgcn.rcp(llvm.sqrt(x)) -> llvm.amdgcn.rsq(x) if contractable and
    // relaxed.
    if (IID == Intrinsic::amdgcn_sqrt || IID == Intrinsic::sqrt) {
      const FPMathOperator *SqrtOp = cast<FPMathOperator>(SrcCI);
      FastMathFlags InnerFMF = SqrtOp->getFastMathFlags();
      if (!InnerFMF.allowContract() || !SrcCI->hasOneUse())
        break;

      if (IID == Intrinsic::sqrt && !canContractSqrtToRsq(SqrtOp))
        break;

      Function *NewDecl = Intrinsic::getOrInsertDeclaration(
          SrcCI->getModule(), Intrinsic::amdgcn_rsq, {SrcCI->getType()});

      InnerFMF |= FMF;
      II.setFastMathFlags(InnerFMF);

      II.setCalledFunction(NewDecl);
      return IC.replaceOperand(II, 0, SrcCI->getArgOperand(0));
    }

    break;
  }
  case Intrinsic::amdgcn_sqrt:
  case Intrinsic::amdgcn_rsq:
  case Intrinsic::amdgcn_tanh: {
    Value *Src = II.getArgOperand(0);
    if (isa<PoisonValue>(Src))
      return IC.replaceInstUsesWith(II, Src);

    // TODO: Move to ConstantFolding/InstSimplify?
    if (isa<UndefValue>(Src)) {
      Type *Ty = II.getType();
      auto *QNaN = ConstantFP::get(Ty, APFloat::getQNaN(Ty->getFltSemantics()));
      return IC.replaceInstUsesWith(II, QNaN);
    }

    // f16 amdgcn.sqrt is identical to regular sqrt.
    if (IID == Intrinsic::amdgcn_sqrt && Src->getType()->isHalfTy()) {
      Function *NewDecl = Intrinsic::getOrInsertDeclaration(
          II.getModule(), Intrinsic::sqrt, {II.getType()});
      II.setCalledFunction(NewDecl);
      return &II;
    }

    break;
  }
  case Intrinsic::amdgcn_log:
  case Intrinsic::amdgcn_exp2: {
    const bool IsLog = IID == Intrinsic::amdgcn_log;
    const bool IsExp = IID == Intrinsic::amdgcn_exp2;
    Value *Src = II.getArgOperand(0);
    Type *Ty = II.getType();

    if (isa<PoisonValue>(Src))
      return IC.replaceInstUsesWith(II, Src);

    if (IC.getSimplifyQuery().isUndefValue(Src))
      return IC.replaceInstUsesWith(II, ConstantFP::getNaN(Ty));

    if (ConstantFP *C = dyn_cast<ConstantFP>(Src)) {
      if (C->isInfinity()) {
        // exp2(+inf) -> +inf
        // log2(+inf) -> +inf
        if (!C->isNegative())
          return IC.replaceInstUsesWith(II, C);

        // exp2(-inf) -> 0
        if (IsExp && C->isNegative())
          return IC.replaceInstUsesWith(II, ConstantFP::getZero(Ty));
      }

      if (II.isStrictFP())
        break;

      if (C->isNaN()) {
        Constant *Quieted = ConstantFP::get(Ty, C->getValue().makeQuiet());
        return IC.replaceInstUsesWith(II, Quieted);
      }

      // f32 instruction doesn't handle denormals, f16 does.
      if (C->isZero() || (C->getValue().isDenormal() && Ty->isFloatTy())) {
        Constant *FoldedValue = IsLog ? ConstantFP::getInfinity(Ty, true)
                                      : ConstantFP::get(Ty, 1.0);
        return IC.replaceInstUsesWith(II, FoldedValue);
      }

      if (IsLog && C->isNegative())
        return IC.replaceInstUsesWith(II, ConstantFP::getNaN(Ty));

      // TODO: Full constant folding matching hardware behavior.
    }

    break;
  }
  case Intrinsic::amdgcn_frexp_mant:
  case Intrinsic::amdgcn_frexp_exp: {
    Value *Src = II.getArgOperand(0);
    if (const ConstantFP *C = dyn_cast<ConstantFP>(Src)) {
      int Exp;
      APFloat Significand =
          frexp(C->getValueAPF(), Exp, APFloat::rmNearestTiesToEven);

      if (IID == Intrinsic::amdgcn_frexp_mant) {
        return IC.replaceInstUsesWith(
            II, ConstantFP::get(II.getContext(), Significand));
      }

      // Match instruction special case behavior.
      if (Exp == APFloat::IEK_NaN || Exp == APFloat::IEK_Inf)
        Exp = 0;

      return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), Exp));
    }

    if (isa<PoisonValue>(Src))
      return IC.replaceInstUsesWith(II, PoisonValue::get(II.getType()));

    if (isa<UndefValue>(Src)) {
      return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));
    }

    break;
  }
  case Intrinsic::amdgcn_class: {
    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);
    const ConstantInt *CMask = dyn_cast<ConstantInt>(Src1);
    if (CMask) {
      II.setCalledOperand(Intrinsic::getOrInsertDeclaration(
          II.getModule(), Intrinsic::is_fpclass, Src0->getType()));

      // Clamp any excess bits, as they're illegal for the generic intrinsic.
      II.setArgOperand(1, ConstantInt::get(Src1->getType(),
                                           CMask->getZExtValue() & fcAllFlags));
      return &II;
    }

    // Propagate poison.
    if (isa<PoisonValue>(Src0) || isa<PoisonValue>(Src1))
      return IC.replaceInstUsesWith(II, PoisonValue::get(II.getType()));

    // llvm.amdgcn.class(_, undef) -> false
    if (IC.getSimplifyQuery().isUndefValue(Src1))
      return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), false));

    // llvm.amdgcn.class(undef, mask) -> mask != 0
    if (IC.getSimplifyQuery().isUndefValue(Src0)) {
      Value *CmpMask = IC.Builder.CreateICmpNE(
          Src1, ConstantInt::getNullValue(Src1->getType()));
      return IC.replaceInstUsesWith(II, CmpMask);
    }
    break;
  }
  case Intrinsic::amdgcn_cvt_pkrtz: {
    auto foldFPTruncToF16RTZ = [](Value *Arg) -> Value * {
      Type *HalfTy = Type::getHalfTy(Arg->getContext());

      if (isa<PoisonValue>(Arg))
        return PoisonValue::get(HalfTy);
      if (isa<UndefValue>(Arg))
        return UndefValue::get(HalfTy);

      ConstantFP *CFP = nullptr;
      if (match(Arg, m_ConstantFP(CFP))) {
        bool LosesInfo;
        APFloat Val(CFP->getValueAPF());
        Val.convert(APFloat::IEEEhalf(), APFloat::rmTowardZero, &LosesInfo);
        return ConstantFP::get(HalfTy, Val);
      }

      Value *Src = nullptr;
      if (match(Arg, m_FPExt(m_Value(Src)))) {
        if (Src->getType()->isHalfTy())
          return Src;
      }

      return nullptr;
    };

    if (Value *Src0 = foldFPTruncToF16RTZ(II.getArgOperand(0))) {
      if (Value *Src1 = foldFPTruncToF16RTZ(II.getArgOperand(1))) {
        Value *V = PoisonValue::get(II.getType());
        V = IC.Builder.CreateInsertElement(V, Src0, (uint64_t)0);
        V = IC.Builder.CreateInsertElement(V, Src1, (uint64_t)1);
        return IC.replaceInstUsesWith(II, V);
      }
    }

    break;
  }
  case Intrinsic::amdgcn_cvt_pknorm_i16:
  case Intrinsic::amdgcn_cvt_pknorm_u16:
  case Intrinsic::amdgcn_cvt_pk_i16:
  case Intrinsic::amdgcn_cvt_pk_u16: {
    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);

    // TODO: Replace call with scalar operation if only one element is poison.
    if (isa<PoisonValue>(Src0) && isa<PoisonValue>(Src1))
      return IC.replaceInstUsesWith(II, PoisonValue::get(II.getType()));

    if (isa<UndefValue>(Src0) && isa<UndefValue>(Src1)) {
      return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));
    }

    break;
  }
  case Intrinsic::amdgcn_cvt_off_f32_i4: {
    Value* Arg = II.getArgOperand(0);
    Type *Ty = II.getType();

    if (isa<PoisonValue>(Arg))
      return IC.replaceInstUsesWith(II, PoisonValue::get(Ty));

    if(IC.getSimplifyQuery().isUndefValue(Arg))
      return IC.replaceInstUsesWith(II, Constant::getNullValue(Ty));

    ConstantInt *CArg = dyn_cast<ConstantInt>(II.getArgOperand(0));
    if (!CArg)
      break;

    // Tabulated 0.0625 * (sext (CArg & 0xf)).
    constexpr size_t ResValsSize = 16;
    static constexpr float ResVals[ResValsSize] = {
        0.0,  0.0625,  0.125,  0.1875,  0.25,  0.3125,  0.375,  0.4375,
        -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625};
    Constant *Res =
        ConstantFP::get(Ty, ResVals[CArg->getZExtValue() & (ResValsSize - 1)]);
    return IC.replaceInstUsesWith(II, Res);
  }
  case Intrinsic::amdgcn_ubfe:
  case Intrinsic::amdgcn_sbfe: {
    // Decompose simple cases into standard shifts.
    Value *Src = II.getArgOperand(0);
    if (isa<UndefValue>(Src)) {
      return IC.replaceInstUsesWith(II, Src);
    }

    unsigned Width;
    Type *Ty = II.getType();
    unsigned IntSize = Ty->getIntegerBitWidth();

    ConstantInt *CWidth = dyn_cast<ConstantInt>(II.getArgOperand(2));
    if (CWidth) {
      Width = CWidth->getZExtValue();
      if ((Width & (IntSize - 1)) == 0) {
        return IC.replaceInstUsesWith(II, ConstantInt::getNullValue(Ty));
      }

      // Hardware ignores high bits, so remove those.
      if (Width >= IntSize) {
        return IC.replaceOperand(
            II, 2, ConstantInt::get(CWidth->getType(), Width & (IntSize - 1)));
      }
    }

    unsigned Offset;
    ConstantInt *COffset = dyn_cast<ConstantInt>(II.getArgOperand(1));
    if (COffset) {
      Offset = COffset->getZExtValue();
      if (Offset >= IntSize) {
        return IC.replaceOperand(
            II, 1,
            ConstantInt::get(COffset->getType(), Offset & (IntSize - 1)));
      }
    }

    bool Signed = IID == Intrinsic::amdgcn_sbfe;

    if (!CWidth || !COffset)
      break;

    // The case of Width == 0 is handled above, which makes this transformation
    // safe.  If Width == 0, then the ashr and lshr instructions become poison
    // value since the shift amount would be equal to the bit size.
    assert(Width != 0);

    // TODO: This allows folding to undef when the hardware has specific
    // behavior?
    if (Offset + Width < IntSize) {
      Value *Shl = IC.Builder.CreateShl(Src, IntSize - Offset - Width);
      Value *RightShift = Signed ? IC.Builder.CreateAShr(Shl, IntSize - Width)
                                 : IC.Builder.CreateLShr(Shl, IntSize - Width);
      RightShift->takeName(&II);
      return IC.replaceInstUsesWith(II, RightShift);
    }

    Value *RightShift = Signed ? IC.Builder.CreateAShr(Src, Offset)
                               : IC.Builder.CreateLShr(Src, Offset);

    RightShift->takeName(&II);
    return IC.replaceInstUsesWith(II, RightShift);
  }
  case Intrinsic::amdgcn_exp:
  case Intrinsic::amdgcn_exp_row:
  case Intrinsic::amdgcn_exp_compr: {
    ConstantInt *En = cast<ConstantInt>(II.getArgOperand(1));
    unsigned EnBits = En->getZExtValue();
    if (EnBits == 0xf)
      break; // All inputs enabled.

    bool IsCompr = IID == Intrinsic::amdgcn_exp_compr;
    bool Changed = false;
    for (int I = 0; I < (IsCompr ? 2 : 4); ++I) {
      if ((!IsCompr && (EnBits & (1 << I)) == 0) ||
          (IsCompr && ((EnBits & (0x3 << (2 * I))) == 0))) {
        Value *Src = II.getArgOperand(I + 2);
        if (!isa<PoisonValue>(Src)) {
          IC.replaceOperand(II, I + 2, PoisonValue::get(Src->getType()));
          Changed = true;
        }
      }
    }

    if (Changed) {
      return &II;
    }

    break;
  }
  case Intrinsic::amdgcn_fmed3: {
    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);
    Value *Src2 = II.getArgOperand(2);

    for (Value *Src : {Src0, Src1, Src2}) {
      if (isa<PoisonValue>(Src))
        return IC.replaceInstUsesWith(II, Src);
    }

    if (II.isStrictFP())
      break;

    // med3 with a nan input acts like
    // v_min_f32(v_min_f32(s0, s1), s2)
    //
    // Signalingness is ignored with ieee=0, so we fold to
    // minimumnum/maximumnum. With ieee=1, the v_min_f32 acts like llvm.minnum
    // with signaling nan handling. With ieee=0, like llvm.minimumnum except a
    // returned signaling nan will not be quieted.

    // ieee=1
    // s0 snan: s2
    // s1 snan: s2
    // s2 snan: qnan

    // s0 qnan: min(s1, s2)
    // s1 qnan: min(s0, s2)
    // s2 qnan: min(s0, s1)

    // ieee=0
    // s0 _nan: min(s1, s2)
    // s1 _nan: min(s0, s2)
    // s2 _nan: min(s0, s1)

    // med3 behavior with infinity
    // s0 +inf: max(s1, s2)
    // s1 +inf: max(s0, s2)
    // s2 +inf: max(s0, s1)
    // s0 -inf: min(s1, s2)
    // s1 -inf: min(s0, s2)
    // s2 -inf: min(s0, s1)

    // Checking for NaN before canonicalization provides better fidelity when
    // mapping other operations onto fmed3 since the order of operands is
    // unchanged.
    Value *V = nullptr;
    const APFloat *ConstSrc0 = nullptr;
    const APFloat *ConstSrc1 = nullptr;
    const APFloat *ConstSrc2 = nullptr;

    if ((match(Src0, m_APFloat(ConstSrc0)) &&
         (ConstSrc0->isNaN() || ConstSrc0->isInfinity())) ||
        isa<UndefValue>(Src0)) {
      const bool IsPosInfinity = ConstSrc0 && ConstSrc0->isPosInfinity();
      switch (fpenvIEEEMode(II)) {
      case KnownIEEEMode::On:
        // TODO: If Src2 is snan, does it need quieting?
        if (ConstSrc0 && ConstSrc0->isNaN() && ConstSrc0->isSignaling())
          return IC.replaceInstUsesWith(II, Src2);

        V = IsPosInfinity ? IC.Builder.CreateMaxNum(Src1, Src2)
                          : IC.Builder.CreateMinNum(Src1, Src2);
        break;
      case KnownIEEEMode::Off:
        V = IsPosInfinity ? IC.Builder.CreateMaximumNum(Src1, Src2)
                          : IC.Builder.CreateMinimumNum(Src1, Src2);
        break;
      case KnownIEEEMode::Unknown:
        break;
      }
    } else if ((match(Src1, m_APFloat(ConstSrc1)) &&
                (ConstSrc1->isNaN() || ConstSrc1->isInfinity())) ||
               isa<UndefValue>(Src1)) {
      const bool IsPosInfinity = ConstSrc1 && ConstSrc1->isPosInfinity();
      switch (fpenvIEEEMode(II)) {
      case KnownIEEEMode::On:
        // TODO: If Src2 is snan, does it need quieting?
        if (ConstSrc1 && ConstSrc1->isNaN() && ConstSrc1->isSignaling())
          return IC.replaceInstUsesWith(II, Src2);

        V = IsPosInfinity ? IC.Builder.CreateMaxNum(Src0, Src2)
                          : IC.Builder.CreateMinNum(Src0, Src2);
        break;
      case KnownIEEEMode::Off:
        V = IsPosInfinity ? IC.Builder.CreateMaximumNum(Src0, Src2)
                          : IC.Builder.CreateMinimumNum(Src0, Src2);
        break;
      case KnownIEEEMode::Unknown:
        break;
      }
    } else if ((match(Src2, m_APFloat(ConstSrc2)) &&
                (ConstSrc2->isNaN() || ConstSrc2->isInfinity())) ||
               isa<UndefValue>(Src2)) {
      switch (fpenvIEEEMode(II)) {
      case KnownIEEEMode::On:
        if (ConstSrc2 && ConstSrc2->isNaN() && ConstSrc2->isSignaling()) {
          auto *Quieted = ConstantFP::get(II.getType(), ConstSrc2->makeQuiet());
          return IC.replaceInstUsesWith(II, Quieted);
        }

        V = (ConstSrc2 && ConstSrc2->isPosInfinity())
                ? IC.Builder.CreateMaxNum(Src0, Src1)
                : IC.Builder.CreateMinNum(Src0, Src1);
        break;
      case KnownIEEEMode::Off:
        V = (ConstSrc2 && ConstSrc2->isNegInfinity())
                ? IC.Builder.CreateMinimumNum(Src0, Src1)
                : IC.Builder.CreateMaximumNum(Src0, Src1);
        break;
      case KnownIEEEMode::Unknown:
        break;
      }
    }

    if (V) {
      if (auto *CI = dyn_cast<CallInst>(V)) {
        CI->copyFastMathFlags(&II);
        CI->takeName(&II);
      }
      return IC.replaceInstUsesWith(II, V);
    }

    bool Swap = false;
    // Canonicalize constants to RHS operands.
    //
    // fmed3(c0, x, c1) -> fmed3(x, c0, c1)
    if (isa<Constant>(Src0) && !isa<Constant>(Src1)) {
      std::swap(Src0, Src1);
      Swap = true;
    }

    if (isa<Constant>(Src1) && !isa<Constant>(Src2)) {
      std::swap(Src1, Src2);
      Swap = true;
    }

    if (isa<Constant>(Src0) && !isa<Constant>(Src1)) {
      std::swap(Src0, Src1);
      Swap = true;
    }

    if (Swap) {
      II.setArgOperand(0, Src0);
      II.setArgOperand(1, Src1);
      II.setArgOperand(2, Src2);
      return &II;
    }

    if (const ConstantFP *C0 = dyn_cast<ConstantFP>(Src0)) {
      if (const ConstantFP *C1 = dyn_cast<ConstantFP>(Src1)) {
        if (const ConstantFP *C2 = dyn_cast<ConstantFP>(Src2)) {
          APFloat Result = fmed3AMDGCN(C0->getValueAPF(), C1->getValueAPF(),
                                       C2->getValueAPF());
          return IC.replaceInstUsesWith(II,
                                        ConstantFP::get(II.getType(), Result));
        }
      }
    }

    if (!ST->hasMed3_16())
      break;

    // Repeat floating-point width reduction done for minnum/maxnum.
    // fmed3((fpext X), (fpext Y), (fpext Z)) -> fpext (fmed3(X, Y, Z))
    if (Value *X = matchFPExtFromF16(Src0)) {
      if (Value *Y = matchFPExtFromF16(Src1)) {
        if (Value *Z = matchFPExtFromF16(Src2)) {
          Value *NewCall = IC.Builder.CreateIntrinsic(
              IID, {X->getType()}, {X, Y, Z}, &II, II.getName());
          return new FPExtInst(NewCall, II.getType());
        }
      }
    }

    break;
  }
  case Intrinsic::amdgcn_icmp:
  case Intrinsic::amdgcn_fcmp: {
    const ConstantInt *CC = cast<ConstantInt>(II.getArgOperand(2));
    // Guard against invalid arguments.
    int64_t CCVal = CC->getZExtValue();
    bool IsInteger = IID == Intrinsic::amdgcn_icmp;
    if ((IsInteger && (CCVal < CmpInst::FIRST_ICMP_PREDICATE ||
                       CCVal > CmpInst::LAST_ICMP_PREDICATE)) ||
        (!IsInteger && (CCVal < CmpInst::FIRST_FCMP_PREDICATE ||
                        CCVal > CmpInst::LAST_FCMP_PREDICATE)))
      break;

    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);

    if (auto *CSrc0 = dyn_cast<Constant>(Src0)) {
      if (auto *CSrc1 = dyn_cast<Constant>(Src1)) {
        Constant *CCmp = ConstantFoldCompareInstOperands(
            (ICmpInst::Predicate)CCVal, CSrc0, CSrc1, DL);
        if (CCmp && CCmp->isNullValue()) {
          return IC.replaceInstUsesWith(
              II, IC.Builder.CreateSExt(CCmp, II.getType()));
        }

        // The result of V_ICMP/V_FCMP assembly instructions (which this
        // intrinsic exposes) is one bit per thread, masked with the EXEC
        // register (which contains the bitmask of live threads). So a
        // comparison that always returns true is the same as a read of the
        // EXEC register.
        Metadata *MDArgs[] = {MDString::get(II.getContext(), "exec")};
        MDNode *MD = MDNode::get(II.getContext(), MDArgs);
        Value *Args[] = {MetadataAsValue::get(II.getContext(), MD)};
        CallInst *NewCall = IC.Builder.CreateIntrinsic(Intrinsic::read_register,
                                                       II.getType(), Args);
        NewCall->addFnAttr(Attribute::Convergent);
        NewCall->takeName(&II);
        return IC.replaceInstUsesWith(II, NewCall);
      }

      // Canonicalize constants to RHS.
      CmpInst::Predicate SwapPred =
          CmpInst::getSwappedPredicate(static_cast<CmpInst::Predicate>(CCVal));
      II.setArgOperand(0, Src1);
      II.setArgOperand(1, Src0);
      II.setArgOperand(
          2, ConstantInt::get(CC->getType(), static_cast<int>(SwapPred)));
      return &II;
    }

    if (CCVal != CmpInst::ICMP_EQ && CCVal != CmpInst::ICMP_NE)
      break;

    // Canonicalize compare eq with true value to compare != 0
    // llvm.amdgcn.icmp(zext (i1 x), 1, eq)
    //   -> llvm.amdgcn.icmp(zext (i1 x), 0, ne)
    // llvm.amdgcn.icmp(sext (i1 x), -1, eq)
    //   -> llvm.amdgcn.icmp(sext (i1 x), 0, ne)
    Value *ExtSrc;
    if (CCVal == CmpInst::ICMP_EQ &&
        ((match(Src1, PatternMatch::m_One()) &&
          match(Src0, m_ZExt(PatternMatch::m_Value(ExtSrc)))) ||
         (match(Src1, PatternMatch::m_AllOnes()) &&
          match(Src0, m_SExt(PatternMatch::m_Value(ExtSrc))))) &&
        ExtSrc->getType()->isIntegerTy(1)) {
      IC.replaceOperand(II, 1, ConstantInt::getNullValue(Src1->getType()));
      IC.replaceOperand(II, 2,
                        ConstantInt::get(CC->getType(), CmpInst::ICMP_NE));
      return &II;
    }

    CmpPredicate SrcPred;
    Value *SrcLHS;
    Value *SrcRHS;

    // Fold compare eq/ne with 0 from a compare result as the predicate to the
    // intrinsic. The typical use is a wave vote function in the library, which
    // will be fed from a user code condition compared with 0. Fold in the
    // redundant compare.

    // llvm.amdgcn.icmp([sz]ext ([if]cmp pred a, b), 0, ne)
    //   -> llvm.amdgcn.[if]cmp(a, b, pred)
    //
    // llvm.amdgcn.icmp([sz]ext ([if]cmp pred a, b), 0, eq)
    //   -> llvm.amdgcn.[if]cmp(a, b, inv pred)
    if (match(Src1, PatternMatch::m_Zero()) &&
        match(Src0, PatternMatch::m_ZExtOrSExt(
                        m_Cmp(SrcPred, PatternMatch::m_Value(SrcLHS),
                              PatternMatch::m_Value(SrcRHS))))) {
      if (CCVal == CmpInst::ICMP_EQ)
        SrcPred = CmpInst::getInversePredicate(SrcPred);

      Intrinsic::ID NewIID = CmpInst::isFPPredicate(SrcPred)
                                 ? Intrinsic::amdgcn_fcmp
                                 : Intrinsic::amdgcn_icmp;

      Type *Ty = SrcLHS->getType();
      if (auto *CmpType = dyn_cast<IntegerType>(Ty)) {
        // Promote to next legal integer type.
        unsigned Width = CmpType->getBitWidth();
        unsigned NewWidth = Width;

        // Don't do anything for i1 comparisons.
        if (Width == 1)
          break;

        if (Width <= 16)
          NewWidth = 16;
        else if (Width <= 32)
          NewWidth = 32;
        else if (Width <= 64)
          NewWidth = 64;
        else
          break; // Can't handle this.

        if (Width != NewWidth) {
          IntegerType *CmpTy = IC.Builder.getIntNTy(NewWidth);
          if (CmpInst::isSigned(SrcPred)) {
            SrcLHS = IC.Builder.CreateSExt(SrcLHS, CmpTy);
            SrcRHS = IC.Builder.CreateSExt(SrcRHS, CmpTy);
          } else {
            SrcLHS = IC.Builder.CreateZExt(SrcLHS, CmpTy);
            SrcRHS = IC.Builder.CreateZExt(SrcRHS, CmpTy);
          }
        }
      } else if (!Ty->isFloatTy() && !Ty->isDoubleTy() && !Ty->isHalfTy())
        break;

      Value *Args[] = {SrcLHS, SrcRHS,
                       ConstantInt::get(CC->getType(), SrcPred)};
      CallInst *NewCall = IC.Builder.CreateIntrinsic(
          NewIID, {II.getType(), SrcLHS->getType()}, Args);
      NewCall->takeName(&II);
      return IC.replaceInstUsesWith(II, NewCall);
    }

    break;
  }
  case Intrinsic::amdgcn_mbcnt_hi: {
    // exec_hi is all 0, so this is just a copy.
    if (ST->isWave32())
      return IC.replaceInstUsesWith(II, II.getArgOperand(1));
    break;
  }
  case Intrinsic::amdgcn_ballot: {
    Value *Arg = II.getArgOperand(0);
    if (isa<PoisonValue>(Arg))
      return IC.replaceInstUsesWith(II, PoisonValue::get(II.getType()));

    if (auto *Src = dyn_cast<ConstantInt>(Arg)) {
      if (Src->isZero()) {
        // amdgcn.ballot(i1 0) is zero.
        return IC.replaceInstUsesWith(II, Constant::getNullValue(II.getType()));
      }
    }
    if (ST->isWave32() && II.getType()->getIntegerBitWidth() == 64) {
      // %b64 = call i64 ballot.i64(...)
      // =>
      // %b32 = call i32 ballot.i32(...)
      // %b64 = zext i32 %b32 to i64
      Value *Call = IC.Builder.CreateZExt(
          IC.Builder.CreateIntrinsic(Intrinsic::amdgcn_ballot,
                                     {IC.Builder.getInt32Ty()},
                                     {II.getArgOperand(0)}),
          II.getType());
      Call->takeName(&II);
      return IC.replaceInstUsesWith(II, Call);
    }
    break;
  }
  case Intrinsic::amdgcn_wavefrontsize: {
    if (ST->isWaveSizeKnown())
      return IC.replaceInstUsesWith(
          II, ConstantInt::get(II.getType(), ST->getWavefrontSize()));
    break;
  }
  case Intrinsic::amdgcn_wqm_vote: {
    // wqm_vote is identity when the argument is constant.
    if (!isa<Constant>(II.getArgOperand(0)))
      break;

    return IC.replaceInstUsesWith(II, II.getArgOperand(0));
  }
  case Intrinsic::amdgcn_kill: {
    const ConstantInt *C = dyn_cast<ConstantInt>(II.getArgOperand(0));
    if (!C || !C->getZExtValue())
      break;

    // amdgcn.kill(i1 1) is a no-op
    return IC.eraseInstFromFunction(II);
  }
  case Intrinsic::amdgcn_update_dpp: {
    Value *Old = II.getArgOperand(0);

    auto *BC = cast<ConstantInt>(II.getArgOperand(5));
    auto *RM = cast<ConstantInt>(II.getArgOperand(3));
    auto *BM = cast<ConstantInt>(II.getArgOperand(4));
    if (BC->isZeroValue() || RM->getZExtValue() != 0xF ||
        BM->getZExtValue() != 0xF || isa<PoisonValue>(Old))
      break;

    // If bound_ctrl = 1, row mask = bank mask = 0xf we can omit old value.
    return IC.replaceOperand(II, 0, PoisonValue::get(Old->getType()));
  }
  case Intrinsic::amdgcn_permlane16:
  case Intrinsic::amdgcn_permlane16_var:
  case Intrinsic::amdgcn_permlanex16:
  case Intrinsic::amdgcn_permlanex16_var: {
    // Discard vdst_in if it's not going to be read.
    Value *VDstIn = II.getArgOperand(0);
    if (isa<PoisonValue>(VDstIn))
      break;

    // FetchInvalid operand idx.
    unsigned int FiIdx = (IID == Intrinsic::amdgcn_permlane16 ||
                          IID == Intrinsic::amdgcn_permlanex16)
                             ? 4  /* for permlane16 and permlanex16 */
                             : 3; /* for permlane16_var and permlanex16_var */

    // BoundCtrl operand idx.
    // For permlane16 and permlanex16 it should be 5
    // For Permlane16_var and permlanex16_var it should be 4
    unsigned int BcIdx = FiIdx + 1;

    ConstantInt *FetchInvalid = cast<ConstantInt>(II.getArgOperand(FiIdx));
    ConstantInt *BoundCtrl = cast<ConstantInt>(II.getArgOperand(BcIdx));
    if (!FetchInvalid->getZExtValue() && !BoundCtrl->getZExtValue())
      break;

    return IC.replaceOperand(II, 0, PoisonValue::get(VDstIn->getType()));
  }
  case Intrinsic::amdgcn_permlane64:
  case Intrinsic::amdgcn_readfirstlane:
  case Intrinsic::amdgcn_readlane:
  case Intrinsic::amdgcn_ds_bpermute: {
    // If the data argument is uniform these intrinsics return it unchanged.
    unsigned SrcIdx = IID == Intrinsic::amdgcn_ds_bpermute ? 1 : 0;
    const Use &Src = II.getArgOperandUse(SrcIdx);
    if (isTriviallyUniform(Src))
      return IC.replaceInstUsesWith(II, Src.get());

    if (IID == Intrinsic::amdgcn_readlane &&
        simplifyDemandedLaneMaskArg(IC, II, 1))
      return &II;

    // If the lane argument of bpermute is uniform, change it to readlane. This
    // generates better code and can enable further optimizations because
    // readlane is AlwaysUniform.
    if (IID == Intrinsic::amdgcn_ds_bpermute) {
      const Use &Lane = II.getArgOperandUse(0);
      if (isTriviallyUniform(Lane)) {
        Value *NewLane = IC.Builder.CreateLShr(Lane, 2);
        Function *NewDecl = Intrinsic::getOrInsertDeclaration(
            II.getModule(), Intrinsic::amdgcn_readlane, II.getType());
        II.setCalledFunction(NewDecl);
        II.setOperand(0, Src);
        II.setOperand(1, NewLane);
        return &II;
      }
    }

    if (IID != Intrinsic::amdgcn_ds_bpermute) {
      if (Instruction *Res = hoistLaneIntrinsicThroughOperand(IC, II))
        return Res;
    }

    return std::nullopt;
  }
  case Intrinsic::amdgcn_writelane: {
    // TODO: Fold bitcast like readlane.
    if (simplifyDemandedLaneMaskArg(IC, II, 1))
      return &II;
    return std::nullopt;
  }
  case Intrinsic::amdgcn_trig_preop: {
    // The intrinsic is declared with name mangling, but currently the
    // instruction only exists for f64
    if (!II.getType()->isDoubleTy())
      break;

    Value *Src = II.getArgOperand(0);
    Value *Segment = II.getArgOperand(1);
    if (isa<PoisonValue>(Src) || isa<PoisonValue>(Segment))
      return IC.replaceInstUsesWith(II, PoisonValue::get(II.getType()));

    if (isa<UndefValue>(Src)) {
      auto *QNaN = ConstantFP::get(
          II.getType(), APFloat::getQNaN(II.getType()->getFltSemantics()));
      return IC.replaceInstUsesWith(II, QNaN);
    }

    const ConstantFP *Csrc = dyn_cast<ConstantFP>(Src);
    if (!Csrc)
      break;

    if (II.isStrictFP())
      break;

    const APFloat &Fsrc = Csrc->getValueAPF();
    if (Fsrc.isNaN()) {
      auto *Quieted = ConstantFP::get(II.getType(), Fsrc.makeQuiet());
      return IC.replaceInstUsesWith(II, Quieted);
    }

    const ConstantInt *Cseg = dyn_cast<ConstantInt>(Segment);
    if (!Cseg)
      break;

    unsigned Exponent = (Fsrc.bitcastToAPInt().getZExtValue() >> 52) & 0x7ff;
    unsigned SegmentVal = Cseg->getValue().trunc(5).getZExtValue();
    unsigned Shift = SegmentVal * 53;
    if (Exponent > 1077)
      Shift += Exponent - 1077;

    // 2.0/PI table.
    static const uint32_t TwoByPi[] = {
        0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0, 0xdb629599, 0x3c439041,
        0xfe5163ab, 0xdebbc561, 0xb7246e3a, 0x424dd2e0, 0x06492eea, 0x09d1921c,
        0xfe1deb1c, 0xb129a73e, 0xe88235f5, 0x2ebb4484, 0xe99c7026, 0xb45f7e41,
        0x3991d639, 0x835339f4, 0x9c845f8b, 0xbdf9283b, 0x1ff897ff, 0xde05980f,
        0xef2f118b, 0x5a0a6d1f, 0x6d367ecf, 0x27cb09b7, 0x4f463f66, 0x9e5fea2d,
        0x7527bac7, 0xebe5f17b, 0x3d0739f7, 0x8a5292ea, 0x6bfb5fb1, 0x1f8d5d08,
        0x56033046};

    // Return 0 for outbound segment (hardware behavior).
    unsigned Idx = Shift >> 5;
    if (Idx + 2 >= std::size(TwoByPi)) {
      APFloat Zero = APFloat::getZero(II.getType()->getFltSemantics());
      return IC.replaceInstUsesWith(II, ConstantFP::get(II.getType(), Zero));
    }

    unsigned BShift = Shift & 0x1f;
    uint64_t Thi = Make_64(TwoByPi[Idx], TwoByPi[Idx + 1]);
    uint64_t Tlo = Make_64(TwoByPi[Idx + 2], 0);
    if (BShift)
      Thi = (Thi << BShift) | (Tlo >> (64 - BShift));
    Thi = Thi >> 11;
    APFloat Result = APFloat((double)Thi);

    int Scale = -53 - Shift;
    if (Exponent >= 1968)
      Scale += 128;

    Result = scalbn(Result, Scale, RoundingMode::NearestTiesToEven);
    return IC.replaceInstUsesWith(II, ConstantFP::get(Src->getType(), Result));
  }
  case Intrinsic::amdgcn_fmul_legacy: {
    Value *Op0 = II.getArgOperand(0);
    Value *Op1 = II.getArgOperand(1);

    for (Value *Src : {Op0, Op1}) {
      if (isa<PoisonValue>(Src))
        return IC.replaceInstUsesWith(II, Src);
    }

    // The legacy behaviour is that multiplying +/-0.0 by anything, even NaN or
    // infinity, gives +0.0.
    // TODO: Move to InstSimplify?
    if (match(Op0, PatternMatch::m_AnyZeroFP()) ||
        match(Op1, PatternMatch::m_AnyZeroFP()))
      return IC.replaceInstUsesWith(II, ConstantFP::getZero(II.getType()));

    // If we can prove we don't have one of the special cases then we can use a
    // normal fmul instruction instead.
    if (canSimplifyLegacyMulToMul(II, Op0, Op1, IC)) {
      auto *FMul = IC.Builder.CreateFMulFMF(Op0, Op1, &II);
      FMul->takeName(&II);
      return IC.replaceInstUsesWith(II, FMul);
    }
    break;
  }
  case Intrinsic::amdgcn_fma_legacy: {
    Value *Op0 = II.getArgOperand(0);
    Value *Op1 = II.getArgOperand(1);
    Value *Op2 = II.getArgOperand(2);

    for (Value *Src : {Op0, Op1, Op2}) {
      if (isa<PoisonValue>(Src))
        return IC.replaceInstUsesWith(II, Src);
    }

    // The legacy behaviour is that multiplying +/-0.0 by anything, even NaN or
    // infinity, gives +0.0.
    // TODO: Move to InstSimplify?
    if (match(Op0, PatternMatch::m_AnyZeroFP()) ||
        match(Op1, PatternMatch::m_AnyZeroFP())) {
      // It's tempting to just return Op2 here, but that would give the wrong
      // result if Op2 was -0.0.
      auto *Zero = ConstantFP::getZero(II.getType());
      auto *FAdd = IC.Builder.CreateFAddFMF(Zero, Op2, &II);
      FAdd->takeName(&II);
      return IC.replaceInstUsesWith(II, FAdd);
    }

    // If we can prove we don't have one of the special cases then we can use a
    // normal fma instead.
    if (canSimplifyLegacyMulToMul(II, Op0, Op1, IC)) {
      II.setCalledOperand(Intrinsic::getOrInsertDeclaration(
          II.getModule(), Intrinsic::fma, II.getType()));
      return &II;
    }
    break;
  }
  case Intrinsic::amdgcn_is_shared:
  case Intrinsic::amdgcn_is_private: {
    Value *Src = II.getArgOperand(0);
    if (isa<PoisonValue>(Src))
      return IC.replaceInstUsesWith(II, PoisonValue::get(II.getType()));
    if (isa<UndefValue>(Src))
      return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));

    if (isa<ConstantPointerNull>(II.getArgOperand(0)))
      return IC.replaceInstUsesWith(II, ConstantInt::getFalse(II.getType()));
    break;
  }
  case Intrinsic::amdgcn_make_buffer_rsrc: {
    Value *Src = II.getArgOperand(0);
    if (isa<PoisonValue>(Src))
      return IC.replaceInstUsesWith(II, PoisonValue::get(II.getType()));
    return std::nullopt;
  }
  case Intrinsic::amdgcn_raw_buffer_store_format:
  case Intrinsic::amdgcn_struct_buffer_store_format:
  case Intrinsic::amdgcn_raw_tbuffer_store:
  case Intrinsic::amdgcn_struct_tbuffer_store:
  case Intrinsic::amdgcn_image_store_1d:
  case Intrinsic::amdgcn_image_store_1darray:
  case Intrinsic::amdgcn_image_store_2d:
  case Intrinsic::amdgcn_image_store_2darray:
  case Intrinsic::amdgcn_image_store_2darraymsaa:
  case Intrinsic::amdgcn_image_store_2dmsaa:
  case Intrinsic::amdgcn_image_store_3d:
  case Intrinsic::amdgcn_image_store_cube:
  case Intrinsic::amdgcn_image_store_mip_1d:
  case Intrinsic::amdgcn_image_store_mip_1darray:
  case Intrinsic::amdgcn_image_store_mip_2d:
  case Intrinsic::amdgcn_image_store_mip_2darray:
  case Intrinsic::amdgcn_image_store_mip_3d:
  case Intrinsic::amdgcn_image_store_mip_cube: {
    if (!isa<FixedVectorType>(II.getArgOperand(0)->getType()))
      break;

    APInt DemandedElts;
    if (ST->hasDefaultComponentBroadcast())
      DemandedElts = defaultComponentBroadcast(II.getArgOperand(0));
    else if (ST->hasDefaultComponentZero())
      DemandedElts = trimTrailingZerosInVector(IC, II.getArgOperand(0), &II);
    else
      break;

    int DMaskIdx = getAMDGPUImageDMaskIntrinsic(II.getIntrinsicID()) ? 1 : -1;
    if (simplifyAMDGCNMemoryIntrinsicDemanded(IC, II, DemandedElts, DMaskIdx,
                                              false)) {
      return IC.eraseInstFromFunction(II);
    }

    break;
  }
  case Intrinsic::amdgcn_prng_b32: {
    auto *Src = II.getArgOperand(0);
    if (isa<UndefValue>(Src)) {
      return IC.replaceInstUsesWith(II, Src);
    }
    return std::nullopt;
  }
  case Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4: {
    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);
    uint64_t CBSZ = cast<ConstantInt>(II.getArgOperand(3))->getZExtValue();
    uint64_t BLGP = cast<ConstantInt>(II.getArgOperand(4))->getZExtValue();
    auto *Src0Ty = cast<FixedVectorType>(Src0->getType());
    auto *Src1Ty = cast<FixedVectorType>(Src1->getType());

    auto getFormatNumRegs = [](unsigned FormatVal) {
      switch (FormatVal) {
      case AMDGPU::MFMAScaleFormats::FP6_E2M3:
      case AMDGPU::MFMAScaleFormats::FP6_E3M2:
        return 6u;
      case AMDGPU::MFMAScaleFormats::FP4_E2M1:
        return 4u;
      case AMDGPU::MFMAScaleFormats::FP8_E4M3:
      case AMDGPU::MFMAScaleFormats::FP8_E5M2:
        return 8u;
      default:
        llvm_unreachable("invalid format value");
      }
    };

    bool MadeChange = false;
    unsigned Src0NumElts = getFormatNumRegs(CBSZ);
    unsigned Src1NumElts = getFormatNumRegs(BLGP);

    // Depending on the used format, fewer registers are required so shrink the
    // vector type.
    if (Src0Ty->getNumElements() > Src0NumElts) {
      Src0 = IC.Builder.CreateExtractVector(
          FixedVectorType::get(Src0Ty->getElementType(), Src0NumElts), Src0,
          uint64_t(0));
      MadeChange = true;
    }

    if (Src1Ty->getNumElements() > Src1NumElts) {
      Src1 = IC.Builder.CreateExtractVector(
          FixedVectorType::get(Src1Ty->getElementType(), Src1NumElts), Src1,
          uint64_t(0));
      MadeChange = true;
    }

    if (!MadeChange)
      return std::nullopt;

    SmallVector<Value *, 10> Args(II.args());
    Args[0] = Src0;
    Args[1] = Src1;

    CallInst *NewII = IC.Builder.CreateIntrinsic(
        IID, {Src0->getType(), Src1->getType()}, Args, &II);
    NewII->takeName(&II);
    return IC.replaceInstUsesWith(II, NewII);
  }
  case Intrinsic::amdgcn_wmma_f32_16x16x128_f8f6f4: {
    Value *Src0 = II.getArgOperand(1);
    Value *Src1 = II.getArgOperand(3);
    unsigned FmtA = cast<ConstantInt>(II.getArgOperand(0))->getZExtValue();
    uint64_t FmtB = cast<ConstantInt>(II.getArgOperand(2))->getZExtValue();
    auto *Src0Ty = cast<FixedVectorType>(Src0->getType());
    auto *Src1Ty = cast<FixedVectorType>(Src1->getType());

    bool MadeChange = false;
    unsigned Src0NumElts = AMDGPU::wmmaScaleF8F6F4FormatToNumRegs(FmtA);
    unsigned Src1NumElts = AMDGPU::wmmaScaleF8F6F4FormatToNumRegs(FmtB);

    // Depending on the used format, fewer registers are required so shrink the
    // vector type.
    if (Src0Ty->getNumElements() > Src0NumElts) {
      Src0 = IC.Builder.CreateExtractVector(
          FixedVectorType::get(Src0Ty->getElementType(), Src0NumElts), Src0,
          IC.Builder.getInt64(0));
      MadeChange = true;
    }

    if (Src1Ty->getNumElements() > Src1NumElts) {
      Src1 = IC.Builder.CreateExtractVector(
          FixedVectorType::get(Src1Ty->getElementType(), Src1NumElts), Src1,
          IC.Builder.getInt64(0));
      MadeChange = true;
    }

    if (!MadeChange)
      return std::nullopt;

    SmallVector<Value *, 13> Args(II.args());
    Args[1] = Src0;
    Args[3] = Src1;

    CallInst *NewII = IC.Builder.CreateIntrinsic(
        IID, {II.getArgOperand(5)->getType(), Src0->getType(), Src1->getType()},
        Args, &II);
    NewII->takeName(&II);
    return IC.replaceInstUsesWith(II, NewII);
  }
  }
  if (const AMDGPU::ImageDimIntrinsicInfo *ImageDimIntr =
            AMDGPU::getImageDimIntrinsicInfo(II.getIntrinsicID())) {
    return simplifyAMDGCNImageIntrinsic(ST, ImageDimIntr, II, IC);
  }
  return std::nullopt;
}

/// Implement SimplifyDemandedVectorElts for amdgcn buffer and image intrinsics.
///
/// The result of simplifying amdgcn image and buffer store intrinsics is updating
/// definitions of the intrinsics vector argument, not Uses of the result like
/// image and buffer loads.
/// Note: This only supports non-TFE/LWE image intrinsic calls; those have
///       struct returns.
static Value *simplifyAMDGCNMemoryIntrinsicDemanded(InstCombiner &IC,
                                                    IntrinsicInst &II,
                                                    APInt DemandedElts,
                                                    int DMaskIdx, bool IsLoad) {

  auto *IIVTy = cast<FixedVectorType>(IsLoad ? II.getType()
                                             : II.getOperand(0)->getType());
  unsigned VWidth = IIVTy->getNumElements();
  if (VWidth == 1)
    return nullptr;
  Type *EltTy = IIVTy->getElementType();

  IRBuilderBase::InsertPointGuard Guard(IC.Builder);
  IC.Builder.SetInsertPoint(&II);

  // Assume the arguments are unchanged and later override them, if needed.
  SmallVector<Value *, 16> Args(II.args());

  if (DMaskIdx < 0) {
    // Buffer case.

    const unsigned ActiveBits = DemandedElts.getActiveBits();
    const unsigned UnusedComponentsAtFront = DemandedElts.countr_zero();

    // Start assuming the prefix of elements is demanded, but possibly clear
    // some other bits if there are trailing zeros (unused components at front)
    // and update offset.
    DemandedElts = (1 << ActiveBits) - 1;

    if (UnusedComponentsAtFront > 0) {
      static const unsigned InvalidOffsetIdx = 0xf;

      unsigned OffsetIdx;
      switch (II.getIntrinsicID()) {
      case Intrinsic::amdgcn_raw_buffer_load:
      case Intrinsic::amdgcn_raw_ptr_buffer_load:
        OffsetIdx = 1;
        break;
      case Intrinsic::amdgcn_s_buffer_load:
        // If resulting type is vec3, there is no point in trimming the
        // load with updated offset, as the vec3 would most likely be widened to
        // vec4 anyway during lowering.
        if (ActiveBits == 4 && UnusedComponentsAtFront == 1)
          OffsetIdx = InvalidOffsetIdx;
        else
          OffsetIdx = 1;
        break;
      case Intrinsic::amdgcn_struct_buffer_load:
      case Intrinsic::amdgcn_struct_ptr_buffer_load:
        OffsetIdx = 2;
        break;
      default:
        // TODO: handle tbuffer* intrinsics.
        OffsetIdx = InvalidOffsetIdx;
        break;
      }

      if (OffsetIdx != InvalidOffsetIdx) {
        // Clear demanded bits and update the offset.
        DemandedElts &= ~((1 << UnusedComponentsAtFront) - 1);
        auto *Offset = Args[OffsetIdx];
        unsigned SingleComponentSizeInBits =
            IC.getDataLayout().getTypeSizeInBits(EltTy);
        unsigned OffsetAdd =
            UnusedComponentsAtFront * SingleComponentSizeInBits / 8;
        auto *OffsetAddVal = ConstantInt::get(Offset->getType(), OffsetAdd);
        Args[OffsetIdx] = IC.Builder.CreateAdd(Offset, OffsetAddVal);
      }
    }
  } else {
    // Image case.

    ConstantInt *DMask = cast<ConstantInt>(Args[DMaskIdx]);
    unsigned DMaskVal = DMask->getZExtValue() & 0xf;

    // dmask 0 has special semantics, do not simplify.
    if (DMaskVal == 0)
      return nullptr;

    // Mask off values that are undefined because the dmask doesn't cover them
    DemandedElts &= (1 << llvm::popcount(DMaskVal)) - 1;

    unsigned NewDMaskVal = 0;
    unsigned OrigLdStIdx = 0;
    for (unsigned SrcIdx = 0; SrcIdx < 4; ++SrcIdx) {
      const unsigned Bit = 1 << SrcIdx;
      if (!!(DMaskVal & Bit)) {
        if (!!DemandedElts[OrigLdStIdx])
          NewDMaskVal |= Bit;
        OrigLdStIdx++;
      }
    }

    if (DMaskVal != NewDMaskVal)
      Args[DMaskIdx] = ConstantInt::get(DMask->getType(), NewDMaskVal);
  }

  unsigned NewNumElts = DemandedElts.popcount();
  if (!NewNumElts)
    return PoisonValue::get(IIVTy);

  if (NewNumElts >= VWidth && DemandedElts.isMask()) {
    if (DMaskIdx >= 0)
      II.setArgOperand(DMaskIdx, Args[DMaskIdx]);
    return nullptr;
  }

  // Validate function argument and return types, extracting overloaded types
  // along the way.
  SmallVector<Type *, 6> OverloadTys;
  if (!Intrinsic::getIntrinsicSignature(II.getCalledFunction(), OverloadTys))
    return nullptr;

  Type *NewTy =
      (NewNumElts == 1) ? EltTy : FixedVectorType::get(EltTy, NewNumElts);
  OverloadTys[0] = NewTy;

  if (!IsLoad) {
    SmallVector<int, 8> EltMask;
    for (unsigned OrigStoreIdx = 0; OrigStoreIdx < VWidth; ++OrigStoreIdx)
      if (DemandedElts[OrigStoreIdx])
        EltMask.push_back(OrigStoreIdx);

    if (NewNumElts == 1)
      Args[0] = IC.Builder.CreateExtractElement(II.getOperand(0), EltMask[0]);
    else
      Args[0] = IC.Builder.CreateShuffleVector(II.getOperand(0), EltMask);
  }

  CallInst *NewCall =
      IC.Builder.CreateIntrinsic(II.getIntrinsicID(), OverloadTys, Args);
  NewCall->takeName(&II);
  NewCall->copyMetadata(II);

  if (IsLoad) {
    if (NewNumElts == 1) {
      return IC.Builder.CreateInsertElement(PoisonValue::get(IIVTy), NewCall,
                                            DemandedElts.countr_zero());
    }

    SmallVector<int, 8> EltMask;
    unsigned NewLoadIdx = 0;
    for (unsigned OrigLoadIdx = 0; OrigLoadIdx < VWidth; ++OrigLoadIdx) {
      if (!!DemandedElts[OrigLoadIdx])
        EltMask.push_back(NewLoadIdx++);
      else
        EltMask.push_back(NewNumElts);
    }

    auto *Shuffle = IC.Builder.CreateShuffleVector(NewCall, EltMask);

    return Shuffle;
  }

  return NewCall;
}

Value *GCNTTIImpl::simplifyAMDGCNLaneIntrinsicDemanded(
    InstCombiner &IC, IntrinsicInst &II, const APInt &DemandedElts,
    APInt &UndefElts) const {
  auto *VT = dyn_cast<FixedVectorType>(II.getType());
  if (!VT)
    return nullptr;

  const unsigned FirstElt = DemandedElts.countr_zero();
  const unsigned LastElt = DemandedElts.getActiveBits() - 1;
  const unsigned MaskLen = LastElt - FirstElt + 1;

  unsigned OldNumElts = VT->getNumElements();
  if (MaskLen == OldNumElts && MaskLen != 1)
    return nullptr;

  Type *EltTy = VT->getElementType();
  Type *NewVT = MaskLen == 1 ? EltTy : FixedVectorType::get(EltTy, MaskLen);

  // Theoretically we should support these intrinsics for any legal type. Avoid
  // introducing cases that aren't direct register types like v3i16.
  if (!isTypeLegal(NewVT))
    return nullptr;

  Value *Src = II.getArgOperand(0);

  // Make sure convergence tokens are preserved.
  // TODO: CreateIntrinsic should allow directly copying bundles
  SmallVector<OperandBundleDef, 2> OpBundles;
  II.getOperandBundlesAsDefs(OpBundles);

  Module *M = IC.Builder.GetInsertBlock()->getModule();
  Function *Remangled =
      Intrinsic::getOrInsertDeclaration(M, II.getIntrinsicID(), {NewVT});

  if (MaskLen == 1) {
    Value *Extract = IC.Builder.CreateExtractElement(Src, FirstElt);

    // TODO: Preserve callsite attributes?
    CallInst *NewCall = IC.Builder.CreateCall(Remangled, {Extract}, OpBundles);

    return IC.Builder.CreateInsertElement(PoisonValue::get(II.getType()),
                                          NewCall, FirstElt);
  }

  SmallVector<int> ExtractMask(MaskLen, -1);
  for (unsigned I = 0; I != MaskLen; ++I) {
    if (DemandedElts[FirstElt + I])
      ExtractMask[I] = FirstElt + I;
  }

  Value *Extract = IC.Builder.CreateShuffleVector(Src, ExtractMask);

  // TODO: Preserve callsite attributes?
  CallInst *NewCall = IC.Builder.CreateCall(Remangled, {Extract}, OpBundles);

  SmallVector<int> InsertMask(OldNumElts, -1);
  for (unsigned I = 0; I != MaskLen; ++I) {
    if (DemandedElts[FirstElt + I])
      InsertMask[FirstElt + I] = I;
  }

  // FIXME: If the call has a convergence bundle, we end up leaving the dead
  // call behind.
  return IC.Builder.CreateShuffleVector(NewCall, InsertMask);
}

std::optional<Value *> GCNTTIImpl::simplifyDemandedVectorEltsIntrinsic(
    InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
    APInt &UndefElts2, APInt &UndefElts3,
    std::function<void(Instruction *, unsigned, APInt, APInt &)>
        SimplifyAndSetOp) const {
  switch (II.getIntrinsicID()) {
  case Intrinsic::amdgcn_readfirstlane:
    SimplifyAndSetOp(&II, 0, DemandedElts, UndefElts);
    return simplifyAMDGCNLaneIntrinsicDemanded(IC, II, DemandedElts, UndefElts);
  case Intrinsic::amdgcn_raw_buffer_load:
  case Intrinsic::amdgcn_raw_ptr_buffer_load:
  case Intrinsic::amdgcn_raw_buffer_load_format:
  case Intrinsic::amdgcn_raw_ptr_buffer_load_format:
  case Intrinsic::amdgcn_raw_tbuffer_load:
  case Intrinsic::amdgcn_raw_ptr_tbuffer_load:
  case Intrinsic::amdgcn_s_buffer_load:
  case Intrinsic::amdgcn_struct_buffer_load:
  case Intrinsic::amdgcn_struct_ptr_buffer_load:
  case Intrinsic::amdgcn_struct_buffer_load_format:
  case Intrinsic::amdgcn_struct_ptr_buffer_load_format:
  case Intrinsic::amdgcn_struct_tbuffer_load:
  case Intrinsic::amdgcn_struct_ptr_tbuffer_load:
    return simplifyAMDGCNMemoryIntrinsicDemanded(IC, II, DemandedElts);
  default: {
    if (getAMDGPUImageDMaskIntrinsic(II.getIntrinsicID())) {
      return simplifyAMDGCNMemoryIntrinsicDemanded(IC, II, DemandedElts, 0);
    }
    break;
  }
  }
  return std::nullopt;
}
