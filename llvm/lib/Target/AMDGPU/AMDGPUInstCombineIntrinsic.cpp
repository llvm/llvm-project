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
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include <optional>

using namespace llvm;

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
  if (isa<FPExtInst>(&V) || isa<SExtInst>(&V) || isa<ZExtInst>(&V))
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

  Function *I = Intrinsic::getDeclaration(OldIntr.getModule(), NewIntr, ArgTys);

  CallInst *NewCall = IC.Builder.CreateCall(I, Args);
  NewCall->takeName(&OldIntr);
  NewCall->copyMetadata(OldIntr);
  if (isa<FPMathOperator>(NewCall))
    NewCall->copyFastMathFlags(&OldIntr);

  // Erase and replace uses
  if (!InstToReplace.getType()->isVoidTy())
    IC.replaceInstUsesWith(InstToReplace, NewCall);

  bool RemoveOldIntr = &OldIntr != &InstToReplace;

  auto RetValue = IC.eraseInstFromFunction(InstToReplace);
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

bool GCNTTIImpl::canSimplifyLegacyMulToMul(const Value *Op0, const Value *Op1,
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
  auto *TLI = &IC.getTargetLibraryInfo();
  if (isKnownNeverInfinity(Op0, TLI) && isKnownNeverNaN(Op0, TLI) &&
      isKnownNeverInfinity(Op1, TLI) && isKnownNeverNaN(Op1, TLI)) {
    // Neither operand is infinity or NaN.
    return true;
  }
  return false;
}

std::optional<Instruction *>
GCNTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  case Intrinsic::amdgcn_rcp: {
    Value *Src = II.getArgOperand(0);

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

    break;
  }
  case Intrinsic::amdgcn_sqrt:
  case Intrinsic::amdgcn_rsq: {
    Value *Src = II.getArgOperand(0);

    // TODO: Move to ConstantFolding/InstSimplify?
    if (isa<UndefValue>(Src)) {
      Type *Ty = II.getType();
      auto *QNaN = ConstantFP::get(Ty, APFloat::getQNaN(Ty->getFltSemantics()));
      return IC.replaceInstUsesWith(II, QNaN);
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

    if (isa<UndefValue>(Src)) {
      return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));
    }

    break;
  }
  case Intrinsic::amdgcn_class: {
    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);
    const ConstantInt *CMask = dyn_cast<ConstantInt>(Src1);
    if (!CMask) {
      if (isa<UndefValue>(Src0)) {
        return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));
      }

      if (isa<UndefValue>(Src1)) {
        return IC.replaceInstUsesWith(II,
                                      ConstantInt::get(II.getType(), false));
      }
      break;
    }

    uint32_t Mask = CMask->getZExtValue();

    // If all tests are made, it doesn't matter what the value is.
    if ((Mask & fcAllFlags) == fcAllFlags) {
      return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), true));
    }

    if ((Mask & fcAllFlags) == 0) {
      return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), false));
    }

    if (Mask == fcNan && !II.isStrictFP()) {
      // Equivalent of isnan. Replace with standard fcmp.
      Value *FCmp = IC.Builder.CreateFCmpUNO(Src0, Src0);
      FCmp->takeName(&II);
      return IC.replaceInstUsesWith(II, FCmp);
    }

    if (Mask == fcZero && !II.isStrictFP()) {
      // Equivalent of == 0.
      Value *FCmp =
          IC.Builder.CreateFCmpOEQ(Src0, ConstantFP::get(Src0->getType(), 0.0));

      FCmp->takeName(&II);
      return IC.replaceInstUsesWith(II, FCmp);
    }

    // fp_class (nnan x), qnan|snan|other -> fp_class (nnan x), other
    if ((Mask & fcNan) && isKnownNeverNaN(Src0, &IC.getTargetLibraryInfo())) {
      return IC.replaceOperand(
          II, 1, ConstantInt::get(Src1->getType(), Mask & ~fcNan));
    }

    const ConstantFP *CVal = dyn_cast<ConstantFP>(Src0);
    if (!CVal) {
      if (isa<UndefValue>(Src0)) {
        return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));
      }

      // Clamp mask to used bits
      if ((Mask & fcAllFlags) != Mask) {
        CallInst *NewCall = IC.Builder.CreateCall(
            II.getCalledFunction(),
            {Src0, ConstantInt::get(Src1->getType(), Mask & fcAllFlags)});

        NewCall->takeName(&II);
        return IC.replaceInstUsesWith(II, NewCall);
      }

      break;
    }

    const APFloat &Val = CVal->getValueAPF();

    bool Result =
        ((Mask & fcSNan) && Val.isNaN() && Val.isSignaling()) ||
        ((Mask & fcQNan) && Val.isNaN() && !Val.isSignaling()) ||
        ((Mask & fcNegInf) && Val.isInfinity() && Val.isNegative()) ||
        ((Mask & fcNegNormal) && Val.isNormal() && Val.isNegative()) ||
        ((Mask & fcNegSubnormal) && Val.isDenormal() && Val.isNegative()) ||
        ((Mask & fcNegZero) && Val.isZero() && Val.isNegative()) ||
        ((Mask & fcPosZero) && Val.isZero() && !Val.isNegative()) ||
        ((Mask & fcPosSubnormal) && Val.isDenormal() && !Val.isNegative()) ||
        ((Mask & fcPosNormal) && Val.isNormal() && !Val.isNegative()) ||
        ((Mask & fcPosInf) && Val.isInfinity() && !Val.isNegative());

    return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), Result));
  }
  case Intrinsic::amdgcn_cvt_pkrtz: {
    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);
    if (const ConstantFP *C0 = dyn_cast<ConstantFP>(Src0)) {
      if (const ConstantFP *C1 = dyn_cast<ConstantFP>(Src1)) {
        const fltSemantics &HalfSem =
            II.getType()->getScalarType()->getFltSemantics();
        bool LosesInfo;
        APFloat Val0 = C0->getValueAPF();
        APFloat Val1 = C1->getValueAPF();
        Val0.convert(HalfSem, APFloat::rmTowardZero, &LosesInfo);
        Val1.convert(HalfSem, APFloat::rmTowardZero, &LosesInfo);

        Constant *Folded =
            ConstantVector::get({ConstantFP::get(II.getContext(), Val0),
                                 ConstantFP::get(II.getContext(), Val1)});
        return IC.replaceInstUsesWith(II, Folded);
      }
    }

    if (isa<UndefValue>(Src0) && isa<UndefValue>(Src1)) {
      return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));
    }

    break;
  }
  case Intrinsic::amdgcn_cvt_pknorm_i16:
  case Intrinsic::amdgcn_cvt_pknorm_u16:
  case Intrinsic::amdgcn_cvt_pk_i16:
  case Intrinsic::amdgcn_cvt_pk_u16: {
    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);

    if (isa<UndefValue>(Src0) && isa<UndefValue>(Src1)) {
      return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));
    }

    break;
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
        if (!isa<UndefValue>(Src)) {
          IC.replaceOperand(II, I + 2, UndefValue::get(Src->getType()));
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
    // Note this does not preserve proper sNaN behavior if IEEE-mode is enabled
    // for the shader.

    Value *Src0 = II.getArgOperand(0);
    Value *Src1 = II.getArgOperand(1);
    Value *Src2 = II.getArgOperand(2);

    // Checking for NaN before canonicalization provides better fidelity when
    // mapping other operations onto fmed3 since the order of operands is
    // unchanged.
    CallInst *NewCall = nullptr;
    if (match(Src0, PatternMatch::m_NaN()) || isa<UndefValue>(Src0)) {
      NewCall = IC.Builder.CreateMinNum(Src1, Src2);
    } else if (match(Src1, PatternMatch::m_NaN()) || isa<UndefValue>(Src1)) {
      NewCall = IC.Builder.CreateMinNum(Src0, Src2);
    } else if (match(Src2, PatternMatch::m_NaN()) || isa<UndefValue>(Src2)) {
      NewCall = IC.Builder.CreateMaxNum(Src0, Src1);
    }

    if (NewCall) {
      NewCall->copyFastMathFlags(&II);
      NewCall->takeName(&II);
      return IC.replaceInstUsesWith(II, NewCall);
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
          return IC.replaceInstUsesWith(
              II, ConstantFP::get(IC.Builder.getContext(), Result));
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
        Constant *CCmp = ConstantExpr::getCompare(CCVal, CSrc0, CSrc1);
        if (CCmp->isNullValue()) {
          return IC.replaceInstUsesWith(
              II, ConstantExpr::getSExt(CCmp, II.getType()));
        }

        // The result of V_ICMP/V_FCMP assembly instructions (which this
        // intrinsic exposes) is one bit per thread, masked with the EXEC
        // register (which contains the bitmask of live threads). So a
        // comparison that always returns true is the same as a read of the
        // EXEC register.
        Function *NewF = Intrinsic::getDeclaration(
            II.getModule(), Intrinsic::read_register, II.getType());
        Metadata *MDArgs[] = {MDString::get(II.getContext(), "exec")};
        MDNode *MD = MDNode::get(II.getContext(), MDArgs);
        Value *Args[] = {MetadataAsValue::get(II.getContext(), MD)};
        CallInst *NewCall = IC.Builder.CreateCall(NewF, Args);
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

    CmpInst::Predicate SrcPred;
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
        else if (Width > 64)
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

      Function *NewF = Intrinsic::getDeclaration(
          II.getModule(), NewIID, {II.getType(), SrcLHS->getType()});
      Value *Args[] = {SrcLHS, SrcRHS,
                       ConstantInt::get(CC->getType(), SrcPred)};
      CallInst *NewCall = IC.Builder.CreateCall(NewF, Args);
      NewCall->takeName(&II);
      return IC.replaceInstUsesWith(II, NewCall);
    }

    break;
  }
  case Intrinsic::amdgcn_ballot: {
    if (auto *Src = dyn_cast<ConstantInt>(II.getArgOperand(0))) {
      if (Src->isZero()) {
        // amdgcn.ballot(i1 0) is zero.
        return IC.replaceInstUsesWith(II, Constant::getNullValue(II.getType()));
      }

      if (Src->isOne()) {
        // amdgcn.ballot(i1 1) is exec.
        const char *RegName = "exec";
        if (II.getType()->isIntegerTy(32))
          RegName = "exec_lo";
        else if (!II.getType()->isIntegerTy(64))
          break;

        Function *NewF = Intrinsic::getDeclaration(
            II.getModule(), Intrinsic::read_register, II.getType());
        Metadata *MDArgs[] = {MDString::get(II.getContext(), RegName)};
        MDNode *MD = MDNode::get(II.getContext(), MDArgs);
        Value *Args[] = {MetadataAsValue::get(II.getContext(), MD)};
        CallInst *NewCall = IC.Builder.CreateCall(NewF, Args);
        NewCall->addFnAttr(Attribute::Convergent);
        NewCall->takeName(&II);
        return IC.replaceInstUsesWith(II, NewCall);
      }
    }
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
        BM->getZExtValue() != 0xF || isa<UndefValue>(Old))
      break;

    // If bound_ctrl = 1, row mask = bank mask = 0xf we can omit old value.
    return IC.replaceOperand(II, 0, UndefValue::get(Old->getType()));
  }
  case Intrinsic::amdgcn_permlane16:
  case Intrinsic::amdgcn_permlanex16: {
    // Discard vdst_in if it's not going to be read.
    Value *VDstIn = II.getArgOperand(0);
    if (isa<UndefValue>(VDstIn))
      break;

    ConstantInt *FetchInvalid = cast<ConstantInt>(II.getArgOperand(4));
    ConstantInt *BoundCtrl = cast<ConstantInt>(II.getArgOperand(5));
    if (!FetchInvalid->getZExtValue() && !BoundCtrl->getZExtValue())
      break;

    return IC.replaceOperand(II, 0, UndefValue::get(VDstIn->getType()));
  }
  case Intrinsic::amdgcn_permlane64:
    // A constant value is trivially uniform.
    if (Constant *C = dyn_cast<Constant>(II.getArgOperand(0))) {
      return IC.replaceInstUsesWith(II, C);
    }
    break;
  case Intrinsic::amdgcn_readfirstlane:
  case Intrinsic::amdgcn_readlane: {
    // A constant value is trivially uniform.
    if (Constant *C = dyn_cast<Constant>(II.getArgOperand(0))) {
      return IC.replaceInstUsesWith(II, C);
    }

    // The rest of these may not be safe if the exec may not be the same between
    // the def and use.
    Value *Src = II.getArgOperand(0);
    Instruction *SrcInst = dyn_cast<Instruction>(Src);
    if (SrcInst && SrcInst->getParent() != II.getParent())
      break;

    // readfirstlane (readfirstlane x) -> readfirstlane x
    // readlane (readfirstlane x), y -> readfirstlane x
    if (match(Src,
              PatternMatch::m_Intrinsic<Intrinsic::amdgcn_readfirstlane>())) {
      return IC.replaceInstUsesWith(II, Src);
    }

    if (IID == Intrinsic::amdgcn_readfirstlane) {
      // readfirstlane (readlane x, y) -> readlane x, y
      if (match(Src, PatternMatch::m_Intrinsic<Intrinsic::amdgcn_readlane>())) {
        return IC.replaceInstUsesWith(II, Src);
      }
    } else {
      // readlane (readlane x, y), y -> readlane x, y
      if (match(Src, PatternMatch::m_Intrinsic<Intrinsic::amdgcn_readlane>(
                         PatternMatch::m_Value(),
                         PatternMatch::m_Specific(II.getArgOperand(1))))) {
        return IC.replaceInstUsesWith(II, Src);
      }
    }

    break;
  }
  case Intrinsic::amdgcn_ldexp: {
    // FIXME: This doesn't introduce new instructions and belongs in
    // InstructionSimplify.
    Type *Ty = II.getType();
    Value *Op0 = II.getArgOperand(0);
    Value *Op1 = II.getArgOperand(1);

    // Folding undef to qnan is safe regardless of the FP mode.
    if (isa<UndefValue>(Op0)) {
      auto *QNaN = ConstantFP::get(Ty, APFloat::getQNaN(Ty->getFltSemantics()));
      return IC.replaceInstUsesWith(II, QNaN);
    }

    const APFloat *C = nullptr;
    match(Op0, PatternMatch::m_APFloat(C));

    // FIXME: Should flush denorms depending on FP mode, but that's ignored
    // everywhere else.
    //
    // These cases should be safe, even with strictfp.
    // ldexp(0.0, x) -> 0.0
    // ldexp(-0.0, x) -> -0.0
    // ldexp(inf, x) -> inf
    // ldexp(-inf, x) -> -inf
    if (C && (C->isZero() || C->isInfinity())) {
      return IC.replaceInstUsesWith(II, Op0);
    }

    // With strictfp, be more careful about possibly needing to flush denormals
    // or not, and snan behavior depends on ieee_mode.
    if (II.isStrictFP())
      break;

    if (C && C->isNaN())
      return IC.replaceInstUsesWith(II, ConstantFP::get(Ty, C->makeQuiet()));

    // ldexp(x, 0) -> x
    // ldexp(x, undef) -> x
    if (isa<UndefValue>(Op1) || match(Op1, PatternMatch::m_ZeroInt())) {
      return IC.replaceInstUsesWith(II, Op0);
    }

    break;
  }
  case Intrinsic::amdgcn_fmul_legacy: {
    Value *Op0 = II.getArgOperand(0);
    Value *Op1 = II.getArgOperand(1);

    // The legacy behaviour is that multiplying +/-0.0 by anything, even NaN or
    // infinity, gives +0.0.
    // TODO: Move to InstSimplify?
    if (match(Op0, PatternMatch::m_AnyZeroFP()) ||
        match(Op1, PatternMatch::m_AnyZeroFP()))
      return IC.replaceInstUsesWith(II, ConstantFP::getNullValue(II.getType()));

    // If we can prove we don't have one of the special cases then we can use a
    // normal fmul instruction instead.
    if (canSimplifyLegacyMulToMul(Op0, Op1, IC)) {
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

    // The legacy behaviour is that multiplying +/-0.0 by anything, even NaN or
    // infinity, gives +0.0.
    // TODO: Move to InstSimplify?
    if (match(Op0, PatternMatch::m_AnyZeroFP()) ||
        match(Op1, PatternMatch::m_AnyZeroFP())) {
      // It's tempting to just return Op2 here, but that would give the wrong
      // result if Op2 was -0.0.
      auto *Zero = ConstantFP::getNullValue(II.getType());
      auto *FAdd = IC.Builder.CreateFAddFMF(Zero, Op2, &II);
      FAdd->takeName(&II);
      return IC.replaceInstUsesWith(II, FAdd);
    }

    // If we can prove we don't have one of the special cases then we can use a
    // normal fma instead.
    if (canSimplifyLegacyMulToMul(Op0, Op1, IC)) {
      II.setCalledOperand(Intrinsic::getDeclaration(
          II.getModule(), Intrinsic::fma, II.getType()));
      return &II;
    }
    break;
  }
  case Intrinsic::amdgcn_is_shared:
  case Intrinsic::amdgcn_is_private: {
    if (isa<UndefValue>(II.getArgOperand(0)))
      return IC.replaceInstUsesWith(II, UndefValue::get(II.getType()));

    if (isa<ConstantPointerNull>(II.getArgOperand(0)))
      return IC.replaceInstUsesWith(II, ConstantInt::getFalse(II.getType()));
    break;
  }
  default: {
    if (const AMDGPU::ImageDimIntrinsicInfo *ImageDimIntr =
            AMDGPU::getImageDimIntrinsicInfo(II.getIntrinsicID())) {
      return simplifyAMDGCNImageIntrinsic(ST, ImageDimIntr, II, IC);
    }
  }
  }
  return std::nullopt;
}

/// Implement SimplifyDemandedVectorElts for amdgcn buffer and image intrinsics.
///
/// Note: This only supports non-TFE/LWE image intrinsic calls; those have
///       struct returns.
static Value *simplifyAMDGCNMemoryIntrinsicDemanded(InstCombiner &IC,
                                                    IntrinsicInst &II,
                                                    APInt DemandedElts,
                                                    int DMaskIdx = -1) {

  auto *IIVTy = cast<FixedVectorType>(II.getType());
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
    const unsigned UnusedComponentsAtFront = DemandedElts.countTrailingZeros();

    // Start assuming the prefix of elements is demanded, but possibly clear
    // some other bits if there are trailing zeros (unused components at front)
    // and update offset.
    DemandedElts = (1 << ActiveBits) - 1;

    if (UnusedComponentsAtFront > 0) {
      static const unsigned InvalidOffsetIdx = 0xf;

      unsigned OffsetIdx;
      switch (II.getIntrinsicID()) {
      case Intrinsic::amdgcn_raw_buffer_load:
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

    // Mask off values that are undefined because the dmask doesn't cover them
    DemandedElts &= (1 << llvm::popcount(DMaskVal)) - 1;

    unsigned NewDMaskVal = 0;
    unsigned OrigLoadIdx = 0;
    for (unsigned SrcIdx = 0; SrcIdx < 4; ++SrcIdx) {
      const unsigned Bit = 1 << SrcIdx;
      if (!!(DMaskVal & Bit)) {
        if (!!DemandedElts[OrigLoadIdx])
          NewDMaskVal |= Bit;
        OrigLoadIdx++;
      }
    }

    if (DMaskVal != NewDMaskVal)
      Args[DMaskIdx] = ConstantInt::get(DMask->getType(), NewDMaskVal);
  }

  unsigned NewNumElts = DemandedElts.countPopulation();
  if (!NewNumElts)
    return UndefValue::get(IIVTy);

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

  Function *NewIntrin = Intrinsic::getDeclaration(
      II.getModule(), II.getIntrinsicID(), OverloadTys);
  CallInst *NewCall = IC.Builder.CreateCall(NewIntrin, Args);
  NewCall->takeName(&II);
  NewCall->copyMetadata(II);

  if (NewNumElts == 1) {
    return IC.Builder.CreateInsertElement(UndefValue::get(IIVTy), NewCall,
                                          DemandedElts.countTrailingZeros());
  }

  SmallVector<int, 8> EltMask;
  unsigned NewLoadIdx = 0;
  for (unsigned OrigLoadIdx = 0; OrigLoadIdx < VWidth; ++OrigLoadIdx) {
    if (!!DemandedElts[OrigLoadIdx])
      EltMask.push_back(NewLoadIdx++);
    else
      EltMask.push_back(NewNumElts);
  }

  Value *Shuffle = IC.Builder.CreateShuffleVector(NewCall, EltMask);

  return Shuffle;
}

std::optional<Value *> GCNTTIImpl::simplifyDemandedVectorEltsIntrinsic(
    InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
    APInt &UndefElts2, APInt &UndefElts3,
    std::function<void(Instruction *, unsigned, APInt, APInt &)>
        SimplifyAndSetOp) const {
  switch (II.getIntrinsicID()) {
  case Intrinsic::amdgcn_buffer_load:
  case Intrinsic::amdgcn_buffer_load_format:
  case Intrinsic::amdgcn_raw_buffer_load:
  case Intrinsic::amdgcn_raw_buffer_load_format:
  case Intrinsic::amdgcn_raw_tbuffer_load:
  case Intrinsic::amdgcn_s_buffer_load:
  case Intrinsic::amdgcn_struct_buffer_load:
  case Intrinsic::amdgcn_struct_buffer_load_format:
  case Intrinsic::amdgcn_struct_tbuffer_load:
  case Intrinsic::amdgcn_tbuffer_load:
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
