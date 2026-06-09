//===-- LoongArchTargetTransformInfo.cpp - LoongArch specific TTI ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// LoongArch target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "LoongArchTargetTransformInfo.h"

using namespace llvm;

#define DEBUG_TYPE "loongarchtti"

TypeSize LoongArchTTIImpl::getRegisterBitWidth(
    TargetTransformInfo::RegisterKind K) const {
  TypeSize DefSize = TargetTransformInfoImplBase::getRegisterBitWidth(K);
  switch (K) {
  case TargetTransformInfo::RGK_Scalar:
    return TypeSize::getFixed(ST->is64Bit() ? 64 : 32);
  case TargetTransformInfo::RGK_FixedWidthVector:
    if (ST->hasExtLASX())
      return TypeSize::getFixed(256);
    if (ST->hasExtLSX())
      return TypeSize::getFixed(128);
    [[fallthrough]];
  case TargetTransformInfo::RGK_ScalableVector:
    return DefSize;
  }

  llvm_unreachable("Unsupported register kind");
}

unsigned LoongArchTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  switch (ClassID) {
  case LoongArchRegisterClass::GPRRC:
    // 30 = 32 GPRs - r0 (zero register) - r21 (non-allocatable)
    return 30;
  case LoongArchRegisterClass::FPRRC:
    return ST->hasBasicF() ? 32 : 0;
  case LoongArchRegisterClass::VRRC:
    return ST->hasExtLSX() ? 32 : 0;
  }
  llvm_unreachable("unknown register class");
}

unsigned LoongArchTTIImpl::getRegisterClassForType(bool Vector,
                                                   Type *Ty) const {
  if (Vector)
    return LoongArchRegisterClass::VRRC;
  if (!Ty)
    return LoongArchRegisterClass::GPRRC;

  Type *ScalarTy = Ty->getScalarType();
  if ((ScalarTy->isFloatTy() && ST->hasBasicF()) ||
      (ScalarTy->isDoubleTy() && ST->hasBasicD())) {
    return LoongArchRegisterClass::FPRRC;
  }

  return LoongArchRegisterClass::GPRRC;
}

unsigned LoongArchTTIImpl::getMaxInterleaveFactor(ElementCount VF) const {
  return ST->getMaxInterleaveFactor();
}

const char *LoongArchTTIImpl::getRegisterClassName(unsigned ClassID) const {
  switch (ClassID) {
  case LoongArchRegisterClass::GPRRC:
    return "LoongArch::GPRRC";
  case LoongArchRegisterClass::FPRRC:
    return "LoongArch::FPRRC";
  case LoongArchRegisterClass::VRRC:
    return "LoongArch::VRRC";
  }
  llvm_unreachable("unknown register class");
}

TargetTransformInfo::PopcntSupportKind
LoongArchTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return ST->hasExtLSX() ? TTI::PSK_FastHardware : TTI::PSK_Software;
}

unsigned LoongArchTTIImpl::getCacheLineSize() const { return 64; }

unsigned LoongArchTTIImpl::getPrefetchDistance() const { return 200; }

bool LoongArchTTIImpl::enableWritePrefetching() const { return true; }

bool LoongArchTTIImpl::shouldExpandReduction(const IntrinsicInst *II) const {
  switch (II->getIntrinsicID()) {
  default:
    return true;
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_and:
  case Intrinsic::vector_reduce_or:
  case Intrinsic::vector_reduce_smax:
  case Intrinsic::vector_reduce_smin:
  case Intrinsic::vector_reduce_umax:
  case Intrinsic::vector_reduce_umin:
  case Intrinsic::vector_reduce_xor:
    return false;
  }
}

LoongArchTTIImpl::TTI::MemCmpExpansionOptions
LoongArchTTIImpl::enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const {
  TTI::MemCmpExpansionOptions Options;

  if (!ST->hasUAL())
    return Options;

  Options.MaxNumLoads = TLI->getMaxExpandSizeMemcmp(OptSize);
  Options.NumLoadsPerBlock = Options.MaxNumLoads;
  Options.AllowOverlappingLoads = true;

  // TODO: Support for vectors.
  if (ST->is64Bit()) {
    Options.LoadSizes = {8, 4, 2, 1};
    Options.AllowedTailExpansions = {3, 5, 6};
  } else {
    Options.LoadSizes = {4, 2, 1};
    Options.AllowedTailExpansions = {3};
  }

  return Options;
}

InstructionCost LoongArchTTIImpl::getPartialReductionCost(
    unsigned Opcode, Type *InputTypeA, Type *InputTypeB, Type *AccumType,
    ElementCount VF, TTI::PartialReductionExtendKind OpAExtend,
    TTI::PartialReductionExtendKind OpBExtend, std::optional<unsigned> BinOp,
    TTI::TargetCostKind CostKind, std::optional<FastMathFlags> FMF) const {
  InstructionCost Invalid = InstructionCost::getInvalid();

  if (CostKind != TTI::TCK_RecipThroughput)
    return Invalid;

  if (Opcode != Instruction::Add)
    return Invalid;

  // LoongArch only supports fixed-length vector now.
  if (!ST->hasExtLSX() || VF.isScalable())
    return Invalid;

  // We only support multiply binary operations for now, and for muls we
  // require the types being extended to be the same.
  if (InputTypeA != InputTypeB)
    return Invalid;

  if (!BinOp || *BinOp != Instruction::Mul)
    return Invalid;

  // Must be sext or zext, different extend type is allowed for SUMLA.
  if (OpAExtend == TTI::PR_None || OpBExtend == TTI::PR_None)
    return Invalid;

  unsigned Ratio =
      AccumType->getScalarSizeInBits() / InputTypeA->getScalarSizeInBits();
  if (VF.getKnownMinValue() <= Ratio)
    return Invalid;

  VectorType *InputVectorType = VectorType::get(InputTypeA, VF);
  VectorType *AccumVectorType =
      VectorType::get(AccumType, VF.divideCoefficientBy(Ratio));
  // We don't yet support all kinds of legalization.
  auto TC = TLI->getTypeConversion(AccumVectorType->getContext(),
                                   EVT::getEVT(AccumVectorType));
  switch (TC.first) {
  default:
    return Invalid;
  case TargetLowering::TypeLegal:
  case TargetLowering::TypePromoteInteger:
  case TargetLowering::TypeSplitVector:
    // The legalised type (e.g. after splitting) must be legal too.
    if (TLI->getTypeAction(AccumVectorType->getContext(), TC.second) !=
        TargetLowering::TypeLegal)
      return Invalid;
    break;
  }

  std::pair<InstructionCost, MVT> AccumLT =
      getTypeLegalizationCost(AccumVectorType);
  std::pair<InstructionCost, MVT> InputLT =
      getTypeLegalizationCost(InputVectorType);

  // A legal i8->i32 partial reduction will be lowered into:
  //   vmulwev.h.b[u] + vmulwod.h.b[u] + 2 x vhaddw.w.h + 2 x vadd.w
  // The estimate recip-throughput is 10.
  InstructionCost Cost = InputLT.first * TTI::TCC_Basic * 10;
  unsigned Bits = AccumLT.second.getSizeInBits();
  if (AccumLT.second.getScalarType() == MVT::i32 &&
      InputLT.second.getScalarType() == MVT::i8)
    if ((ST->hasExtLASX() && Bits <= 256) || (ST->hasExtLSX() && Bits <= 128))
      return Cost;

  return BaseT::getPartialReductionCost(Opcode, InputTypeA, InputTypeB,
                                        AccumType, VF, OpAExtend, OpBExtend,
                                        BinOp, CostKind, FMF);
}
