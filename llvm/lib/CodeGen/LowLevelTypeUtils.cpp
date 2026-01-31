//===-- llvm/CodeGen/LowLevelTypeUtils.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the more header-heavy bits of the LLT class to
/// avoid polluting users' namespaces.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
using namespace llvm;

LLT llvm::getLLTForType(Type &Ty, const DataLayout &DL) {
  if (auto *VTy = dyn_cast<VectorType>(&Ty)) {
    auto EC = VTy->getElementCount();
    LLT ScalarTy = getLLTForType(*VTy->getElementType(), DL);
    if (EC.isScalar())
      return ScalarTy;
    return LLT::vector(EC, ScalarTy);
  }

  if (auto *PTy = dyn_cast<PointerType>(&Ty)) {
    unsigned AddrSpace = PTy->getAddressSpace();
    return LLT::pointer(AddrSpace, DL.getPointerSizeInBits(AddrSpace));
  }

  if (Ty.isSized() && !Ty.isScalableTargetExtTy()) {
    // Aggregates are no different from real scalars as far as GlobalISel is
    // concerned.
    auto SizeInBits = DL.getTypeSizeInBits(&Ty);
    assert(SizeInBits != 0 && "invalid zero-sized type");

    // Return simple scalar
    if (!LLT::getUseExtended())
      return LLT::scalar(SizeInBits);

    // Choose more precise LLT variant
    if (Ty.isFloatingPointTy())
      switch (Ty.getTypeID()) {
      default:
        llvm_unreachable("Unhandled LLVM IR floating point type");
      case Type::HalfTyID:
        return LLT::float16();
      case Type::BFloatTyID:
        return LLT::bfloat16();
      case Type::FloatTyID:
        return LLT::float32();
      case Type::DoubleTyID:
        return LLT::float64();
      case Type::X86_FP80TyID:
        return LLT::x86fp80();
      case Type::FP128TyID:
        return LLT::float128();
      case Type::PPC_FP128TyID:
        return LLT::ppcf128();
      }

    if (Ty.isIntegerTy())
      return LLT::integer(SizeInBits);

    return LLT::scalar(SizeInBits);
  }

  if (Ty.isTokenTy())
    return LLT::token();

  return LLT();
}

MVT llvm::getMVTForLLT(LLT Ty) {
  if (Ty.isVector())
    return MVT::getVectorVT(getMVTForLLT(Ty.getElementType()),
                            Ty.getElementCount());

  if (Ty.isFloat()) {
    if (Ty.isBFloat16())
      return MVT::bf16;

    if (Ty.isX86FP80())
      return MVT::f80;

    if (Ty.isPPCF128())
      return MVT::ppcf128;

    return MVT::getFloatingPointVT(Ty.getSizeInBits());
  }

  return MVT::getIntegerVT(Ty.getSizeInBits());
}

EVT llvm::getApproximateEVTForLLT(LLT Ty, LLVMContext &Ctx) {
  if (Ty.isVector()) {
    EVT EltVT = getApproximateEVTForLLT(Ty.getElementType(), Ctx);
    return EVT::getVectorVT(Ctx, EltVT, Ty.getElementCount());
  }

  return EVT::getIntegerVT(Ctx, Ty.getSizeInBits());
}

LLT llvm::getLLTForMVT(MVT VT) { return LLT(VT); }

const llvm::fltSemantics &llvm::getFltSemanticForLLT(LLT Ty) {
  assert((Ty.isAnyScalar() || Ty.isFloat()) &&
         "Expected a any scalar or float type.");

  // Any scalar type always matches IEEE format
  if (Ty.isAnyScalar()) {
    switch (Ty.getSizeInBits()) {
    default:
      llvm_unreachable("Invalid FP type size.");
    case 16:
      return APFloat::IEEEhalf();
    case 32:
      return APFloat::IEEEsingle();
    case 64:
      return APFloat::IEEEdouble();
    case 128:
      return APFloat::IEEEquad();
    }
  }

  return APFloat::EnumToSemantics(Ty.getFpSemantics());
}
