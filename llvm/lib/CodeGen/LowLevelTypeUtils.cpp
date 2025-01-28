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
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

LLT llvm::getLLTForType(Type &Ty, const DataLayout &DL, bool EnableFPInfo) {
  if (auto VTy = dyn_cast<VectorType>(&Ty)) {
    auto EC = VTy->getElementCount();
    LLT ScalarTy = getLLTForType(*VTy->getElementType(), DL, EnableFPInfo);
    if (EC.isScalar())
      return ScalarTy;
    return LLT::vector(EC, ScalarTy);
  }

  if (auto PTy = dyn_cast<PointerType>(&Ty)) {
    unsigned AddrSpace = PTy->getAddressSpace();
    return LLT::pointer(AddrSpace, DL.getPointerSizeInBits(AddrSpace));
  }

  if (Ty.isSized() && !Ty.isScalableTargetExtTy()) {
    // Aggregates are no different from real scalars as far as GlobalISel is
    // concerned.
    auto SizeInBits = DL.getTypeSizeInBits(&Ty);
    assert(SizeInBits != 0 && "invalid zero-sized type");

    if (Ty.isFloatingPointTy()) {
      if (Ty.isHalfTy())
        return LLT::float16();

      if (Ty.isBFloatTy())
        return LLT::bfloat();

      if (Ty.isFloatTy())
        return LLT::float32();

      if (Ty.isDoubleTy())
        return LLT::float64();

      if (Ty.isX86_FP80Ty())
        return LLT::x86fp80();

      if (Ty.isFP128Ty())
        return LLT::float128();

      if (Ty.isPPC_FP128Ty())
        return LLT::ppcf128();

      llvm_unreachable("Unhandled LLVM IR floating point type");
    }

    if (Ty.isIntegerTy()) {
      return LLT::integer(SizeInBits);
    }

    return LLT::integer(SizeInBits);
  }

  if (Ty.isTokenTy())
    return LLT::token();

  return LLT();
}

MVT llvm::getMVTForLLT(LLT Ty, bool EnableFPInfo) {
  if (Ty.isVector()) {
    return MVT::getVectorVT(getMVTForLLT(Ty.getElementType()), Ty.getElementCount());
  }

  if (Ty.isFloat()) {
    if (Ty == LLT::bfloat())
      return MVT::bf16;

    if (Ty == LLT::x86fp80())
      return MVT::f80;

    if (Ty == LLT::ppcf128())
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

LLT llvm::getLLTForMVT(MVT Ty, bool EnableFPInfo) {
  if (EnableFPInfo)
    return LLT(Ty, EnableFPInfo);

  if (!Ty.isVector())
    return LLT::integer(Ty.getSizeInBits());

  return LLT::scalarOrVector(Ty.getVectorElementCount(),
                             LLT::integer(Ty.getVectorElementType().getSizeInBits()));
}

const llvm::fltSemantics &llvm::getFltSemanticForLLT(LLT Ty) {
  assert(Ty.isScalar() && "Expected a scalar type.");

  if (Ty.isBFloat())
    return APFloat::BFloat();
  if (Ty.isX86FP80())
    return APFloat::x87DoubleExtended();
  if (Ty.isPPCF128())
    return APFloat::PPCDoubleDouble();

  assert(!Ty.isVariantFloat() && "Unhandled variant float type");

  switch (Ty.getSizeInBits()) {
  case 16:
    return APFloat::IEEEhalf();
  case 32:
    return APFloat::IEEEsingle();
  case 64:
    return APFloat::IEEEdouble();
  case 128:
    return APFloat::IEEEquad();
  }
  llvm_unreachable("Invalid FP type size.");
}
