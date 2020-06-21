//===----------- ValueTypes.cpp - Implementation of EVT methods -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TypeSize.h"
using namespace llvm;

EVT EVT::changeExtendedTypeToInteger() const {
  LLVMContext &Context = LLVMTy->getContext();
  return getIntegerVT(Context, getSizeInBits());
}

EVT EVT::changeExtendedVectorElementTypeToInteger() const {
  LLVMContext &Context = LLVMTy->getContext();
  EVT IntTy = getIntegerVT(Context, getScalarSizeInBits());
  return getVectorVT(Context, IntTy, getVectorNumElements(),
                     isScalableVector());
}

EVT EVT::changeExtendedVectorElementType(EVT EltVT) const {
  LLVMContext &Context = LLVMTy->getContext();
  return getVectorVT(Context, EltVT, getVectorElementCount());
}

EVT EVT::getExtendedIntegerVT(LLVMContext &Context, unsigned BitWidth) {
  EVT VT;
  VT.LLVMTy = IntegerType::get(Context, BitWidth);
  assert(VT.isExtended() && "Type is not extended!");
  return VT;
}

EVT EVT::getExtendedVectorVT(LLVMContext &Context, EVT VT, unsigned NumElements,
                             bool IsScalable) {
  EVT ResultVT;
  ResultVT.LLVMTy =
      VectorType::get(VT.getTypeForEVT(Context), NumElements, IsScalable);
  assert(ResultVT.isExtended() && "Type is not extended!");
  return ResultVT;
}

EVT EVT::getExtendedVectorVT(LLVMContext &Context, EVT VT, ElementCount EC) {
  EVT ResultVT;
  ResultVT.LLVMTy =
      VectorType::get(VT.getTypeForEVT(Context), {EC.Min, EC.Scalable});
  assert(ResultVT.isExtended() && "Type is not extended!");
  return ResultVT;
}

bool EVT::isExtendedFloatingPoint() const {
  assert(isExtended() && "Type is not extended!");
  return LLVMTy->isFPOrFPVectorTy();
}

bool EVT::isExtendedInteger() const {
  assert(isExtended() && "Type is not extended!");
  return LLVMTy->isIntOrIntVectorTy();
}

bool EVT::isExtendedScalarInteger() const {
  assert(isExtended() && "Type is not extended!");
  return LLVMTy->isIntegerTy();
}

bool EVT::isExtendedVector() const {
  assert(isExtended() && "Type is not extended!");
  return LLVMTy->isVectorTy();
}

bool EVT::isExtended16BitVector() const {
  return isExtendedVector() && getExtendedSizeInBits() == 16;
}

bool EVT::isExtended32BitVector() const {
  return isExtendedVector() && getExtendedSizeInBits() == 32;
}

bool EVT::isExtended64BitVector() const {
  return isExtendedVector() && getExtendedSizeInBits() == 64;
}

bool EVT::isExtended128BitVector() const {
  return isExtendedVector() && getExtendedSizeInBits() == 128;
}

bool EVT::isExtended256BitVector() const {
  return isExtendedVector() && getExtendedSizeInBits() == 256;
}

bool EVT::isExtended512BitVector() const {
  return isExtendedVector() && getExtendedSizeInBits() == 512;
}

bool EVT::isExtended1024BitVector() const {
  return isExtendedVector() && getExtendedSizeInBits() == 1024;
}

bool EVT::isExtended2048BitVector() const {
  return isExtendedVector() && getExtendedSizeInBits() == 2048;
}

bool EVT::isExtendedFixedLengthVector() const {
  return isExtendedVector() && isa<FixedVectorType>(LLVMTy);
}

bool EVT::isExtendedScalableVector() const {
  return isExtendedVector() && isa<ScalableVectorType>(LLVMTy);
}

EVT EVT::getExtendedVectorElementType() const {
  assert(isExtended() && "Type is not extended!");
  return EVT::getEVT(cast<VectorType>(LLVMTy)->getElementType());
}

unsigned EVT::getExtendedVectorNumElements() const {
  assert(isExtended() && "Type is not extended!");
  return cast<VectorType>(LLVMTy)->getNumElements();
}

ElementCount EVT::getExtendedVectorElementCount() const {
  assert(isExtended() && "Type is not extended!");
  return cast<VectorType>(LLVMTy)->getElementCount();
}

TypeSize EVT::getExtendedSizeInBits() const {
  assert(isExtended() && "Type is not extended!");
  if (IntegerType *ITy = dyn_cast<IntegerType>(LLVMTy))
    return TypeSize::Fixed(ITy->getBitWidth());
  if (VectorType *VTy = dyn_cast<VectorType>(LLVMTy))
    return VTy->getPrimitiveSizeInBits();
  llvm_unreachable("Unrecognized extended type!");
}

/// getEVTString - This function returns value type as a string, e.g. "i32".
std::string EVT::getEVTString() const {
  switch (V.SimpleTy) {
  default:
    if (isVector())
      return (isScalableVector() ? "nxv" : "v")
             + utostr(getVectorElementCount().Min)
             + getVectorElementType().getEVTString();
    if (isInteger())
      return "i" + utostr(getSizeInBits());
    if (isFloatingPoint())
      return "f" + utostr(getSizeInBits());
    llvm_unreachable("Invalid EVT!");
  case MVT::bf16:    return "bf16";
  case MVT::ppcf128: return "ppcf128";
  case MVT::isVoid:  return "isVoid";
  case MVT::Other:   return "ch";
  case MVT::Glue:    return "glue";
  case MVT::x86mmx:  return "x86mmx";
  case MVT::Metadata:return "Metadata";
  case MVT::Untyped: return "Untyped";
  case MVT::exnref : return "exnref";
  }
}

/// getTypeForEVT - This method returns an LLVM type corresponding to the
/// specified EVT.  For integer types, this returns an unsigned type.  Note
/// that this will abort for types that cannot be represented.
Type *EVT::getTypeForEVT(LLVMContext &Context) const {
  switch (V.SimpleTy) {
  default:
    assert(isExtended() && "Type is not extended!");
    return LLVMTy;
  case MVT::isVoid:  return Type::getVoidTy(Context);
  case MVT::i1:      return Type::getInt1Ty(Context);
  case MVT::i8:      return Type::getInt8Ty(Context);
  case MVT::i16:     return Type::getInt16Ty(Context);
  case MVT::i32:     return Type::getInt32Ty(Context);
  case MVT::i64:     return Type::getInt64Ty(Context);
  case MVT::i128:    return IntegerType::get(Context, 128);
  case MVT::f16:     return Type::getHalfTy(Context);
  case MVT::bf16:     return Type::getBFloatTy(Context);
  case MVT::f32:     return Type::getFloatTy(Context);
  case MVT::f64:     return Type::getDoubleTy(Context);
  case MVT::f80:     return Type::getX86_FP80Ty(Context);
  case MVT::f128:    return Type::getFP128Ty(Context);
  case MVT::ppcf128: return Type::getPPC_FP128Ty(Context);
  case MVT::x86mmx:  return Type::getX86_MMXTy(Context);
  case MVT::v1i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 1);
  case MVT::v2i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 2);
  case MVT::v4i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 4);
  case MVT::v8i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 8);
  case MVT::v16i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 16);
  case MVT::v32i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 32);
  case MVT::v64i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 64);
  case MVT::v128i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 128);
  case MVT::v256i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 256);
  case MVT::v512i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 512);
  case MVT::v1024i1:
    return FixedVectorType::get(Type::getInt1Ty(Context), 1024);
  case MVT::v1i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 1);
  case MVT::v2i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 2);
  case MVT::v4i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 4);
  case MVT::v8i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 8);
  case MVT::v16i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 16);
  case MVT::v32i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 32);
  case MVT::v64i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 64);
  case MVT::v128i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 128);
  case MVT::v256i8:
    return FixedVectorType::get(Type::getInt8Ty(Context), 256);
  case MVT::v1i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 1);
  case MVT::v2i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 2);
  case MVT::v3i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 3);
  case MVT::v4i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 4);
  case MVT::v8i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 8);
  case MVT::v16i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 16);
  case MVT::v32i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 32);
  case MVT::v64i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 64);
  case MVT::v128i16:
    return FixedVectorType::get(Type::getInt16Ty(Context), 128);
  case MVT::v1i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 1);
  case MVT::v2i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 2);
  case MVT::v3i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 3);
  case MVT::v4i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 4);
  case MVT::v5i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 5);
  case MVT::v8i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 8);
  case MVT::v16i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 16);
  case MVT::v32i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 32);
  case MVT::v64i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 64);
  case MVT::v128i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 128);
  case MVT::v256i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 256);
  case MVT::v512i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 512);
  case MVT::v1024i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 1024);
  case MVT::v2048i32:
    return FixedVectorType::get(Type::getInt32Ty(Context), 2048);
  case MVT::v1i64:
    return FixedVectorType::get(Type::getInt64Ty(Context), 1);
  case MVT::v2i64:
    return FixedVectorType::get(Type::getInt64Ty(Context), 2);
  case MVT::v4i64:
    return FixedVectorType::get(Type::getInt64Ty(Context), 4);
  case MVT::v8i64:
    return FixedVectorType::get(Type::getInt64Ty(Context), 8);
  case MVT::v16i64:
    return FixedVectorType::get(Type::getInt64Ty(Context), 16);
  case MVT::v32i64:
    return FixedVectorType::get(Type::getInt64Ty(Context), 32);
  case MVT::v1i128:
    return FixedVectorType::get(Type::getInt128Ty(Context), 1);
  case MVT::v2f16:
    return FixedVectorType::get(Type::getHalfTy(Context), 2);
  case MVT::v3f16:
    return FixedVectorType::get(Type::getHalfTy(Context), 3);
  case MVT::v4f16:
    return FixedVectorType::get(Type::getHalfTy(Context), 4);
  case MVT::v8f16:
    return FixedVectorType::get(Type::getHalfTy(Context), 8);
  case MVT::v16f16:
    return FixedVectorType::get(Type::getHalfTy(Context), 16);
  case MVT::v32f16:
    return FixedVectorType::get(Type::getHalfTy(Context), 32);
  case MVT::v64f16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 64);
  case MVT::v128f16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 128);
  case MVT::v2bf16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 2);
  case MVT::v3bf16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 3);
  case MVT::v4bf16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 4);
  case MVT::v8bf16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 8);
  case MVT::v16bf16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 16);
  case MVT::v32bf16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 32);
  case MVT::v64bf16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 64);
  case MVT::v128bf16:
    return FixedVectorType::get(Type::getBFloatTy(Context), 128);
  case MVT::v1f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 1);
  case MVT::v2f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 2);
  case MVT::v3f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 3);
  case MVT::v4f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 4);
  case MVT::v5f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 5);
  case MVT::v8f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 8);
  case MVT::v16f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 16);
  case MVT::v32f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 32);
  case MVT::v64f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 64);
  case MVT::v128f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 128);
  case MVT::v256f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 256);
  case MVT::v512f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 512);
  case MVT::v1024f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 1024);
  case MVT::v2048f32:
    return FixedVectorType::get(Type::getFloatTy(Context), 2048);
  case MVT::v1f64:
    return FixedVectorType::get(Type::getDoubleTy(Context), 1);
  case MVT::v2f64:
    return FixedVectorType::get(Type::getDoubleTy(Context), 2);
  case MVT::v4f64:
    return FixedVectorType::get(Type::getDoubleTy(Context), 4);
  case MVT::v8f64:
    return FixedVectorType::get(Type::getDoubleTy(Context), 8);
  case MVT::v16f64:
    return FixedVectorType::get(Type::getDoubleTy(Context), 16);
  case MVT::v32f64:
    return FixedVectorType::get(Type::getDoubleTy(Context), 32);
  case MVT::nxv1i1:
    return ScalableVectorType::get(Type::getInt1Ty(Context), 1);
  case MVT::nxv2i1:
    return ScalableVectorType::get(Type::getInt1Ty(Context), 2);
  case MVT::nxv4i1:
    return ScalableVectorType::get(Type::getInt1Ty(Context), 4);
  case MVT::nxv8i1:
    return ScalableVectorType::get(Type::getInt1Ty(Context), 8);
  case MVT::nxv16i1:
    return ScalableVectorType::get(Type::getInt1Ty(Context), 16);
  case MVT::nxv32i1:
    return ScalableVectorType::get(Type::getInt1Ty(Context), 32);
  case MVT::nxv1i8:
    return ScalableVectorType::get(Type::getInt8Ty(Context), 1);
  case MVT::nxv2i8:
    return ScalableVectorType::get(Type::getInt8Ty(Context), 2);
  case MVT::nxv4i8:
    return ScalableVectorType::get(Type::getInt8Ty(Context), 4);
  case MVT::nxv8i8:
    return ScalableVectorType::get(Type::getInt8Ty(Context), 8);
  case MVT::nxv16i8:
    return ScalableVectorType::get(Type::getInt8Ty(Context), 16);
  case MVT::nxv32i8:
    return ScalableVectorType::get(Type::getInt8Ty(Context), 32);
  case MVT::nxv1i16:
    return ScalableVectorType::get(Type::getInt16Ty(Context), 1);
  case MVT::nxv2i16:
    return ScalableVectorType::get(Type::getInt16Ty(Context), 2);
  case MVT::nxv4i16:
    return ScalableVectorType::get(Type::getInt16Ty(Context), 4);
  case MVT::nxv8i16:
    return ScalableVectorType::get(Type::getInt16Ty(Context), 8);
  case MVT::nxv16i16:
    return ScalableVectorType::get(Type::getInt16Ty(Context), 16);
  case MVT::nxv32i16:
    return ScalableVectorType::get(Type::getInt16Ty(Context), 32);
  case MVT::nxv1i32:
    return ScalableVectorType::get(Type::getInt32Ty(Context), 1);
  case MVT::nxv2i32:
    return ScalableVectorType::get(Type::getInt32Ty(Context), 2);
  case MVT::nxv4i32:
    return ScalableVectorType::get(Type::getInt32Ty(Context), 4);
  case MVT::nxv8i32:
    return ScalableVectorType::get(Type::getInt32Ty(Context), 8);
  case MVT::nxv16i32:
    return ScalableVectorType::get(Type::getInt32Ty(Context), 16);
  case MVT::nxv32i32:
    return ScalableVectorType::get(Type::getInt32Ty(Context), 32);
  case MVT::nxv1i64:
    return ScalableVectorType::get(Type::getInt64Ty(Context), 1);
  case MVT::nxv2i64:
    return ScalableVectorType::get(Type::getInt64Ty(Context), 2);
  case MVT::nxv4i64:
    return ScalableVectorType::get(Type::getInt64Ty(Context), 4);
  case MVT::nxv8i64:
    return ScalableVectorType::get(Type::getInt64Ty(Context), 8);
  case MVT::nxv16i64:
    return ScalableVectorType::get(Type::getInt64Ty(Context), 16);
  case MVT::nxv32i64:
    return ScalableVectorType::get(Type::getInt64Ty(Context), 32);
  case MVT::nxv2f16:
    return ScalableVectorType::get(Type::getHalfTy(Context), 2);
  case MVT::nxv4f16:
    return ScalableVectorType::get(Type::getHalfTy(Context), 4);
  case MVT::nxv8f16:
    return ScalableVectorType::get(Type::getHalfTy(Context), 8);
  case MVT::nxv2bf16:
    return ScalableVectorType::get(Type::getBFloatTy(Context), 2);
  case MVT::nxv4bf16:
    return ScalableVectorType::get(Type::getBFloatTy(Context), 4);
  case MVT::nxv8bf16:
    return ScalableVectorType::get(Type::getBFloatTy(Context), 8);
  case MVT::nxv1f32:
    return ScalableVectorType::get(Type::getFloatTy(Context), 1);
  case MVT::nxv2f32:
    return ScalableVectorType::get(Type::getFloatTy(Context), 2);
  case MVT::nxv4f32:
    return ScalableVectorType::get(Type::getFloatTy(Context), 4);
  case MVT::nxv8f32:
    return ScalableVectorType::get(Type::getFloatTy(Context), 8);
  case MVT::nxv16f32:
    return ScalableVectorType::get(Type::getFloatTy(Context), 16);
  case MVT::nxv1f64:
    return ScalableVectorType::get(Type::getDoubleTy(Context), 1);
  case MVT::nxv2f64:
    return ScalableVectorType::get(Type::getDoubleTy(Context), 2);
  case MVT::nxv4f64:
    return ScalableVectorType::get(Type::getDoubleTy(Context), 4);
  case MVT::nxv8f64:
    return ScalableVectorType::get(Type::getDoubleTy(Context), 8);
  case MVT::Metadata: return Type::getMetadataTy(Context);
  }
}

/// Return the value type corresponding to the specified type.  This returns all
/// pointers as MVT::iPTR.  If HandleUnknown is true, unknown types are returned
/// as Other, otherwise they are invalid.
MVT MVT::getVT(Type *Ty, bool HandleUnknown){
  switch (Ty->getTypeID()) {
  default:
    if (HandleUnknown) return MVT(MVT::Other);
    llvm_unreachable("Unknown type!");
  case Type::VoidTyID:
    return MVT::isVoid;
  case Type::IntegerTyID:
    return getIntegerVT(cast<IntegerType>(Ty)->getBitWidth());
  case Type::HalfTyID:      return MVT(MVT::f16);
  case Type::BFloatTyID:    return MVT(MVT::bf16);
  case Type::FloatTyID:     return MVT(MVT::f32);
  case Type::DoubleTyID:    return MVT(MVT::f64);
  case Type::X86_FP80TyID:  return MVT(MVT::f80);
  case Type::X86_MMXTyID:   return MVT(MVT::x86mmx);
  case Type::FP128TyID:     return MVT(MVT::f128);
  case Type::PPC_FP128TyID: return MVT(MVT::ppcf128);
  case Type::PointerTyID:   return MVT(MVT::iPTR);
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID: {
    VectorType *VTy = cast<VectorType>(Ty);
    return getVectorVT(
      getVT(VTy->getElementType(), /*HandleUnknown=*/ false),
            VTy->getElementCount());
  }
  }
}

/// getEVT - Return the value type corresponding to the specified type.  This
/// returns all pointers as MVT::iPTR.  If HandleUnknown is true, unknown types
/// are returned as Other, otherwise they are invalid.
EVT EVT::getEVT(Type *Ty, bool HandleUnknown){
  switch (Ty->getTypeID()) {
  default:
    return MVT::getVT(Ty, HandleUnknown);
  case Type::IntegerTyID:
    return getIntegerVT(Ty->getContext(), cast<IntegerType>(Ty)->getBitWidth());
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID: {
    VectorType *VTy = cast<VectorType>(Ty);
    return getVectorVT(Ty->getContext(),
                       getEVT(VTy->getElementType(), /*HandleUnknown=*/ false),
                       VTy->getElementCount());
  }
  }
}
