//===- Type.cpp - Sandbox IR Type -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Type.h"
#include "llvm/SandboxIR/Context.h"

using namespace llvm::sandboxir;

Type *Type::getScalarType() const {
  return Ctx.getType(LLVMTy->getScalarType());
}

Type *Type::getInt64Ty(Context &Ctx) {
  return Ctx.getType(llvm::Type::getInt64Ty(Ctx.LLVMCtx));
}
Type *Type::getInt32Ty(Context &Ctx) {
  return Ctx.getType(llvm::Type::getInt32Ty(Ctx.LLVMCtx));
}
Type *Type::getInt16Ty(Context &Ctx) {
  return Ctx.getType(llvm::Type::getInt16Ty(Ctx.LLVMCtx));
}
Type *Type::getInt8Ty(Context &Ctx) {
  return Ctx.getType(llvm::Type::getInt8Ty(Ctx.LLVMCtx));
}
Type *Type::getInt1Ty(Context &Ctx) {
  return Ctx.getType(llvm::Type::getInt1Ty(Ctx.LLVMCtx));
}
Type *Type::getDoubleTy(Context &Ctx) {
  return Ctx.getType(llvm::Type::getDoubleTy(Ctx.LLVMCtx));
}
Type *Type::getFloatTy(Context &Ctx) {
  return Ctx.getType(llvm::Type::getFloatTy(Ctx.LLVMCtx));
}

#ifndef NDEBUG
void Type::dumpOS(raw_ostream &OS) { LLVMTy->print(OS); }
void Type::dump() {
  dumpOS(dbgs());
  dbgs() << "\n";
}
#endif

PointerType *PointerType::get(Context &Ctx, unsigned AddressSpace) {
  return cast<PointerType>(
      Ctx.getType(llvm::PointerType::get(Ctx.LLVMCtx, AddressSpace)));
}

ArrayType *ArrayType::get(Type *ElementType, uint64_t NumElements) {
  return cast<ArrayType>(ElementType->getContext().getType(
      llvm::ArrayType::get(ElementType->LLVMTy, NumElements)));
}

StructType *StructType::get(Context &Ctx, ArrayRef<Type *> Elements,
                            bool IsPacked) {
  SmallVector<llvm::Type *> LLVMElements;
  LLVMElements.reserve(Elements.size());
  for (Type *Elm : Elements)
    LLVMElements.push_back(Elm->LLVMTy);
  return cast<StructType>(
      Ctx.getType(llvm::StructType::get(Ctx.LLVMCtx, LLVMElements, IsPacked)));
}

VectorType *VectorType::get(Type *ElementType, ElementCount EC) {
  return cast<VectorType>(ElementType->getContext().getType(
      llvm::VectorType::get(ElementType->LLVMTy, EC)));
}

Type *VectorType::getElementType() const {
  return Ctx.getType(cast<llvm::VectorType>(LLVMTy)->getElementType());
}
VectorType *VectorType::getInteger(VectorType *VTy) {
  return cast<VectorType>(VTy->getContext().getType(
      llvm::VectorType::getInteger(cast<llvm::VectorType>(VTy->LLVMTy))));
}
VectorType *VectorType::getExtendedElementVectorType(VectorType *VTy) {
  return cast<VectorType>(
      VTy->getContext().getType(llvm::VectorType::getExtendedElementVectorType(
          cast<llvm::VectorType>(VTy->LLVMTy))));
}
VectorType *VectorType::getTruncatedElementVectorType(VectorType *VTy) {
  return cast<VectorType>(
      VTy->getContext().getType(llvm::VectorType::getTruncatedElementVectorType(
          cast<llvm::VectorType>(VTy->LLVMTy))));
}
VectorType *VectorType::getSubdividedVectorType(VectorType *VTy,
                                                int NumSubdivs) {
  return cast<VectorType>(
      VTy->getContext().getType(llvm::VectorType::getSubdividedVectorType(
          cast<llvm::VectorType>(VTy->LLVMTy), NumSubdivs)));
}
VectorType *VectorType::getHalfElementsVectorType(VectorType *VTy) {
  return cast<VectorType>(
      VTy->getContext().getType(llvm::VectorType::getHalfElementsVectorType(
          cast<llvm::VectorType>(VTy->LLVMTy))));
}
VectorType *VectorType::getDoubleElementsVectorType(VectorType *VTy) {
  return cast<VectorType>(
      VTy->getContext().getType(llvm::VectorType::getDoubleElementsVectorType(
          cast<llvm::VectorType>(VTy->LLVMTy))));
}
bool VectorType::isValidElementType(Type *ElemTy) {
  return llvm::VectorType::isValidElementType(ElemTy->LLVMTy);
}

FixedVectorType *FixedVectorType::get(Type *ElementType, unsigned NumElts) {
  return cast<FixedVectorType>(ElementType->getContext().getType(
      llvm::FixedVectorType::get(ElementType->LLVMTy, NumElts)));
}

ScalableVectorType *ScalableVectorType::get(Type *ElementType,
                                            unsigned NumElts) {
  return cast<ScalableVectorType>(ElementType->getContext().getType(
      llvm::ScalableVectorType::get(ElementType->LLVMTy, NumElts)));
}

IntegerType *IntegerType::get(Context &Ctx, unsigned NumBits) {
  return cast<IntegerType>(
      Ctx.getType(llvm::IntegerType::get(Ctx.LLVMCtx, NumBits)));
}
