//===- Type.cpp - Sandbox IR Type -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Type.h"
#include "llvm/SandboxIR/SandboxIR.h"

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

PointerType *PointerType::get(Type *ElementType, unsigned AddressSpace) {
  return cast<PointerType>(ElementType->getContext().getType(
      llvm::PointerType::get(ElementType->LLVMTy, AddressSpace)));
}

PointerType *PointerType::get(Context &Ctx, unsigned AddressSpace) {
  return cast<PointerType>(
      Ctx.getType(llvm::PointerType::get(Ctx.LLVMCtx, AddressSpace)));
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

IntegerType *IntegerType::get(Context &Ctx, unsigned NumBits) {
  return cast<IntegerType>(
      Ctx.getType(llvm::IntegerType::get(Ctx.LLVMCtx, NumBits)));
}
