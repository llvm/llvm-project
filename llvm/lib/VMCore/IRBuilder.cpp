//===---- IRBuilder.cpp - Builder for LLVM Instrs -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IRBuilder class, which is used as a convenient way
// to create LLVM instructions with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/IRBuilder.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
using namespace llvm;

/// CreateGlobalString - Make a new global variable with an initializer that
/// has array of i8 type filled in with the nul terminated string value
/// specified.  If Name is specified, it is the name of the global variable
/// created.
Value *IRBuilderBase::CreateGlobalString(StringRef Str, const Twine &Name) {
  Constant *StrConstant = ConstantDataArray::getString(Context, Str);
  Module &M = *BB->getParent()->getParent();
  GlobalVariable *GV = new GlobalVariable(M, StrConstant->getType(),
                                          true, GlobalValue::PrivateLinkage,
                                          StrConstant);
  GV->setName(Name);
  GV->setUnnamedAddr(true);
  return GV;
}

Type *IRBuilderBase::getCurrentFunctionReturnType() const {
  assert(BB && BB->getParent() && "No current function!");
  return BB->getParent()->getReturnType();
}

Value *IRBuilderBase::getCastedInt8PtrValue(Value *Ptr) {
  PointerType *PT = cast<PointerType>(Ptr->getType());
  if (PT->getElementType()->isIntegerTy(8))
    return Ptr;
  
  // Otherwise, we need to insert a bitcast.
  PT = getInt8PtrTy(PT->getAddressSpace());
  BitCastInst *BCI = new BitCastInst(Ptr, PT, "");
  BB->getInstList().insert(InsertPt, BCI);
  SetInstDebugLocation(BCI);
  return BCI;
}

static CallInst *createCallHelper(Value *Callee, ArrayRef<Value *> Ops,
                                  IRBuilderBase *Builder) {
  CallInst *CI = CallInst::Create(Callee, Ops, "");
  Builder->GetInsertBlock()->getInstList().insert(Builder->GetInsertPoint(),CI);
  Builder->SetInstDebugLocation(CI);
  return CI;  
}

CallInst *IRBuilderBase::
CreateMemSet(Value *Ptr, Value *Val, Value *Size, unsigned Align,
             bool isVolatile, MDNode *TBAATag) {
  Ptr = getCastedInt8PtrValue(Ptr);
  Value *Ops[] = { Ptr, Val, Size, getInt32(Align), getInt1(isVolatile) };
  Type *Tys[] = { Ptr->getType(), Size->getType() };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::memset, Tys);
  
  CallInst *CI = createCallHelper(TheFn, Ops, this);
  
  // Set the TBAA info if present.
  if (TBAATag)
    CI->setMetadata(LLVMContext::MD_tbaa, TBAATag);
  
  return CI;
}

CallInst *IRBuilderBase::
CreateMemCpy(Value *Dst, Value *Src, Value *Size, unsigned Align,
             bool isVolatile, MDNode *TBAATag, MDNode *TBAAStructTag) {
  Dst = getCastedInt8PtrValue(Dst);
  Src = getCastedInt8PtrValue(Src);

  Value *Ops[] = { Dst, Src, Size, getInt32(Align), getInt1(isVolatile) };
  Type *Tys[] = { Dst->getType(), Src->getType(), Size->getType() };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::memcpy, Tys);
  
  CallInst *CI = createCallHelper(TheFn, Ops, this);
  
  // Set the TBAA info if present.
  if (TBAATag)
    CI->setMetadata(LLVMContext::MD_tbaa, TBAATag);

  // Set the TBAA Struct info if present.
  if (TBAAStructTag)
    CI->setMetadata(LLVMContext::MD_tbaa_struct, TBAAStructTag);
  
  return CI;  
}

CallInst *IRBuilderBase::
CreateMemMove(Value *Dst, Value *Src, Value *Size, unsigned Align,
              bool isVolatile, MDNode *TBAATag) {
  Dst = getCastedInt8PtrValue(Dst);
  Src = getCastedInt8PtrValue(Src);
  
  Value *Ops[] = { Dst, Src, Size, getInt32(Align), getInt1(isVolatile) };
  Type *Tys[] = { Dst->getType(), Src->getType(), Size->getType() };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::memmove, Tys);
  
  CallInst *CI = createCallHelper(TheFn, Ops, this);
  
  // Set the TBAA info if present.
  if (TBAATag)
    CI->setMetadata(LLVMContext::MD_tbaa, TBAATag);
  
  return CI;  
}

CallInst *IRBuilderBase::CreateLifetimeStart(Value *Ptr, ConstantInt *Size) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "lifetime.start only applies to pointers.");
  Ptr = getCastedInt8PtrValue(Ptr);
  if (!Size)
    Size = getInt64(-1);
  else
    assert(Size->getType() == getInt64Ty() &&
           "lifetime.start requires the size to be an i64");
  Value *Ops[] = { Size, Ptr };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::lifetime_start);
  return createCallHelper(TheFn, Ops, this);
}

CallInst *IRBuilderBase::CreateLifetimeEnd(Value *Ptr, ConstantInt *Size) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "lifetime.end only applies to pointers.");
  Ptr = getCastedInt8PtrValue(Ptr);
  if (!Size)
    Size = getInt64(-1);
  else
    assert(Size->getType() == getInt64Ty() &&
           "lifetime.end requires the size to be an i64");
  Value *Ops[] = { Size, Ptr };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::lifetime_end);
  return createCallHelper(TheFn, Ops, this);
}
