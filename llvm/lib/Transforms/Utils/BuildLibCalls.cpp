//===- BuildLibCalls.cpp - Utility builder for libcalls -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements some functions that will create standard C libcalls.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace llvm;

Value *llvm::castToCStr(Value *V, IRBuilder<> &B) {
  unsigned AS = V->getType()->getPointerAddressSpace();
  return B.CreateBitCast(V, B.getInt8PtrTy(AS), "cstr");
}

Value *llvm::emitStrLen(Value *Ptr, IRBuilder<> &B, const DataLayout &DL,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strlen))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Constant *StrLen = M->getOrInsertFunction(
      "strlen", AttributeSet::get(M->getContext(), AS),
      DL.getIntPtrType(Context), B.getInt8PtrTy(), nullptr);
  CallInst *CI = B.CreateCall(StrLen, castToCStr(Ptr, B), "strlen");
  if (const Function *F = dyn_cast<Function>(StrLen->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitStrChr(Value *Ptr, char C, IRBuilder<> &B,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strchr))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AttributeSet AS =
    AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  Type *I8Ptr = B.getInt8PtrTy();
  Type *I32Ty = B.getInt32Ty();
  Constant *StrChr = M->getOrInsertFunction("strchr",
                                            AttributeSet::get(M->getContext(),
                                                             AS),
                                            I8Ptr, I8Ptr, I32Ty, nullptr);
  CallInst *CI = B.CreateCall(
      StrChr, {castToCStr(Ptr, B), ConstantInt::get(I32Ty, C)}, "strchr");
  if (const Function *F = dyn_cast<Function>(StrChr->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitStrNCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                         const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strncmp))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[3];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS[2] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *StrNCmp = M->getOrInsertFunction(
      "strncmp", AttributeSet::get(M->getContext(), AS), B.getInt32Ty(),
      B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context), nullptr);
  CallInst *CI = B.CreateCall(
      StrNCmp, {castToCStr(Ptr1, B), castToCStr(Ptr2, B), Len}, "strncmp");

  if (const Function *F = dyn_cast<Function>(StrNCmp->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitStrCpy(Value *Dst, Value *Src, IRBuilder<> &B,
                        const TargetLibraryInfo *TLI, StringRef Name) {
  if (!TLI->has(LibFunc::strcpy))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  Type *I8Ptr = B.getInt8PtrTy();
  Value *StrCpy = M->getOrInsertFunction(Name,
                                         AttributeSet::get(M->getContext(), AS),
                                         I8Ptr, I8Ptr, I8Ptr, nullptr);
  CallInst *CI =
      B.CreateCall(StrCpy, {castToCStr(Dst, B), castToCStr(Src, B)}, Name);
  if (const Function *F = dyn_cast<Function>(StrCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitStrNCpy(Value *Dst, Value *Src, Value *Len, IRBuilder<> &B,
                         const TargetLibraryInfo *TLI, StringRef Name) {
  if (!TLI->has(LibFunc::strncpy))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  Type *I8Ptr = B.getInt8PtrTy();
  Value *StrNCpy = M->getOrInsertFunction(Name,
                                          AttributeSet::get(M->getContext(),
                                                            AS),
                                          I8Ptr, I8Ptr, I8Ptr,
                                          Len->getType(), nullptr);
  CallInst *CI = B.CreateCall(
      StrNCpy, {castToCStr(Dst, B), castToCStr(Src, B), Len}, "strncpy");
  if (const Function *F = dyn_cast<Function>(StrNCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitMemCpyChk(Value *Dst, Value *Src, Value *Len, Value *ObjSize,
                           IRBuilder<> &B, const DataLayout &DL,
                           const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memcpy_chk))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS;
  AS = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                         Attribute::NoUnwind);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemCpy = M->getOrInsertFunction(
      "__memcpy_chk", AttributeSet::get(M->getContext(), AS), B.getInt8PtrTy(),
      B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context),
      DL.getIntPtrType(Context), nullptr);
  Dst = castToCStr(Dst, B);
  Src = castToCStr(Src, B);
  CallInst *CI = B.CreateCall(MemCpy, {Dst, Src, Len, ObjSize});
  if (const Function *F = dyn_cast<Function>(MemCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitMemChr(Value *Ptr, Value *Val, Value *Len, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memchr))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS;
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemChr = M->getOrInsertFunction(
      "memchr", AttributeSet::get(M->getContext(), AS), B.getInt8PtrTy(),
      B.getInt8PtrTy(), B.getInt32Ty(), DL.getIntPtrType(Context), nullptr);
  CallInst *CI = B.CreateCall(MemChr, {castToCStr(Ptr, B), Val, Len}, "memchr");

  if (const Function *F = dyn_cast<Function>(MemChr->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitMemCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memcmp))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[3];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS[2] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemCmp = M->getOrInsertFunction(
      "memcmp", AttributeSet::get(M->getContext(), AS), B.getInt32Ty(),
      B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context), nullptr);
  CallInst *CI = B.CreateCall(
      MemCmp, {castToCStr(Ptr1, B), castToCStr(Ptr2, B), Len}, "memcmp");

  if (const Function *F = dyn_cast<Function>(MemCmp->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// Append a suffix to the function name according to the type of 'Op'.
static void appendTypeSuffix(Value *Op, StringRef &Name,
                             SmallString<20> &NameBuffer) {
  if (!Op->getType()->isDoubleTy()) {
      NameBuffer += Name;

    if (Op->getType()->isFloatTy())
      NameBuffer += 'f';
    else
      NameBuffer += 'l';

    Name = NameBuffer;
  }  
  return;
}

Value *llvm::emitUnaryFloatFnCall(Value *Op, StringRef Name, IRBuilder<> &B,
                                  const AttributeSet &Attrs) {
  SmallString<20> NameBuffer;
  appendTypeSuffix(Op, Name, NameBuffer);

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Value *Callee = M->getOrInsertFunction(Name, Op->getType(),
                                         Op->getType(), nullptr);
  CallInst *CI = B.CreateCall(Callee, Op, Name);
  CI->setAttributes(Attrs);
  if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitBinaryFloatFnCall(Value *Op1, Value *Op2, StringRef Name,
                                  IRBuilder<> &B, const AttributeSet &Attrs) {
  SmallString<20> NameBuffer;
  appendTypeSuffix(Op1, Name, NameBuffer);

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Value *Callee = M->getOrInsertFunction(Name, Op1->getType(), Op1->getType(),
                                         Op2->getType(), nullptr);
  CallInst *CI = B.CreateCall(Callee, {Op1, Op2}, Name);
  CI->setAttributes(Attrs);
  if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

Value *llvm::emitPutChar(Value *Char, IRBuilder<> &B,
                         const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::putchar))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Value *PutChar = M->getOrInsertFunction("putchar", B.getInt32Ty(),
                                          B.getInt32Ty(), nullptr);
  CallInst *CI = B.CreateCall(PutChar,
                              B.CreateIntCast(Char,
                              B.getInt32Ty(),
                              /*isSigned*/true,
                              "chari"),
                              "putchar");

  if (const Function *F = dyn_cast<Function>(PutChar->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitPutS(Value *Str, IRBuilder<> &B,
                      const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::puts))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);

  Value *PutS = M->getOrInsertFunction("puts",
                                       AttributeSet::get(M->getContext(), AS),
                                       B.getInt32Ty(),
                                       B.getInt8PtrTy(),
                                       nullptr);
  CallInst *CI = B.CreateCall(PutS, castToCStr(Str, B), "puts");
  if (const Function *F = dyn_cast<Function>(PutS->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

Value *llvm::emitFPutC(Value *Char, Value *File, IRBuilder<> &B,
                       const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fputc))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction("fputc",
                               AttributeSet::get(M->getContext(), AS),
                               B.getInt32Ty(),
                               B.getInt32Ty(), File->getType(),
                               nullptr);
  else
    F = M->getOrInsertFunction("fputc",
                               B.getInt32Ty(),
                               B.getInt32Ty(),
                               File->getType(), nullptr);
  Char = B.CreateIntCast(Char, B.getInt32Ty(), /*isSigned*/true,
                         "chari");
  CallInst *CI = B.CreateCall(F, {Char, File}, "fputc");

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

Value *llvm::emitFPutS(Value *Str, Value *File, IRBuilder<> &B,
                       const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fputs))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[3];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  AS[2] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  StringRef FPutsName = TLI->getName(LibFunc::fputs);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction(FPutsName,
                               AttributeSet::get(M->getContext(), AS),
                               B.getInt32Ty(),
                               B.getInt8PtrTy(),
                               File->getType(), nullptr);
  else
    F = M->getOrInsertFunction(FPutsName, B.getInt32Ty(),
                               B.getInt8PtrTy(),
                               File->getType(), nullptr);
  CallInst *CI = B.CreateCall(F, {castToCStr(Str, B), File}, "fputs");

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

Value *llvm::emitFWrite(Value *Ptr, Value *Size, Value *File, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fwrite))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[3];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), 4, Attribute::NoCapture);
  AS[2] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  StringRef FWriteName = TLI->getName(LibFunc::fwrite);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction(
        FWriteName, AttributeSet::get(M->getContext(), AS),
        DL.getIntPtrType(Context), B.getInt8PtrTy(), DL.getIntPtrType(Context),
        DL.getIntPtrType(Context), File->getType(), nullptr);
  else
    F = M->getOrInsertFunction(FWriteName, DL.getIntPtrType(Context),
                               B.getInt8PtrTy(), DL.getIntPtrType(Context),
                               DL.getIntPtrType(Context), File->getType(),
                               nullptr);
  CallInst *CI =
      B.CreateCall(F, {castToCStr(Ptr, B), Size,
                       ConstantInt::get(DL.getIntPtrType(Context), 1), File});

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}
