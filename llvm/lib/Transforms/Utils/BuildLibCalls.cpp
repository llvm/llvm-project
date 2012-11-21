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
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/Intrinsics.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"

using namespace llvm;

/// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
Value *llvm::CastToCStr(Value *V, IRBuilder<> &B) {
  return B.CreateBitCast(V, B.getInt8PtrTy(), "cstr");
}

/// EmitStrLen - Emit a call to the strlen function to the builder, for the
/// specified pointer.  This always returns an integer value of size intptr_t.
Value *llvm::EmitStrLen(Value *Ptr, IRBuilder<> &B, const DataLayout *TD,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strlen))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 1, Attributes::NoCapture);
  Attributes::AttrVal AVs[2] = { Attributes::ReadOnly, Attributes::NoUnwind };
  AWI[1] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   ArrayRef<Attributes::AttrVal>(AVs, 2));

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Constant *StrLen = M->getOrInsertFunction("strlen", AttrListPtr::get(AWI),
                                            TD->getIntPtrType(Context),
                                            B.getInt8PtrTy(),
                                            NULL);
  CallInst *CI = B.CreateCall(StrLen, CastToCStr(Ptr, B), "strlen");
  if (const Function *F = dyn_cast<Function>(StrLen->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitStrNLen - Emit a call to the strnlen function to the builder, for the
/// specified pointer.  Ptr is required to be some pointer type, MaxLen must
/// be of size_t type, and the return value has 'intptr_t' type.
Value *llvm::EmitStrNLen(Value *Ptr, Value *MaxLen, IRBuilder<> &B,
                         const DataLayout *TD, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strnlen))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 1, Attributes::NoCapture);
  Attributes::AttrVal AVs[2] = { Attributes::ReadOnly, Attributes::NoUnwind };
  AWI[1] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   ArrayRef<Attributes::AttrVal>(AVs, 2));

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Constant *StrNLen = M->getOrInsertFunction("strnlen", AttrListPtr::get(AWI),
                                             TD->getIntPtrType(Context),
                                             B.getInt8PtrTy(),
                                             TD->getIntPtrType(Context),
                                             NULL);
  CallInst *CI = B.CreateCall2(StrNLen, CastToCStr(Ptr, B), MaxLen, "strnlen");
  if (const Function *F = dyn_cast<Function>(StrNLen->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitStrChr - Emit a call to the strchr function to the builder, for the
/// specified pointer and character.  Ptr is required to be some pointer type,
/// and the return value has 'i8*' type.
Value *llvm::EmitStrChr(Value *Ptr, char C, IRBuilder<> &B,
                        const DataLayout *TD, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strchr))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Attributes::AttrVal AVs[2] = { Attributes::ReadOnly, Attributes::NoUnwind };
  AttributeWithIndex AWI =
    AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                            ArrayRef<Attributes::AttrVal>(AVs, 2));

  Type *I8Ptr = B.getInt8PtrTy();
  Type *I32Ty = B.getInt32Ty();
  Constant *StrChr = M->getOrInsertFunction("strchr", AttrListPtr::get(AWI),
                                            I8Ptr, I8Ptr, I32Ty, NULL);
  CallInst *CI = B.CreateCall2(StrChr, CastToCStr(Ptr, B),
                               ConstantInt::get(I32Ty, C), "strchr");
  if (const Function *F = dyn_cast<Function>(StrChr->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitStrNCmp - Emit a call to the strncmp function to the builder.
Value *llvm::EmitStrNCmp(Value *Ptr1, Value *Ptr2, Value *Len,
                         IRBuilder<> &B, const DataLayout *TD,
                         const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strncmp))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[3];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 1, Attributes::NoCapture);
  AWI[1] = AttributeWithIndex::get(M->getContext(), 2, Attributes::NoCapture);
  Attributes::AttrVal AVs[2] = { Attributes::ReadOnly, Attributes::NoUnwind };
  AWI[2] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   ArrayRef<Attributes::AttrVal>(AVs, 2));

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *StrNCmp = M->getOrInsertFunction("strncmp", AttrListPtr::get(AWI),
                                          B.getInt32Ty(),
                                          B.getInt8PtrTy(),
                                          B.getInt8PtrTy(),
                                          TD->getIntPtrType(Context), NULL);
  CallInst *CI = B.CreateCall3(StrNCmp, CastToCStr(Ptr1, B),
                               CastToCStr(Ptr2, B), Len, "strncmp");

  if (const Function *F = dyn_cast<Function>(StrNCmp->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitStrCpy - Emit a call to the strcpy function to the builder, for the
/// specified pointer arguments.
Value *llvm::EmitStrCpy(Value *Dst, Value *Src, IRBuilder<> &B,
                        const DataLayout *TD, const TargetLibraryInfo *TLI,
                        StringRef Name) {
  if (!TLI->has(LibFunc::strcpy))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 2, Attributes::NoCapture);
  AWI[1] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   Attributes::NoUnwind);
  Type *I8Ptr = B.getInt8PtrTy();
  Value *StrCpy = M->getOrInsertFunction(Name, AttrListPtr::get(AWI),
                                         I8Ptr, I8Ptr, I8Ptr, NULL);
  CallInst *CI = B.CreateCall2(StrCpy, CastToCStr(Dst, B), CastToCStr(Src, B),
                               Name);
  if (const Function *F = dyn_cast<Function>(StrCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitStrNCpy - Emit a call to the strncpy function to the builder, for the
/// specified pointer arguments.
Value *llvm::EmitStrNCpy(Value *Dst, Value *Src, Value *Len,
                         IRBuilder<> &B, const DataLayout *TD,
                         const TargetLibraryInfo *TLI, StringRef Name) {
  if (!TLI->has(LibFunc::strncpy))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 2, Attributes::NoCapture);
  AWI[1] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   Attributes::NoUnwind);
  Type *I8Ptr = B.getInt8PtrTy();
  Value *StrNCpy = M->getOrInsertFunction(Name, AttrListPtr::get(AWI),
                                          I8Ptr, I8Ptr, I8Ptr,
                                          Len->getType(), NULL);
  CallInst *CI = B.CreateCall3(StrNCpy, CastToCStr(Dst, B), CastToCStr(Src, B),
                               Len, "strncpy");
  if (const Function *F = dyn_cast<Function>(StrNCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitMemCpyChk - Emit a call to the __memcpy_chk function to the builder.
/// This expects that the Len and ObjSize have type 'intptr_t' and Dst/Src
/// are pointers.
Value *llvm::EmitMemCpyChk(Value *Dst, Value *Src, Value *Len, Value *ObjSize,
                           IRBuilder<> &B, const DataLayout *TD,
                           const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memcpy_chk))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI;
  AWI = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                Attributes::NoUnwind);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemCpy = M->getOrInsertFunction("__memcpy_chk",
                                         AttrListPtr::get(AWI),
                                         B.getInt8PtrTy(),
                                         B.getInt8PtrTy(),
                                         B.getInt8PtrTy(),
                                         TD->getIntPtrType(Context),
                                         TD->getIntPtrType(Context), NULL);
  Dst = CastToCStr(Dst, B);
  Src = CastToCStr(Src, B);
  CallInst *CI = B.CreateCall4(MemCpy, Dst, Src, Len, ObjSize);
  if (const Function *F = dyn_cast<Function>(MemCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitMemChr - Emit a call to the memchr function.  This assumes that Ptr is
/// a pointer, Val is an i32 value, and Len is an 'intptr_t' value.
Value *llvm::EmitMemChr(Value *Ptr, Value *Val,
                        Value *Len, IRBuilder<> &B, const DataLayout *TD,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memchr))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI;
  Attributes::AttrVal AVs[2] = { Attributes::ReadOnly, Attributes::NoUnwind };
  AWI = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                ArrayRef<Attributes::AttrVal>(AVs, 2));
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemChr = M->getOrInsertFunction("memchr", AttrListPtr::get(AWI),
                                         B.getInt8PtrTy(),
                                         B.getInt8PtrTy(),
                                         B.getInt32Ty(),
                                         TD->getIntPtrType(Context),
                                         NULL);
  CallInst *CI = B.CreateCall3(MemChr, CastToCStr(Ptr, B), Val, Len, "memchr");

  if (const Function *F = dyn_cast<Function>(MemChr->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitMemCmp - Emit a call to the memcmp function.
Value *llvm::EmitMemCmp(Value *Ptr1, Value *Ptr2,
                        Value *Len, IRBuilder<> &B, const DataLayout *TD,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memcmp))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[3];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 1, Attributes::NoCapture);
  AWI[1] = AttributeWithIndex::get(M->getContext(), 2, Attributes::NoCapture);
  Attributes::AttrVal AVs[2] = { Attributes::ReadOnly, Attributes::NoUnwind };
  AWI[2] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   ArrayRef<Attributes::AttrVal>(AVs, 2));

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemCmp = M->getOrInsertFunction("memcmp", AttrListPtr::get(AWI),
                                         B.getInt32Ty(),
                                         B.getInt8PtrTy(),
                                         B.getInt8PtrTy(),
                                         TD->getIntPtrType(Context), NULL);
  CallInst *CI = B.CreateCall3(MemCmp, CastToCStr(Ptr1, B), CastToCStr(Ptr2, B),
                               Len, "memcmp");

  if (const Function *F = dyn_cast<Function>(MemCmp->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitUnaryFloatFnCall - Emit a call to the unary function named 'Name' (e.g.
/// 'floor').  This function is known to take a single of type matching 'Op' and
/// returns one value with the same type.  If 'Op' is a long double, 'l' is
/// added as the suffix of name, if 'Op' is a float, we add a 'f' suffix.
Value *llvm::EmitUnaryFloatFnCall(Value *Op, StringRef Name, IRBuilder<> &B,
                                  const AttrListPtr &Attrs) {
  SmallString<20> NameBuffer;
  if (!Op->getType()->isDoubleTy()) {
    // If we need to add a suffix, copy into NameBuffer.
    NameBuffer += Name;
    if (Op->getType()->isFloatTy())
      NameBuffer += 'f'; // floorf
    else
      NameBuffer += 'l'; // floorl
    Name = NameBuffer;
  }

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Value *Callee = M->getOrInsertFunction(Name, Op->getType(),
                                         Op->getType(), NULL);
  CallInst *CI = B.CreateCall(Callee, Op, Name);
  CI->setAttributes(Attrs);
  if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitPutChar - Emit a call to the putchar function.  This assumes that Char
/// is an integer.
Value *llvm::EmitPutChar(Value *Char, IRBuilder<> &B, const DataLayout *TD,
                         const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::putchar))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Value *PutChar = M->getOrInsertFunction("putchar", B.getInt32Ty(),
                                          B.getInt32Ty(), NULL);
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

/// EmitPutS - Emit a call to the puts function.  This assumes that Str is
/// some pointer.
Value *llvm::EmitPutS(Value *Str, IRBuilder<> &B, const DataLayout *TD,
                      const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::puts))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 1, Attributes::NoCapture);
  AWI[1] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   Attributes::NoUnwind);

  Value *PutS = M->getOrInsertFunction("puts", AttrListPtr::get(AWI),
                                       B.getInt32Ty(),
                                       B.getInt8PtrTy(),
                                       NULL);
  CallInst *CI = B.CreateCall(PutS, CastToCStr(Str, B), "puts");
  if (const Function *F = dyn_cast<Function>(PutS->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitFPutC - Emit a call to the fputc function.  This assumes that Char is
/// an integer and File is a pointer to FILE.
Value *llvm::EmitFPutC(Value *Char, Value *File, IRBuilder<> &B,
                       const DataLayout *TD, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fputc))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 2, Attributes::NoCapture);
  AWI[1] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   Attributes::NoUnwind);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction("fputc", AttrListPtr::get(AWI),
                               B.getInt32Ty(),
                               B.getInt32Ty(), File->getType(),
                               NULL);
  else
    F = M->getOrInsertFunction("fputc",
                               B.getInt32Ty(),
                               B.getInt32Ty(),
                               File->getType(), NULL);
  Char = B.CreateIntCast(Char, B.getInt32Ty(), /*isSigned*/true,
                         "chari");
  CallInst *CI = B.CreateCall2(F, Char, File, "fputc");

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

/// EmitFPutS - Emit a call to the puts function.  Str is required to be a
/// pointer and File is a pointer to FILE.
Value *llvm::EmitFPutS(Value *Str, Value *File, IRBuilder<> &B,
                       const DataLayout *TD, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fputs))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[3];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 1, Attributes::NoCapture);
  AWI[1] = AttributeWithIndex::get(M->getContext(), 2, Attributes::NoCapture);
  AWI[2] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   Attributes::NoUnwind);
  StringRef FPutsName = TLI->getName(LibFunc::fputs);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction(FPutsName, AttrListPtr::get(AWI),
                               B.getInt32Ty(),
                               B.getInt8PtrTy(),
                               File->getType(), NULL);
  else
    F = M->getOrInsertFunction(FPutsName, B.getInt32Ty(),
                               B.getInt8PtrTy(),
                               File->getType(), NULL);
  CallInst *CI = B.CreateCall2(F, CastToCStr(Str, B), File, "fputs");

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

/// EmitFWrite - Emit a call to the fwrite function.  This assumes that Ptr is
/// a pointer, Size is an 'intptr_t', and File is a pointer to FILE.
Value *llvm::EmitFWrite(Value *Ptr, Value *Size, Value *File,
                        IRBuilder<> &B, const DataLayout *TD,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fwrite))
    return 0;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeWithIndex AWI[3];
  AWI[0] = AttributeWithIndex::get(M->getContext(), 1, Attributes::NoCapture);
  AWI[1] = AttributeWithIndex::get(M->getContext(), 4, Attributes::NoCapture);
  AWI[2] = AttributeWithIndex::get(M->getContext(), AttrListPtr::FunctionIndex,
                                   Attributes::NoUnwind);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  StringRef FWriteName = TLI->getName(LibFunc::fwrite);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction(FWriteName, AttrListPtr::get(AWI),
                               TD->getIntPtrType(Context),
                               B.getInt8PtrTy(),
                               TD->getIntPtrType(Context),
                               TD->getIntPtrType(Context),
                               File->getType(), NULL);
  else
    F = M->getOrInsertFunction(FWriteName, TD->getIntPtrType(Context),
                               B.getInt8PtrTy(),
                               TD->getIntPtrType(Context),
                               TD->getIntPtrType(Context),
                               File->getType(), NULL);
  CallInst *CI = B.CreateCall4(F, CastToCStr(Ptr, B), Size,
                        ConstantInt::get(TD->getIntPtrType(Context), 1), File);

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

SimplifyFortifiedLibCalls::~SimplifyFortifiedLibCalls() { }

bool SimplifyFortifiedLibCalls::fold(CallInst *CI, const DataLayout *TD,
                                     const TargetLibraryInfo *TLI) {
  // We really need DataLayout for later.
  if (!TD) return false;
  
  this->CI = CI;
  Function *Callee = CI->getCalledFunction();
  StringRef Name = Callee->getName();
  FunctionType *FT = Callee->getFunctionType();
  LLVMContext &Context = CI->getParent()->getContext();
  IRBuilder<> B(CI);

  if (Name == "__memcpy_chk") {
    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(Context) ||
        FT->getParamType(3) != TD->getIntPtrType(Context))
      return false;

    if (isFoldable(3, 2, false)) {
      B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                     CI->getArgOperand(2), 1);
      replaceCall(CI->getArgOperand(0));
      return true;
    }
    return false;
  }

  // Should be similar to memcpy.
  if (Name == "__mempcpy_chk") {
    return false;
  }

  if (Name == "__memmove_chk") {
    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(Context) ||
        FT->getParamType(3) != TD->getIntPtrType(Context))
      return false;

    if (isFoldable(3, 2, false)) {
      B.CreateMemMove(CI->getArgOperand(0), CI->getArgOperand(1),
                      CI->getArgOperand(2), 1);
      replaceCall(CI->getArgOperand(0));
      return true;
    }
    return false;
  }

  if (Name == "__memset_chk") {
    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(Context) ||
        FT->getParamType(3) != TD->getIntPtrType(Context))
      return false;

    if (isFoldable(3, 2, false)) {
      Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(),
                                   false);
      B.CreateMemSet(CI->getArgOperand(0), Val, CI->getArgOperand(2), 1);
      replaceCall(CI->getArgOperand(0));
      return true;
    }
    return false;
  }

  if (Name == "__strcpy_chk" || Name == "__stpcpy_chk") {
    // Check if this has the right signature.
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        FT->getParamType(2) != TD->getIntPtrType(Context))
      return 0;
    
    
    // If a) we don't have any length information, or b) we know this will
    // fit then just lower to a plain st[rp]cpy. Otherwise we'll keep our
    // st[rp]cpy_chk call which may fail at runtime if the size is too long.
    // TODO: It might be nice to get a maximum length out of the possible
    // string lengths for varying.
    if (isFoldable(2, 1, true)) {
      Value *Ret = EmitStrCpy(CI->getArgOperand(0), CI->getArgOperand(1), B, TD,
                              TLI, Name.substr(2, 6));
      if (!Ret)
        return false;
      replaceCall(Ret);
      return true;
    }
    return false;
  }

  if (Name == "__strncpy_chk" || Name == "__stpncpy_chk") {
    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        !FT->getParamType(2)->isIntegerTy() ||
        FT->getParamType(3) != TD->getIntPtrType(Context))
      return false;

    if (isFoldable(3, 2, false)) {
      Value *Ret = EmitStrNCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                               CI->getArgOperand(2), B, TD, TLI,
                               Name.substr(2, 7));
      if (!Ret)
        return false;
      replaceCall(Ret);
      return true;
    }
    return false;
  }

  if (Name == "__strcat_chk") {
    return false;
  }

  if (Name == "__strncat_chk") {
    return false;
  }

  return false;
}
