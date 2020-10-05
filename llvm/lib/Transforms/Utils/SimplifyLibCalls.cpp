//===------ SimplifyLibCalls.cpp - Library calls simplifier ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the library calls simplifier. It does not implement
// any pass, but can't be used by other passes to do simplifications.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/SizeOpts.h"

using namespace llvm;
using namespace PatternMatch;

static cl::opt<bool>
    EnableUnsafeFPShrink("enable-double-float-shrink", cl::Hidden,
                         cl::init(false),
                         cl::desc("Enable unsafe double to float "
                                  "shrinking for math lib calls"));

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static bool ignoreCallingConv(LibFunc Func) {
  return Func == LibFunc_abs || Func == LibFunc_labs ||
         Func == LibFunc_llabs || Func == LibFunc_strlen;
}

static bool isCallingConvCCompatible(CallInst *CI) {
  switch(CI->getCallingConv()) {
  default:
    return false;
  case llvm::CallingConv::C:
    return true;
  case llvm::CallingConv::ARM_APCS:
  case llvm::CallingConv::ARM_AAPCS:
  case llvm::CallingConv::ARM_AAPCS_VFP: {

    // The iOS ABI diverges from the standard in some cases, so for now don't
    // try to simplify those calls.
    if (Triple(CI->getModule()->getTargetTriple()).isiOS())
      return false;

    auto *FuncTy = CI->getFunctionType();

    if (!FuncTy->getReturnType()->isPointerTy() &&
        !FuncTy->getReturnType()->isIntegerTy() &&
        !FuncTy->getReturnType()->isVoidTy())
      return false;

    for (auto Param : FuncTy->params()) {
      if (!Param->isPointerTy() && !Param->isIntegerTy())
        return false;
    }
    return true;
  }
  }
  return false;
}

/// Return true if it is only used in equality comparisons with With.
static bool isOnlyUsedInEqualityComparison(Value *V, Value *With) {
  for (User *U : V->users()) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(U))
      if (IC->isEquality() && IC->getOperand(1) == With)
        continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

static bool callHasFloatingPointArgument(const CallInst *CI) {
  return any_of(CI->operands(), [](const Use &OI) {
    return OI->getType()->isFloatingPointTy();
  });
}

static bool callHasFP128Argument(const CallInst *CI) {
  return any_of(CI->operands(), [](const Use &OI) {
    return OI->getType()->isFP128Ty();
  });
}

static Value *convertStrToNumber(CallInst *CI, StringRef &Str, int64_t Base) {
  if (Base < 2 || Base > 36)
    // handle special zero base
    if (Base != 0)
      return nullptr;

  char *End;
  std::string nptr = Str.str();
  errno = 0;
  long long int Result = strtoll(nptr.c_str(), &End, Base);
  if (errno)
    return nullptr;

  // if we assume all possible target locales are ASCII supersets,
  // then if strtoll successfully parses a number on the host,
  // it will also successfully parse the same way on the target
  if (*End != '\0')
    return nullptr;

  if (!isIntN(CI->getType()->getPrimitiveSizeInBits(), Result))
    return nullptr;

  return ConstantInt::get(CI->getType(), Result);
}

static bool isOnlyUsedInComparisonWithZero(Value *V) {
  for (User *U : V->users()) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(U))
      if (Constant *C = dyn_cast<Constant>(IC->getOperand(1)))
        if (C->isNullValue())
          continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

static bool canTransformToMemCmp(CallInst *CI, Value *Str, uint64_t Len,
                                 const DataLayout &DL) {
  if (!isOnlyUsedInComparisonWithZero(CI))
    return false;

  if (!isDereferenceableAndAlignedPointer(Str, Align(1), APInt(64, Len), DL))
    return false;

  if (CI->getFunction()->hasFnAttribute(Attribute::SanitizeMemory))
    return false;

  return true;
}

static void annotateDereferenceableBytes(CallInst *CI,
                                         ArrayRef<unsigned> ArgNos,
                                         uint64_t DereferenceableBytes) {
  const Function *F = CI->getCaller();
  if (!F)
    return;
  for (unsigned ArgNo : ArgNos) {
    uint64_t DerefBytes = DereferenceableBytes;
    unsigned AS = CI->getArgOperand(ArgNo)->getType()->getPointerAddressSpace();
    if (!llvm::NullPointerIsDefined(F, AS) ||
        CI->paramHasAttr(ArgNo, Attribute::NonNull))
      DerefBytes = std::max(CI->getDereferenceableOrNullBytes(
                                ArgNo + AttributeList::FirstArgIndex),
                            DereferenceableBytes);
  
    if (CI->getDereferenceableBytes(ArgNo + AttributeList::FirstArgIndex) <
        DerefBytes) {
      CI->removeParamAttr(ArgNo, Attribute::Dereferenceable);
      if (!llvm::NullPointerIsDefined(F, AS) ||
          CI->paramHasAttr(ArgNo, Attribute::NonNull))
        CI->removeParamAttr(ArgNo, Attribute::DereferenceableOrNull);
      CI->addParamAttr(ArgNo, Attribute::getWithDereferenceableBytes(
                                  CI->getContext(), DerefBytes));
    }
  }
}

static void annotateNonNullBasedOnAccess(CallInst *CI,
                                         ArrayRef<unsigned> ArgNos) {
  Function *F = CI->getCaller();
  if (!F)
    return;

  for (unsigned ArgNo : ArgNos) {
    if (CI->paramHasAttr(ArgNo, Attribute::NonNull))
      continue;
    unsigned AS = CI->getArgOperand(ArgNo)->getType()->getPointerAddressSpace();
    if (llvm::NullPointerIsDefined(F, AS))
      continue;

    CI->addParamAttr(ArgNo, Attribute::NonNull);
    annotateDereferenceableBytes(CI, ArgNo, 1);
  }
}

static void annotateNonNullAndDereferenceable(CallInst *CI, ArrayRef<unsigned> ArgNos,
                               Value *Size, const DataLayout &DL) {
  if (ConstantInt *LenC = dyn_cast<ConstantInt>(Size)) {
    annotateNonNullBasedOnAccess(CI, ArgNos);
    annotateDereferenceableBytes(CI, ArgNos, LenC->getZExtValue());
  } else if (isKnownNonZero(Size, DL)) {
    annotateNonNullBasedOnAccess(CI, ArgNos);
    const APInt *X, *Y;
    uint64_t DerefMin = 1;
    if (match(Size, m_Select(m_Value(), m_APInt(X), m_APInt(Y)))) {
      DerefMin = std::min(X->getZExtValue(), Y->getZExtValue());
      annotateDereferenceableBytes(CI, ArgNos, DerefMin);
    }
  }
}

//===----------------------------------------------------------------------===//
// String and Memory Library Call Optimizations
//===----------------------------------------------------------------------===//

Value *LibCallSimplifier::optimizeStrCat(CallInst *CI, IRBuilderBase &B) {
  // Extract some information from the instruction
  Value *Dst = CI->getArgOperand(0);
  Value *Src = CI->getArgOperand(1);
  annotateNonNullBasedOnAccess(CI, {0, 1});

  // See if we can get the length of the input string.
  uint64_t Len = GetStringLength(Src);
  if (Len)
    annotateDereferenceableBytes(CI, 1, Len);
  else
    return nullptr;
  --Len; // Unbias length.

  // Handle the simple, do-nothing case: strcat(x, "") -> x
  if (Len == 0)
    return Dst;

  return emitStrLenMemCpy(Src, Dst, Len, B);
}

Value *LibCallSimplifier::emitStrLenMemCpy(Value *Src, Value *Dst, uint64_t Len,
                                           IRBuilderBase &B) {
  // We need to find the end of the destination string.  That's where the
  // memory is to be moved to. We just generate a call to strlen.
  Value *DstLen = emitStrLen(Dst, B, DL, TLI);
  if (!DstLen)
    return nullptr;

  // Now that we have the destination's length, we must index into the
  // destination's pointer to get the actual memcpy destination (end of
  // the string .. we're concatenating).
  Value *CpyDst = B.CreateGEP(B.getInt8Ty(), Dst, DstLen, "endptr");

  // We have enough information to now generate the memcpy call to do the
  // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
  B.CreateMemCpy(
      CpyDst, Align(1), Src, Align(1),
      ConstantInt::get(DL.getIntPtrType(Src->getContext()), Len + 1));
  return Dst;
}

Value *LibCallSimplifier::optimizeStrNCat(CallInst *CI, IRBuilderBase &B) {
  // Extract some information from the instruction.
  Value *Dst = CI->getArgOperand(0);
  Value *Src = CI->getArgOperand(1);
  Value *Size = CI->getArgOperand(2);
  uint64_t Len;
  annotateNonNullBasedOnAccess(CI, 0);
  if (isKnownNonZero(Size, DL))
    annotateNonNullBasedOnAccess(CI, 1);

  // We don't do anything if length is not constant.
  ConstantInt *LengthArg = dyn_cast<ConstantInt>(Size);
  if (LengthArg) {
    Len = LengthArg->getZExtValue();
    // strncat(x, c, 0) -> x
    if (!Len)
      return Dst;
  } else {
    return nullptr;
  }

  // See if we can get the length of the input string.
  uint64_t SrcLen = GetStringLength(Src);
  if (SrcLen) {
    annotateDereferenceableBytes(CI, 1, SrcLen);
    --SrcLen; // Unbias length.
  } else {
    return nullptr;
  }

  // strncat(x, "", c) -> x
  if (SrcLen == 0)
    return Dst;

  // We don't optimize this case.
  if (Len < SrcLen)
    return nullptr;

  // strncat(x, s, c) -> strcat(x, s)
  // s is constant so the strcat can be optimized further.
  return emitStrLenMemCpy(Src, Dst, SrcLen, B);
}

Value *LibCallSimplifier::optimizeStrChr(CallInst *CI, IRBuilderBase &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  Value *SrcStr = CI->getArgOperand(0);
  annotateNonNullBasedOnAccess(CI, 0);

  // If the second operand is non-constant, see if we can compute the length
  // of the input string and turn this into memchr.
  ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!CharC) {
    uint64_t Len = GetStringLength(SrcStr);
    if (Len)
      annotateDereferenceableBytes(CI, 0, Len);
    else
      return nullptr;
    if (!FT->getParamType(1)->isIntegerTy(32)) // memchr needs i32.
      return nullptr;

    return emitMemChr(SrcStr, CI->getArgOperand(1), // include nul.
                      ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len),
                      B, DL, TLI);
  }

  // Otherwise, the character is a constant, see if the first argument is
  // a string literal.  If so, we can constant fold.
  StringRef Str;
  if (!getConstantStringInfo(SrcStr, Str)) {
    if (CharC->isZero()) // strchr(p, 0) -> p + strlen(p)
      if (Value *StrLen = emitStrLen(SrcStr, B, DL, TLI))
        return B.CreateGEP(B.getInt8Ty(), SrcStr, StrLen, "strchr");
    return nullptr;
  }

  // Compute the offset, make sure to handle the case when we're searching for
  // zero (a weird way to spell strlen).
  size_t I = (0xFF & CharC->getSExtValue()) == 0
                 ? Str.size()
                 : Str.find(CharC->getSExtValue());
  if (I == StringRef::npos) // Didn't find the char.  strchr returns null.
    return Constant::getNullValue(CI->getType());

  // strchr(s+n,c)  -> gep(s+n+i,c)
  return B.CreateGEP(B.getInt8Ty(), SrcStr, B.getInt64(I), "strchr");
}

Value *LibCallSimplifier::optimizeStrRChr(CallInst *CI, IRBuilderBase &B) {
  Value *SrcStr = CI->getArgOperand(0);
  ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  annotateNonNullBasedOnAccess(CI, 0);

  // Cannot fold anything if we're not looking for a constant.
  if (!CharC)
    return nullptr;

  StringRef Str;
  if (!getConstantStringInfo(SrcStr, Str)) {
    // strrchr(s, 0) -> strchr(s, 0)
    if (CharC->isZero())
      return emitStrChr(SrcStr, '\0', B, TLI);
    return nullptr;
  }

  // Compute the offset.
  size_t I = (0xFF & CharC->getSExtValue()) == 0
                 ? Str.size()
                 : Str.rfind(CharC->getSExtValue());
  if (I == StringRef::npos) // Didn't find the char. Return null.
    return Constant::getNullValue(CI->getType());

  // strrchr(s+n,c) -> gep(s+n+i,c)
  return B.CreateGEP(B.getInt8Ty(), SrcStr, B.getInt64(I), "strrchr");
}

Value *LibCallSimplifier::optimizeStrCmp(CallInst *CI, IRBuilderBase &B) {
  Value *Str1P = CI->getArgOperand(0), *Str2P = CI->getArgOperand(1);
  if (Str1P == Str2P) // strcmp(x,x)  -> 0
    return ConstantInt::get(CI->getType(), 0);

  StringRef Str1, Str2;
  bool HasStr1 = getConstantStringInfo(Str1P, Str1);
  bool HasStr2 = getConstantStringInfo(Str2P, Str2);

  // strcmp(x, y)  -> cnst  (if both x and y are constant strings)
  if (HasStr1 && HasStr2)
    return ConstantInt::get(CI->getType(), Str1.compare(Str2));

  if (HasStr1 && Str1.empty()) // strcmp("", x) -> -*x
    return B.CreateNeg(B.CreateZExt(
        B.CreateLoad(B.getInt8Ty(), Str2P, "strcmpload"), CI->getType()));

  if (HasStr2 && Str2.empty()) // strcmp(x,"") -> *x
    return B.CreateZExt(B.CreateLoad(B.getInt8Ty(), Str1P, "strcmpload"),
                        CI->getType());

  // strcmp(P, "x") -> memcmp(P, "x", 2)
  uint64_t Len1 = GetStringLength(Str1P);
  if (Len1)
    annotateDereferenceableBytes(CI, 0, Len1);
  uint64_t Len2 = GetStringLength(Str2P);
  if (Len2)
    annotateDereferenceableBytes(CI, 1, Len2);

  if (Len1 && Len2) {
    return emitMemCmp(Str1P, Str2P,
                      ConstantInt::get(DL.getIntPtrType(CI->getContext()),
                                       std::min(Len1, Len2)),
                      B, DL, TLI);
  }

  // strcmp to memcmp
  if (!HasStr1 && HasStr2) {
    if (canTransformToMemCmp(CI, Str1P, Len2, DL))
      return emitMemCmp(
          Str1P, Str2P,
          ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len2), B, DL,
          TLI);
  } else if (HasStr1 && !HasStr2) {
    if (canTransformToMemCmp(CI, Str2P, Len1, DL))
      return emitMemCmp(
          Str1P, Str2P,
          ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len1), B, DL,
          TLI);
  }

  annotateNonNullBasedOnAccess(CI, {0, 1});
  return nullptr;
}

Value *LibCallSimplifier::optimizeStrNCmp(CallInst *CI, IRBuilderBase &B) {
  Value *Str1P = CI->getArgOperand(0);
  Value *Str2P = CI->getArgOperand(1);
  Value *Size = CI->getArgOperand(2);
  if (Str1P == Str2P) // strncmp(x,x,n)  -> 0
    return ConstantInt::get(CI->getType(), 0);

  if (isKnownNonZero(Size, DL))
    annotateNonNullBasedOnAccess(CI, {0, 1});
  // Get the length argument if it is constant.
  uint64_t Length;
  if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(Size))
    Length = LengthArg->getZExtValue();
  else
    return nullptr;

  if (Length == 0) // strncmp(x,y,0)   -> 0
    return ConstantInt::get(CI->getType(), 0);

  if (Length == 1) // strncmp(x,y,1) -> memcmp(x,y,1)
    return emitMemCmp(Str1P, Str2P, Size, B, DL, TLI);

  StringRef Str1, Str2;
  bool HasStr1 = getConstantStringInfo(Str1P, Str1);
  bool HasStr2 = getConstantStringInfo(Str2P, Str2);

  // strncmp(x, y)  -> cnst  (if both x and y are constant strings)
  if (HasStr1 && HasStr2) {
    StringRef SubStr1 = Str1.substr(0, Length);
    StringRef SubStr2 = Str2.substr(0, Length);
    return ConstantInt::get(CI->getType(), SubStr1.compare(SubStr2));
  }

  if (HasStr1 && Str1.empty()) // strncmp("", x, n) -> -*x
    return B.CreateNeg(B.CreateZExt(
        B.CreateLoad(B.getInt8Ty(), Str2P, "strcmpload"), CI->getType()));

  if (HasStr2 && Str2.empty()) // strncmp(x, "", n) -> *x
    return B.CreateZExt(B.CreateLoad(B.getInt8Ty(), Str1P, "strcmpload"),
                        CI->getType());

  uint64_t Len1 = GetStringLength(Str1P);
  if (Len1)
    annotateDereferenceableBytes(CI, 0, Len1);
  uint64_t Len2 = GetStringLength(Str2P);
  if (Len2)
    annotateDereferenceableBytes(CI, 1, Len2);

  // strncmp to memcmp
  if (!HasStr1 && HasStr2) {
    Len2 = std::min(Len2, Length);
    if (canTransformToMemCmp(CI, Str1P, Len2, DL))
      return emitMemCmp(
          Str1P, Str2P,
          ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len2), B, DL,
          TLI);
  } else if (HasStr1 && !HasStr2) {
    Len1 = std::min(Len1, Length);
    if (canTransformToMemCmp(CI, Str2P, Len1, DL))
      return emitMemCmp(
          Str1P, Str2P,
          ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len1), B, DL,
          TLI);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrNDup(CallInst *CI, IRBuilderBase &B) {
  Value *Src = CI->getArgOperand(0);
  ConstantInt *Size = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  uint64_t SrcLen = GetStringLength(Src);
  if (SrcLen && Size) {
    annotateDereferenceableBytes(CI, 0, SrcLen);
    if (SrcLen <= Size->getZExtValue() + 1)
      return emitStrDup(Src, B, TLI);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrCpy(CallInst *CI, IRBuilderBase &B) {
  Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
  if (Dst == Src) // strcpy(x,x)  -> x
    return Src;
  
  annotateNonNullBasedOnAccess(CI, {0, 1});
  // See if we can get the length of the input string.
  uint64_t Len = GetStringLength(Src);
  if (Len)
    annotateDereferenceableBytes(CI, 1, Len);
  else
    return nullptr;

  // We have enough information to now generate the memcpy call to do the
  // copy for us.  Make a memcpy to copy the nul byte with align = 1.
  CallInst *NewCI =
      B.CreateMemCpy(Dst, Align(1), Src, Align(1),
                     ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len));
  NewCI->setAttributes(CI->getAttributes());
  return Dst;
}

Value *LibCallSimplifier::optimizeStpCpy(CallInst *CI, IRBuilderBase &B) {
  Function *Callee = CI->getCalledFunction();
  Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
  if (Dst == Src) { // stpcpy(x,x)  -> x+strlen(x)
    Value *StrLen = emitStrLen(Src, B, DL, TLI);
    return StrLen ? B.CreateInBoundsGEP(B.getInt8Ty(), Dst, StrLen) : nullptr;
  }

  // See if we can get the length of the input string.
  uint64_t Len = GetStringLength(Src);
  if (Len)
    annotateDereferenceableBytes(CI, 1, Len);
  else
    return nullptr;

  Type *PT = Callee->getFunctionType()->getParamType(0);
  Value *LenV = ConstantInt::get(DL.getIntPtrType(PT), Len);
  Value *DstEnd = B.CreateGEP(B.getInt8Ty(), Dst,
                              ConstantInt::get(DL.getIntPtrType(PT), Len - 1));

  // We have enough information to now generate the memcpy call to do the
  // copy for us.  Make a memcpy to copy the nul byte with align = 1.
  CallInst *NewCI = B.CreateMemCpy(Dst, Align(1), Src, Align(1), LenV);
  NewCI->setAttributes(CI->getAttributes());
  return DstEnd;
}

Value *LibCallSimplifier::optimizeStrNCpy(CallInst *CI, IRBuilderBase &B) {
  Function *Callee = CI->getCalledFunction();
  Value *Dst = CI->getArgOperand(0);
  Value *Src = CI->getArgOperand(1);
  Value *Size = CI->getArgOperand(2);
  annotateNonNullBasedOnAccess(CI, 0);
  if (isKnownNonZero(Size, DL))
    annotateNonNullBasedOnAccess(CI, 1);

  uint64_t Len;
  if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(Size))
    Len = LengthArg->getZExtValue();
  else
    return nullptr;

  // strncpy(x, y, 0) -> x
  if (Len == 0)
    return Dst;

  // See if we can get the length of the input string.
  uint64_t SrcLen = GetStringLength(Src);
  if (SrcLen) {
    annotateDereferenceableBytes(CI, 1, SrcLen);
    --SrcLen; // Unbias length.
  } else {
    return nullptr;
  }

  if (SrcLen == 0) {
    // strncpy(x, "", y) -> memset(align 1 x, '\0', y)
    CallInst *NewCI = B.CreateMemSet(Dst, B.getInt8('\0'), Size, Align(1));
    AttrBuilder ArgAttrs(CI->getAttributes().getParamAttributes(0));
    NewCI->setAttributes(NewCI->getAttributes().addParamAttributes(
        CI->getContext(), 0, ArgAttrs));
    return Dst;
  }

  // strncpy(a, "a", 4) - > memcpy(a, "a\0\0\0", 4)
  if (Len > SrcLen + 1) {
    if (Len <= 128) {
      StringRef Str;
      if (!getConstantStringInfo(Src, Str))
        return nullptr;
      std::string SrcStr = Str.str();
      SrcStr.resize(Len, '\0');
      Src = B.CreateGlobalString(SrcStr, "str");
    } else {
      return nullptr;
    }
  }

  Type *PT = Callee->getFunctionType()->getParamType(0);
  // strncpy(x, s, c) -> memcpy(align 1 x, align 1 s, c) [s and c are constant]
  CallInst *NewCI = B.CreateMemCpy(Dst, Align(1), Src, Align(1),
                                   ConstantInt::get(DL.getIntPtrType(PT), Len));
  NewCI->setAttributes(CI->getAttributes());
  return Dst;
}

Value *LibCallSimplifier::optimizeStringLength(CallInst *CI, IRBuilderBase &B,
                                               unsigned CharSize) {
  Value *Src = CI->getArgOperand(0);

  // Constant folding: strlen("xyz") -> 3
  if (uint64_t Len = GetStringLength(Src, CharSize))
    return ConstantInt::get(CI->getType(), Len - 1);

  // If s is a constant pointer pointing to a string literal, we can fold
  // strlen(s + x) to strlen(s) - x, when x is known to be in the range
  // [0, strlen(s)] or the string has a single null terminator '\0' at the end.
  // We only try to simplify strlen when the pointer s points to an array
  // of i8. Otherwise, we would need to scale the offset x before doing the
  // subtraction. This will make the optimization more complex, and it's not
  // very useful because calling strlen for a pointer of other types is
  // very uncommon.
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(Src)) {
    if (!isGEPBasedOnPointerToString(GEP, CharSize))
      return nullptr;

    ConstantDataArraySlice Slice;
    if (getConstantDataArrayInfo(GEP->getOperand(0), Slice, CharSize)) {
      uint64_t NullTermIdx;
      if (Slice.Array == nullptr) {
        NullTermIdx = 0;
      } else {
        NullTermIdx = ~((uint64_t)0);
        for (uint64_t I = 0, E = Slice.Length; I < E; ++I) {
          if (Slice.Array->getElementAsInteger(I + Slice.Offset) == 0) {
            NullTermIdx = I;
            break;
          }
        }
        // If the string does not have '\0', leave it to strlen to compute
        // its length.
        if (NullTermIdx == ~((uint64_t)0))
          return nullptr;
      }

      Value *Offset = GEP->getOperand(2);
      KnownBits Known = computeKnownBits(Offset, DL, 0, nullptr, CI, nullptr);
      Known.Zero.flipAllBits();
      uint64_t ArrSize =
             cast<ArrayType>(GEP->getSourceElementType())->getNumElements();

      // KnownZero's bits are flipped, so zeros in KnownZero now represent
      // bits known to be zeros in Offset, and ones in KnowZero represent
      // bits unknown in Offset. Therefore, Offset is known to be in range
      // [0, NullTermIdx] when the flipped KnownZero is non-negative and
      // unsigned-less-than NullTermIdx.
      //
      // If Offset is not provably in the range [0, NullTermIdx], we can still
      // optimize if we can prove that the program has undefined behavior when
      // Offset is outside that range. That is the case when GEP->getOperand(0)
      // is a pointer to an object whose memory extent is NullTermIdx+1.
      if ((Known.Zero.isNonNegative() && Known.Zero.ule(NullTermIdx)) ||
          (GEP->isInBounds() && isa<GlobalVariable>(GEP->getOperand(0)) &&
           NullTermIdx == ArrSize - 1)) {
        Offset = B.CreateSExtOrTrunc(Offset, CI->getType());
        return B.CreateSub(ConstantInt::get(CI->getType(), NullTermIdx),
                           Offset);
      }
    }
  }

  // strlen(x?"foo":"bars") --> x ? 3 : 4
  if (SelectInst *SI = dyn_cast<SelectInst>(Src)) {
    uint64_t LenTrue = GetStringLength(SI->getTrueValue(), CharSize);
    uint64_t LenFalse = GetStringLength(SI->getFalseValue(), CharSize);
    if (LenTrue && LenFalse) {
      ORE.emit([&]() {
        return OptimizationRemark("instcombine", "simplify-libcalls", CI)
               << "folded strlen(select) to select of constants";
      });
      return B.CreateSelect(SI->getCondition(),
                            ConstantInt::get(CI->getType(), LenTrue - 1),
                            ConstantInt::get(CI->getType(), LenFalse - 1));
    }
  }

  // strlen(x) != 0 --> *x != 0
  // strlen(x) == 0 --> *x == 0
  if (isOnlyUsedInZeroEqualityComparison(CI))
    return B.CreateZExt(B.CreateLoad(B.getIntNTy(CharSize), Src, "strlenfirst"),
                        CI->getType());

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrLen(CallInst *CI, IRBuilderBase &B) {
  if (Value *V = optimizeStringLength(CI, B, 8))
    return V;
  annotateNonNullBasedOnAccess(CI, 0);
  return nullptr;
}

Value *LibCallSimplifier::optimizeWcslen(CallInst *CI, IRBuilderBase &B) {
  Module &M = *CI->getModule();
  unsigned WCharSize = TLI->getWCharSize(M) * 8;
  // We cannot perform this optimization without wchar_size metadata.
  if (WCharSize == 0)
    return nullptr;

  return optimizeStringLength(CI, B, WCharSize);
}

Value *LibCallSimplifier::optimizeStrPBrk(CallInst *CI, IRBuilderBase &B) {
  StringRef S1, S2;
  bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
  bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

  // strpbrk(s, "") -> nullptr
  // strpbrk("", s) -> nullptr
  if ((HasS1 && S1.empty()) || (HasS2 && S2.empty()))
    return Constant::getNullValue(CI->getType());

  // Constant folding.
  if (HasS1 && HasS2) {
    size_t I = S1.find_first_of(S2);
    if (I == StringRef::npos) // No match.
      return Constant::getNullValue(CI->getType());

    return B.CreateGEP(B.getInt8Ty(), CI->getArgOperand(0), B.getInt64(I),
                       "strpbrk");
  }

  // strpbrk(s, "a") -> strchr(s, 'a')
  if (HasS2 && S2.size() == 1)
    return emitStrChr(CI->getArgOperand(0), S2[0], B, TLI);

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrTo(CallInst *CI, IRBuilderBase &B) {
  Value *EndPtr = CI->getArgOperand(1);
  if (isa<ConstantPointerNull>(EndPtr)) {
    // With a null EndPtr, this function won't capture the main argument.
    // It would be readonly too, except that it still may write to errno.
    CI->addParamAttr(0, Attribute::NoCapture);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrSpn(CallInst *CI, IRBuilderBase &B) {
  StringRef S1, S2;
  bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
  bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

  // strspn(s, "") -> 0
  // strspn("", s) -> 0
  if ((HasS1 && S1.empty()) || (HasS2 && S2.empty()))
    return Constant::getNullValue(CI->getType());

  // Constant folding.
  if (HasS1 && HasS2) {
    size_t Pos = S1.find_first_not_of(S2);
    if (Pos == StringRef::npos)
      Pos = S1.size();
    return ConstantInt::get(CI->getType(), Pos);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrCSpn(CallInst *CI, IRBuilderBase &B) {
  StringRef S1, S2;
  bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
  bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

  // strcspn("", s) -> 0
  if (HasS1 && S1.empty())
    return Constant::getNullValue(CI->getType());

  // Constant folding.
  if (HasS1 && HasS2) {
    size_t Pos = S1.find_first_of(S2);
    if (Pos == StringRef::npos)
      Pos = S1.size();
    return ConstantInt::get(CI->getType(), Pos);
  }

  // strcspn(s, "") -> strlen(s)
  if (HasS2 && S2.empty())
    return emitStrLen(CI->getArgOperand(0), B, DL, TLI);

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrStr(CallInst *CI, IRBuilderBase &B) {
  // fold strstr(x, x) -> x.
  if (CI->getArgOperand(0) == CI->getArgOperand(1))
    return B.CreateBitCast(CI->getArgOperand(0), CI->getType());

  // fold strstr(a, b) == a -> strncmp(a, b, strlen(b)) == 0
  if (isOnlyUsedInEqualityComparison(CI, CI->getArgOperand(0))) {
    Value *StrLen = emitStrLen(CI->getArgOperand(1), B, DL, TLI);
    if (!StrLen)
      return nullptr;
    Value *StrNCmp = emitStrNCmp(CI->getArgOperand(0), CI->getArgOperand(1),
                                 StrLen, B, DL, TLI);
    if (!StrNCmp)
      return nullptr;
    for (auto UI = CI->user_begin(), UE = CI->user_end(); UI != UE;) {
      ICmpInst *Old = cast<ICmpInst>(*UI++);
      Value *Cmp =
          B.CreateICmp(Old->getPredicate(), StrNCmp,
                       ConstantInt::getNullValue(StrNCmp->getType()), "cmp");
      replaceAllUsesWith(Old, Cmp);
    }
    return CI;
  }

  // See if either input string is a constant string.
  StringRef SearchStr, ToFindStr;
  bool HasStr1 = getConstantStringInfo(CI->getArgOperand(0), SearchStr);
  bool HasStr2 = getConstantStringInfo(CI->getArgOperand(1), ToFindStr);

  // fold strstr(x, "") -> x.
  if (HasStr2 && ToFindStr.empty())
    return B.CreateBitCast(CI->getArgOperand(0), CI->getType());

  // If both strings are known, constant fold it.
  if (HasStr1 && HasStr2) {
    size_t Offset = SearchStr.find(ToFindStr);

    if (Offset == StringRef::npos) // strstr("foo", "bar") -> null
      return Constant::getNullValue(CI->getType());

    // strstr("abcd", "bc") -> gep((char*)"abcd", 1)
    Value *Result = castToCStr(CI->getArgOperand(0), B);
    Result =
        B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Result, Offset, "strstr");
    return B.CreateBitCast(Result, CI->getType());
  }

  // fold strstr(x, "y") -> strchr(x, 'y').
  if (HasStr2 && ToFindStr.size() == 1) {
    Value *StrChr = emitStrChr(CI->getArgOperand(0), ToFindStr[0], B, TLI);
    return StrChr ? B.CreateBitCast(StrChr, CI->getType()) : nullptr;
  }

  annotateNonNullBasedOnAccess(CI, {0, 1});
  return nullptr;
}

Value *LibCallSimplifier::optimizeMemRChr(CallInst *CI, IRBuilderBase &B) {
  if (isKnownNonZero(CI->getOperand(2), DL))
    annotateNonNullBasedOnAccess(CI, 0);
  return nullptr;
}

Value *LibCallSimplifier::optimizeMemChr(CallInst *CI, IRBuilderBase &B) {
  Value *SrcStr = CI->getArgOperand(0);
  Value *Size = CI->getArgOperand(2);
  annotateNonNullAndDereferenceable(CI, 0, Size, DL);
  ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  ConstantInt *LenC = dyn_cast<ConstantInt>(Size);

  // memchr(x, y, 0) -> null
  if (LenC) {
    if (LenC->isZero())
      return Constant::getNullValue(CI->getType());
  } else {
    // From now on we need at least constant length and string.
    return nullptr;
  }

  StringRef Str;
  if (!getConstantStringInfo(SrcStr, Str, 0, /*TrimAtNul=*/false))
    return nullptr;

  // Truncate the string to LenC. If Str is smaller than LenC we will still only
  // scan the string, as reading past the end of it is undefined and we can just
  // return null if we don't find the char.
  Str = Str.substr(0, LenC->getZExtValue());

  // If the char is variable but the input str and length are not we can turn
  // this memchr call into a simple bit field test. Of course this only works
  // when the return value is only checked against null.
  //
  // It would be really nice to reuse switch lowering here but we can't change
  // the CFG at this point.
  //
  // memchr("\r\n", C, 2) != nullptr -> (1 << C & ((1 << '\r') | (1 << '\n')))
  // != 0
  //   after bounds check.
  if (!CharC && !Str.empty() && isOnlyUsedInZeroEqualityComparison(CI)) {
    unsigned char Max =
        *std::max_element(reinterpret_cast<const unsigned char *>(Str.begin()),
                          reinterpret_cast<const unsigned char *>(Str.end()));

    // Make sure the bit field we're about to create fits in a register on the
    // target.
    // FIXME: On a 64 bit architecture this prevents us from using the
    // interesting range of alpha ascii chars. We could do better by emitting
    // two bitfields or shifting the range by 64 if no lower chars are used.
    if (!DL.fitsInLegalInteger(Max + 1))
      return nullptr;

    // For the bit field use a power-of-2 type with at least 8 bits to avoid
    // creating unnecessary illegal types.
    unsigned char Width = NextPowerOf2(std::max((unsigned char)7, Max));

    // Now build the bit field.
    APInt Bitfield(Width, 0);
    for (char C : Str)
      Bitfield.setBit((unsigned char)C);
    Value *BitfieldC = B.getInt(Bitfield);

    // Adjust width of "C" to the bitfield width, then mask off the high bits.
    Value *C = B.CreateZExtOrTrunc(CI->getArgOperand(1), BitfieldC->getType());
    C = B.CreateAnd(C, B.getIntN(Width, 0xFF));

    // First check that the bit field access is within bounds.
    Value *Bounds = B.CreateICmp(ICmpInst::ICMP_ULT, C, B.getIntN(Width, Width),
                                 "memchr.bounds");

    // Create code that checks if the given bit is set in the field.
    Value *Shl = B.CreateShl(B.getIntN(Width, 1ULL), C);
    Value *Bits = B.CreateIsNotNull(B.CreateAnd(Shl, BitfieldC), "memchr.bits");

    // Finally merge both checks and cast to pointer type. The inttoptr
    // implicitly zexts the i1 to intptr type.
    return B.CreateIntToPtr(B.CreateAnd(Bounds, Bits, "memchr"), CI->getType());
  }

  // Check if all arguments are constants.  If so, we can constant fold.
  if (!CharC)
    return nullptr;

  // Compute the offset.
  size_t I = Str.find(CharC->getSExtValue() & 0xFF);
  if (I == StringRef::npos) // Didn't find the char.  memchr returns null.
    return Constant::getNullValue(CI->getType());

  // memchr(s+n,c,l) -> gep(s+n+i,c)
  return B.CreateGEP(B.getInt8Ty(), SrcStr, B.getInt64(I), "memchr");
}

static Value *optimizeMemCmpConstantSize(CallInst *CI, Value *LHS, Value *RHS,
                                         uint64_t Len, IRBuilderBase &B,
                                         const DataLayout &DL) {
  if (Len == 0) // memcmp(s1,s2,0) -> 0
    return Constant::getNullValue(CI->getType());

  // memcmp(S1,S2,1) -> *(unsigned char*)LHS - *(unsigned char*)RHS
  if (Len == 1) {
    Value *LHSV =
        B.CreateZExt(B.CreateLoad(B.getInt8Ty(), castToCStr(LHS, B), "lhsc"),
                     CI->getType(), "lhsv");
    Value *RHSV =
        B.CreateZExt(B.CreateLoad(B.getInt8Ty(), castToCStr(RHS, B), "rhsc"),
                     CI->getType(), "rhsv");
    return B.CreateSub(LHSV, RHSV, "chardiff");
  }

  // memcmp(S1,S2,N/8)==0 -> (*(intN_t*)S1 != *(intN_t*)S2)==0
  // TODO: The case where both inputs are constants does not need to be limited
  // to legal integers or equality comparison. See block below this.
  if (DL.isLegalInteger(Len * 8) && isOnlyUsedInZeroEqualityComparison(CI)) {
    IntegerType *IntType = IntegerType::get(CI->getContext(), Len * 8);
    unsigned PrefAlignment = DL.getPrefTypeAlignment(IntType);

    // First, see if we can fold either argument to a constant.
    Value *LHSV = nullptr;
    if (auto *LHSC = dyn_cast<Constant>(LHS)) {
      LHSC = ConstantExpr::getBitCast(LHSC, IntType->getPointerTo());
      LHSV = ConstantFoldLoadFromConstPtr(LHSC, IntType, DL);
    }
    Value *RHSV = nullptr;
    if (auto *RHSC = dyn_cast<Constant>(RHS)) {
      RHSC = ConstantExpr::getBitCast(RHSC, IntType->getPointerTo());
      RHSV = ConstantFoldLoadFromConstPtr(RHSC, IntType, DL);
    }

    // Don't generate unaligned loads. If either source is constant data,
    // alignment doesn't matter for that source because there is no load.
    if ((LHSV || getKnownAlignment(LHS, DL, CI) >= PrefAlignment) &&
        (RHSV || getKnownAlignment(RHS, DL, CI) >= PrefAlignment)) {
      if (!LHSV) {
        Type *LHSPtrTy =
            IntType->getPointerTo(LHS->getType()->getPointerAddressSpace());
        LHSV = B.CreateLoad(IntType, B.CreateBitCast(LHS, LHSPtrTy), "lhsv");
      }
      if (!RHSV) {
        Type *RHSPtrTy =
            IntType->getPointerTo(RHS->getType()->getPointerAddressSpace());
        RHSV = B.CreateLoad(IntType, B.CreateBitCast(RHS, RHSPtrTy), "rhsv");
      }
      return B.CreateZExt(B.CreateICmpNE(LHSV, RHSV), CI->getType(), "memcmp");
    }
  }

  // Constant folding: memcmp(x, y, Len) -> constant (all arguments are const).
  // TODO: This is limited to i8 arrays.
  StringRef LHSStr, RHSStr;
  if (getConstantStringInfo(LHS, LHSStr) &&
      getConstantStringInfo(RHS, RHSStr)) {
    // Make sure we're not reading out-of-bounds memory.
    if (Len > LHSStr.size() || Len > RHSStr.size())
      return nullptr;
    // Fold the memcmp and normalize the result.  This way we get consistent
    // results across multiple platforms.
    uint64_t Ret = 0;
    int Cmp = memcmp(LHSStr.data(), RHSStr.data(), Len);
    if (Cmp < 0)
      Ret = -1;
    else if (Cmp > 0)
      Ret = 1;
    return ConstantInt::get(CI->getType(), Ret);
  }

  return nullptr;
}

// Most simplifications for memcmp also apply to bcmp.
Value *LibCallSimplifier::optimizeMemCmpBCmpCommon(CallInst *CI,
                                                   IRBuilderBase &B) {
  Value *LHS = CI->getArgOperand(0), *RHS = CI->getArgOperand(1);
  Value *Size = CI->getArgOperand(2);

  if (LHS == RHS) // memcmp(s,s,x) -> 0
    return Constant::getNullValue(CI->getType());

  annotateNonNullAndDereferenceable(CI, {0, 1}, Size, DL);
  // Handle constant lengths.
  ConstantInt *LenC = dyn_cast<ConstantInt>(Size);
  if (!LenC)
    return nullptr;

  // memcmp(d,s,0) -> 0
  if (LenC->getZExtValue() == 0)
    return Constant::getNullValue(CI->getType());

  if (Value *Res =
          optimizeMemCmpConstantSize(CI, LHS, RHS, LenC->getZExtValue(), B, DL))
    return Res;
  return nullptr;
}

Value *LibCallSimplifier::optimizeMemCmp(CallInst *CI, IRBuilderBase &B) {
  if (Value *V = optimizeMemCmpBCmpCommon(CI, B))
    return V;

  // memcmp(x, y, Len) == 0 -> bcmp(x, y, Len) == 0
  // bcmp can be more efficient than memcmp because it only has to know that
  // there is a difference, not how different one is to the other.
  if (TLI->has(LibFunc_bcmp) && isOnlyUsedInZeroEqualityComparison(CI)) {
    Value *LHS = CI->getArgOperand(0);
    Value *RHS = CI->getArgOperand(1);
    Value *Size = CI->getArgOperand(2);
    return emitBCmp(LHS, RHS, Size, B, DL, TLI);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeBCmp(CallInst *CI, IRBuilderBase &B) {
  return optimizeMemCmpBCmpCommon(CI, B);
}

Value *LibCallSimplifier::optimizeMemCpy(CallInst *CI, IRBuilderBase &B) {
  Value *Size = CI->getArgOperand(2);
  annotateNonNullAndDereferenceable(CI, {0, 1}, Size, DL);
  if (isa<IntrinsicInst>(CI))
    return nullptr;

  // memcpy(x, y, n) -> llvm.memcpy(align 1 x, align 1 y, n)
  CallInst *NewCI = B.CreateMemCpy(CI->getArgOperand(0), Align(1),
                                   CI->getArgOperand(1), Align(1), Size);
  NewCI->setAttributes(CI->getAttributes());
  return CI->getArgOperand(0);
}

Value *LibCallSimplifier::optimizeMemCCpy(CallInst *CI, IRBuilderBase &B) {
  Value *Dst = CI->getArgOperand(0);
  Value *Src = CI->getArgOperand(1);
  ConstantInt *StopChar = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  ConstantInt *N = dyn_cast<ConstantInt>(CI->getArgOperand(3));
  StringRef SrcStr;
  if (CI->use_empty() && Dst == Src)
    return Dst;
  // memccpy(d, s, c, 0) -> nullptr
  if (N) {
    if (N->isNullValue())
      return Constant::getNullValue(CI->getType());
    if (!getConstantStringInfo(Src, SrcStr, /*Offset=*/0,
                               /*TrimAtNul=*/false) ||
        !StopChar)
      return nullptr;
  } else {
    return nullptr;
  }

  // Wrap arg 'c' of type int to char
  size_t Pos = SrcStr.find(StopChar->getSExtValue() & 0xFF);
  if (Pos == StringRef::npos) {
    if (N->getZExtValue() <= SrcStr.size()) {
      B.CreateMemCpy(Dst, Align(1), Src, Align(1), CI->getArgOperand(3));
      return Constant::getNullValue(CI->getType());
    }
    return nullptr;
  }

  Value *NewN =
      ConstantInt::get(N->getType(), std::min(uint64_t(Pos + 1), N->getZExtValue()));
  // memccpy -> llvm.memcpy
  B.CreateMemCpy(Dst, Align(1), Src, Align(1), NewN);
  return Pos + 1 <= N->getZExtValue()
             ? B.CreateInBoundsGEP(B.getInt8Ty(), Dst, NewN)
             : Constant::getNullValue(CI->getType());
}

Value *LibCallSimplifier::optimizeMemPCpy(CallInst *CI, IRBuilderBase &B) {
  Value *Dst = CI->getArgOperand(0);
  Value *N = CI->getArgOperand(2);
  // mempcpy(x, y, n) -> llvm.memcpy(align 1 x, align 1 y, n), x + n
  CallInst *NewCI =
      B.CreateMemCpy(Dst, Align(1), CI->getArgOperand(1), Align(1), N);
  NewCI->setAttributes(CI->getAttributes());
  return B.CreateInBoundsGEP(B.getInt8Ty(), Dst, N);
}

Value *LibCallSimplifier::optimizeMemMove(CallInst *CI, IRBuilderBase &B) {
  Value *Size = CI->getArgOperand(2);
  annotateNonNullAndDereferenceable(CI, {0, 1}, Size, DL);
  if (isa<IntrinsicInst>(CI))
    return nullptr;

  // memmove(x, y, n) -> llvm.memmove(align 1 x, align 1 y, n)
  CallInst *NewCI = B.CreateMemMove(CI->getArgOperand(0), Align(1),
                                    CI->getArgOperand(1), Align(1), Size);
  NewCI->setAttributes(CI->getAttributes());
  return CI->getArgOperand(0);
}

/// Fold memset[_chk](malloc(n), 0, n) --> calloc(1, n).
Value *LibCallSimplifier::foldMallocMemset(CallInst *Memset, IRBuilderBase &B) {
  // This has to be a memset of zeros (bzero).
  auto *FillValue = dyn_cast<ConstantInt>(Memset->getArgOperand(1));
  if (!FillValue || FillValue->getZExtValue() != 0)
    return nullptr;

  // TODO: We should handle the case where the malloc has more than one use.
  // This is necessary to optimize common patterns such as when the result of
  // the malloc is checked against null or when a memset intrinsic is used in
  // place of a memset library call.
  auto *Malloc = dyn_cast<CallInst>(Memset->getArgOperand(0));
  if (!Malloc || !Malloc->hasOneUse())
    return nullptr;

  // Is the inner call really malloc()?
  Function *InnerCallee = Malloc->getCalledFunction();
  if (!InnerCallee)
    return nullptr;

  LibFunc Func;
  if (!TLI->getLibFunc(*InnerCallee, Func) || !TLI->has(Func) ||
      Func != LibFunc_malloc)
    return nullptr;

  // The memset must cover the same number of bytes that are malloc'd.
  if (Memset->getArgOperand(2) != Malloc->getArgOperand(0))
    return nullptr;

  // Replace the malloc with a calloc. We need the data layout to know what the
  // actual size of a 'size_t' parameter is.
  B.SetInsertPoint(Malloc->getParent(), ++Malloc->getIterator());
  const DataLayout &DL = Malloc->getModule()->getDataLayout();
  IntegerType *SizeType = DL.getIntPtrType(B.GetInsertBlock()->getContext());
  if (Value *Calloc = emitCalloc(ConstantInt::get(SizeType, 1),
                                 Malloc->getArgOperand(0),
                                 Malloc->getAttributes(), B, *TLI)) {
    substituteInParent(Malloc, Calloc);
    return Calloc;
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeMemSet(CallInst *CI, IRBuilderBase &B) {
  Value *Size = CI->getArgOperand(2);
  annotateNonNullAndDereferenceable(CI, 0, Size, DL);
  if (isa<IntrinsicInst>(CI))
    return nullptr;

  if (auto *Calloc = foldMallocMemset(CI, B))
    return Calloc;

  // memset(p, v, n) -> llvm.memset(align 1 p, v, n)
  Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(), false);
  CallInst *NewCI = B.CreateMemSet(CI->getArgOperand(0), Val, Size, Align(1));
  NewCI->setAttributes(CI->getAttributes());
  return CI->getArgOperand(0);
}

Value *LibCallSimplifier::optimizeRealloc(CallInst *CI, IRBuilderBase &B) {
  if (isa<ConstantPointerNull>(CI->getArgOperand(0)))
    return emitMalloc(CI->getArgOperand(1), B, DL, TLI);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Math Library Optimizations
//===----------------------------------------------------------------------===//

// Replace a libcall \p CI with a call to intrinsic \p IID
static Value *replaceUnaryCall(CallInst *CI, IRBuilderBase &B,
                               Intrinsic::ID IID) {
  // Propagate fast-math flags from the existing call to the new call.
  IRBuilderBase::FastMathFlagGuard Guard(B);
  B.setFastMathFlags(CI->getFastMathFlags());

  Module *M = CI->getModule();
  Value *V = CI->getArgOperand(0);
  Function *F = Intrinsic::getDeclaration(M, IID, CI->getType());
  CallInst *NewCall = B.CreateCall(F, V);
  NewCall->takeName(CI);
  return NewCall;
}

/// Return a variant of Val with float type.
/// Currently this works in two cases: If Val is an FPExtension of a float
/// value to something bigger, simply return the operand.
/// If Val is a ConstantFP but can be converted to a float ConstantFP without
/// loss of precision do so.
static Value *valueHasFloatPrecision(Value *Val) {
  if (FPExtInst *Cast = dyn_cast<FPExtInst>(Val)) {
    Value *Op = Cast->getOperand(0);
    if (Op->getType()->isFloatTy())
      return Op;
  }
  if (ConstantFP *Const = dyn_cast<ConstantFP>(Val)) {
    APFloat F = Const->getValueAPF();
    bool losesInfo;
    (void)F.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
    if (!losesInfo)
      return ConstantFP::get(Const->getContext(), F);
  }
  return nullptr;
}

/// Shrink double -> float functions.
static Value *optimizeDoubleFP(CallInst *CI, IRBuilderBase &B,
                               bool isBinary, bool isPrecise = false) {
  Function *CalleeFn = CI->getCalledFunction();
  if (!CI->getType()->isDoubleTy() || !CalleeFn)
    return nullptr;

  // If not all the uses of the function are converted to float, then bail out.
  // This matters if the precision of the result is more important than the
  // precision of the arguments.
  if (isPrecise)
    for (User *U : CI->users()) {
      FPTruncInst *Cast = dyn_cast<FPTruncInst>(U);
      if (!Cast || !Cast->getType()->isFloatTy())
        return nullptr;
    }

  // If this is something like 'g((double) float)', convert to 'gf(float)'.
  Value *V[2];
  V[0] = valueHasFloatPrecision(CI->getArgOperand(0));
  V[1] = isBinary ? valueHasFloatPrecision(CI->getArgOperand(1)) : nullptr;
  if (!V[0] || (isBinary && !V[1]))
    return nullptr;

  // If call isn't an intrinsic, check that it isn't within a function with the
  // same name as the float version of this call, otherwise the result is an
  // infinite loop.  For example, from MinGW-w64:
  //
  // float expf(float val) { return (float) exp((double) val); }
  StringRef CalleeName = CalleeFn->getName();
  bool IsIntrinsic = CalleeFn->isIntrinsic();
  if (!IsIntrinsic) {
    StringRef CallerName = CI->getFunction()->getName();
    if (!CallerName.empty() && CallerName.back() == 'f' &&
        CallerName.size() == (CalleeName.size() + 1) &&
        CallerName.startswith(CalleeName))
      return nullptr;
  }

  // Propagate the math semantics from the current function to the new function.
  IRBuilderBase::FastMathFlagGuard Guard(B);
  B.setFastMathFlags(CI->getFastMathFlags());

  // g((double) float) -> (double) gf(float)
  Value *R;
  if (IsIntrinsic) {
    Module *M = CI->getModule();
    Intrinsic::ID IID = CalleeFn->getIntrinsicID();
    Function *Fn = Intrinsic::getDeclaration(M, IID, B.getFloatTy());
    R = isBinary ? B.CreateCall(Fn, V) : B.CreateCall(Fn, V[0]);
  } else {
    AttributeList CalleeAttrs = CalleeFn->getAttributes();
    R = isBinary ? emitBinaryFloatFnCall(V[0], V[1], CalleeName, B, CalleeAttrs)
                 : emitUnaryFloatFnCall(V[0], CalleeName, B, CalleeAttrs);
  }
  return B.CreateFPExt(R, B.getDoubleTy());
}

/// Shrink double -> float for unary functions.
static Value *optimizeUnaryDoubleFP(CallInst *CI, IRBuilderBase &B,
                                    bool isPrecise = false) {
  return optimizeDoubleFP(CI, B, false, isPrecise);
}

/// Shrink double -> float for binary functions.
static Value *optimizeBinaryDoubleFP(CallInst *CI, IRBuilderBase &B,
                                     bool isPrecise = false) {
  return optimizeDoubleFP(CI, B, true, isPrecise);
}

// cabs(z) -> sqrt((creal(z)*creal(z)) + (cimag(z)*cimag(z)))
Value *LibCallSimplifier::optimizeCAbs(CallInst *CI, IRBuilderBase &B) {
  if (!CI->isFast())
    return nullptr;

  // Propagate fast-math flags from the existing call to new instructions.
  IRBuilderBase::FastMathFlagGuard Guard(B);
  B.setFastMathFlags(CI->getFastMathFlags());

  Value *Real, *Imag;
  if (CI->getNumArgOperands() == 1) {
    Value *Op = CI->getArgOperand(0);
    assert(Op->getType()->isArrayTy() && "Unexpected signature for cabs!");
    Real = B.CreateExtractValue(Op, 0, "real");
    Imag = B.CreateExtractValue(Op, 1, "imag");
  } else {
    assert(CI->getNumArgOperands() == 2 && "Unexpected signature for cabs!");
    Real = CI->getArgOperand(0);
    Imag = CI->getArgOperand(1);
  }

  Value *RealReal = B.CreateFMul(Real, Real);
  Value *ImagImag = B.CreateFMul(Imag, Imag);

  Function *FSqrt = Intrinsic::getDeclaration(CI->getModule(), Intrinsic::sqrt,
                                              CI->getType());
  return B.CreateCall(FSqrt, B.CreateFAdd(RealReal, ImagImag), "cabs");
}

static Value *optimizeTrigReflections(CallInst *Call, LibFunc Func,
                                      IRBuilderBase &B) {
  if (!isa<FPMathOperator>(Call))
    return nullptr;

  IRBuilderBase::FastMathFlagGuard Guard(B);
  B.setFastMathFlags(Call->getFastMathFlags());

  // TODO: Can this be shared to also handle LLVM intrinsics?
  Value *X;
  switch (Func) {
  case LibFunc_sin:
  case LibFunc_sinf:
  case LibFunc_sinl:
  case LibFunc_tan:
  case LibFunc_tanf:
  case LibFunc_tanl:
    // sin(-X) --> -sin(X)
    // tan(-X) --> -tan(X)
    if (match(Call->getArgOperand(0), m_OneUse(m_FNeg(m_Value(X)))))
      return B.CreateFNeg(B.CreateCall(Call->getCalledFunction(), X));
    break;
  case LibFunc_cos:
  case LibFunc_cosf:
  case LibFunc_cosl:
    // cos(-X) --> cos(X)
    if (match(Call->getArgOperand(0), m_FNeg(m_Value(X))))
      return B.CreateCall(Call->getCalledFunction(), X, "cos");
    break;
  default:
    break;
  }
  return nullptr;
}

static Value *getPow(Value *InnerChain[33], unsigned Exp, IRBuilderBase &B) {
  // Multiplications calculated using Addition Chains.
  // Refer: http://wwwhomes.uni-bielefeld.de/achim/addition_chain.html

  assert(Exp != 0 && "Incorrect exponent 0 not handled");

  if (InnerChain[Exp])
    return InnerChain[Exp];

  static const unsigned AddChain[33][2] = {
      {0, 0}, // Unused.
      {0, 0}, // Unused (base case = pow1).
      {1, 1}, // Unused (pre-computed).
      {1, 2},  {2, 2},   {2, 3},  {3, 3},   {2, 5},  {4, 4},
      {1, 8},  {5, 5},   {1, 10}, {6, 6},   {4, 9},  {7, 7},
      {3, 12}, {8, 8},   {8, 9},  {2, 16},  {1, 18}, {10, 10},
      {6, 15}, {11, 11}, {3, 20}, {12, 12}, {8, 17}, {13, 13},
      {3, 24}, {14, 14}, {4, 25}, {15, 15}, {3, 28}, {16, 16},
  };

  InnerChain[Exp] = B.CreateFMul(getPow(InnerChain, AddChain[Exp][0], B),
                                 getPow(InnerChain, AddChain[Exp][1], B));
  return InnerChain[Exp];
}

// Return a properly extended 32-bit integer if the operation is an itofp.
static Value *getIntToFPVal(Value *I2F, IRBuilderBase &B) {
  if (isa<SIToFPInst>(I2F) || isa<UIToFPInst>(I2F)) {
    Value *Op = cast<Instruction>(I2F)->getOperand(0);
    // Make sure that the exponent fits inside an int32_t,
    // thus avoiding any range issues that FP has not.
    unsigned BitWidth = Op->getType()->getPrimitiveSizeInBits();
    if (BitWidth < 32 ||
        (BitWidth == 32 && isa<SIToFPInst>(I2F)))
      return isa<SIToFPInst>(I2F) ? B.CreateSExt(Op, B.getInt32Ty())
                                  : B.CreateZExt(Op, B.getInt32Ty());
  }

  return nullptr;
}

/// Use exp{,2}(x * y) for pow(exp{,2}(x), y);
/// ldexp(1.0, x) for pow(2.0, itofp(x)); exp2(n * x) for pow(2.0 ** n, x);
/// exp10(x) for pow(10.0, x); exp2(log2(n) * x) for pow(n, x).
Value *LibCallSimplifier::replacePowWithExp(CallInst *Pow, IRBuilderBase &B) {
  Value *Base = Pow->getArgOperand(0), *Expo = Pow->getArgOperand(1);
  AttributeList Attrs; // Attributes are only meaningful on the original call
  Module *Mod = Pow->getModule();
  Type *Ty = Pow->getType();
  bool Ignored;

  // Evaluate special cases related to a nested function as the base.

  // pow(exp(x), y) -> exp(x * y)
  // pow(exp2(x), y) -> exp2(x * y)
  // If exp{,2}() is used only once, it is better to fold two transcendental
  // math functions into one.  If used again, exp{,2}() would still have to be
  // called with the original argument, then keep both original transcendental
  // functions.  However, this transformation is only safe with fully relaxed
  // math semantics, since, besides rounding differences, it changes overflow
  // and underflow behavior quite dramatically.  For example:
  //   pow(exp(1000), 0.001) = pow(inf, 0.001) = inf
  // Whereas:
  //   exp(1000 * 0.001) = exp(1)
  // TODO: Loosen the requirement for fully relaxed math semantics.
  // TODO: Handle exp10() when more targets have it available.
  CallInst *BaseFn = dyn_cast<CallInst>(Base);
  if (BaseFn && BaseFn->hasOneUse() && BaseFn->isFast() && Pow->isFast()) {
    LibFunc LibFn;

    Function *CalleeFn = BaseFn->getCalledFunction();
    if (CalleeFn &&
        TLI->getLibFunc(CalleeFn->getName(), LibFn) && TLI->has(LibFn)) {
      StringRef ExpName;
      Intrinsic::ID ID;
      Value *ExpFn;
      LibFunc LibFnFloat, LibFnDouble, LibFnLongDouble;

      switch (LibFn) {
      default:
        return nullptr;
      case LibFunc_expf:  case LibFunc_exp:  case LibFunc_expl:
        ExpName = TLI->getName(LibFunc_exp);
        ID = Intrinsic::exp;
        LibFnFloat = LibFunc_expf;
        LibFnDouble = LibFunc_exp;
        LibFnLongDouble = LibFunc_expl;
        break;
      case LibFunc_exp2f: case LibFunc_exp2: case LibFunc_exp2l:
        ExpName = TLI->getName(LibFunc_exp2);
        ID = Intrinsic::exp2;
        LibFnFloat = LibFunc_exp2f;
        LibFnDouble = LibFunc_exp2;
        LibFnLongDouble = LibFunc_exp2l;
        break;
      }

      // Create new exp{,2}() with the product as its argument.
      Value *FMul = B.CreateFMul(BaseFn->getArgOperand(0), Expo, "mul");
      ExpFn = BaseFn->doesNotAccessMemory()
              ? B.CreateCall(Intrinsic::getDeclaration(Mod, ID, Ty),
                             FMul, ExpName)
              : emitUnaryFloatFnCall(FMul, TLI, LibFnDouble, LibFnFloat,
                                     LibFnLongDouble, B,
                                     BaseFn->getAttributes());

      // Since the new exp{,2}() is different from the original one, dead code
      // elimination cannot be trusted to remove it, since it may have side
      // effects (e.g., errno).  When the only consumer for the original
      // exp{,2}() is pow(), then it has to be explicitly erased.
      substituteInParent(BaseFn, ExpFn);
      return ExpFn;
    }
  }

  // Evaluate special cases related to a constant base.

  const APFloat *BaseF;
  if (!match(Pow->getArgOperand(0), m_APFloat(BaseF)))
    return nullptr;

  // pow(2.0, itofp(x)) -> ldexp(1.0, x)
  if (match(Base, m_SpecificFP(2.0)) &&
      (isa<SIToFPInst>(Expo) || isa<UIToFPInst>(Expo)) &&
      hasFloatFn(TLI, Ty, LibFunc_ldexp, LibFunc_ldexpf, LibFunc_ldexpl)) {
    if (Value *ExpoI = getIntToFPVal(Expo, B))
      return emitBinaryFloatFnCall(ConstantFP::get(Ty, 1.0), ExpoI, TLI,
                                   LibFunc_ldexp, LibFunc_ldexpf, LibFunc_ldexpl,
                                   B, Attrs);
  }

  // pow(2.0 ** n, x) -> exp2(n * x)
  if (hasFloatFn(TLI, Ty, LibFunc_exp2, LibFunc_exp2f, LibFunc_exp2l)) {
    APFloat BaseR = APFloat(1.0);
    BaseR.convert(BaseF->getSemantics(), APFloat::rmTowardZero, &Ignored);
    BaseR = BaseR / *BaseF;
    bool IsInteger = BaseF->isInteger(), IsReciprocal = BaseR.isInteger();
    const APFloat *NF = IsReciprocal ? &BaseR : BaseF;
    APSInt NI(64, false);
    if ((IsInteger || IsReciprocal) &&
        NF->convertToInteger(NI, APFloat::rmTowardZero, &Ignored) ==
            APFloat::opOK &&
        NI > 1 && NI.isPowerOf2()) {
      double N = NI.logBase2() * (IsReciprocal ? -1.0 : 1.0);
      Value *FMul = B.CreateFMul(Expo, ConstantFP::get(Ty, N), "mul");
      if (Pow->doesNotAccessMemory())
        return B.CreateCall(Intrinsic::getDeclaration(Mod, Intrinsic::exp2, Ty),
                            FMul, "exp2");
      else
        return emitUnaryFloatFnCall(FMul, TLI, LibFunc_exp2, LibFunc_exp2f,
                                    LibFunc_exp2l, B, Attrs);
    }
  }

  // pow(10.0, x) -> exp10(x)
  // TODO: There is no exp10() intrinsic yet, but some day there shall be one.
  if (match(Base, m_SpecificFP(10.0)) &&
      hasFloatFn(TLI, Ty, LibFunc_exp10, LibFunc_exp10f, LibFunc_exp10l))
    return emitUnaryFloatFnCall(Expo, TLI, LibFunc_exp10, LibFunc_exp10f,
                                LibFunc_exp10l, B, Attrs);

  // pow(x, y) -> exp2(log2(x) * y)
  if (Pow->hasApproxFunc() && Pow->hasNoNaNs() && BaseF->isFiniteNonZero() &&
      !BaseF->isNegative()) {
    // pow(1, inf) is defined to be 1 but exp2(log2(1) * inf) evaluates to NaN.
    // Luckily optimizePow has already handled the x == 1 case.
    assert(!match(Base, m_FPOne()) &&
           "pow(1.0, y) should have been simplified earlier!");

    Value *Log = nullptr;
    if (Ty->isFloatTy())
      Log = ConstantFP::get(Ty, std::log2(BaseF->convertToFloat()));
    else if (Ty->isDoubleTy())
      Log = ConstantFP::get(Ty, std::log2(BaseF->convertToDouble()));

    if (Log) {
      Value *FMul = B.CreateFMul(Log, Expo, "mul");
      if (Pow->doesNotAccessMemory())
        return B.CreateCall(Intrinsic::getDeclaration(Mod, Intrinsic::exp2, Ty),
                            FMul, "exp2");
      else if (hasFloatFn(TLI, Ty, LibFunc_exp2, LibFunc_exp2f, LibFunc_exp2l))
        return emitUnaryFloatFnCall(FMul, TLI, LibFunc_exp2, LibFunc_exp2f,
                                    LibFunc_exp2l, B, Attrs);
    }
  }

  return nullptr;
}

static Value *getSqrtCall(Value *V, AttributeList Attrs, bool NoErrno,
                          Module *M, IRBuilderBase &B,
                          const TargetLibraryInfo *TLI) {
  // If errno is never set, then use the intrinsic for sqrt().
  if (NoErrno) {
    Function *SqrtFn =
        Intrinsic::getDeclaration(M, Intrinsic::sqrt, V->getType());
    return B.CreateCall(SqrtFn, V, "sqrt");
  }

  // Otherwise, use the libcall for sqrt().
  if (hasFloatFn(TLI, V->getType(), LibFunc_sqrt, LibFunc_sqrtf, LibFunc_sqrtl))
    // TODO: We also should check that the target can in fact lower the sqrt()
    // libcall. We currently have no way to ask this question, so we ask if
    // the target has a sqrt() libcall, which is not exactly the same.
    return emitUnaryFloatFnCall(V, TLI, LibFunc_sqrt, LibFunc_sqrtf,
                                LibFunc_sqrtl, B, Attrs);

  return nullptr;
}

/// Use square root in place of pow(x, +/-0.5).
Value *LibCallSimplifier::replacePowWithSqrt(CallInst *Pow, IRBuilderBase &B) {
  Value *Sqrt, *Base = Pow->getArgOperand(0), *Expo = Pow->getArgOperand(1);
  AttributeList Attrs; // Attributes are only meaningful on the original call
  Module *Mod = Pow->getModule();
  Type *Ty = Pow->getType();

  const APFloat *ExpoF;
  if (!match(Expo, m_APFloat(ExpoF)) ||
      (!ExpoF->isExactlyValue(0.5) && !ExpoF->isExactlyValue(-0.5)))
    return nullptr;

  // Converting pow(X, -0.5) to 1/sqrt(X) may introduce an extra rounding step,
  // so that requires fast-math-flags (afn or reassoc).
  if (ExpoF->isNegative() && (!Pow->hasApproxFunc() && !Pow->hasAllowReassoc()))
    return nullptr;

  // If we have a pow() library call (accesses memory) and we can't guarantee
  // that the base is not an infinity, give up:
  // pow(-Inf, 0.5) is optionally required to have a result of +Inf (not setting
  // errno), but sqrt(-Inf) is required by various standards to set errno.
  if (!Pow->doesNotAccessMemory() && !Pow->hasNoInfs() &&
      !isKnownNeverInfinity(Base, TLI))
    return nullptr;

  Sqrt = getSqrtCall(Base, Attrs, Pow->doesNotAccessMemory(), Mod, B, TLI);
  if (!Sqrt)
    return nullptr;

  // Handle signed zero base by expanding to fabs(sqrt(x)).
  if (!Pow->hasNoSignedZeros()) {
    Function *FAbsFn = Intrinsic::getDeclaration(Mod, Intrinsic::fabs, Ty);
    Sqrt = B.CreateCall(FAbsFn, Sqrt, "abs");
  }

  // Handle non finite base by expanding to
  // (x == -infinity ? +infinity : sqrt(x)).
  if (!Pow->hasNoInfs()) {
    Value *PosInf = ConstantFP::getInfinity(Ty),
          *NegInf = ConstantFP::getInfinity(Ty, true);
    Value *FCmp = B.CreateFCmpOEQ(Base, NegInf, "isinf");
    Sqrt = B.CreateSelect(FCmp, PosInf, Sqrt);
  }

  // If the exponent is negative, then get the reciprocal.
  if (ExpoF->isNegative())
    Sqrt = B.CreateFDiv(ConstantFP::get(Ty, 1.0), Sqrt, "reciprocal");

  return Sqrt;
}

static Value *createPowWithIntegerExponent(Value *Base, Value *Expo, Module *M,
                                           IRBuilderBase &B) {
  Value *Args[] = {Base, Expo};
  Function *F = Intrinsic::getDeclaration(M, Intrinsic::powi, Base->getType());
  return B.CreateCall(F, Args);
}

Value *LibCallSimplifier::optimizePow(CallInst *Pow, IRBuilderBase &B) {
  Value *Base = Pow->getArgOperand(0);
  Value *Expo = Pow->getArgOperand(1);
  Function *Callee = Pow->getCalledFunction();
  StringRef Name = Callee->getName();
  Type *Ty = Pow->getType();
  Module *M = Pow->getModule();
  Value *Shrunk = nullptr;
  bool AllowApprox = Pow->hasApproxFunc();
  bool Ignored;

  // Propagate the math semantics from the call to any created instructions.
  IRBuilderBase::FastMathFlagGuard Guard(B);
  B.setFastMathFlags(Pow->getFastMathFlags());

  // Shrink pow() to powf() if the arguments are single precision,
  // unless the result is expected to be double precision.
  if (UnsafeFPShrink && Name == TLI->getName(LibFunc_pow) &&
      hasFloatVersion(Name))
    Shrunk = optimizeBinaryDoubleFP(Pow, B, true);

  // Evaluate special cases related to the base.

  // pow(1.0, x) -> 1.0
  if (match(Base, m_FPOne()))
    return Base;

  if (Value *Exp = replacePowWithExp(Pow, B))
    return Exp;

  // Evaluate special cases related to the exponent.

  // pow(x, -1.0) -> 1.0 / x
  if (match(Expo, m_SpecificFP(-1.0)))
    return B.CreateFDiv(ConstantFP::get(Ty, 1.0), Base, "reciprocal");

  // pow(x, +/-0.0) -> 1.0
  if (match(Expo, m_AnyZeroFP()))
    return ConstantFP::get(Ty, 1.0);

  // pow(x, 1.0) -> x
  if (match(Expo, m_FPOne()))
    return Base;

  // pow(x, 2.0) -> x * x
  if (match(Expo, m_SpecificFP(2.0)))
    return B.CreateFMul(Base, Base, "square");

  if (Value *Sqrt = replacePowWithSqrt(Pow, B))
    return Sqrt;

  // pow(x, n) -> x * x * x * ...
  const APFloat *ExpoF;
  if (AllowApprox && match(Expo, m_APFloat(ExpoF)) &&
      !ExpoF->isExactlyValue(0.5) && !ExpoF->isExactlyValue(-0.5)) {
    // We limit to a max of 7 multiplications, thus the maximum exponent is 32.
    // If the exponent is an integer+0.5 we generate a call to sqrt and an
    // additional fmul.
    // TODO: This whole transformation should be backend specific (e.g. some
    //       backends might prefer libcalls or the limit for the exponent might
    //       be different) and it should also consider optimizing for size.
    APFloat LimF(ExpoF->getSemantics(), 33),
            ExpoA(abs(*ExpoF));
    if (ExpoA < LimF) {
      // This transformation applies to integer or integer+0.5 exponents only.
      // For integer+0.5, we create a sqrt(Base) call.
      Value *Sqrt = nullptr;
      if (!ExpoA.isInteger()) {
        APFloat Expo2 = ExpoA;
        // To check if ExpoA is an integer + 0.5, we add it to itself. If there
        // is no floating point exception and the result is an integer, then
        // ExpoA == integer + 0.5
        if (Expo2.add(ExpoA, APFloat::rmNearestTiesToEven) != APFloat::opOK)
          return nullptr;

        if (!Expo2.isInteger())
          return nullptr;

        Sqrt = getSqrtCall(Base, Pow->getCalledFunction()->getAttributes(),
                           Pow->doesNotAccessMemory(), M, B, TLI);
        if (!Sqrt)
          return nullptr;
      }

      // We will memoize intermediate products of the Addition Chain.
      Value *InnerChain[33] = {nullptr};
      InnerChain[1] = Base;
      InnerChain[2] = B.CreateFMul(Base, Base, "square");

      // We cannot readily convert a non-double type (like float) to a double.
      // So we first convert it to something which could be converted to double.
      ExpoA.convert(APFloat::IEEEdouble(), APFloat::rmTowardZero, &Ignored);
      Value *FMul = getPow(InnerChain, ExpoA.convertToDouble(), B);

      // Expand pow(x, y+0.5) to pow(x, y) * sqrt(x).
      if (Sqrt)
        FMul = B.CreateFMul(FMul, Sqrt);

      // If the exponent is negative, then get the reciprocal.
      if (ExpoF->isNegative())
        FMul = B.CreateFDiv(ConstantFP::get(Ty, 1.0), FMul, "reciprocal");

      return FMul;
    }

    APSInt IntExpo(32, /*isUnsigned=*/false);
    // powf(x, n) -> powi(x, n) if n is a constant signed integer value
    if (ExpoF->isInteger() &&
        ExpoF->convertToInteger(IntExpo, APFloat::rmTowardZero, &Ignored) ==
            APFloat::opOK) {
      return createPowWithIntegerExponent(
          Base, ConstantInt::get(B.getInt32Ty(), IntExpo), M, B);
    }
  }

  // powf(x, itofp(y)) -> powi(x, y)
  if (AllowApprox && (isa<SIToFPInst>(Expo) || isa<UIToFPInst>(Expo))) {
    if (Value *ExpoI = getIntToFPVal(Expo, B))
      return createPowWithIntegerExponent(Base, ExpoI, M, B);
  }

  return Shrunk;
}

Value *LibCallSimplifier::optimizeExp2(CallInst *CI, IRBuilderBase &B) {
  Function *Callee = CI->getCalledFunction();
  AttributeList Attrs; // Attributes are only meaningful on the original call
  StringRef Name = Callee->getName();
  Value *Ret = nullptr;
  if (UnsafeFPShrink && Name == TLI->getName(LibFunc_exp2) &&
      hasFloatVersion(Name))
    Ret = optimizeUnaryDoubleFP(CI, B, true);

  Type *Ty = CI->getType();
  Value *Op = CI->getArgOperand(0);

  // Turn exp2(sitofp(x)) -> ldexp(1.0, sext(x))  if sizeof(x) <= 32
  // Turn exp2(uitofp(x)) -> ldexp(1.0, zext(x))  if sizeof(x) < 32
  if ((isa<SIToFPInst>(Op) || isa<UIToFPInst>(Op)) &&
      hasFloatFn(TLI, Ty, LibFunc_ldexp, LibFunc_ldexpf, LibFunc_ldexpl)) {
    if (Value *Exp = getIntToFPVal(Op, B))
      return emitBinaryFloatFnCall(ConstantFP::get(Ty, 1.0), Exp, TLI,
                                   LibFunc_ldexp, LibFunc_ldexpf, LibFunc_ldexpl,
                                   B, Attrs);
  }

  return Ret;
}

Value *LibCallSimplifier::optimizeFMinFMax(CallInst *CI, IRBuilderBase &B) {
  // If we can shrink the call to a float function rather than a double
  // function, do that first.
  Function *Callee = CI->getCalledFunction();
  StringRef Name = Callee->getName();
  if ((Name == "fmin" || Name == "fmax") && hasFloatVersion(Name))
    if (Value *Ret = optimizeBinaryDoubleFP(CI, B))
      return Ret;

  // The LLVM intrinsics minnum/maxnum correspond to fmin/fmax. Canonicalize to
  // the intrinsics for improved optimization (for example, vectorization).
  // No-signed-zeros is implied by the definitions of fmax/fmin themselves.
  // From the C standard draft WG14/N1256:
  // "Ideally, fmax would be sensitive to the sign of zero, for example
  // fmax(-0.0, +0.0) would return +0; however, implementation in software
  // might be impractical."
  IRBuilderBase::FastMathFlagGuard Guard(B);
  FastMathFlags FMF = CI->getFastMathFlags();
  FMF.setNoSignedZeros();
  B.setFastMathFlags(FMF);

  Intrinsic::ID IID = Callee->getName().startswith("fmin") ? Intrinsic::minnum
                                                           : Intrinsic::maxnum;
  Function *F = Intrinsic::getDeclaration(CI->getModule(), IID, CI->getType());
  return B.CreateCall(F, { CI->getArgOperand(0), CI->getArgOperand(1) });
}

Value *LibCallSimplifier::optimizeLog(CallInst *Log, IRBuilderBase &B) {
  Function *LogFn = Log->getCalledFunction();
  AttributeList Attrs; // Attributes are only meaningful on the original call
  StringRef LogNm = LogFn->getName();
  Intrinsic::ID LogID = LogFn->getIntrinsicID();
  Module *Mod = Log->getModule();
  Type *Ty = Log->getType();
  Value *Ret = nullptr;

  if (UnsafeFPShrink && hasFloatVersion(LogNm))
    Ret = optimizeUnaryDoubleFP(Log, B, true);

  // The earlier call must also be 'fast' in order to do these transforms.
  CallInst *Arg = dyn_cast<CallInst>(Log->getArgOperand(0));
  if (!Log->isFast() || !Arg || !Arg->isFast() || !Arg->hasOneUse())
    return Ret;

  LibFunc LogLb, ExpLb, Exp2Lb, Exp10Lb, PowLb;

  // This is only applicable to log(), log2(), log10().
  if (TLI->getLibFunc(LogNm, LogLb))
    switch (LogLb) {
    case LibFunc_logf:
      LogID = Intrinsic::log;
      ExpLb = LibFunc_expf;
      Exp2Lb = LibFunc_exp2f;
      Exp10Lb = LibFunc_exp10f;
      PowLb = LibFunc_powf;
      break;
    case LibFunc_log:
      LogID = Intrinsic::log;
      ExpLb = LibFunc_exp;
      Exp2Lb = LibFunc_exp2;
      Exp10Lb = LibFunc_exp10;
      PowLb = LibFunc_pow;
      break;
    case LibFunc_logl:
      LogID = Intrinsic::log;
      ExpLb = LibFunc_expl;
      Exp2Lb = LibFunc_exp2l;
      Exp10Lb = LibFunc_exp10l;
      PowLb = LibFunc_powl;
      break;
    case LibFunc_log2f:
      LogID = Intrinsic::log2;
      ExpLb = LibFunc_expf;
      Exp2Lb = LibFunc_exp2f;
      Exp10Lb = LibFunc_exp10f;
      PowLb = LibFunc_powf;
      break;
    case LibFunc_log2:
      LogID = Intrinsic::log2;
      ExpLb = LibFunc_exp;
      Exp2Lb = LibFunc_exp2;
      Exp10Lb = LibFunc_exp10;
      PowLb = LibFunc_pow;
      break;
    case LibFunc_log2l:
      LogID = Intrinsic::log2;
      ExpLb = LibFunc_expl;
      Exp2Lb = LibFunc_exp2l;
      Exp10Lb = LibFunc_exp10l;
      PowLb = LibFunc_powl;
      break;
    case LibFunc_log10f:
      LogID = Intrinsic::log10;
      ExpLb = LibFunc_expf;
      Exp2Lb = LibFunc_exp2f;
      Exp10Lb = LibFunc_exp10f;
      PowLb = LibFunc_powf;
      break;
    case LibFunc_log10:
      LogID = Intrinsic::log10;
      ExpLb = LibFunc_exp;
      Exp2Lb = LibFunc_exp2;
      Exp10Lb = LibFunc_exp10;
      PowLb = LibFunc_pow;
      break;
    case LibFunc_log10l:
      LogID = Intrinsic::log10;
      ExpLb = LibFunc_expl;
      Exp2Lb = LibFunc_exp2l;
      Exp10Lb = LibFunc_exp10l;
      PowLb = LibFunc_powl;
      break;
    default:
      return Ret;
    }
  else if (LogID == Intrinsic::log || LogID == Intrinsic::log2 ||
           LogID == Intrinsic::log10) {
    if (Ty->getScalarType()->isFloatTy()) {
      ExpLb = LibFunc_expf;
      Exp2Lb = LibFunc_exp2f;
      Exp10Lb = LibFunc_exp10f;
      PowLb = LibFunc_powf;
    } else if (Ty->getScalarType()->isDoubleTy()) {
      ExpLb = LibFunc_exp;
      Exp2Lb = LibFunc_exp2;
      Exp10Lb = LibFunc_exp10;
      PowLb = LibFunc_pow;
    } else
      return Ret;
  } else
    return Ret;

  IRBuilderBase::FastMathFlagGuard Guard(B);
  B.setFastMathFlags(FastMathFlags::getFast());

  Intrinsic::ID ArgID = Arg->getIntrinsicID();
  LibFunc ArgLb = NotLibFunc;
  TLI->getLibFunc(*Arg, ArgLb);

  // log(pow(x,y)) -> y*log(x)
  if (ArgLb == PowLb || ArgID == Intrinsic::pow) {
    Value *LogX =
        Log->doesNotAccessMemory()
            ? B.CreateCall(Intrinsic::getDeclaration(Mod, LogID, Ty),
                           Arg->getOperand(0), "log")
            : emitUnaryFloatFnCall(Arg->getOperand(0), LogNm, B, Attrs);
    Value *MulY = B.CreateFMul(Arg->getArgOperand(1), LogX, "mul");
    // Since pow() may have side effects, e.g. errno,
    // dead code elimination may not be trusted to remove it.
    substituteInParent(Arg, MulY);
    return MulY;
  }

  // log(exp{,2,10}(y)) -> y*log({e,2,10})
  // TODO: There is no exp10() intrinsic yet.
  if (ArgLb == ExpLb || ArgLb == Exp2Lb || ArgLb == Exp10Lb ||
           ArgID == Intrinsic::exp || ArgID == Intrinsic::exp2) {
    Constant *Eul;
    if (ArgLb == ExpLb || ArgID == Intrinsic::exp)
      // FIXME: Add more precise value of e for long double.
      Eul = ConstantFP::get(Log->getType(), numbers::e);
    else if (ArgLb == Exp2Lb || ArgID == Intrinsic::exp2)
      Eul = ConstantFP::get(Log->getType(), 2.0);
    else
      Eul = ConstantFP::get(Log->getType(), 10.0);
    Value *LogE = Log->doesNotAccessMemory()
                      ? B.CreateCall(Intrinsic::getDeclaration(Mod, LogID, Ty),
                                     Eul, "log")
                      : emitUnaryFloatFnCall(Eul, LogNm, B, Attrs);
    Value *MulY = B.CreateFMul(Arg->getArgOperand(0), LogE, "mul");
    // Since exp() may have side effects, e.g. errno,
    // dead code elimination may not be trusted to remove it.
    substituteInParent(Arg, MulY);
    return MulY;
  }

  return Ret;
}

Value *LibCallSimplifier::optimizeSqrt(CallInst *CI, IRBuilderBase &B) {
  Function *Callee = CI->getCalledFunction();
  Value *Ret = nullptr;
  // TODO: Once we have a way (other than checking for the existince of the
  // libcall) to tell whether our target can lower @llvm.sqrt, relax the
  // condition below.
  if (TLI->has(LibFunc_sqrtf) && (Callee->getName() == "sqrt" ||
                                  Callee->getIntrinsicID() == Intrinsic::sqrt))
    Ret = optimizeUnaryDoubleFP(CI, B, true);

  if (!CI->isFast())
    return Ret;

  Instruction *I = dyn_cast<Instruction>(CI->getArgOperand(0));
  if (!I || I->getOpcode() != Instruction::FMul || !I->isFast())
    return Ret;

  // We're looking for a repeated factor in a multiplication tree,
  // so we can do this fold: sqrt(x * x) -> fabs(x);
  // or this fold: sqrt((x * x) * y) -> fabs(x) * sqrt(y).
  Value *Op0 = I->getOperand(0);
  Value *Op1 = I->getOperand(1);
  Value *RepeatOp = nullptr;
  Value *OtherOp = nullptr;
  if (Op0 == Op1) {
    // Simple match: the operands of the multiply are identical.
    RepeatOp = Op0;
  } else {
    // Look for a more complicated pattern: one of the operands is itself
    // a multiply, so search for a common factor in that multiply.
    // Note: We don't bother looking any deeper than this first level or for
    // variations of this pattern because instcombine's visitFMUL and/or the
    // reassociation pass should give us this form.
    Value *OtherMul0, *OtherMul1;
    if (match(Op0, m_FMul(m_Value(OtherMul0), m_Value(OtherMul1)))) {
      // Pattern: sqrt((x * y) * z)
      if (OtherMul0 == OtherMul1 && cast<Instruction>(Op0)->isFast()) {
        // Matched: sqrt((x * x) * z)
        RepeatOp = OtherMul0;
        OtherOp = Op1;
      }
    }
  }
  if (!RepeatOp)
    return Ret;

  // Fast math flags for any created instructions should match the sqrt
  // and multiply.
  IRBuilderBase::FastMathFlagGuard Guard(B);
  B.setFastMathFlags(I->getFastMathFlags());

  // If we found a repeated factor, hoist it out of the square root and
  // replace it with the fabs of that factor.
  Module *M = Callee->getParent();
  Type *ArgType = I->getType();
  Function *Fabs = Intrinsic::getDeclaration(M, Intrinsic::fabs, ArgType);
  Value *FabsCall = B.CreateCall(Fabs, RepeatOp, "fabs");
  if (OtherOp) {
    // If we found a non-repeated factor, we still need to get its square
    // root. We then multiply that by the value that was simplified out
    // of the square root calculation.
    Function *Sqrt = Intrinsic::getDeclaration(M, Intrinsic::sqrt, ArgType);
    Value *SqrtCall = B.CreateCall(Sqrt, OtherOp, "sqrt");
    return B.CreateFMul(FabsCall, SqrtCall);
  }
  return FabsCall;
}

// TODO: Generalize to handle any trig function and its inverse.
Value *LibCallSimplifier::optimizeTan(CallInst *CI, IRBuilderBase &B) {
  Function *Callee = CI->getCalledFunction();
  Value *Ret = nullptr;
  StringRef Name = Callee->getName();
  if (UnsafeFPShrink && Name == "tan" && hasFloatVersion(Name))
    Ret = optimizeUnaryDoubleFP(CI, B, true);

  Value *Op1 = CI->getArgOperand(0);
  auto *OpC = dyn_cast<CallInst>(Op1);
  if (!OpC)
    return Ret;

  // Both calls must be 'fast' in order to remove them.
  if (!CI->isFast() || !OpC->isFast())
    return Ret;

  // tan(atan(x)) -> x
  // tanf(atanf(x)) -> x
  // tanl(atanl(x)) -> x
  LibFunc Func;
  Function *F = OpC->getCalledFunction();
  if (F && TLI->getLibFunc(F->getName(), Func) && TLI->has(Func) &&
      ((Func == LibFunc_atan && Callee->getName() == "tan") ||
       (Func == LibFunc_atanf && Callee->getName() == "tanf") ||
       (Func == LibFunc_atanl && Callee->getName() == "tanl")))
    Ret = OpC->getArgOperand(0);
  return Ret;
}

static bool isTrigLibCall(CallInst *CI) {
  // We can only hope to do anything useful if we can ignore things like errno
  // and floating-point exceptions.
  // We already checked the prototype.
  return CI->hasFnAttr(Attribute::NoUnwind) &&
         CI->hasFnAttr(Attribute::ReadNone);
}

static void insertSinCosCall(IRBuilderBase &B, Function *OrigCallee, Value *Arg,
                             bool UseFloat, Value *&Sin, Value *&Cos,
                             Value *&SinCos) {
  Type *ArgTy = Arg->getType();
  Type *ResTy;
  StringRef Name;

  Triple T(OrigCallee->getParent()->getTargetTriple());
  if (UseFloat) {
    Name = "__sincospif_stret";

    assert(T.getArch() != Triple::x86 && "x86 messy and unsupported for now");
    // x86_64 can't use {float, float} since that would be returned in both
    // xmm0 and xmm1, which isn't what a real struct would do.
    ResTy = T.getArch() == Triple::x86_64
                ? static_cast<Type *>(FixedVectorType::get(ArgTy, 2))
                : static_cast<Type *>(StructType::get(ArgTy, ArgTy));
  } else {
    Name = "__sincospi_stret";
    ResTy = StructType::get(ArgTy, ArgTy);
  }

  Module *M = OrigCallee->getParent();
  FunctionCallee Callee =
      M->getOrInsertFunction(Name, OrigCallee->getAttributes(), ResTy, ArgTy);

  if (Instruction *ArgInst = dyn_cast<Instruction>(Arg)) {
    // If the argument is an instruction, it must dominate all uses so put our
    // sincos call there.
    B.SetInsertPoint(ArgInst->getParent(), ++ArgInst->getIterator());
  } else {
    // Otherwise (e.g. for a constant) the beginning of the function is as
    // good a place as any.
    BasicBlock &EntryBB = B.GetInsertBlock()->getParent()->getEntryBlock();
    B.SetInsertPoint(&EntryBB, EntryBB.begin());
  }

  SinCos = B.CreateCall(Callee, Arg, "sincospi");

  if (SinCos->getType()->isStructTy()) {
    Sin = B.CreateExtractValue(SinCos, 0, "sinpi");
    Cos = B.CreateExtractValue(SinCos, 1, "cospi");
  } else {
    Sin = B.CreateExtractElement(SinCos, ConstantInt::get(B.getInt32Ty(), 0),
                                 "sinpi");
    Cos = B.CreateExtractElement(SinCos, ConstantInt::get(B.getInt32Ty(), 1),
                                 "cospi");
  }
}

Value *LibCallSimplifier::optimizeSinCosPi(CallInst *CI, IRBuilderBase &B) {
  // Make sure the prototype is as expected, otherwise the rest of the
  // function is probably invalid and likely to abort.
  if (!isTrigLibCall(CI))
    return nullptr;

  Value *Arg = CI->getArgOperand(0);
  SmallVector<CallInst *, 1> SinCalls;
  SmallVector<CallInst *, 1> CosCalls;
  SmallVector<CallInst *, 1> SinCosCalls;

  bool IsFloat = Arg->getType()->isFloatTy();

  // Look for all compatible sinpi, cospi and sincospi calls with the same
  // argument. If there are enough (in some sense) we can make the
  // substitution.
  Function *F = CI->getFunction();
  for (User *U : Arg->users())
    classifyArgUse(U, F, IsFloat, SinCalls, CosCalls, SinCosCalls);

  // It's only worthwhile if both sinpi and cospi are actually used.
  if (SinCosCalls.empty() && (SinCalls.empty() || CosCalls.empty()))
    return nullptr;

  Value *Sin, *Cos, *SinCos;
  insertSinCosCall(B, CI->getCalledFunction(), Arg, IsFloat, Sin, Cos, SinCos);

  auto replaceTrigInsts = [this](SmallVectorImpl<CallInst *> &Calls,
                                 Value *Res) {
    for (CallInst *C : Calls)
      replaceAllUsesWith(C, Res);
  };

  replaceTrigInsts(SinCalls, Sin);
  replaceTrigInsts(CosCalls, Cos);
  replaceTrigInsts(SinCosCalls, SinCos);

  return nullptr;
}

void LibCallSimplifier::classifyArgUse(
    Value *Val, Function *F, bool IsFloat,
    SmallVectorImpl<CallInst *> &SinCalls,
    SmallVectorImpl<CallInst *> &CosCalls,
    SmallVectorImpl<CallInst *> &SinCosCalls) {
  CallInst *CI = dyn_cast<CallInst>(Val);

  if (!CI)
    return;

  // Don't consider calls in other functions.
  if (CI->getFunction() != F)
    return;

  Function *Callee = CI->getCalledFunction();
  LibFunc Func;
  if (!Callee || !TLI->getLibFunc(*Callee, Func) || !TLI->has(Func) ||
      !isTrigLibCall(CI))
    return;

  if (IsFloat) {
    if (Func == LibFunc_sinpif)
      SinCalls.push_back(CI);
    else if (Func == LibFunc_cospif)
      CosCalls.push_back(CI);
    else if (Func == LibFunc_sincospif_stret)
      SinCosCalls.push_back(CI);
  } else {
    if (Func == LibFunc_sinpi)
      SinCalls.push_back(CI);
    else if (Func == LibFunc_cospi)
      CosCalls.push_back(CI);
    else if (Func == LibFunc_sincospi_stret)
      SinCosCalls.push_back(CI);
  }
}

//===----------------------------------------------------------------------===//
// Integer Library Call Optimizations
//===----------------------------------------------------------------------===//

Value *LibCallSimplifier::optimizeFFS(CallInst *CI, IRBuilderBase &B) {
  // ffs(x) -> x != 0 ? (i32)llvm.cttz(x)+1 : 0
  Value *Op = CI->getArgOperand(0);
  Type *ArgType = Op->getType();
  Function *F = Intrinsic::getDeclaration(CI->getCalledFunction()->getParent(),
                                          Intrinsic::cttz, ArgType);
  Value *V = B.CreateCall(F, {Op, B.getTrue()}, "cttz");
  V = B.CreateAdd(V, ConstantInt::get(V->getType(), 1));
  V = B.CreateIntCast(V, B.getInt32Ty(), false);

  Value *Cond = B.CreateICmpNE(Op, Constant::getNullValue(ArgType));
  return B.CreateSelect(Cond, V, B.getInt32(0));
}

Value *LibCallSimplifier::optimizeFls(CallInst *CI, IRBuilderBase &B) {
  // fls(x) -> (i32)(sizeInBits(x) - llvm.ctlz(x, false))
  Value *Op = CI->getArgOperand(0);
  Type *ArgType = Op->getType();
  Function *F = Intrinsic::getDeclaration(CI->getCalledFunction()->getParent(),
                                          Intrinsic::ctlz, ArgType);
  Value *V = B.CreateCall(F, {Op, B.getFalse()}, "ctlz");
  V = B.CreateSub(ConstantInt::get(V->getType(), ArgType->getIntegerBitWidth()),
                  V);
  return B.CreateIntCast(V, CI->getType(), false);
}

Value *LibCallSimplifier::optimizeAbs(CallInst *CI, IRBuilderBase &B) {
  // abs(x) -> x <s 0 ? -x : x
  // The negation has 'nsw' because abs of INT_MIN is undefined.
  Value *X = CI->getArgOperand(0);
  Value *IsNeg = B.CreateICmpSLT(X, Constant::getNullValue(X->getType()));
  Value *NegX = B.CreateNSWNeg(X, "neg");
  return B.CreateSelect(IsNeg, NegX, X);
}

Value *LibCallSimplifier::optimizeIsDigit(CallInst *CI, IRBuilderBase &B) {
  // isdigit(c) -> (c-'0') <u 10
  Value *Op = CI->getArgOperand(0);
  Op = B.CreateSub(Op, B.getInt32('0'), "isdigittmp");
  Op = B.CreateICmpULT(Op, B.getInt32(10), "isdigit");
  return B.CreateZExt(Op, CI->getType());
}

Value *LibCallSimplifier::optimizeIsAscii(CallInst *CI, IRBuilderBase &B) {
  // isascii(c) -> c <u 128
  Value *Op = CI->getArgOperand(0);
  Op = B.CreateICmpULT(Op, B.getInt32(128), "isascii");
  return B.CreateZExt(Op, CI->getType());
}

Value *LibCallSimplifier::optimizeToAscii(CallInst *CI, IRBuilderBase &B) {
  // toascii(c) -> c & 0x7f
  return B.CreateAnd(CI->getArgOperand(0),
                     ConstantInt::get(CI->getType(), 0x7F));
}

Value *LibCallSimplifier::optimizeAtoi(CallInst *CI, IRBuilderBase &B) {
  StringRef Str;
  if (!getConstantStringInfo(CI->getArgOperand(0), Str))
    return nullptr;

  return convertStrToNumber(CI, Str, 10);
}

Value *LibCallSimplifier::optimizeStrtol(CallInst *CI, IRBuilderBase &B) {
  StringRef Str;
  if (!getConstantStringInfo(CI->getArgOperand(0), Str))
    return nullptr;

  if (!isa<ConstantPointerNull>(CI->getArgOperand(1)))
    return nullptr;

  if (ConstantInt *CInt = dyn_cast<ConstantInt>(CI->getArgOperand(2))) {
    return convertStrToNumber(CI, Str, CInt->getSExtValue());
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Formatting and IO Library Call Optimizations
//===----------------------------------------------------------------------===//

static bool isReportingError(Function *Callee, CallInst *CI, int StreamArg);

Value *LibCallSimplifier::optimizeErrorReporting(CallInst *CI, IRBuilderBase &B,
                                                 int StreamArg) {
  Function *Callee = CI->getCalledFunction();
  // Error reporting calls should be cold, mark them as such.
  // This applies even to non-builtin calls: it is only a hint and applies to
  // functions that the frontend might not understand as builtins.

  // This heuristic was suggested in:
  // Improving Static Branch Prediction in a Compiler
  // Brian L. Deitrich, Ben-Chung Cheng, Wen-mei W. Hwu
  // Proceedings of PACT'98, Oct. 1998, IEEE
  if (!CI->hasFnAttr(Attribute::Cold) &&
      isReportingError(Callee, CI, StreamArg)) {
    CI->addAttribute(AttributeList::FunctionIndex, Attribute::Cold);
  }

  return nullptr;
}

static bool isReportingError(Function *Callee, CallInst *CI, int StreamArg) {
  if (!Callee || !Callee->isDeclaration())
    return false;

  if (StreamArg < 0)
    return true;

  // These functions might be considered cold, but only if their stream
  // argument is stderr.

  if (StreamArg >= (int)CI->getNumArgOperands())
    return false;
  LoadInst *LI = dyn_cast<LoadInst>(CI->getArgOperand(StreamArg));
  if (!LI)
    return false;
  GlobalVariable *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand());
  if (!GV || !GV->isDeclaration())
    return false;
  return GV->getName() == "stderr";
}

Value *LibCallSimplifier::optimizePrintFString(CallInst *CI, IRBuilderBase &B) {
  // Check for a fixed format string.
  StringRef FormatStr;
  if (!getConstantStringInfo(CI->getArgOperand(0), FormatStr))
    return nullptr;

  // Empty format string -> noop.
  if (FormatStr.empty()) // Tolerate printf's declared void.
    return CI->use_empty() ? (Value *)CI : ConstantInt::get(CI->getType(), 0);

  // Do not do any of the following transformations if the printf return value
  // is used, in general the printf return value is not compatible with either
  // putchar() or puts().
  if (!CI->use_empty())
    return nullptr;

  // printf("x") -> putchar('x'), even for "%" and "%%".
  if (FormatStr.size() == 1 || FormatStr == "%%")
    return emitPutChar(B.getInt32(FormatStr[0]), B, TLI);

  // printf("%s", "a") --> putchar('a')
  if (FormatStr == "%s" && CI->getNumArgOperands() > 1) {
    StringRef ChrStr;
    if (!getConstantStringInfo(CI->getOperand(1), ChrStr))
      return nullptr;
    if (ChrStr.size() != 1)
      return nullptr;
    return emitPutChar(B.getInt32(ChrStr[0]), B, TLI);
  }

  // printf("foo\n") --> puts("foo")
  if (FormatStr[FormatStr.size() - 1] == '\n' &&
      FormatStr.find('%') == StringRef::npos) { // No format characters.
    // Create a string literal with no \n on it.  We expect the constant merge
    // pass to be run after this pass, to merge duplicate strings.
    FormatStr = FormatStr.drop_back();
    Value *GV = B.CreateGlobalString(FormatStr, "str");
    return emitPutS(GV, B, TLI);
  }

  // Optimize specific format strings.
  // printf("%c", chr) --> putchar(chr)
  if (FormatStr == "%c" && CI->getNumArgOperands() > 1 &&
      CI->getArgOperand(1)->getType()->isIntegerTy())
    return emitPutChar(CI->getArgOperand(1), B, TLI);

  // printf("%s\n", str) --> puts(str)
  if (FormatStr == "%s\n" && CI->getNumArgOperands() > 1 &&
      CI->getArgOperand(1)->getType()->isPointerTy())
    return emitPutS(CI->getArgOperand(1), B, TLI);
  return nullptr;
}

Value *LibCallSimplifier::optimizePrintF(CallInst *CI, IRBuilderBase &B) {

  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (Value *V = optimizePrintFString(CI, B)) {
    return V;
  }

  // printf(format, ...) -> iprintf(format, ...) if no floating point
  // arguments.
  if (TLI->has(LibFunc_iprintf) && !callHasFloatingPointArgument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    FunctionCallee IPrintFFn =
        M->getOrInsertFunction("iprintf", FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(IPrintFFn);
    B.Insert(New);
    return New;
  }

  // printf(format, ...) -> __small_printf(format, ...) if no 128-bit floating point
  // arguments.
  if (TLI->has(LibFunc_small_printf) && !callHasFP128Argument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    auto SmallPrintFFn =
        M->getOrInsertFunction(TLI->getName(LibFunc_small_printf),
                               FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(SmallPrintFFn);
    B.Insert(New);
    return New;
  }

  annotateNonNullBasedOnAccess(CI, 0);
  return nullptr;
}

Value *LibCallSimplifier::optimizeSPrintFString(CallInst *CI,
                                                IRBuilderBase &B) {
  // Check for a fixed format string.
  StringRef FormatStr;
  if (!getConstantStringInfo(CI->getArgOperand(1), FormatStr))
    return nullptr;

  // If we just have a format string (nothing else crazy) transform it.
  if (CI->getNumArgOperands() == 2) {
    // Make sure there's no % in the constant array.  We could try to handle
    // %% -> % in the future if we cared.
    if (FormatStr.find('%') != StringRef::npos)
      return nullptr; // we found a format specifier, bail out.

    // sprintf(str, fmt) -> llvm.memcpy(align 1 str, align 1 fmt, strlen(fmt)+1)
    B.CreateMemCpy(
        CI->getArgOperand(0), Align(1), CI->getArgOperand(1), Align(1),
        ConstantInt::get(DL.getIntPtrType(CI->getContext()),
                         FormatStr.size() + 1)); // Copy the null byte.
    return ConstantInt::get(CI->getType(), FormatStr.size());
  }

  // The remaining optimizations require the format string to be "%s" or "%c"
  // and have an extra operand.
  if (FormatStr.size() != 2 || FormatStr[0] != '%' ||
      CI->getNumArgOperands() < 3)
    return nullptr;

  // Decode the second character of the format string.
  if (FormatStr[1] == 'c') {
    // sprintf(dst, "%c", chr) --> *(i8*)dst = chr; *((i8*)dst+1) = 0
    if (!CI->getArgOperand(2)->getType()->isIntegerTy())
      return nullptr;
    Value *V = B.CreateTrunc(CI->getArgOperand(2), B.getInt8Ty(), "char");
    Value *Ptr = castToCStr(CI->getArgOperand(0), B);
    B.CreateStore(V, Ptr);
    Ptr = B.CreateGEP(B.getInt8Ty(), Ptr, B.getInt32(1), "nul");
    B.CreateStore(B.getInt8(0), Ptr);

    return ConstantInt::get(CI->getType(), 1);
  }

  if (FormatStr[1] == 's') {
    // sprintf(dest, "%s", str) -> llvm.memcpy(align 1 dest, align 1 str,
    // strlen(str)+1)
    if (!CI->getArgOperand(2)->getType()->isPointerTy())
      return nullptr;

    if (CI->use_empty())
      // sprintf(dest, "%s", str) -> strcpy(dest, str)
      return emitStrCpy(CI->getArgOperand(0), CI->getArgOperand(2), B, TLI);

    uint64_t SrcLen = GetStringLength(CI->getArgOperand(2));
    if (SrcLen) {
      B.CreateMemCpy(
          CI->getArgOperand(0), Align(1), CI->getArgOperand(2), Align(1),
          ConstantInt::get(DL.getIntPtrType(CI->getContext()), SrcLen));
      // Returns total number of characters written without null-character.
      return ConstantInt::get(CI->getType(), SrcLen - 1);
    } else if (Value *V = emitStpCpy(CI->getArgOperand(0), CI->getArgOperand(2),
                                     B, TLI)) {
      // sprintf(dest, "%s", str) -> stpcpy(dest, str) - dest
      Value *PtrDiff = B.CreatePtrDiff(V, CI->getArgOperand(0));
      return B.CreateIntCast(PtrDiff, CI->getType(), false);
    }

    bool OptForSize = CI->getFunction()->hasOptSize() ||
                      llvm::shouldOptimizeForSize(CI->getParent(), PSI, BFI,
                                                  PGSOQueryType::IRPass);
    if (OptForSize)
      return nullptr;

    Value *Len = emitStrLen(CI->getArgOperand(2), B, DL, TLI);
    if (!Len)
      return nullptr;
    Value *IncLen =
        B.CreateAdd(Len, ConstantInt::get(Len->getType(), 1), "leninc");
    B.CreateMemCpy(CI->getArgOperand(0), Align(1), CI->getArgOperand(2),
                   Align(1), IncLen);

    // The sprintf result is the unincremented number of bytes in the string.
    return B.CreateIntCast(Len, CI->getType(), false);
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeSPrintF(CallInst *CI, IRBuilderBase &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (Value *V = optimizeSPrintFString(CI, B)) {
    return V;
  }

  // sprintf(str, format, ...) -> siprintf(str, format, ...) if no floating
  // point arguments.
  if (TLI->has(LibFunc_siprintf) && !callHasFloatingPointArgument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    FunctionCallee SIPrintFFn =
        M->getOrInsertFunction("siprintf", FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(SIPrintFFn);
    B.Insert(New);
    return New;
  }

  // sprintf(str, format, ...) -> __small_sprintf(str, format, ...) if no 128-bit
  // floating point arguments.
  if (TLI->has(LibFunc_small_sprintf) && !callHasFP128Argument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    auto SmallSPrintFFn =
        M->getOrInsertFunction(TLI->getName(LibFunc_small_sprintf),
                               FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(SmallSPrintFFn);
    B.Insert(New);
    return New;
  }

  annotateNonNullBasedOnAccess(CI, {0, 1});
  return nullptr;
}

Value *LibCallSimplifier::optimizeSnPrintFString(CallInst *CI,
                                                 IRBuilderBase &B) {
  // Check for size
  ConstantInt *Size = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!Size)
    return nullptr;

  uint64_t N = Size->getZExtValue();
  // Check for a fixed format string.
  StringRef FormatStr;
  if (!getConstantStringInfo(CI->getArgOperand(2), FormatStr))
    return nullptr;

  // If we just have a format string (nothing else crazy) transform it.
  if (CI->getNumArgOperands() == 3) {
    // Make sure there's no % in the constant array.  We could try to handle
    // %% -> % in the future if we cared.
    if (FormatStr.find('%') != StringRef::npos)
      return nullptr; // we found a format specifier, bail out.

    if (N == 0)
      return ConstantInt::get(CI->getType(), FormatStr.size());
    else if (N < FormatStr.size() + 1)
      return nullptr;

    // snprintf(dst, size, fmt) -> llvm.memcpy(align 1 dst, align 1 fmt,
    // strlen(fmt)+1)
    B.CreateMemCpy(
        CI->getArgOperand(0), Align(1), CI->getArgOperand(2), Align(1),
        ConstantInt::get(DL.getIntPtrType(CI->getContext()),
                         FormatStr.size() + 1)); // Copy the null byte.
    return ConstantInt::get(CI->getType(), FormatStr.size());
  }

  // The remaining optimizations require the format string to be "%s" or "%c"
  // and have an extra operand.
  if (FormatStr.size() == 2 && FormatStr[0] == '%' &&
      CI->getNumArgOperands() == 4) {

    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      if (N == 0)
        return ConstantInt::get(CI->getType(), 1);
      else if (N == 1)
        return nullptr;

      // snprintf(dst, size, "%c", chr) --> *(i8*)dst = chr; *((i8*)dst+1) = 0
      if (!CI->getArgOperand(3)->getType()->isIntegerTy())
        return nullptr;
      Value *V = B.CreateTrunc(CI->getArgOperand(3), B.getInt8Ty(), "char");
      Value *Ptr = castToCStr(CI->getArgOperand(0), B);
      B.CreateStore(V, Ptr);
      Ptr = B.CreateGEP(B.getInt8Ty(), Ptr, B.getInt32(1), "nul");
      B.CreateStore(B.getInt8(0), Ptr);

      return ConstantInt::get(CI->getType(), 1);
    }

    if (FormatStr[1] == 's') {
      // snprintf(dest, size, "%s", str) to llvm.memcpy(dest, str, len+1, 1)
      StringRef Str;
      if (!getConstantStringInfo(CI->getArgOperand(3), Str))
        return nullptr;

      if (N == 0)
        return ConstantInt::get(CI->getType(), Str.size());
      else if (N < Str.size() + 1)
        return nullptr;

      B.CreateMemCpy(CI->getArgOperand(0), Align(1), CI->getArgOperand(3),
                     Align(1), ConstantInt::get(CI->getType(), Str.size() + 1));

      // The snprintf result is the unincremented number of bytes in the string.
      return ConstantInt::get(CI->getType(), Str.size());
    }
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeSnPrintF(CallInst *CI, IRBuilderBase &B) {
  if (Value *V = optimizeSnPrintFString(CI, B)) {
    return V;
  }

  if (isKnownNonZero(CI->getOperand(1), DL))
    annotateNonNullBasedOnAccess(CI, 0);
  return nullptr;
}

Value *LibCallSimplifier::optimizeFPrintFString(CallInst *CI,
                                                IRBuilderBase &B) {
  optimizeErrorReporting(CI, B, 0);

  // All the optimizations depend on the format string.
  StringRef FormatStr;
  if (!getConstantStringInfo(CI->getArgOperand(1), FormatStr))
    return nullptr;

  // Do not do any of the following transformations if the fprintf return
  // value is used, in general the fprintf return value is not compatible
  // with fwrite(), fputc() or fputs().
  if (!CI->use_empty())
    return nullptr;

  // fprintf(F, "foo") --> fwrite("foo", 3, 1, F)
  if (CI->getNumArgOperands() == 2) {
    // Could handle %% -> % if we cared.
    if (FormatStr.find('%') != StringRef::npos)
      return nullptr; // We found a format specifier.

    return emitFWrite(
        CI->getArgOperand(1),
        ConstantInt::get(DL.getIntPtrType(CI->getContext()), FormatStr.size()),
        CI->getArgOperand(0), B, DL, TLI);
  }

  // The remaining optimizations require the format string to be "%s" or "%c"
  // and have an extra operand.
  if (FormatStr.size() != 2 || FormatStr[0] != '%' ||
      CI->getNumArgOperands() < 3)
    return nullptr;

  // Decode the second character of the format string.
  if (FormatStr[1] == 'c') {
    // fprintf(F, "%c", chr) --> fputc(chr, F)
    if (!CI->getArgOperand(2)->getType()->isIntegerTy())
      return nullptr;
    return emitFPutC(CI->getArgOperand(2), CI->getArgOperand(0), B, TLI);
  }

  if (FormatStr[1] == 's') {
    // fprintf(F, "%s", str) --> fputs(str, F)
    if (!CI->getArgOperand(2)->getType()->isPointerTy())
      return nullptr;
    return emitFPutS(CI->getArgOperand(2), CI->getArgOperand(0), B, TLI);
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeFPrintF(CallInst *CI, IRBuilderBase &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (Value *V = optimizeFPrintFString(CI, B)) {
    return V;
  }

  // fprintf(stream, format, ...) -> fiprintf(stream, format, ...) if no
  // floating point arguments.
  if (TLI->has(LibFunc_fiprintf) && !callHasFloatingPointArgument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    FunctionCallee FIPrintFFn =
        M->getOrInsertFunction("fiprintf", FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(FIPrintFFn);
    B.Insert(New);
    return New;
  }

  // fprintf(stream, format, ...) -> __small_fprintf(stream, format, ...) if no
  // 128-bit floating point arguments.
  if (TLI->has(LibFunc_small_fprintf) && !callHasFP128Argument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    auto SmallFPrintFFn =
        M->getOrInsertFunction(TLI->getName(LibFunc_small_fprintf),
                               FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(SmallFPrintFFn);
    B.Insert(New);
    return New;
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeFWrite(CallInst *CI, IRBuilderBase &B) {
  optimizeErrorReporting(CI, B, 3);

  // Get the element size and count.
  ConstantInt *SizeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  ConstantInt *CountC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  if (SizeC && CountC) {
    uint64_t Bytes = SizeC->getZExtValue() * CountC->getZExtValue();

    // If this is writing zero records, remove the call (it's a noop).
    if (Bytes == 0)
      return ConstantInt::get(CI->getType(), 0);

    // If this is writing one byte, turn it into fputc.
    // This optimisation is only valid, if the return value is unused.
    if (Bytes == 1 && CI->use_empty()) { // fwrite(S,1,1,F) -> fputc(S[0],F)
      Value *Char = B.CreateLoad(B.getInt8Ty(),
                                 castToCStr(CI->getArgOperand(0), B), "char");
      Value *NewCI = emitFPutC(Char, CI->getArgOperand(3), B, TLI);
      return NewCI ? ConstantInt::get(CI->getType(), 1) : nullptr;
    }
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeFPuts(CallInst *CI, IRBuilderBase &B) {
  optimizeErrorReporting(CI, B, 1);

  // Don't rewrite fputs to fwrite when optimising for size because fwrite
  // requires more arguments and thus extra MOVs are required.
  bool OptForSize = CI->getFunction()->hasOptSize() ||
                    llvm::shouldOptimizeForSize(CI->getParent(), PSI, BFI,
                                                PGSOQueryType::IRPass);
  if (OptForSize)
    return nullptr;

  // We can't optimize if return value is used.
  if (!CI->use_empty())
    return nullptr;

  // fputs(s,F) --> fwrite(s,strlen(s),1,F)
  uint64_t Len = GetStringLength(CI->getArgOperand(0));
  if (!Len)
    return nullptr;

  // Known to have no uses (see above).
  return emitFWrite(
      CI->getArgOperand(0),
      ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len - 1),
      CI->getArgOperand(1), B, DL, TLI);
}

Value *LibCallSimplifier::optimizePuts(CallInst *CI, IRBuilderBase &B) {
  annotateNonNullBasedOnAccess(CI, 0);
  if (!CI->use_empty())
    return nullptr;

  // Check for a constant string.
  // puts("") -> putchar('\n')
  StringRef Str;
  if (getConstantStringInfo(CI->getArgOperand(0), Str) && Str.empty())
    return emitPutChar(B.getInt32('\n'), B, TLI);

  return nullptr;
}

Value *LibCallSimplifier::optimizeBCopy(CallInst *CI, IRBuilderBase &B) {
  // bcopy(src, dst, n) -> llvm.memmove(dst, src, n)
  return B.CreateMemMove(CI->getArgOperand(1), Align(1), CI->getArgOperand(0),
                         Align(1), CI->getArgOperand(2));
}

bool LibCallSimplifier::hasFloatVersion(StringRef FuncName) {
  LibFunc Func;
  SmallString<20> FloatFuncName = FuncName;
  FloatFuncName += 'f';
  if (TLI->getLibFunc(FloatFuncName, Func))
    return TLI->has(Func);
  return false;
}

Value *LibCallSimplifier::optimizeStringMemoryLibCall(CallInst *CI,
                                                      IRBuilderBase &Builder) {
  LibFunc Func;
  Function *Callee = CI->getCalledFunction();
  // Check for string/memory library functions.
  if (TLI->getLibFunc(*Callee, Func) && TLI->has(Func)) {
    // Make sure we never change the calling convention.
    assert((ignoreCallingConv(Func) ||
            isCallingConvCCompatible(CI)) &&
      "Optimizing string/memory libcall would change the calling convention");
    switch (Func) {
    case LibFunc_strcat:
      return optimizeStrCat(CI, Builder);
    case LibFunc_strncat:
      return optimizeStrNCat(CI, Builder);
    case LibFunc_strchr:
      return optimizeStrChr(CI, Builder);
    case LibFunc_strrchr:
      return optimizeStrRChr(CI, Builder);
    case LibFunc_strcmp:
      return optimizeStrCmp(CI, Builder);
    case LibFunc_strncmp:
      return optimizeStrNCmp(CI, Builder);
    case LibFunc_strcpy:
      return optimizeStrCpy(CI, Builder);
    case LibFunc_stpcpy:
      return optimizeStpCpy(CI, Builder);
    case LibFunc_strncpy:
      return optimizeStrNCpy(CI, Builder);
    case LibFunc_strlen:
      return optimizeStrLen(CI, Builder);
    case LibFunc_strpbrk:
      return optimizeStrPBrk(CI, Builder);
    case LibFunc_strndup:
      return optimizeStrNDup(CI, Builder);
    case LibFunc_strtol:
    case LibFunc_strtod:
    case LibFunc_strtof:
    case LibFunc_strtoul:
    case LibFunc_strtoll:
    case LibFunc_strtold:
    case LibFunc_strtoull:
      return optimizeStrTo(CI, Builder);
    case LibFunc_strspn:
      return optimizeStrSpn(CI, Builder);
    case LibFunc_strcspn:
      return optimizeStrCSpn(CI, Builder);
    case LibFunc_strstr:
      return optimizeStrStr(CI, Builder);
    case LibFunc_memchr:
      return optimizeMemChr(CI, Builder);
    case LibFunc_memrchr:
      return optimizeMemRChr(CI, Builder);
    case LibFunc_bcmp:
      return optimizeBCmp(CI, Builder);
    case LibFunc_memcmp:
      return optimizeMemCmp(CI, Builder);
    case LibFunc_memcpy:
      return optimizeMemCpy(CI, Builder);
    case LibFunc_memccpy:
      return optimizeMemCCpy(CI, Builder);
    case LibFunc_mempcpy:
      return optimizeMemPCpy(CI, Builder);
    case LibFunc_memmove:
      return optimizeMemMove(CI, Builder);
    case LibFunc_memset:
      return optimizeMemSet(CI, Builder);
    case LibFunc_realloc:
      return optimizeRealloc(CI, Builder);
    case LibFunc_wcslen:
      return optimizeWcslen(CI, Builder);
    case LibFunc_bcopy:
      return optimizeBCopy(CI, Builder);
    default:
      break;
    }
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeFloatingPointLibCall(CallInst *CI,
                                                       LibFunc Func,
                                                       IRBuilderBase &Builder) {
  // Don't optimize calls that require strict floating point semantics.
  if (CI->isStrictFP())
    return nullptr;

  if (Value *V = optimizeTrigReflections(CI, Func, Builder))
    return V;

  switch (Func) {
  case LibFunc_sinpif:
  case LibFunc_sinpi:
  case LibFunc_cospif:
  case LibFunc_cospi:
    return optimizeSinCosPi(CI, Builder);
  case LibFunc_powf:
  case LibFunc_pow:
  case LibFunc_powl:
    return optimizePow(CI, Builder);
  case LibFunc_exp2l:
  case LibFunc_exp2:
  case LibFunc_exp2f:
    return optimizeExp2(CI, Builder);
  case LibFunc_fabsf:
  case LibFunc_fabs:
  case LibFunc_fabsl:
    return replaceUnaryCall(CI, Builder, Intrinsic::fabs);
  case LibFunc_sqrtf:
  case LibFunc_sqrt:
  case LibFunc_sqrtl:
    return optimizeSqrt(CI, Builder);
  case LibFunc_logf:
  case LibFunc_log:
  case LibFunc_logl:
  case LibFunc_log10f:
  case LibFunc_log10:
  case LibFunc_log10l:
  case LibFunc_log1pf:
  case LibFunc_log1p:
  case LibFunc_log1pl:
  case LibFunc_log2f:
  case LibFunc_log2:
  case LibFunc_log2l:
  case LibFunc_logbf:
  case LibFunc_logb:
  case LibFunc_logbl:
    return optimizeLog(CI, Builder);
  case LibFunc_tan:
  case LibFunc_tanf:
  case LibFunc_tanl:
    return optimizeTan(CI, Builder);
  case LibFunc_ceil:
    return replaceUnaryCall(CI, Builder, Intrinsic::ceil);
  case LibFunc_floor:
    return replaceUnaryCall(CI, Builder, Intrinsic::floor);
  case LibFunc_round:
    return replaceUnaryCall(CI, Builder, Intrinsic::round);
  case LibFunc_roundeven:
    return replaceUnaryCall(CI, Builder, Intrinsic::roundeven);
  case LibFunc_nearbyint:
    return replaceUnaryCall(CI, Builder, Intrinsic::nearbyint);
  case LibFunc_rint:
    return replaceUnaryCall(CI, Builder, Intrinsic::rint);
  case LibFunc_trunc:
    return replaceUnaryCall(CI, Builder, Intrinsic::trunc);
  case LibFunc_acos:
  case LibFunc_acosh:
  case LibFunc_asin:
  case LibFunc_asinh:
  case LibFunc_atan:
  case LibFunc_atanh:
  case LibFunc_cbrt:
  case LibFunc_cosh:
  case LibFunc_exp:
  case LibFunc_exp10:
  case LibFunc_expm1:
  case LibFunc_cos:
  case LibFunc_sin:
  case LibFunc_sinh:
  case LibFunc_tanh:
    if (UnsafeFPShrink && hasFloatVersion(CI->getCalledFunction()->getName()))
      return optimizeUnaryDoubleFP(CI, Builder, true);
    return nullptr;
  case LibFunc_copysign:
    if (hasFloatVersion(CI->getCalledFunction()->getName()))
      return optimizeBinaryDoubleFP(CI, Builder);
    return nullptr;
  case LibFunc_fminf:
  case LibFunc_fmin:
  case LibFunc_fminl:
  case LibFunc_fmaxf:
  case LibFunc_fmax:
  case LibFunc_fmaxl:
    return optimizeFMinFMax(CI, Builder);
  case LibFunc_cabs:
  case LibFunc_cabsf:
  case LibFunc_cabsl:
    return optimizeCAbs(CI, Builder);
  default:
    return nullptr;
  }
}

Value *LibCallSimplifier::optimizeCall(CallInst *CI, IRBuilderBase &Builder) {
  // TODO: Split out the code below that operates on FP calls so that
  //       we can all non-FP calls with the StrictFP attribute to be
  //       optimized.
  if (CI->isNoBuiltin())
    return nullptr;

  LibFunc Func;
  Function *Callee = CI->getCalledFunction();
  bool isCallingConvC = isCallingConvCCompatible(CI);

  SmallVector<OperandBundleDef, 2> OpBundles;
  CI->getOperandBundlesAsDefs(OpBundles);

  IRBuilderBase::OperandBundlesGuard Guard(Builder);
  Builder.setDefaultOperandBundles(OpBundles);

  // Command-line parameter overrides instruction attribute.
  // This can't be moved to optimizeFloatingPointLibCall() because it may be
  // used by the intrinsic optimizations.
  if (EnableUnsafeFPShrink.getNumOccurrences() > 0)
    UnsafeFPShrink = EnableUnsafeFPShrink;
  else if (isa<FPMathOperator>(CI) && CI->isFast())
    UnsafeFPShrink = true;

  // First, check for intrinsics.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
    if (!isCallingConvC)
      return nullptr;
    // The FP intrinsics have corresponding constrained versions so we don't
    // need to check for the StrictFP attribute here.
    switch (II->getIntrinsicID()) {
    case Intrinsic::pow:
      return optimizePow(CI, Builder);
    case Intrinsic::exp2:
      return optimizeExp2(CI, Builder);
    case Intrinsic::log:
    case Intrinsic::log2:
    case Intrinsic::log10:
      return optimizeLog(CI, Builder);
    case Intrinsic::sqrt:
      return optimizeSqrt(CI, Builder);
    // TODO: Use foldMallocMemset() with memset intrinsic.
    case Intrinsic::memset:
      return optimizeMemSet(CI, Builder);
    case Intrinsic::memcpy:
      return optimizeMemCpy(CI, Builder);
    case Intrinsic::memmove:
      return optimizeMemMove(CI, Builder);
    default:
      return nullptr;
    }
  }

  // Also try to simplify calls to fortified library functions.
  if (Value *SimplifiedFortifiedCI =
          FortifiedSimplifier.optimizeCall(CI, Builder)) {
    // Try to further simplify the result.
    CallInst *SimplifiedCI = dyn_cast<CallInst>(SimplifiedFortifiedCI);
    if (SimplifiedCI && SimplifiedCI->getCalledFunction()) {
      // Ensure that SimplifiedCI's uses are complete, since some calls have
      // their uses analyzed.
      replaceAllUsesWith(CI, SimplifiedCI);

      // Set insertion point to SimplifiedCI to guarantee we reach all uses
      // we might replace later on.
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(SimplifiedCI);
      if (Value *V = optimizeStringMemoryLibCall(SimplifiedCI, Builder)) {
        // If we were able to further simplify, remove the now redundant call.
        substituteInParent(SimplifiedCI, V);
        return V;
      }
    }
    return SimplifiedFortifiedCI;
  }

  // Then check for known library functions.
  if (TLI->getLibFunc(*Callee, Func) && TLI->has(Func)) {
    // We never change the calling convention.
    if (!ignoreCallingConv(Func) && !isCallingConvC)
      return nullptr;
    if (Value *V = optimizeStringMemoryLibCall(CI, Builder))
      return V;
    if (Value *V = optimizeFloatingPointLibCall(CI, Func, Builder))
      return V;
    switch (Func) {
    case LibFunc_ffs:
    case LibFunc_ffsl:
    case LibFunc_ffsll:
      return optimizeFFS(CI, Builder);
    case LibFunc_fls:
    case LibFunc_flsl:
    case LibFunc_flsll:
      return optimizeFls(CI, Builder);
    case LibFunc_abs:
    case LibFunc_labs:
    case LibFunc_llabs:
      return optimizeAbs(CI, Builder);
    case LibFunc_isdigit:
      return optimizeIsDigit(CI, Builder);
    case LibFunc_isascii:
      return optimizeIsAscii(CI, Builder);
    case LibFunc_toascii:
      return optimizeToAscii(CI, Builder);
    case LibFunc_atoi:
    case LibFunc_atol:
    case LibFunc_atoll:
      return optimizeAtoi(CI, Builder);
    case LibFunc_strtol:
    case LibFunc_strtoll:
      return optimizeStrtol(CI, Builder);
    case LibFunc_printf:
      return optimizePrintF(CI, Builder);
    case LibFunc_sprintf:
      return optimizeSPrintF(CI, Builder);
    case LibFunc_snprintf:
      return optimizeSnPrintF(CI, Builder);
    case LibFunc_fprintf:
      return optimizeFPrintF(CI, Builder);
    case LibFunc_fwrite:
      return optimizeFWrite(CI, Builder);
    case LibFunc_fputs:
      return optimizeFPuts(CI, Builder);
    case LibFunc_puts:
      return optimizePuts(CI, Builder);
    case LibFunc_perror:
      return optimizeErrorReporting(CI, Builder);
    case LibFunc_vfprintf:
    case LibFunc_fiprintf:
      return optimizeErrorReporting(CI, Builder, 0);
    default:
      return nullptr;
    }
  }
  return nullptr;
}

LibCallSimplifier::LibCallSimplifier(
    const DataLayout &DL, const TargetLibraryInfo *TLI,
    OptimizationRemarkEmitter &ORE,
    BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
    function_ref<void(Instruction *, Value *)> Replacer,
    function_ref<void(Instruction *)> Eraser)
    : FortifiedSimplifier(TLI), DL(DL), TLI(TLI), ORE(ORE), BFI(BFI), PSI(PSI),
      UnsafeFPShrink(false), Replacer(Replacer), Eraser(Eraser) {}

void LibCallSimplifier::replaceAllUsesWith(Instruction *I, Value *With) {
  // Indirect through the replacer used in this instance.
  Replacer(I, With);
}

void LibCallSimplifier::eraseFromParent(Instruction *I) {
  Eraser(I);
}

// TODO:
//   Additional cases that we need to add to this file:
//
// cbrt:
//   * cbrt(expN(X))  -> expN(x/3)
//   * cbrt(sqrt(x))  -> pow(x,1/6)
//   * cbrt(cbrt(x))  -> pow(x,1/9)
//
// exp, expf, expl:
//   * exp(log(x))  -> x
//
// log, logf, logl:
//   * log(exp(x))   -> x
//   * log(exp(y))   -> y*log(e)
//   * log(exp10(y)) -> y*log(10)
//   * log(sqrt(x))  -> 0.5*log(x)
//
// pow, powf, powl:
//   * pow(sqrt(x),y) -> pow(x,y*0.5)
//   * pow(pow(x,y),z)-> pow(x,y*z)
//
// signbit:
//   * signbit(cnst) -> cnst'
//   * signbit(nncst) -> 0 (if pstv is a non-negative constant)
//
// sqrt, sqrtf, sqrtl:
//   * sqrt(expN(x))  -> expN(x*0.5)
//   * sqrt(Nroot(x)) -> pow(x,1/(2*N))
//   * sqrt(pow(x,y)) -> pow(|x|,y*0.5)
//

//===----------------------------------------------------------------------===//
// Fortified Library Call Optimizations
//===----------------------------------------------------------------------===//

bool
FortifiedLibCallSimplifier::isFortifiedCallFoldable(CallInst *CI,
                                                    unsigned ObjSizeOp,
                                                    Optional<unsigned> SizeOp,
                                                    Optional<unsigned> StrOp,
                                                    Optional<unsigned> FlagOp) {
  // If this function takes a flag argument, the implementation may use it to
  // perform extra checks. Don't fold into the non-checking variant.
  if (FlagOp) {
    ConstantInt *Flag = dyn_cast<ConstantInt>(CI->getArgOperand(*FlagOp));
    if (!Flag || !Flag->isZero())
      return false;
  }

  if (SizeOp && CI->getArgOperand(ObjSizeOp) == CI->getArgOperand(*SizeOp))
    return true;

  if (ConstantInt *ObjSizeCI =
          dyn_cast<ConstantInt>(CI->getArgOperand(ObjSizeOp))) {
    if (ObjSizeCI->isMinusOne())
      return true;
    // If the object size wasn't -1 (unknown), bail out if we were asked to.
    if (OnlyLowerUnknownSize)
      return false;
    if (StrOp) {
      uint64_t Len = GetStringLength(CI->getArgOperand(*StrOp));
      // If the length is 0 we don't know how long it is and so we can't
      // remove the check.
      if (Len)
        annotateDereferenceableBytes(CI, *StrOp, Len);
      else
        return false;
      return ObjSizeCI->getZExtValue() >= Len;
    }

    if (SizeOp) {
      if (ConstantInt *SizeCI =
              dyn_cast<ConstantInt>(CI->getArgOperand(*SizeOp)))
        return ObjSizeCI->getZExtValue() >= SizeCI->getZExtValue();
    }
  }
  return false;
}

Value *FortifiedLibCallSimplifier::optimizeMemCpyChk(CallInst *CI,
                                                     IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 3, 2)) {
    CallInst *NewCI =
        B.CreateMemCpy(CI->getArgOperand(0), Align(1), CI->getArgOperand(1),
                       Align(1), CI->getArgOperand(2));
    NewCI->setAttributes(CI->getAttributes());
    return CI->getArgOperand(0);
  }
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeMemMoveChk(CallInst *CI,
                                                      IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 3, 2)) {
    CallInst *NewCI =
        B.CreateMemMove(CI->getArgOperand(0), Align(1), CI->getArgOperand(1),
                        Align(1), CI->getArgOperand(2));
    NewCI->setAttributes(CI->getAttributes());
    return CI->getArgOperand(0);
  }
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeMemSetChk(CallInst *CI,
                                                     IRBuilderBase &B) {
  // TODO: Try foldMallocMemset() here.

  if (isFortifiedCallFoldable(CI, 3, 2)) {
    Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(), false);
    CallInst *NewCI = B.CreateMemSet(CI->getArgOperand(0), Val,
                                     CI->getArgOperand(2), Align(1));
    NewCI->setAttributes(CI->getAttributes());
    return CI->getArgOperand(0);
  }
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeMemPCpyChk(CallInst *CI,
                                                      IRBuilderBase &B) {
  const DataLayout &DL = CI->getModule()->getDataLayout();
  if (isFortifiedCallFoldable(CI, 3, 2))
    if (Value *Call = emitMemPCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                                  CI->getArgOperand(2), B, DL, TLI)) {
      CallInst *NewCI = cast<CallInst>(Call);
      NewCI->setAttributes(CI->getAttributes());
      return NewCI;
    }
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeStrpCpyChk(CallInst *CI,
                                                      IRBuilderBase &B,
                                                      LibFunc Func) {
  const DataLayout &DL = CI->getModule()->getDataLayout();
  Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1),
        *ObjSize = CI->getArgOperand(2);

  // __stpcpy_chk(x,x,...)  -> x+strlen(x)
  if (Func == LibFunc_stpcpy_chk && !OnlyLowerUnknownSize && Dst == Src) {
    Value *StrLen = emitStrLen(Src, B, DL, TLI);
    return StrLen ? B.CreateInBoundsGEP(B.getInt8Ty(), Dst, StrLen) : nullptr;
  }

  // If a) we don't have any length information, or b) we know this will
  // fit then just lower to a plain st[rp]cpy. Otherwise we'll keep our
  // st[rp]cpy_chk call which may fail at runtime if the size is too long.
  // TODO: It might be nice to get a maximum length out of the possible
  // string lengths for varying.
  if (isFortifiedCallFoldable(CI, 2, None, 1)) {
    if (Func == LibFunc_strcpy_chk)
      return emitStrCpy(Dst, Src, B, TLI);
    else
      return emitStpCpy(Dst, Src, B, TLI);
  }

  if (OnlyLowerUnknownSize)
    return nullptr;

  // Maybe we can stil fold __st[rp]cpy_chk to __memcpy_chk.
  uint64_t Len = GetStringLength(Src);
  if (Len)
    annotateDereferenceableBytes(CI, 1, Len);
  else
    return nullptr;

  Type *SizeTTy = DL.getIntPtrType(CI->getContext());
  Value *LenV = ConstantInt::get(SizeTTy, Len);
  Value *Ret = emitMemCpyChk(Dst, Src, LenV, ObjSize, B, DL, TLI);
  // If the function was an __stpcpy_chk, and we were able to fold it into
  // a __memcpy_chk, we still need to return the correct end pointer.
  if (Ret && Func == LibFunc_stpcpy_chk)
    return B.CreateGEP(B.getInt8Ty(), Dst, ConstantInt::get(SizeTTy, Len - 1));
  return Ret;
}

Value *FortifiedLibCallSimplifier::optimizeStrLenChk(CallInst *CI,
                                                     IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 1, None, 0))
    return emitStrLen(CI->getArgOperand(0), B, CI->getModule()->getDataLayout(),
                      TLI);
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeStrpNCpyChk(CallInst *CI,
                                                       IRBuilderBase &B,
                                                       LibFunc Func) {
  if (isFortifiedCallFoldable(CI, 3, 2)) {
    if (Func == LibFunc_strncpy_chk)
      return emitStrNCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                               CI->getArgOperand(2), B, TLI);
    else
      return emitStpNCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                         CI->getArgOperand(2), B, TLI);
  }

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeMemCCpyChk(CallInst *CI,
                                                      IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 4, 3))
    return emitMemCCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                       CI->getArgOperand(2), CI->getArgOperand(3), B, TLI);

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeSNPrintfChk(CallInst *CI,
                                                       IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 3, 1, None, 2)) {
    SmallVector<Value *, 8> VariadicArgs(CI->arg_begin() + 5, CI->arg_end());
    return emitSNPrintf(CI->getArgOperand(0), CI->getArgOperand(1),
                        CI->getArgOperand(4), VariadicArgs, B, TLI);
  }

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeSPrintfChk(CallInst *CI,
                                                      IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 2, None, None, 1)) {
    SmallVector<Value *, 8> VariadicArgs(CI->arg_begin() + 4, CI->arg_end());
    return emitSPrintf(CI->getArgOperand(0), CI->getArgOperand(3), VariadicArgs,
                       B, TLI);
  }

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeStrCatChk(CallInst *CI,
                                                     IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 2))
    return emitStrCat(CI->getArgOperand(0), CI->getArgOperand(1), B, TLI);

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeStrLCat(CallInst *CI,
                                                   IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 3))
    return emitStrLCat(CI->getArgOperand(0), CI->getArgOperand(1),
                       CI->getArgOperand(2), B, TLI);

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeStrNCatChk(CallInst *CI,
                                                      IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 3))
    return emitStrNCat(CI->getArgOperand(0), CI->getArgOperand(1),
                       CI->getArgOperand(2), B, TLI);

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeStrLCpyChk(CallInst *CI,
                                                      IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 3))
    return emitStrLCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                       CI->getArgOperand(2), B, TLI);

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeVSNPrintfChk(CallInst *CI,
                                                        IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 3, 1, None, 2))
    return emitVSNPrintf(CI->getArgOperand(0), CI->getArgOperand(1),
                         CI->getArgOperand(4), CI->getArgOperand(5), B, TLI);

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeVSPrintfChk(CallInst *CI,
                                                       IRBuilderBase &B) {
  if (isFortifiedCallFoldable(CI, 2, None, None, 1))
    return emitVSPrintf(CI->getArgOperand(0), CI->getArgOperand(3),
                        CI->getArgOperand(4), B, TLI);

  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeCall(CallInst *CI,
                                                IRBuilderBase &Builder) {
  // FIXME: We shouldn't be changing "nobuiltin" or TLI unavailable calls here.
  // Some clang users checked for _chk libcall availability using:
  //   __has_builtin(__builtin___memcpy_chk)
  // When compiling with -fno-builtin, this is always true.
  // When passing -ffreestanding/-mkernel, which both imply -fno-builtin, we
  // end up with fortified libcalls, which isn't acceptable in a freestanding
  // environment which only provides their non-fortified counterparts.
  //
  // Until we change clang and/or teach external users to check for availability
  // differently, disregard the "nobuiltin" attribute and TLI::has.
  //
  // PR23093.

  LibFunc Func;
  Function *Callee = CI->getCalledFunction();
  bool isCallingConvC = isCallingConvCCompatible(CI);

  SmallVector<OperandBundleDef, 2> OpBundles;
  CI->getOperandBundlesAsDefs(OpBundles);

  IRBuilderBase::OperandBundlesGuard Guard(Builder);
  Builder.setDefaultOperandBundles(OpBundles);

  // First, check that this is a known library functions and that the prototype
  // is correct.
  if (!TLI->getLibFunc(*Callee, Func))
    return nullptr;

  // We never change the calling convention.
  if (!ignoreCallingConv(Func) && !isCallingConvC)
    return nullptr;

  switch (Func) {
  case LibFunc_memcpy_chk:
    return optimizeMemCpyChk(CI, Builder);
  case LibFunc_mempcpy_chk:
    return optimizeMemPCpyChk(CI, Builder);
  case LibFunc_memmove_chk:
    return optimizeMemMoveChk(CI, Builder);
  case LibFunc_memset_chk:
    return optimizeMemSetChk(CI, Builder);
  case LibFunc_stpcpy_chk:
  case LibFunc_strcpy_chk:
    return optimizeStrpCpyChk(CI, Builder, Func);
  case LibFunc_strlen_chk:
    return optimizeStrLenChk(CI, Builder);
  case LibFunc_stpncpy_chk:
  case LibFunc_strncpy_chk:
    return optimizeStrpNCpyChk(CI, Builder, Func);
  case LibFunc_memccpy_chk:
    return optimizeMemCCpyChk(CI, Builder);
  case LibFunc_snprintf_chk:
    return optimizeSNPrintfChk(CI, Builder);
  case LibFunc_sprintf_chk:
    return optimizeSPrintfChk(CI, Builder);
  case LibFunc_strcat_chk:
    return optimizeStrCatChk(CI, Builder);
  case LibFunc_strlcat_chk:
    return optimizeStrLCat(CI, Builder);
  case LibFunc_strncat_chk:
    return optimizeStrNCatChk(CI, Builder);
  case LibFunc_strlcpy_chk:
    return optimizeStrLCpyChk(CI, Builder);
  case LibFunc_vsnprintf_chk:
    return optimizeVSNPrintfChk(CI, Builder);
  case LibFunc_vsprintf_chk:
    return optimizeVSPrintfChk(CI, Builder);
  default:
    break;
  }
  return nullptr;
}

FortifiedLibCallSimplifier::FortifiedLibCallSimplifier(
    const TargetLibraryInfo *TLI, bool OnlyLowerUnknownSize)
    : TLI(TLI), OnlyLowerUnknownSize(OnlyLowerUnknownSize) {}
