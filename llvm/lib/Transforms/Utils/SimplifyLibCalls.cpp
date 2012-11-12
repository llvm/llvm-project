//===------ SimplifyLibCalls.cpp - Library calls simplifier ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility pass used for testing the InstructionSimplify analysis.
// The analysis is applied to every instruction, and if it simplifies then the
// instruction is replaced by the simplification.  If you are looking for a pass
// that performs serious instruction folding, use the instcombine pass instead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include "llvm/DataLayout.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/LLVMContext.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

using namespace llvm;

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
class LibCallOptimization {
protected:
  Function *Caller;
  const DataLayout *TD;
  const TargetLibraryInfo *TLI;
  const LibCallSimplifier *LCS;
  LLVMContext* Context;
public:
  LibCallOptimization() { }
  virtual ~LibCallOptimization() {}

  /// callOptimizer - This pure virtual method is implemented by base classes to
  /// do various optimizations.  If this returns null then no transformation was
  /// performed.  If it returns CI, then it transformed the call and CI is to be
  /// deleted.  If it returns something else, replace CI with the new value and
  /// delete CI.
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B)
    =0;

  Value *optimizeCall(CallInst *CI, const DataLayout *TD,
                      const TargetLibraryInfo *TLI,
                      const LibCallSimplifier *LCS, IRBuilder<> &B) {
    Caller = CI->getParent()->getParent();
    this->TD = TD;
    this->TLI = TLI;
    this->LCS = LCS;
    if (CI->getCalledFunction())
      Context = &CI->getCalledFunction()->getContext();

    // We never change the calling convention.
    if (CI->getCallingConv() != llvm::CallingConv::C)
      return NULL;

    return callOptimizer(CI->getCalledFunction(), CI, B);
  }
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// isOnlyUsedInZeroEqualityComparison - Return true if it only matters that the
/// value is equal or not-equal to zero.
static bool isOnlyUsedInZeroEqualityComparison(Value *V) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end();
       UI != E; ++UI) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(*UI))
      if (IC->isEquality())
        if (Constant *C = dyn_cast<Constant>(IC->getOperand(1)))
          if (C->isNullValue())
            continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

/// isOnlyUsedInEqualityComparison - Return true if it is only used in equality
/// comparisons with With.
static bool isOnlyUsedInEqualityComparison(Value *V, Value *With) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end();
       UI != E; ++UI) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(*UI))
      if (IC->isEquality() && IC->getOperand(1) == With)
        continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Fortified Library Call Optimizations
//===----------------------------------------------------------------------===//

struct FortifiedLibCallOptimization : public LibCallOptimization {
protected:
  virtual bool isFoldable(unsigned SizeCIOp, unsigned SizeArgOp,
			  bool isString) const = 0;
};

struct InstFortifiedLibCallOptimization : public FortifiedLibCallOptimization {
  CallInst *CI;

  bool isFoldable(unsigned SizeCIOp, unsigned SizeArgOp, bool isString) const {
    if (CI->getArgOperand(SizeCIOp) == CI->getArgOperand(SizeArgOp))
      return true;
    if (ConstantInt *SizeCI =
                           dyn_cast<ConstantInt>(CI->getArgOperand(SizeCIOp))) {
      if (SizeCI->isAllOnesValue())
        return true;
      if (isString) {
        uint64_t Len = GetStringLength(CI->getArgOperand(SizeArgOp));
        // If the length is 0 we don't know how long it is and so we can't
        // remove the check.
        if (Len == 0) return false;
        return SizeCI->getZExtValue() >= Len;
      }
      if (ConstantInt *Arg = dyn_cast<ConstantInt>(
                                                  CI->getArgOperand(SizeArgOp)))
        return SizeCI->getZExtValue() >= Arg->getZExtValue();
    }
    return false;
  }
};

struct MemCpyChkOpt : public InstFortifiedLibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(Context) ||
        FT->getParamType(3) != TD->getIntPtrType(Context))
      return 0;

    if (isFoldable(3, 2, false)) {
      B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                     CI->getArgOperand(2), 1);
      return CI->getArgOperand(0);
    }
    return 0;
  }
};

struct MemMoveChkOpt : public InstFortifiedLibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(Context) ||
        FT->getParamType(3) != TD->getIntPtrType(Context))
      return 0;

    if (isFoldable(3, 2, false)) {
      B.CreateMemMove(CI->getArgOperand(0), CI->getArgOperand(1),
                      CI->getArgOperand(2), 1);
      return CI->getArgOperand(0);
    }
    return 0;
  }
};

struct MemSetChkOpt : public InstFortifiedLibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(Context) ||
        FT->getParamType(3) != TD->getIntPtrType(Context))
      return 0;

    if (isFoldable(3, 2, false)) {
      Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(),
                                   false);
      B.CreateMemSet(CI->getArgOperand(0), Val, CI->getArgOperand(2), 1);
      return CI->getArgOperand(0);
    }
    return 0;
  }
};

struct StrCpyChkOpt : public InstFortifiedLibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        FT->getParamType(2) != TD->getIntPtrType(Context))
      return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src)      // __strcpy_chk(x,x)  -> x
      return Src;

    // If a) we don't have any length information, or b) we know this will
    // fit then just lower to a plain strcpy. Otherwise we'll keep our
    // strcpy_chk call which may fail at runtime if the size is too long.
    // TODO: It might be nice to get a maximum length out of the possible
    // string lengths for varying.
    if (isFoldable(2, 1, true)) {
      Value *Ret = EmitStrCpy(Dst, Src, B, TD, TLI, Name.substr(2, 6));
      return Ret;
    } else {
      // Maybe we can stil fold __strcpy_chk to __memcpy_chk.
      uint64_t Len = GetStringLength(Src);
      if (Len == 0) return 0;

      // This optimization require DataLayout.
      if (!TD) return 0;

      Value *Ret =
	EmitMemCpyChk(Dst, Src,
                      ConstantInt::get(TD->getIntPtrType(Context), Len),
                      CI->getArgOperand(2), B, TD, TLI);
      return Ret;
    }
    return 0;
  }
};

struct StpCpyChkOpt : public InstFortifiedLibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        FT->getParamType(2) != TD->getIntPtrType(FT->getParamType(0)))
      return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src) {  // stpcpy(x,x)  -> x+strlen(x)
      Value *StrLen = EmitStrLen(Src, B, TD, TLI);
      return StrLen ? B.CreateInBoundsGEP(Dst, StrLen) : 0;
    }

    // If a) we don't have any length information, or b) we know this will
    // fit then just lower to a plain stpcpy. Otherwise we'll keep our
    // stpcpy_chk call which may fail at runtime if the size is too long.
    // TODO: It might be nice to get a maximum length out of the possible
    // string lengths for varying.
    if (isFoldable(2, 1, true)) {
      Value *Ret = EmitStrCpy(Dst, Src, B, TD, TLI, Name.substr(2, 6));
      return Ret;
    } else {
      // Maybe we can stil fold __stpcpy_chk to __memcpy_chk.
      uint64_t Len = GetStringLength(Src);
      if (Len == 0) return 0;

      // This optimization require DataLayout.
      if (!TD) return 0;

      Type *PT = FT->getParamType(0);
      Value *LenV = ConstantInt::get(TD->getIntPtrType(PT), Len);
      Value *DstEnd = B.CreateGEP(Dst,
                                  ConstantInt::get(TD->getIntPtrType(PT),
                                                   Len - 1));
      if (!EmitMemCpyChk(Dst, Src, LenV, CI->getArgOperand(2), B, TD, TLI))
        return 0;
      return DstEnd;
    }
    return 0;
  }
};

struct StrNCpyChkOpt : public InstFortifiedLibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        !FT->getParamType(2)->isIntegerTy() ||
        FT->getParamType(3) != TD->getIntPtrType(Context))
      return 0;

    if (isFoldable(3, 2, false)) {
      Value *Ret = EmitStrNCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                               CI->getArgOperand(2), B, TD, TLI,
                               Name.substr(2, 7));
      return Ret;
    }
    return 0;
  }
};

//===----------------------------------------------------------------------===//
// String and Memory Library Call Optimizations
//===----------------------------------------------------------------------===//

struct StrCatOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strcat" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != B.getInt8PtrTy() ||
        FT->getParamType(0) != FT->getReturnType() ||
        FT->getParamType(1) != FT->getReturnType())
      return 0;

    // Extract some information from the instruction
    Value *Dst = CI->getArgOperand(0);
    Value *Src = CI->getArgOperand(1);

    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;
    --Len;  // Unbias length.

    // Handle the simple, do-nothing case: strcat(x, "") -> x
    if (Len == 0)
      return Dst;

    // These optimizations require DataLayout.
    if (!TD) return 0;

    return emitStrLenMemCpy(Src, Dst, Len, B);
  }

  Value *emitStrLenMemCpy(Value *Src, Value *Dst, uint64_t Len,
                          IRBuilder<> &B) {
    // We need to find the end of the destination string.  That's where the
    // memory is to be moved to. We just generate a call to strlen.
    Value *DstLen = EmitStrLen(Dst, B, TD, TLI);
    if (!DstLen)
      return 0;

    // Now that we have the destination's length, we must index into the
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    Value *CpyDst = B.CreateGEP(Dst, DstLen, "endptr");

    // We have enough information to now generate the memcpy call to do the
    // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(CpyDst, Src,
                   ConstantInt::get(TD->getIntPtrType(*Context), Len + 1), 1);
    return Dst;
  }
};

struct StrNCatOpt : public StrCatOpt {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strncat" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != B.getInt8PtrTy() ||
        FT->getParamType(0) != FT->getReturnType() ||
        FT->getParamType(1) != FT->getReturnType() ||
        !FT->getParamType(2)->isIntegerTy())
      return 0;

    // Extract some information from the instruction
    Value *Dst = CI->getArgOperand(0);
    Value *Src = CI->getArgOperand(1);
    uint64_t Len;

    // We don't do anything if length is not constant
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
      Len = LengthArg->getZExtValue();
    else
      return 0;

    // See if we can get the length of the input string.
    uint64_t SrcLen = GetStringLength(Src);
    if (SrcLen == 0) return 0;
    --SrcLen;  // Unbias length.

    // Handle the simple, do-nothing cases:
    // strncat(x, "", c) -> x
    // strncat(x,  c, 0) -> x
    if (SrcLen == 0 || Len == 0) return Dst;

    // These optimizations require DataLayout.
    if (!TD) return 0;

    // We don't optimize this case
    if (Len < SrcLen) return 0;

    // strncat(x, s, c) -> strcat(x, s)
    // s is constant so the strcat can be optimized further
    return emitStrLenMemCpy(Src, Dst, SrcLen, B);
  }
};

struct StrChrOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strchr" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != B.getInt8PtrTy() ||
        FT->getParamType(0) != FT->getReturnType() ||
        !FT->getParamType(1)->isIntegerTy(32))
      return 0;

    Value *SrcStr = CI->getArgOperand(0);

    // If the second operand is non-constant, see if we can compute the length
    // of the input string and turn this into memchr.
    ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (CharC == 0) {
      // These optimizations require DataLayout.
      if (!TD) return 0;

      uint64_t Len = GetStringLength(SrcStr);
      if (Len == 0 || !FT->getParamType(1)->isIntegerTy(32))// memchr needs i32.
        return 0;

      return EmitMemChr(SrcStr, CI->getArgOperand(1), // include nul.
                        ConstantInt::get(TD->getIntPtrType(*Context), Len),
                        B, TD, TLI);
    }

    // Otherwise, the character is a constant, see if the first argument is
    // a string literal.  If so, we can constant fold.
    StringRef Str;
    if (!getConstantStringInfo(SrcStr, Str))
      return 0;

    // Compute the offset, make sure to handle the case when we're searching for
    // zero (a weird way to spell strlen).
    size_t I = CharC->getSExtValue() == 0 ?
        Str.size() : Str.find(CharC->getSExtValue());
    if (I == StringRef::npos) // Didn't find the char.  strchr returns null.
      return Constant::getNullValue(CI->getType());

    // strchr(s+n,c)  -> gep(s+n+i,c)
    return B.CreateGEP(SrcStr, B.getInt64(I), "strchr");
  }
};

struct StrRChrOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strrchr" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != B.getInt8PtrTy() ||
        FT->getParamType(0) != FT->getReturnType() ||
        !FT->getParamType(1)->isIntegerTy(32))
      return 0;

    Value *SrcStr = CI->getArgOperand(0);
    ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));

    // Cannot fold anything if we're not looking for a constant.
    if (!CharC)
      return 0;

    StringRef Str;
    if (!getConstantStringInfo(SrcStr, Str)) {
      // strrchr(s, 0) -> strchr(s, 0)
      if (TD && CharC->isZero())
        return EmitStrChr(SrcStr, '\0', B, TD, TLI);
      return 0;
    }

    // Compute the offset.
    size_t I = CharC->getSExtValue() == 0 ?
        Str.size() : Str.rfind(CharC->getSExtValue());
    if (I == StringRef::npos) // Didn't find the char. Return null.
      return Constant::getNullValue(CI->getType());

    // strrchr(s+n,c) -> gep(s+n+i,c)
    return B.CreateGEP(SrcStr, B.getInt64(I), "strrchr");
  }
};

struct StrCmpOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strcmp" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        !FT->getReturnType()->isIntegerTy(32) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy())
      return 0;

    Value *Str1P = CI->getArgOperand(0), *Str2P = CI->getArgOperand(1);
    if (Str1P == Str2P)      // strcmp(x,x)  -> 0
      return ConstantInt::get(CI->getType(), 0);

    StringRef Str1, Str2;
    bool HasStr1 = getConstantStringInfo(Str1P, Str1);
    bool HasStr2 = getConstantStringInfo(Str2P, Str2);

    // strcmp(x, y)  -> cnst  (if both x and y are constant strings)
    if (HasStr1 && HasStr2)
      return ConstantInt::get(CI->getType(), Str1.compare(Str2));

    if (HasStr1 && Str1.empty()) // strcmp("", x) -> -*x
      return B.CreateNeg(B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"),
                                      CI->getType()));

    if (HasStr2 && Str2.empty()) // strcmp(x,"") -> *x
      return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());

    // strcmp(P, "x") -> memcmp(P, "x", 2)
    uint64_t Len1 = GetStringLength(Str1P);
    uint64_t Len2 = GetStringLength(Str2P);
    if (Len1 && Len2) {
      // These optimizations require DataLayout.
      if (!TD) return 0;

      return EmitMemCmp(Str1P, Str2P,
                        ConstantInt::get(TD->getIntPtrType(*Context),
                        std::min(Len1, Len2)), B, TD, TLI);
    }

    return 0;
  }
};

struct StrNCmpOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strncmp" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 ||
        !FT->getReturnType()->isIntegerTy(32) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        !FT->getParamType(2)->isIntegerTy())
      return 0;

    Value *Str1P = CI->getArgOperand(0), *Str2P = CI->getArgOperand(1);
    if (Str1P == Str2P)      // strncmp(x,x,n)  -> 0
      return ConstantInt::get(CI->getType(), 0);

    // Get the length argument if it is constant.
    uint64_t Length;
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
      Length = LengthArg->getZExtValue();
    else
      return 0;

    if (Length == 0) // strncmp(x,y,0)   -> 0
      return ConstantInt::get(CI->getType(), 0);

    if (TD && Length == 1) // strncmp(x,y,1) -> memcmp(x,y,1)
      return EmitMemCmp(Str1P, Str2P, CI->getArgOperand(2), B, TD, TLI);

    StringRef Str1, Str2;
    bool HasStr1 = getConstantStringInfo(Str1P, Str1);
    bool HasStr2 = getConstantStringInfo(Str2P, Str2);

    // strncmp(x, y)  -> cnst  (if both x and y are constant strings)
    if (HasStr1 && HasStr2) {
      StringRef SubStr1 = Str1.substr(0, Length);
      StringRef SubStr2 = Str2.substr(0, Length);
      return ConstantInt::get(CI->getType(), SubStr1.compare(SubStr2));
    }

    if (HasStr1 && Str1.empty())  // strncmp("", x, n) -> -*x
      return B.CreateNeg(B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"),
                                      CI->getType()));

    if (HasStr2 && Str2.empty())  // strncmp(x, "", n) -> *x
      return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());

    return 0;
  }
};

struct StrCpyOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strcpy" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy())
      return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src)      // strcpy(x,x)  -> x
      return Src;

    // These optimizations require DataLayout.
    if (!TD) return 0;

    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;

    // We have enough information to now generate the memcpy call to do the
    // copy for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(Dst, Src,
		   ConstantInt::get(TD->getIntPtrType(*Context), Len), 1);
    return Dst;
  }
};

struct StpCpyOpt: public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "stpcpy" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy())
      return 0;

    // These optimizations require DataLayout.
    if (!TD) return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src) {  // stpcpy(x,x)  -> x+strlen(x)
      Value *StrLen = EmitStrLen(Src, B, TD, TLI);
      return StrLen ? B.CreateInBoundsGEP(Dst, StrLen) : 0;
    }

    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;

    Type *PT = FT->getParamType(0);
    Value *LenV = ConstantInt::get(TD->getIntPtrType(PT), Len);
    Value *DstEnd = B.CreateGEP(Dst,
                                ConstantInt::get(TD->getIntPtrType(PT),
                                                 Len - 1));

    // We have enough information to now generate the memcpy call to do the
    // copy for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(Dst, Src, LenV, 1);
    return DstEnd;
  }
};

struct StrNCpyOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        !FT->getParamType(2)->isIntegerTy())
      return 0;

    Value *Dst = CI->getArgOperand(0);
    Value *Src = CI->getArgOperand(1);
    Value *LenOp = CI->getArgOperand(2);

    // See if we can get the length of the input string.
    uint64_t SrcLen = GetStringLength(Src);
    if (SrcLen == 0) return 0;
    --SrcLen;

    if (SrcLen == 0) {
      // strncpy(x, "", y) -> memset(x, '\0', y, 1)
      B.CreateMemSet(Dst, B.getInt8('\0'), LenOp, 1);
      return Dst;
    }

    uint64_t Len;
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(LenOp))
      Len = LengthArg->getZExtValue();
    else
      return 0;

    if (Len == 0) return Dst; // strncpy(x, y, 0) -> x

    // These optimizations require DataLayout.
    if (!TD) return 0;

    // Let strncpy handle the zero padding
    if (Len > SrcLen+1) return 0;

    Type *PT = FT->getParamType(0);
    // strncpy(x, s, c) -> memcpy(x, s, c, 1) [s and c are constant]
    B.CreateMemCpy(Dst, Src,
                   ConstantInt::get(TD->getIntPtrType(PT), Len), 1);

    return Dst;
  }
};

struct StrLenOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 1 ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    Value *Src = CI->getArgOperand(0);

    // Constant folding: strlen("xyz") -> 3
    if (uint64_t Len = GetStringLength(Src))
      return ConstantInt::get(CI->getType(), Len-1);

    // strlen(x) != 0 --> *x != 0
    // strlen(x) == 0 --> *x == 0
    if (isOnlyUsedInZeroEqualityComparison(CI))
      return B.CreateZExt(B.CreateLoad(Src, "strlenfirst"), CI->getType());
    return 0;
  }
};

struct StrPBrkOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        FT->getParamType(1) != FT->getParamType(0) ||
        FT->getReturnType() != FT->getParamType(0))
      return 0;

    StringRef S1, S2;
    bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
    bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

    // strpbrk(s, "") -> NULL
    // strpbrk("", s) -> NULL
    if ((HasS1 && S1.empty()) || (HasS2 && S2.empty()))
      return Constant::getNullValue(CI->getType());

    // Constant folding.
    if (HasS1 && HasS2) {
      size_t I = S1.find_first_of(S2);
      if (I == std::string::npos) // No match.
        return Constant::getNullValue(CI->getType());

      return B.CreateGEP(CI->getArgOperand(0), B.getInt64(I), "strpbrk");
    }

    // strpbrk(s, "a") -> strchr(s, 'a')
    if (TD && HasS2 && S2.size() == 1)
      return EmitStrChr(CI->getArgOperand(0), S2[0], B, TD, TLI);

    return 0;
  }
};

struct StrToOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if ((FT->getNumParams() != 2 && FT->getNumParams() != 3) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy())
      return 0;

    Value *EndPtr = CI->getArgOperand(1);
    if (isa<ConstantPointerNull>(EndPtr)) {
      // With a null EndPtr, this function won't capture the main argument.
      // It would be readonly too, except that it still may write to errno.
      CI->addAttribute(1, Attributes::get(Callee->getContext(),
                                          Attributes::NoCapture));
    }

    return 0;
  }
};

struct StrSpnOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        FT->getParamType(1) != FT->getParamType(0) ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

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
      if (Pos == StringRef::npos) Pos = S1.size();
      return ConstantInt::get(CI->getType(), Pos);
    }

    return 0;
  }
};

struct StrCSpnOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        FT->getParamType(1) != FT->getParamType(0) ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    StringRef S1, S2;
    bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
    bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

    // strcspn("", s) -> 0
    if (HasS1 && S1.empty())
      return Constant::getNullValue(CI->getType());

    // Constant folding.
    if (HasS1 && HasS2) {
      size_t Pos = S1.find_first_of(S2);
      if (Pos == StringRef::npos) Pos = S1.size();
      return ConstantInt::get(CI->getType(), Pos);
    }

    // strcspn(s, "") -> strlen(s)
    if (TD && HasS2 && S2.empty())
      return EmitStrLen(CI->getArgOperand(0), B, TD, TLI);

    return 0;
  }
};

struct StrStrOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !FT->getReturnType()->isPointerTy())
      return 0;

    // fold strstr(x, x) -> x.
    if (CI->getArgOperand(0) == CI->getArgOperand(1))
      return B.CreateBitCast(CI->getArgOperand(0), CI->getType());

    // fold strstr(a, b) == a -> strncmp(a, b, strlen(b)) == 0
    if (TD && isOnlyUsedInEqualityComparison(CI, CI->getArgOperand(0))) {
      Value *StrLen = EmitStrLen(CI->getArgOperand(1), B, TD, TLI);
      if (!StrLen)
        return 0;
      Value *StrNCmp = EmitStrNCmp(CI->getArgOperand(0), CI->getArgOperand(1),
                                   StrLen, B, TD, TLI);
      if (!StrNCmp)
        return 0;
      for (Value::use_iterator UI = CI->use_begin(), UE = CI->use_end();
           UI != UE; ) {
        ICmpInst *Old = cast<ICmpInst>(*UI++);
        Value *Cmp = B.CreateICmp(Old->getPredicate(), StrNCmp,
                                  ConstantInt::getNullValue(StrNCmp->getType()),
                                  "cmp");
        LCS->replaceAllUsesWith(Old, Cmp);
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
      std::string::size_type Offset = SearchStr.find(ToFindStr);

      if (Offset == StringRef::npos) // strstr("foo", "bar") -> null
        return Constant::getNullValue(CI->getType());

      // strstr("abcd", "bc") -> gep((char*)"abcd", 1)
      Value *Result = CastToCStr(CI->getArgOperand(0), B);
      Result = B.CreateConstInBoundsGEP1_64(Result, Offset, "strstr");
      return B.CreateBitCast(Result, CI->getType());
    }

    // fold strstr(x, "y") -> strchr(x, 'y').
    if (HasStr2 && ToFindStr.size() == 1) {
      Value *StrChr= EmitStrChr(CI->getArgOperand(0), ToFindStr[0], B, TD, TLI);
      return StrChr ? B.CreateBitCast(StrChr, CI->getType()) : 0;
    }
    return 0;
  }
};

struct MemCmpOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !FT->getReturnType()->isIntegerTy(32))
      return 0;

    Value *LHS = CI->getArgOperand(0), *RHS = CI->getArgOperand(1);

    if (LHS == RHS)  // memcmp(s,s,x) -> 0
      return Constant::getNullValue(CI->getType());

    // Make sure we have a constant length.
    ConstantInt *LenC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!LenC) return 0;
    uint64_t Len = LenC->getZExtValue();

    if (Len == 0) // memcmp(s1,s2,0) -> 0
      return Constant::getNullValue(CI->getType());

    // memcmp(S1,S2,1) -> *(unsigned char*)LHS - *(unsigned char*)RHS
    if (Len == 1) {
      Value *LHSV = B.CreateZExt(B.CreateLoad(CastToCStr(LHS, B), "lhsc"),
                                 CI->getType(), "lhsv");
      Value *RHSV = B.CreateZExt(B.CreateLoad(CastToCStr(RHS, B), "rhsc"),
                                 CI->getType(), "rhsv");
      return B.CreateSub(LHSV, RHSV, "chardiff");
    }

    // Constant folding: memcmp(x, y, l) -> cnst (all arguments are constant)
    StringRef LHSStr, RHSStr;
    if (getConstantStringInfo(LHS, LHSStr) &&
        getConstantStringInfo(RHS, RHSStr)) {
      // Make sure we're not reading out-of-bounds memory.
      if (Len > LHSStr.size() || Len > RHSStr.size())
        return 0;
      uint64_t Ret = memcmp(LHSStr.data(), RHSStr.data(), Len);
      return ConstantInt::get(CI->getType(), Ret);
    }

    return 0;
  }
};

struct MemCpyOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // These optimizations require DataLayout.
    if (!TD) return 0;

    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(*Context))
      return 0;

    // memcpy(x, y, n) -> llvm.memcpy(x, y, n, 1)
    B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                   CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
};

struct MemMoveOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // These optimizations require DataLayout.
    if (!TD) return 0;

    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(*Context))
      return 0;

    // memmove(x, y, n) -> llvm.memmove(x, y, n, 1)
    B.CreateMemMove(CI->getArgOperand(0), CI->getArgOperand(1),
                    CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
};

struct MemSetOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // These optimizations require DataLayout.
    if (!TD) return 0;

    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(*Context))
      return 0;

    // memset(p, v, n) -> llvm.memset(p, v, n, 1)
    Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(), false);
    B.CreateMemSet(CI->getArgOperand(0), Val, CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
};

} // End anonymous namespace.

namespace llvm {

class LibCallSimplifierImpl {
  const DataLayout *TD;
  const TargetLibraryInfo *TLI;
  const LibCallSimplifier *LCS;
  StringMap<LibCallOptimization*> Optimizations;

  // Fortified library call optimizations.
  MemCpyChkOpt MemCpyChk;
  MemMoveChkOpt MemMoveChk;
  MemSetChkOpt MemSetChk;
  StrCpyChkOpt StrCpyChk;
  StpCpyChkOpt StpCpyChk;
  StrNCpyChkOpt StrNCpyChk;

  // String library call optimizations.
  StrCatOpt StrCat;
  StrNCatOpt StrNCat;
  StrChrOpt StrChr;
  StrRChrOpt StrRChr;
  StrCmpOpt StrCmp;
  StrNCmpOpt StrNCmp;
  StrCpyOpt StrCpy;
  StpCpyOpt StpCpy;
  StrNCpyOpt StrNCpy;
  StrLenOpt StrLen;
  StrPBrkOpt StrPBrk;
  StrToOpt StrTo;
  StrSpnOpt StrSpn;
  StrCSpnOpt StrCSpn;
  StrStrOpt StrStr;

  // Memory library call optimizations.
  MemCmpOpt MemCmp;
  MemCpyOpt MemCpy;
  MemMoveOpt MemMove;
  MemSetOpt MemSet;

  void initOptimizations();
  void addOpt(LibFunc::Func F, LibCallOptimization* Opt);
public:
  LibCallSimplifierImpl(const DataLayout *TD, const TargetLibraryInfo *TLI,
                        const LibCallSimplifier *LCS) {
    this->TD = TD;
    this->TLI = TLI;
    this->LCS = LCS;
  }

  Value *optimizeCall(CallInst *CI);
};

void LibCallSimplifierImpl::initOptimizations() {
  // Fortified library call optimizations.
  Optimizations["__memcpy_chk"] = &MemCpyChk;
  Optimizations["__memmove_chk"] = &MemMoveChk;
  Optimizations["__memset_chk"] = &MemSetChk;
  Optimizations["__strcpy_chk"] = &StrCpyChk;
  Optimizations["__stpcpy_chk"] = &StpCpyChk;
  Optimizations["__strncpy_chk"] = &StrNCpyChk;
  Optimizations["__stpncpy_chk"] = &StrNCpyChk;

  // String library call optimizations.
  addOpt(LibFunc::strcat, &StrCat);
  addOpt(LibFunc::strncat, &StrNCat);
  addOpt(LibFunc::strchr, &StrChr);
  addOpt(LibFunc::strrchr, &StrRChr);
  addOpt(LibFunc::strcmp, &StrCmp);
  addOpt(LibFunc::strncmp, &StrNCmp);
  addOpt(LibFunc::strcpy, &StrCpy);
  addOpt(LibFunc::stpcpy, &StpCpy);
  addOpt(LibFunc::strncpy, &StrNCpy);
  addOpt(LibFunc::strlen, &StrLen);
  addOpt(LibFunc::strpbrk, &StrPBrk);
  addOpt(LibFunc::strtol, &StrTo);
  addOpt(LibFunc::strtod, &StrTo);
  addOpt(LibFunc::strtof, &StrTo);
  addOpt(LibFunc::strtoul, &StrTo);
  addOpt(LibFunc::strtoll, &StrTo);
  addOpt(LibFunc::strtold, &StrTo);
  addOpt(LibFunc::strtoull, &StrTo);
  addOpt(LibFunc::strspn, &StrSpn);
  addOpt(LibFunc::strcspn, &StrCSpn);
  addOpt(LibFunc::strstr, &StrStr);

  // Memory library call optimizations.
  addOpt(LibFunc::memcmp, &MemCmp);
  addOpt(LibFunc::memcpy, &MemCpy);
  addOpt(LibFunc::memmove, &MemMove);
  addOpt(LibFunc::memset, &MemSet);
}

Value *LibCallSimplifierImpl::optimizeCall(CallInst *CI) {
  if (Optimizations.empty())
    initOptimizations();

  Function *Callee = CI->getCalledFunction();
  LibCallOptimization *LCO = Optimizations.lookup(Callee->getName());
  if (LCO) {
    IRBuilder<> Builder(CI);
    return LCO->optimizeCall(CI, TD, TLI, LCS, Builder);
  }
  return 0;
}

void LibCallSimplifierImpl::addOpt(LibFunc::Func F, LibCallOptimization* Opt) {
  if (TLI->has(F))
    Optimizations[TLI->getName(F)] = Opt;
}

LibCallSimplifier::LibCallSimplifier(const DataLayout *TD,
                                     const TargetLibraryInfo *TLI) {
  Impl = new LibCallSimplifierImpl(TD, TLI, this);
}

LibCallSimplifier::~LibCallSimplifier() {
  delete Impl;
}

Value *LibCallSimplifier::optimizeCall(CallInst *CI) {
  return Impl->optimizeCall(CI);
}

void LibCallSimplifier::replaceAllUsesWith(Instruction *I, Value *With) const {
  I->replaceAllUsesWith(With);
  I->eraseFromParent();
}

}
