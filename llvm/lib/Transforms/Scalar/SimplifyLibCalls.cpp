//===- SimplifyLibCalls.cpp - Optimize specific well-known library calls --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple pass that applies a variety of small
// optimizations for calls to specific well-known function calls (e.g. runtime
// library functions).   Any optimization that takes the very simple form
// "replace call to library function with simpler code that provides the same
// result" belongs in this file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplify-libcalls"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/IRBuilder.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Config/config.h"            // FIXME: Shouldn't depend on host!
using namespace llvm;

STATISTIC(NumSimplified, "Number of library calls simplified");
STATISTIC(NumAnnotated, "Number of attributes added to library functions");

static cl::opt<bool> UnsafeFPShrink("enable-double-float-shrink", cl::Hidden,
                                   cl::init(false),
                                   cl::desc("Enable unsafe double to float "
                                            "shrinking for math lib calls"));
//===----------------------------------------------------------------------===//
// Optimizer Base Class
//===----------------------------------------------------------------------===//

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
class LibCallOptimization {
protected:
  Function *Caller;
  const DataLayout *TD;
  const TargetLibraryInfo *TLI;
  LLVMContext* Context;
public:
  LibCallOptimization() { }
  virtual ~LibCallOptimization() {}

  /// CallOptimizer - This pure virtual method is implemented by base classes to
  /// do various optimizations.  If this returns null then no transformation was
  /// performed.  If it returns CI, then it transformed the call and CI is to be
  /// deleted.  If it returns something else, replace CI with the new value and
  /// delete CI.
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B)
    =0;

  Value *OptimizeCall(CallInst *CI, const DataLayout *TD,
                      const TargetLibraryInfo *TLI, IRBuilder<> &B) {
    Caller = CI->getParent()->getParent();
    this->TD = TD;
    this->TLI = TLI;
    if (CI->getCalledFunction())
      Context = &CI->getCalledFunction()->getContext();

    // We never change the calling convention.
    if (CI->getCallingConv() != llvm::CallingConv::C)
      return NULL;

    return CallOptimizer(CI->getCalledFunction(), CI, B);
  }
};
} // End anonymous namespace.


//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static bool CallHasFloatingPointArgument(const CallInst *CI) {
  for (CallInst::const_op_iterator it = CI->op_begin(), e = CI->op_end();
       it != e; ++it) {
    if ((*it)->getType()->isFloatingPointTy())
      return true;
  }
  return false;
}

namespace {
//===----------------------------------------------------------------------===//
// Math Library Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// Double -> Float Shrinking Optimizations for Unary Functions like 'floor'

struct UnaryDoubleFPOpt : public LibCallOptimization {
  bool CheckRetType;
  UnaryDoubleFPOpt(bool CheckReturnType): CheckRetType(CheckReturnType) {}
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 1 || !FT->getReturnType()->isDoubleTy() ||
        !FT->getParamType(0)->isDoubleTy())
      return 0;

    if (CheckRetType) {
      // Check if all the uses for function like 'sin' are converted to float.
      for (Value::use_iterator UseI = CI->use_begin(); UseI != CI->use_end();
          ++UseI) {
        FPTruncInst *Cast = dyn_cast<FPTruncInst>(*UseI);
        if (Cast == 0 || !Cast->getType()->isFloatTy())
          return 0;
      }
    }

    // If this is something like 'floor((double)floatval)', convert to floorf.
    FPExtInst *Cast = dyn_cast<FPExtInst>(CI->getArgOperand(0));
    if (Cast == 0 || !Cast->getOperand(0)->getType()->isFloatTy())
      return 0;

    // floor((double)floatval) -> (double)floorf(floatval)
    Value *V = Cast->getOperand(0);
    V = EmitUnaryFloatFnCall(V, Callee->getName(), B, Callee->getAttributes());
    return B.CreateFPExt(V, B.getDoubleTy());
  }
};

//===---------------------------------------===//
// 'cos*' Optimizations
struct CosOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    Value *Ret = NULL;
    if (UnsafeFPShrink && Callee->getName() == "cos" &&
        TLI->has(LibFunc::cosf)) {
      UnaryDoubleFPOpt UnsafeUnaryDoubleFP(true);
      Ret = UnsafeUnaryDoubleFP.CallOptimizer(Callee, CI, B);
    }

    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 1 argument of FP type, which matches the
    // result type.
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isFloatingPointTy())
      return Ret;

    // cos(-x) -> cos(x)
    Value *Op1 = CI->getArgOperand(0);
    if (BinaryOperator::isFNeg(Op1)) {
      BinaryOperator *BinExpr = cast<BinaryOperator>(Op1);
      return B.CreateCall(Callee, BinExpr->getOperand(1), "cos");
    }
    return Ret;
  }
};

//===---------------------------------------===//
// 'pow*' Optimizations

struct PowOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    Value *Ret = NULL;
    if (UnsafeFPShrink && Callee->getName() == "pow" &&
        TLI->has(LibFunc::powf)) {
      UnaryDoubleFPOpt UnsafeUnaryDoubleFP(true);
      Ret = UnsafeUnaryDoubleFP.CallOptimizer(Callee, CI, B);
    }

    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        !FT->getParamType(0)->isFloatingPointTy())
      return Ret;

    Value *Op1 = CI->getArgOperand(0), *Op2 = CI->getArgOperand(1);
    if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1)) {
      if (Op1C->isExactlyValue(1.0))  // pow(1.0, x) -> 1.0
        return Op1C;
      if (Op1C->isExactlyValue(2.0))  // pow(2.0, x) -> exp2(x)
        return EmitUnaryFloatFnCall(Op2, "exp2", B, Callee->getAttributes());
    }

    ConstantFP *Op2C = dyn_cast<ConstantFP>(Op2);
    if (Op2C == 0) return Ret;

    if (Op2C->getValueAPF().isZero())  // pow(x, 0.0) -> 1.0
      return ConstantFP::get(CI->getType(), 1.0);

    if (Op2C->isExactlyValue(0.5)) {
      // Expand pow(x, 0.5) to (x == -infinity ? +infinity : fabs(sqrt(x))).
      // This is faster than calling pow, and still handles negative zero
      // and negative infinity correctly.
      // TODO: In fast-math mode, this could be just sqrt(x).
      // TODO: In finite-only mode, this could be just fabs(sqrt(x)).
      Value *Inf = ConstantFP::getInfinity(CI->getType());
      Value *NegInf = ConstantFP::getInfinity(CI->getType(), true);
      Value *Sqrt = EmitUnaryFloatFnCall(Op1, "sqrt", B,
                                         Callee->getAttributes());
      Value *FAbs = EmitUnaryFloatFnCall(Sqrt, "fabs", B,
                                         Callee->getAttributes());
      Value *FCmp = B.CreateFCmpOEQ(Op1, NegInf);
      Value *Sel = B.CreateSelect(FCmp, Inf, FAbs);
      return Sel;
    }

    if (Op2C->isExactlyValue(1.0))  // pow(x, 1.0) -> x
      return Op1;
    if (Op2C->isExactlyValue(2.0))  // pow(x, 2.0) -> x*x
      return B.CreateFMul(Op1, Op1, "pow2");
    if (Op2C->isExactlyValue(-1.0)) // pow(x, -1.0) -> 1.0/x
      return B.CreateFDiv(ConstantFP::get(CI->getType(), 1.0),
                          Op1, "powrecip");
    return 0;
  }
};

//===---------------------------------------===//
// 'exp2' Optimizations

struct Exp2Opt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    Value *Ret = NULL;
    if (UnsafeFPShrink && Callee->getName() == "exp2" &&
        TLI->has(LibFunc::exp2)) {
      UnaryDoubleFPOpt UnsafeUnaryDoubleFP(true);
      Ret = UnsafeUnaryDoubleFP.CallOptimizer(Callee, CI, B);
    }

    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 1 argument of FP type, which matches the
    // result type.
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isFloatingPointTy())
      return Ret;

    Value *Op = CI->getArgOperand(0);
    // Turn exp2(sitofp(x)) -> ldexp(1.0, sext(x))  if sizeof(x) <= 32
    // Turn exp2(uitofp(x)) -> ldexp(1.0, zext(x))  if sizeof(x) < 32
    Value *LdExpArg = 0;
    if (SIToFPInst *OpC = dyn_cast<SIToFPInst>(Op)) {
      if (OpC->getOperand(0)->getType()->getPrimitiveSizeInBits() <= 32)
        LdExpArg = B.CreateSExt(OpC->getOperand(0), B.getInt32Ty());
    } else if (UIToFPInst *OpC = dyn_cast<UIToFPInst>(Op)) {
      if (OpC->getOperand(0)->getType()->getPrimitiveSizeInBits() < 32)
        LdExpArg = B.CreateZExt(OpC->getOperand(0), B.getInt32Ty());
    }

    if (LdExpArg) {
      const char *Name;
      if (Op->getType()->isFloatTy())
        Name = "ldexpf";
      else if (Op->getType()->isDoubleTy())
        Name = "ldexp";
      else
        Name = "ldexpl";

      Constant *One = ConstantFP::get(*Context, APFloat(1.0f));
      if (!Op->getType()->isFloatTy())
        One = ConstantExpr::getFPExtend(One, Op->getType());

      Module *M = Caller->getParent();
      Value *Callee = M->getOrInsertFunction(Name, Op->getType(),
                                             Op->getType(),
                                             B.getInt32Ty(), NULL);
      CallInst *CI = B.CreateCall2(Callee, One, LdExpArg);
      if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
        CI->setCallingConv(F->getCallingConv());

      return CI;
    }
    return Ret;
  }
};

//===----------------------------------------------------------------------===//
// Integer Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'ffs*' Optimizations

struct FFSOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 1 ||
        !FT->getReturnType()->isIntegerTy(32) ||
        !FT->getParamType(0)->isIntegerTy())
      return 0;

    Value *Op = CI->getArgOperand(0);

    // Constant fold.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op)) {
      if (CI->isZero()) // ffs(0) -> 0.
        return B.getInt32(0);
      // ffs(c) -> cttz(c)+1
      return B.getInt32(CI->getValue().countTrailingZeros() + 1);
    }

    // ffs(x) -> x != 0 ? (i32)llvm.cttz(x)+1 : 0
    Type *ArgType = Op->getType();
    Value *F = Intrinsic::getDeclaration(Callee->getParent(),
                                         Intrinsic::cttz, ArgType);
    Value *V = B.CreateCall2(F, Op, B.getFalse(), "cttz");
    V = B.CreateAdd(V, ConstantInt::get(V->getType(), 1));
    V = B.CreateIntCast(V, B.getInt32Ty(), false);

    Value *Cond = B.CreateICmpNE(Op, Constant::getNullValue(ArgType));
    return B.CreateSelect(Cond, V, B.getInt32(0));
  }
};

//===---------------------------------------===//
// 'isdigit' Optimizations

struct IsDigitOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    // We require integer(i32)
    if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
        !FT->getParamType(0)->isIntegerTy(32))
      return 0;

    // isdigit(c) -> (c-'0') <u 10
    Value *Op = CI->getArgOperand(0);
    Op = B.CreateSub(Op, B.getInt32('0'), "isdigittmp");
    Op = B.CreateICmpULT(Op, B.getInt32(10), "isdigit");
    return B.CreateZExt(Op, CI->getType());
  }
};

//===---------------------------------------===//
// 'isascii' Optimizations

struct IsAsciiOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    // We require integer(i32)
    if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
        !FT->getParamType(0)->isIntegerTy(32))
      return 0;

    // isascii(c) -> c <u 128
    Value *Op = CI->getArgOperand(0);
    Op = B.CreateICmpULT(Op, B.getInt32(128), "isascii");
    return B.CreateZExt(Op, CI->getType());
  }
};

//===---------------------------------------===//
// 'abs', 'labs', 'llabs' Optimizations

struct AbsOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    // We require integer(integer) where the types agree.
    if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
        FT->getParamType(0) != FT->getReturnType())
      return 0;

    // abs(x) -> x >s -1 ? x : -x
    Value *Op = CI->getArgOperand(0);
    Value *Pos = B.CreateICmpSGT(Op, Constant::getAllOnesValue(Op->getType()),
                                 "ispos");
    Value *Neg = B.CreateNeg(Op, "neg");
    return B.CreateSelect(Pos, Op, Neg);
  }
};


//===---------------------------------------===//
// 'toascii' Optimizations

struct ToAsciiOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    FunctionType *FT = Callee->getFunctionType();
    // We require i32(i32)
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isIntegerTy(32))
      return 0;

    // isascii(c) -> c & 0x7f
    return B.CreateAnd(CI->getArgOperand(0),
                       ConstantInt::get(CI->getType(),0x7F));
  }
};

//===----------------------------------------------------------------------===//
// Formatting and IO Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'printf' Optimizations

struct PrintFOpt : public LibCallOptimization {
  Value *OptimizeFixedFormatString(Function *Callee, CallInst *CI,
                                   IRBuilder<> &B) {
    // Check for a fixed format string.
    StringRef FormatStr;
    if (!getConstantStringInfo(CI->getArgOperand(0), FormatStr))
      return 0;

    // Empty format string -> noop.
    if (FormatStr.empty())  // Tolerate printf's declared void.
      return CI->use_empty() ? (Value*)CI :
                               ConstantInt::get(CI->getType(), 0);

    // Do not do any of the following transformations if the printf return value
    // is used, in general the printf return value is not compatible with either
    // putchar() or puts().
    if (!CI->use_empty())
      return 0;

    // printf("x") -> putchar('x'), even for '%'.
    if (FormatStr.size() == 1) {
      Value *Res = EmitPutChar(B.getInt32(FormatStr[0]), B, TD, TLI);
      if (CI->use_empty() || !Res) return Res;
      return B.CreateIntCast(Res, CI->getType(), true);
    }

    // printf("foo\n") --> puts("foo")
    if (FormatStr[FormatStr.size()-1] == '\n' &&
        FormatStr.find('%') == std::string::npos) {  // no format characters.
      // Create a string literal with no \n on it.  We expect the constant merge
      // pass to be run after this pass, to merge duplicate strings.
      FormatStr = FormatStr.drop_back();
      Value *GV = B.CreateGlobalString(FormatStr, "str");
      Value *NewCI = EmitPutS(GV, B, TD, TLI);
      return (CI->use_empty() || !NewCI) ?
              NewCI :
              ConstantInt::get(CI->getType(), FormatStr.size()+1);
    }

    // Optimize specific format strings.
    // printf("%c", chr) --> putchar(chr)
    if (FormatStr == "%c" && CI->getNumArgOperands() > 1 &&
        CI->getArgOperand(1)->getType()->isIntegerTy()) {
      Value *Res = EmitPutChar(CI->getArgOperand(1), B, TD, TLI);

      if (CI->use_empty() || !Res) return Res;
      return B.CreateIntCast(Res, CI->getType(), true);
    }

    // printf("%s\n", str) --> puts(str)
    if (FormatStr == "%s\n" && CI->getNumArgOperands() > 1 &&
        CI->getArgOperand(1)->getType()->isPointerTy()) {
      return EmitPutS(CI->getArgOperand(1), B, TD, TLI);
    }
    return 0;
  }

  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require one fixed pointer argument and an integer/void result.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() < 1 || !FT->getParamType(0)->isPointerTy() ||
        !(FT->getReturnType()->isIntegerTy() ||
          FT->getReturnType()->isVoidTy()))
      return 0;

    if (Value *V = OptimizeFixedFormatString(Callee, CI, B)) {
      return V;
    }

    // printf(format, ...) -> iprintf(format, ...) if no floating point
    // arguments.
    if (TLI->has(LibFunc::iprintf) && !CallHasFloatingPointArgument(CI)) {
      Module *M = B.GetInsertBlock()->getParent()->getParent();
      Constant *IPrintFFn =
        M->getOrInsertFunction("iprintf", FT, Callee->getAttributes());
      CallInst *New = cast<CallInst>(CI->clone());
      New->setCalledFunction(IPrintFFn);
      B.Insert(New);
      return New;
    }
    return 0;
  }
};

//===---------------------------------------===//
// 'sprintf' Optimizations

struct SPrintFOpt : public LibCallOptimization {
  Value *OptimizeFixedFormatString(Function *Callee, CallInst *CI,
                                   IRBuilder<> &B) {
    // Check for a fixed format string.
    StringRef FormatStr;
    if (!getConstantStringInfo(CI->getArgOperand(1), FormatStr))
      return 0;

    // If we just have a format string (nothing else crazy) transform it.
    if (CI->getNumArgOperands() == 2) {
      // Make sure there's no % in the constant array.  We could try to handle
      // %% -> % in the future if we cared.
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')
          return 0; // we found a format specifier, bail out.

      // These optimizations require DataLayout.
      if (!TD) return 0;

      // sprintf(str, fmt) -> llvm.memcpy(str, fmt, strlen(fmt)+1, 1)
      B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                     ConstantInt::get(TD->getIntPtrType(*Context), // Copy the
                                      FormatStr.size() + 1), 1);   // nul byte.
      return ConstantInt::get(CI->getType(), FormatStr.size());
    }

    // The remaining optimizations require the format string to be "%s" or "%c"
    // and have an extra operand.
    if (FormatStr.size() != 2 || FormatStr[0] != '%' ||
        CI->getNumArgOperands() < 3)
      return 0;

    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      // sprintf(dst, "%c", chr) --> *(i8*)dst = chr; *((i8*)dst+1) = 0
      if (!CI->getArgOperand(2)->getType()->isIntegerTy()) return 0;
      Value *V = B.CreateTrunc(CI->getArgOperand(2), B.getInt8Ty(), "char");
      Value *Ptr = CastToCStr(CI->getArgOperand(0), B);
      B.CreateStore(V, Ptr);
      Ptr = B.CreateGEP(Ptr, B.getInt32(1), "nul");
      B.CreateStore(B.getInt8(0), Ptr);

      return ConstantInt::get(CI->getType(), 1);
    }

    if (FormatStr[1] == 's') {
      // These optimizations require DataLayout.
      if (!TD) return 0;

      // sprintf(dest, "%s", str) -> llvm.memcpy(dest, str, strlen(str)+1, 1)
      if (!CI->getArgOperand(2)->getType()->isPointerTy()) return 0;

      Value *Len = EmitStrLen(CI->getArgOperand(2), B, TD, TLI);
      if (!Len)
        return 0;
      Value *IncLen = B.CreateAdd(Len,
                                  ConstantInt::get(Len->getType(), 1),
                                  "leninc");
      B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(2), IncLen, 1);

      // The sprintf result is the unincremented number of bytes in the string.
      return B.CreateIntCast(Len, CI->getType(), false);
    }
    return 0;
  }

  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require two fixed pointer arguments and an integer result.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    if (Value *V = OptimizeFixedFormatString(Callee, CI, B)) {
      return V;
    }

    // sprintf(str, format, ...) -> siprintf(str, format, ...) if no floating
    // point arguments.
    if (TLI->has(LibFunc::siprintf) && !CallHasFloatingPointArgument(CI)) {
      Module *M = B.GetInsertBlock()->getParent()->getParent();
      Constant *SIPrintFFn =
        M->getOrInsertFunction("siprintf", FT, Callee->getAttributes());
      CallInst *New = cast<CallInst>(CI->clone());
      New->setCalledFunction(SIPrintFFn);
      B.Insert(New);
      return New;
    }
    return 0;
  }
};

//===---------------------------------------===//
// 'fwrite' Optimizations

struct FWriteOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require a pointer, an integer, an integer, a pointer, returning integer.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 4 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() ||
        !FT->getParamType(2)->isIntegerTy() ||
        !FT->getParamType(3)->isPointerTy() ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    // Get the element size and count.
    ConstantInt *SizeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    ConstantInt *CountC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!SizeC || !CountC) return 0;
    uint64_t Bytes = SizeC->getZExtValue()*CountC->getZExtValue();

    // If this is writing zero records, remove the call (it's a noop).
    if (Bytes == 0)
      return ConstantInt::get(CI->getType(), 0);

    // If this is writing one byte, turn it into fputc.
    // This optimisation is only valid, if the return value is unused.
    if (Bytes == 1 && CI->use_empty()) {  // fwrite(S,1,1,F) -> fputc(S[0],F)
      Value *Char = B.CreateLoad(CastToCStr(CI->getArgOperand(0), B), "char");
      Value *NewCI = EmitFPutC(Char, CI->getArgOperand(3), B, TD, TLI);
      return NewCI ? ConstantInt::get(CI->getType(), 1) : 0;
    }

    return 0;
  }
};

//===---------------------------------------===//
// 'fputs' Optimizations

struct FPutsOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // These optimizations require DataLayout.
    if (!TD) return 0;

    // Require two pointers.  Also, we can't optimize if return value is used.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !CI->use_empty())
      return 0;

    // fputs(s,F) --> fwrite(s,1,strlen(s),F)
    uint64_t Len = GetStringLength(CI->getArgOperand(0));
    if (!Len) return 0;
    // Known to have no uses (see above).
    return EmitFWrite(CI->getArgOperand(0),
                      ConstantInt::get(TD->getIntPtrType(*Context), Len-1),
                      CI->getArgOperand(1), B, TD, TLI);
  }
};

//===---------------------------------------===//
// 'fprintf' Optimizations

struct FPrintFOpt : public LibCallOptimization {
  Value *OptimizeFixedFormatString(Function *Callee, CallInst *CI,
                                   IRBuilder<> &B) {
    // All the optimizations depend on the format string.
    StringRef FormatStr;
    if (!getConstantStringInfo(CI->getArgOperand(1), FormatStr))
      return 0;

    // fprintf(F, "foo") --> fwrite("foo", 3, 1, F)
    if (CI->getNumArgOperands() == 2) {
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')  // Could handle %% -> % if we cared.
          return 0; // We found a format specifier.

      // These optimizations require DataLayout.
      if (!TD) return 0;

      Value *NewCI = EmitFWrite(CI->getArgOperand(1),
                                ConstantInt::get(TD->getIntPtrType(*Context),
                                                 FormatStr.size()),
                                CI->getArgOperand(0), B, TD, TLI);
      return NewCI ? ConstantInt::get(CI->getType(), FormatStr.size()) : 0;
    }

    // The remaining optimizations require the format string to be "%s" or "%c"
    // and have an extra operand.
    if (FormatStr.size() != 2 || FormatStr[0] != '%' ||
        CI->getNumArgOperands() < 3)
      return 0;

    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      // fprintf(F, "%c", chr) --> fputc(chr, F)
      if (!CI->getArgOperand(2)->getType()->isIntegerTy()) return 0;
      Value *NewCI = EmitFPutC(CI->getArgOperand(2), CI->getArgOperand(0), B,
                               TD, TLI);
      return NewCI ? ConstantInt::get(CI->getType(), 1) : 0;
    }

    if (FormatStr[1] == 's') {
      // fprintf(F, "%s", str) --> fputs(str, F)
      if (!CI->getArgOperand(2)->getType()->isPointerTy() || !CI->use_empty())
        return 0;
      return EmitFPutS(CI->getArgOperand(2), CI->getArgOperand(0), B, TD, TLI);
    }
    return 0;
  }

  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require two fixed paramters as pointers and integer result.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    if (Value *V = OptimizeFixedFormatString(Callee, CI, B)) {
      return V;
    }

    // fprintf(stream, format, ...) -> fiprintf(stream, format, ...) if no
    // floating point arguments.
    if (TLI->has(LibFunc::fiprintf) && !CallHasFloatingPointArgument(CI)) {
      Module *M = B.GetInsertBlock()->getParent()->getParent();
      Constant *FIPrintFFn =
        M->getOrInsertFunction("fiprintf", FT, Callee->getAttributes());
      CallInst *New = cast<CallInst>(CI->clone());
      New->setCalledFunction(FIPrintFFn);
      B.Insert(New);
      return New;
    }
    return 0;
  }
};

//===---------------------------------------===//
// 'puts' Optimizations

struct PutsOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require one fixed pointer argument and an integer/void result.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() < 1 || !FT->getParamType(0)->isPointerTy() ||
        !(FT->getReturnType()->isIntegerTy() ||
          FT->getReturnType()->isVoidTy()))
      return 0;

    // Check for a constant string.
    StringRef Str;
    if (!getConstantStringInfo(CI->getArgOperand(0), Str))
      return 0;

    if (Str.empty() && CI->use_empty()) {
      // puts("") -> putchar('\n')
      Value *Res = EmitPutChar(B.getInt32('\n'), B, TD, TLI);
      if (CI->use_empty() || !Res) return Res;
      return B.CreateIntCast(Res, CI->getType(), true);
    }

    return 0;
  }
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// SimplifyLibCalls Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// This pass optimizes well known library functions from libc and libm.
  ///
  class SimplifyLibCalls : public FunctionPass {
    TargetLibraryInfo *TLI;

    StringMap<LibCallOptimization*> Optimizations;
    // Math Library Optimizations
    CosOpt Cos; PowOpt Pow; Exp2Opt Exp2;
    UnaryDoubleFPOpt UnaryDoubleFP, UnsafeUnaryDoubleFP;
    // Integer Optimizations
    FFSOpt FFS; AbsOpt Abs; IsDigitOpt IsDigit; IsAsciiOpt IsAscii;
    ToAsciiOpt ToAscii;
    // Formatting and IO Optimizations
    SPrintFOpt SPrintF; PrintFOpt PrintF;
    FWriteOpt FWrite; FPutsOpt FPuts; FPrintFOpt FPrintF;
    PutsOpt Puts;

    bool Modified;  // This is only used by doInitialization.
  public:
    static char ID; // Pass identification
    SimplifyLibCalls() : FunctionPass(ID), UnaryDoubleFP(false),
                         UnsafeUnaryDoubleFP(true) {
      initializeSimplifyLibCallsPass(*PassRegistry::getPassRegistry());
    }
    void AddOpt(LibFunc::Func F, LibCallOptimization* Opt);
    void AddOpt(LibFunc::Func F1, LibFunc::Func F2, LibCallOptimization* Opt);

    void InitOptimizations();
    bool runOnFunction(Function &F);

    void setDoesNotAccessMemory(Function &F);
    void setOnlyReadsMemory(Function &F);
    void setDoesNotThrow(Function &F);
    void setDoesNotCapture(Function &F, unsigned n);
    void setDoesNotAlias(Function &F, unsigned n);
    bool doInitialization(Module &M);

    void inferPrototypeAttributes(Function &F);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetLibraryInfo>();
    }
  };
} // end anonymous namespace.

char SimplifyLibCalls::ID = 0;

INITIALIZE_PASS_BEGIN(SimplifyLibCalls, "simplify-libcalls",
                      "Simplify well-known library calls", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(SimplifyLibCalls, "simplify-libcalls",
                    "Simplify well-known library calls", false, false)

// Public interface to the Simplify LibCalls pass.
FunctionPass *llvm::createSimplifyLibCallsPass() {
  return new SimplifyLibCalls();
}

void SimplifyLibCalls::AddOpt(LibFunc::Func F, LibCallOptimization* Opt) {
  if (TLI->has(F))
    Optimizations[TLI->getName(F)] = Opt;
}

void SimplifyLibCalls::AddOpt(LibFunc::Func F1, LibFunc::Func F2,
                              LibCallOptimization* Opt) {
  if (TLI->has(F1) && TLI->has(F2))
    Optimizations[TLI->getName(F1)] = Opt;
}

/// Optimizations - Populate the Optimizations map with all the optimizations
/// we know.
void SimplifyLibCalls::InitOptimizations() {
  // Math Library Optimizations
  Optimizations["cosf"] = &Cos;
  Optimizations["cos"] = &Cos;
  Optimizations["cosl"] = &Cos;
  Optimizations["powf"] = &Pow;
  Optimizations["pow"] = &Pow;
  Optimizations["powl"] = &Pow;
  Optimizations["llvm.pow.f32"] = &Pow;
  Optimizations["llvm.pow.f64"] = &Pow;
  Optimizations["llvm.pow.f80"] = &Pow;
  Optimizations["llvm.pow.f128"] = &Pow;
  Optimizations["llvm.pow.ppcf128"] = &Pow;
  Optimizations["exp2l"] = &Exp2;
  Optimizations["exp2"] = &Exp2;
  Optimizations["exp2f"] = &Exp2;
  Optimizations["llvm.exp2.ppcf128"] = &Exp2;
  Optimizations["llvm.exp2.f128"] = &Exp2;
  Optimizations["llvm.exp2.f80"] = &Exp2;
  Optimizations["llvm.exp2.f64"] = &Exp2;
  Optimizations["llvm.exp2.f32"] = &Exp2;

  AddOpt(LibFunc::ceil, LibFunc::ceilf, &UnaryDoubleFP);
  AddOpt(LibFunc::fabs, LibFunc::fabsf, &UnaryDoubleFP);
  AddOpt(LibFunc::floor, LibFunc::floorf, &UnaryDoubleFP);
  AddOpt(LibFunc::rint, LibFunc::rintf, &UnaryDoubleFP);
  AddOpt(LibFunc::round, LibFunc::roundf, &UnaryDoubleFP);
  AddOpt(LibFunc::nearbyint, LibFunc::nearbyintf, &UnaryDoubleFP);
  AddOpt(LibFunc::trunc, LibFunc::truncf, &UnaryDoubleFP);

  if(UnsafeFPShrink) {
    AddOpt(LibFunc::acos, LibFunc::acosf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::acosh, LibFunc::acoshf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::asin, LibFunc::asinf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::asinh, LibFunc::asinhf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::atan, LibFunc::atanf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::atanh, LibFunc::atanhf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::cbrt, LibFunc::cbrtf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::cosh, LibFunc::coshf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::exp, LibFunc::expf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::exp10, LibFunc::exp10f, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::expm1, LibFunc::expm1f, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::log, LibFunc::logf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::log10, LibFunc::log10f, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::log1p, LibFunc::log1pf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::log2, LibFunc::log2f, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::logb, LibFunc::logbf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::sin, LibFunc::sinf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::sinh, LibFunc::sinhf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::sqrt, LibFunc::sqrtf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::tan, LibFunc::tanf, &UnsafeUnaryDoubleFP);
    AddOpt(LibFunc::tanh, LibFunc::tanhf, &UnsafeUnaryDoubleFP);
  }

  // Integer Optimizations
  Optimizations["ffs"] = &FFS;
  Optimizations["ffsl"] = &FFS;
  Optimizations["ffsll"] = &FFS;
  Optimizations["abs"] = &Abs;
  Optimizations["labs"] = &Abs;
  Optimizations["llabs"] = &Abs;
  Optimizations["isdigit"] = &IsDigit;
  Optimizations["isascii"] = &IsAscii;
  Optimizations["toascii"] = &ToAscii;

  // Formatting and IO Optimizations
  Optimizations["sprintf"] = &SPrintF;
  Optimizations["printf"] = &PrintF;
  AddOpt(LibFunc::fwrite, &FWrite);
  AddOpt(LibFunc::fputs, &FPuts);
  Optimizations["fprintf"] = &FPrintF;
  Optimizations["puts"] = &Puts;
}


/// runOnFunction - Top level algorithm.
///
bool SimplifyLibCalls::runOnFunction(Function &F) {
  TLI = &getAnalysis<TargetLibraryInfo>();

  if (Optimizations.empty())
    InitOptimizations();

  const DataLayout *TD = getAnalysisIfAvailable<DataLayout>();

  IRBuilder<> Builder(F.getContext());

  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      // Ignore non-calls.
      CallInst *CI = dyn_cast<CallInst>(I++);
      if (!CI) continue;

      // Ignore indirect calls and calls to non-external functions.
      Function *Callee = CI->getCalledFunction();
      if (Callee == 0 || !Callee->isDeclaration() ||
          !(Callee->hasExternalLinkage() || Callee->hasDLLImportLinkage()))
        continue;

      // Ignore unknown calls.
      LibCallOptimization *LCO = Optimizations.lookup(Callee->getName());
      if (!LCO) continue;

      // Set the builder to the instruction after the call.
      Builder.SetInsertPoint(BB, I);

      // Use debug location of CI for all new instructions.
      Builder.SetCurrentDebugLocation(CI->getDebugLoc());

      // Try to optimize this call.
      Value *Result = LCO->OptimizeCall(CI, TD, TLI, Builder);
      if (Result == 0) continue;

      DEBUG(dbgs() << "SimplifyLibCalls simplified: " << *CI;
            dbgs() << "  into: " << *Result << "\n");

      // Something changed!
      Changed = true;
      ++NumSimplified;

      // Inspect the instruction after the call (which was potentially just
      // added) next.
      I = CI; ++I;

      if (CI != Result && !CI->use_empty()) {
        CI->replaceAllUsesWith(Result);
        if (!Result->hasName())
          Result->takeName(CI);
      }
      CI->eraseFromParent();
    }
  }
  return Changed;
}

// Utility methods for doInitialization.

void SimplifyLibCalls::setDoesNotAccessMemory(Function &F) {
  if (!F.doesNotAccessMemory()) {
    F.setDoesNotAccessMemory();
    ++NumAnnotated;
    Modified = true;
  }
}
void SimplifyLibCalls::setOnlyReadsMemory(Function &F) {
  if (!F.onlyReadsMemory()) {
    F.setOnlyReadsMemory();
    ++NumAnnotated;
    Modified = true;
  }
}
void SimplifyLibCalls::setDoesNotThrow(Function &F) {
  if (!F.doesNotThrow()) {
    F.setDoesNotThrow();
    ++NumAnnotated;
    Modified = true;
  }
}
void SimplifyLibCalls::setDoesNotCapture(Function &F, unsigned n) {
  if (!F.doesNotCapture(n)) {
    F.setDoesNotCapture(n);
    ++NumAnnotated;
    Modified = true;
  }
}
void SimplifyLibCalls::setDoesNotAlias(Function &F, unsigned n) {
  if (!F.doesNotAlias(n)) {
    F.setDoesNotAlias(n);
    ++NumAnnotated;
    Modified = true;
  }
}


void SimplifyLibCalls::inferPrototypeAttributes(Function &F) {
  FunctionType *FTy = F.getFunctionType();

  StringRef Name = F.getName();
  switch (Name[0]) {
  case 's':
    if (Name == "strlen") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setOnlyReadsMemory(F);
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "strchr" ||
               Name == "strrchr") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isIntegerTy())
        return;
      setOnlyReadsMemory(F);
      setDoesNotThrow(F);
    } else if (Name == "strcpy" ||
               Name == "stpcpy" ||
               Name == "strcat" ||
               Name == "strtol" ||
               Name == "strtod" ||
               Name == "strtof" ||
               Name == "strtoul" ||
               Name == "strtoll" ||
               Name == "strtold" ||
               Name == "strncat" ||
               Name == "strncpy" ||
               Name == "stpncpy" ||
               Name == "strtoull") {
      if (FTy->getNumParams() < 2 ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "strxfrm") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "strcmp" ||
               Name == "strspn" ||
               Name == "strncmp" ||
               Name == "strcspn" ||
               Name == "strcoll" ||
               Name == "strcasecmp" ||
               Name == "strncasecmp") {
      if (FTy->getNumParams() < 2 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setOnlyReadsMemory(F);
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "strstr" ||
               Name == "strpbrk") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setOnlyReadsMemory(F);
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "strtok" ||
               Name == "strtok_r") {
      if (FTy->getNumParams() < 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "scanf" ||
               Name == "setbuf" ||
               Name == "setvbuf") {
      if (FTy->getNumParams() < 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "strdup" ||
               Name == "strndup") {
      if (FTy->getNumParams() < 1 || !FTy->getReturnType()->isPointerTy() ||
          !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
      setDoesNotCapture(F, 1);
    } else if (Name == "stat" ||
               Name == "sscanf" ||
               Name == "sprintf" ||
               Name == "statvfs") {
      if (FTy->getNumParams() < 2 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "snprintf") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(2)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 3);
    } else if (Name == "setitimer") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(1)->isPointerTy() ||
          !FTy->getParamType(2)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
      setDoesNotCapture(F, 3);
    } else if (Name == "system") {
      if (FTy->getNumParams() != 1 ||
          !FTy->getParamType(0)->isPointerTy())
        return;
      // May throw; "system" is a valid pthread cancellation point.
      setDoesNotCapture(F, 1);
    }
    break;
  case 'm':
    if (Name == "malloc") {
      if (FTy->getNumParams() != 1 ||
          !FTy->getReturnType()->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
    } else if (Name == "memcmp") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setOnlyReadsMemory(F);
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "memchr" ||
               Name == "memrchr") {
      if (FTy->getNumParams() != 3)
        return;
      setOnlyReadsMemory(F);
      setDoesNotThrow(F);
    } else if (Name == "modf" ||
               Name == "modff" ||
               Name == "modfl" ||
               Name == "memcpy" ||
               Name == "memccpy" ||
               Name == "memmove") {
      if (FTy->getNumParams() < 2 ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "memalign") {
      if (!FTy->getReturnType()->isPointerTy())
        return;
      setDoesNotAlias(F, 0);
    } else if (Name == "mkdir" ||
               Name == "mktime") {
      if (FTy->getNumParams() == 0 ||
          !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    }
    break;
  case 'r':
    if (Name == "realloc") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getReturnType()->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
      setDoesNotCapture(F, 1);
    } else if (Name == "read") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      // May throw; "read" is a valid pthread cancellation point.
      setDoesNotCapture(F, 2);
    } else if (Name == "rmdir" ||
               Name == "rewind" ||
               Name == "remove" ||
               Name == "realpath") {
      if (FTy->getNumParams() < 1 ||
          !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "rename" ||
               Name == "readlink") {
      if (FTy->getNumParams() < 2 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    }
    break;
  case 'w':
    if (Name == "write") {
      if (FTy->getNumParams() != 3 || !FTy->getParamType(1)->isPointerTy())
        return;
      // May throw; "write" is a valid pthread cancellation point.
      setDoesNotCapture(F, 2);
    }
    break;
  case 'b':
    if (Name == "bcopy") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "bcmp") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setOnlyReadsMemory(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "bzero") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    }
    break;
  case 'c':
    if (Name == "calloc") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getReturnType()->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
    } else if (Name == "chmod" ||
               Name == "chown" ||
               Name == "ctermid" ||
               Name == "clearerr" ||
               Name == "closedir") {
      if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    }
    break;
  case 'a':
    if (Name == "atoi" ||
        Name == "atol" ||
        Name == "atof" ||
        Name == "atoll") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setOnlyReadsMemory(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "access") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    }
    break;
  case 'f':
    if (Name == "fopen") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getReturnType()->isPointerTy() ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "fdopen") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getReturnType()->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
      setDoesNotCapture(F, 2);
    } else if (Name == "feof" ||
               Name == "free" ||
               Name == "fseek" ||
               Name == "ftell" ||
               Name == "fgetc" ||
               Name == "fseeko" ||
               Name == "ftello" ||
               Name == "fileno" ||
               Name == "fflush" ||
               Name == "fclose" ||
               Name == "fsetpos" ||
               Name == "flockfile" ||
               Name == "funlockfile" ||
               Name == "ftrylockfile") {
      if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "ferror") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setOnlyReadsMemory(F);
    } else if (Name == "fputc" ||
               Name == "fstat" ||
               Name == "frexp" ||
               Name == "frexpf" ||
               Name == "frexpl" ||
               Name == "fstatvfs") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "fgets") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(2)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 3);
    } else if (Name == "fread" ||
               Name == "fwrite") {
      if (FTy->getNumParams() != 4 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(3)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 4);
    } else if (Name == "fputs" ||
               Name == "fscanf" ||
               Name == "fprintf" ||
               Name == "fgetpos") {
      if (FTy->getNumParams() < 2 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    }
    break;
  case 'g':
    if (Name == "getc" ||
        Name == "getlogin_r" ||
        Name == "getc_unlocked") {
      if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "getenv") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setOnlyReadsMemory(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "gets" ||
               Name == "getchar") {
      setDoesNotThrow(F);
    } else if (Name == "getitimer") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "getpwnam") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    }
    break;
  case 'u':
    if (Name == "ungetc") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "uname" ||
               Name == "unlink" ||
               Name == "unsetenv") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "utime" ||
               Name == "utimes") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    }
    break;
  case 'p':
    if (Name == "putc") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "puts" ||
               Name == "printf" ||
               Name == "perror") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "pread" ||
               Name == "pwrite") {
      if (FTy->getNumParams() != 4 || !FTy->getParamType(1)->isPointerTy())
        return;
      // May throw; these are valid pthread cancellation points.
      setDoesNotCapture(F, 2);
    } else if (Name == "putchar") {
      setDoesNotThrow(F);
    } else if (Name == "popen") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getReturnType()->isPointerTy() ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "pclose") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    }
    break;
  case 'v':
    if (Name == "vscanf") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "vsscanf" ||
               Name == "vfscanf") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(1)->isPointerTy() ||
          !FTy->getParamType(2)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "valloc") {
      if (!FTy->getReturnType()->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
    } else if (Name == "vprintf") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "vfprintf" ||
               Name == "vsprintf") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "vsnprintf") {
      if (FTy->getNumParams() != 4 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(2)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 3);
    }
    break;
  case 'o':
    if (Name == "open") {
      if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy())
        return;
      // May throw; "open" is a valid pthread cancellation point.
      setDoesNotCapture(F, 1);
    } else if (Name == "opendir") {
      if (FTy->getNumParams() != 1 ||
          !FTy->getReturnType()->isPointerTy() ||
          !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
      setDoesNotCapture(F, 1);
    }
    break;
  case 't':
    if (Name == "tmpfile") {
      if (!FTy->getReturnType()->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
    } else if (Name == "times") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    }
    break;
  case 'h':
    if (Name == "htonl" ||
        Name == "htons") {
      setDoesNotThrow(F);
      setDoesNotAccessMemory(F);
    }
    break;
  case 'n':
    if (Name == "ntohl" ||
        Name == "ntohs") {
      setDoesNotThrow(F);
      setDoesNotAccessMemory(F);
    }
    break;
  case 'l':
    if (Name == "lstat") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "lchown") {
      if (FTy->getNumParams() != 3 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    }
    break;
  case 'q':
    if (Name == "qsort") {
      if (FTy->getNumParams() != 4 || !FTy->getParamType(3)->isPointerTy())
        return;
      // May throw; places call through function pointer.
      setDoesNotCapture(F, 4);
    }
    break;
  case '_':
    if (Name == "__strdup" ||
        Name == "__strndup") {
      if (FTy->getNumParams() < 1 ||
          !FTy->getReturnType()->isPointerTy() ||
          !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
      setDoesNotCapture(F, 1);
    } else if (Name == "__strtok_r") {
      if (FTy->getNumParams() != 3 ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "_IO_getc") {
      if (FTy->getNumParams() != 1 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "_IO_putc") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    }
    break;
  case 1:
    if (Name == "\1__isoc99_scanf") {
      if (FTy->getNumParams() < 1 ||
          !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "\1stat64" ||
               Name == "\1lstat64" ||
               Name == "\1statvfs64" ||
               Name == "\1__isoc99_sscanf") {
      if (FTy->getNumParams() < 1 ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "\1fopen64") {
      if (FTy->getNumParams() != 2 ||
          !FTy->getReturnType()->isPointerTy() ||
          !FTy->getParamType(0)->isPointerTy() ||
          !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
      setDoesNotCapture(F, 1);
      setDoesNotCapture(F, 2);
    } else if (Name == "\1fseeko64" ||
               Name == "\1ftello64") {
      if (FTy->getNumParams() == 0 || !FTy->getParamType(0)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 1);
    } else if (Name == "\1tmpfile64") {
      if (!FTy->getReturnType()->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotAlias(F, 0);
    } else if (Name == "\1fstat64" ||
               Name == "\1fstatvfs64") {
      if (FTy->getNumParams() != 2 || !FTy->getParamType(1)->isPointerTy())
        return;
      setDoesNotThrow(F);
      setDoesNotCapture(F, 2);
    } else if (Name == "\1open64") {
      if (FTy->getNumParams() < 2 || !FTy->getParamType(0)->isPointerTy())
        return;
      // May throw; "open" is a valid pthread cancellation point.
      setDoesNotCapture(F, 1);
    }
    break;
  }
}

/// doInitialization - Add attributes to well-known functions.
///
bool SimplifyLibCalls::doInitialization(Module &M) {
  Modified = false;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Function &F = *I;
    if (F.isDeclaration() && F.hasName())
      inferPrototypeAttributes(F);
  }
  return Modified;
}

// TODO:
//   Additional cases that we need to add to this file:
//
// cbrt:
//   * cbrt(expN(X))  -> expN(x/3)
//   * cbrt(sqrt(x))  -> pow(x,1/6)
//   * cbrt(sqrt(x))  -> pow(x,1/9)
//
// exp, expf, expl:
//   * exp(log(x))  -> x
//
// log, logf, logl:
//   * log(exp(x))   -> x
//   * log(x**y)     -> y*log(x)
//   * log(exp(y))   -> y*log(e)
//   * log(exp2(y))  -> y*log(2)
//   * log(exp10(y)) -> y*log(10)
//   * log(sqrt(x))  -> 0.5*log(x)
//   * log(pow(x,y)) -> y*log(x)
//
// lround, lroundf, lroundl:
//   * lround(cnst) -> cnst'
//
// pow, powf, powl:
//   * pow(exp(x),y)  -> exp(x*y)
//   * pow(sqrt(x),y) -> pow(x,y*0.5)
//   * pow(pow(x,y),z)-> pow(x,y*z)
//
// round, roundf, roundl:
//   * round(cnst) -> cnst'
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
// strchr:
//   * strchr(p, 0) -> strlen(p)
// tan, tanf, tanl:
//   * tan(atan(x)) -> x
//
// trunc, truncf, truncl:
//   * trunc(cnst) -> cnst'
//
//
