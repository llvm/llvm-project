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
// library functions). For example, a call to the function "exit(3)" that
// occurs within the main() function can be transformed into a simple "return 3"
// instruction. Any optimization that takes this form (replace call to library
// function with simpler code that provides the same result) belongs in this
// file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplify-libcalls"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Config/config.h"
using namespace llvm;

STATISTIC(NumSimplified, "Number of library calls simplified");
STATISTIC(NumAnnotated, "Number of attributes added to library functions");

//===----------------------------------------------------------------------===//
// Optimizer Base Class
//===----------------------------------------------------------------------===//

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
class VISIBILITY_HIDDEN LibCallOptimization {
protected:
  Function *Caller;
  const TargetData *TD;
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
  
  Value *OptimizeCall(CallInst *CI, const TargetData &TD, IRBuilder<> &B) {
    Caller = CI->getParent()->getParent();
    this->TD = &TD;
    return CallOptimizer(CI->getCalledFunction(), CI, B);
  }

  /// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
  Value *CastToCStr(Value *V, IRBuilder<> &B);

  /// EmitStrLen - Emit a call to the strlen function to the builder, for the
  /// specified pointer.  Ptr is required to be some pointer type, and the
  /// return value has 'intptr_t' type.
  Value *EmitStrLen(Value *Ptr, IRBuilder<> &B);
  
  /// EmitMemCpy - Emit a call to the memcpy function to the builder.  This
  /// always expects that the size has type 'intptr_t' and Dst/Src are pointers.
  Value *EmitMemCpy(Value *Dst, Value *Src, Value *Len, 
                    unsigned Align, IRBuilder<> &B);
  
  /// EmitMemChr - Emit a call to the memchr function.  This assumes that Ptr is
  /// a pointer, Val is an i32 value, and Len is an 'intptr_t' value.
  Value *EmitMemChr(Value *Ptr, Value *Val, Value *Len, IRBuilder<> &B);

  /// EmitMemCmp - Emit a call to the memcmp function.
  Value *EmitMemCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B);

  /// EmitUnaryFloatFnCall - Emit a call to the unary function named 'Name' (e.g.
  /// 'floor').  This function is known to take a single of type matching 'Op'
  /// and returns one value with the same type.  If 'Op' is a long double, 'l'
  /// is added as the suffix of name, if 'Op' is a float, we add a 'f' suffix.
  Value *EmitUnaryFloatFnCall(Value *Op, const char *Name, IRBuilder<> &B);
  
  /// EmitPutChar - Emit a call to the putchar function.  This assumes that Char
  /// is an integer.
  void EmitPutChar(Value *Char, IRBuilder<> &B);
  
  /// EmitPutS - Emit a call to the puts function.  This assumes that Str is
  /// some pointer.
  void EmitPutS(Value *Str, IRBuilder<> &B);
    
  /// EmitFPutC - Emit a call to the fputc function.  This assumes that Char is
  /// an i32, and File is a pointer to FILE.
  void EmitFPutC(Value *Char, Value *File, IRBuilder<> &B);
  
  /// EmitFPutS - Emit a call to the puts function.  Str is required to be a
  /// pointer and File is a pointer to FILE.
  void EmitFPutS(Value *Str, Value *File, IRBuilder<> &B);
  
  /// EmitFWrite - Emit a call to the fwrite function.  This assumes that Ptr is
  /// a pointer, Size is an 'intptr_t', and File is a pointer to FILE.
  void EmitFWrite(Value *Ptr, Value *Size, Value *File, IRBuilder<> &B);
  
};
} // End anonymous namespace.

/// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
Value *LibCallOptimization::CastToCStr(Value *V, IRBuilder<> &B) {
  return B.CreateBitCast(V, PointerType::getUnqual(Type::Int8Ty), "cstr");
}

/// EmitStrLen - Emit a call to the strlen function to the builder, for the
/// specified pointer.  This always returns an integer value of size intptr_t.
Value *LibCallOptimization::EmitStrLen(Value *Ptr, IRBuilder<> &B) {
  Module *M = Caller->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(1, Attribute::NoCapture);
  AWI[1] = AttributeWithIndex::get(~0u, Attribute::ReadOnly |
                                   Attribute::NoUnwind);

  Constant *StrLen =M->getOrInsertFunction("strlen", AttrListPtr::get(AWI, 2),
                                           TD->getIntPtrType(),
                                           PointerType::getUnqual(Type::Int8Ty),
                                           NULL);
  return B.CreateCall(StrLen, CastToCStr(Ptr, B), "strlen");
}

/// EmitMemCpy - Emit a call to the memcpy function to the builder.  This always
/// expects that the size has type 'intptr_t' and Dst/Src are pointers.
Value *LibCallOptimization::EmitMemCpy(Value *Dst, Value *Src, Value *Len,
                                       unsigned Align, IRBuilder<> &B) {
  Module *M = Caller->getParent();
  Intrinsic::ID IID = Intrinsic::memcpy;
  const Type *Tys[1];
  Tys[0] = Len->getType();
  Value *MemCpy = Intrinsic::getDeclaration(M, IID, Tys, 1);
  return B.CreateCall4(MemCpy, CastToCStr(Dst, B), CastToCStr(Src, B), Len,
                       ConstantInt::get(Type::Int32Ty, Align));
}

/// EmitMemChr - Emit a call to the memchr function.  This assumes that Ptr is
/// a pointer, Val is an i32 value, and Len is an 'intptr_t' value.
Value *LibCallOptimization::EmitMemChr(Value *Ptr, Value *Val,
                                       Value *Len, IRBuilder<> &B) {
  Module *M = Caller->getParent();
  AttributeWithIndex AWI;
  AWI = AttributeWithIndex::get(~0u, Attribute::ReadOnly | Attribute::NoUnwind);

  Value *MemChr = M->getOrInsertFunction("memchr", AttrListPtr::get(&AWI, 1),
                                         PointerType::getUnqual(Type::Int8Ty),
                                         PointerType::getUnqual(Type::Int8Ty),
                                         Type::Int32Ty, TD->getIntPtrType(),
                                         NULL);
  return B.CreateCall3(MemChr, CastToCStr(Ptr, B), Val, Len, "memchr");
}

/// EmitMemCmp - Emit a call to the memcmp function.
Value *LibCallOptimization::EmitMemCmp(Value *Ptr1, Value *Ptr2,
                                       Value *Len, IRBuilder<> &B) {
  Module *M = Caller->getParent();
  AttributeWithIndex AWI[3];
  AWI[0] = AttributeWithIndex::get(1, Attribute::NoCapture);
  AWI[1] = AttributeWithIndex::get(2, Attribute::NoCapture);
  AWI[2] = AttributeWithIndex::get(~0u, Attribute::ReadOnly |
                                   Attribute::NoUnwind);

  Value *MemCmp = M->getOrInsertFunction("memcmp", AttrListPtr::get(AWI, 3),
                                         Type::Int32Ty,
                                         PointerType::getUnqual(Type::Int8Ty),
                                         PointerType::getUnqual(Type::Int8Ty),
                                         TD->getIntPtrType(), NULL);
  return B.CreateCall3(MemCmp, CastToCStr(Ptr1, B), CastToCStr(Ptr2, B),
                       Len, "memcmp");
}

/// EmitUnaryFloatFnCall - Emit a call to the unary function named 'Name' (e.g.
/// 'floor').  This function is known to take a single of type matching 'Op' and
/// returns one value with the same type.  If 'Op' is a long double, 'l' is
/// added as the suffix of name, if 'Op' is a float, we add a 'f' suffix.
Value *LibCallOptimization::EmitUnaryFloatFnCall(Value *Op, const char *Name,
                                                 IRBuilder<> &B) {
  char NameBuffer[20];
  if (Op->getType() != Type::DoubleTy) {
    // If we need to add a suffix, copy into NameBuffer.
    unsigned NameLen = strlen(Name);
    assert(NameLen < sizeof(NameBuffer)-2);
    memcpy(NameBuffer, Name, NameLen);
    if (Op->getType() == Type::FloatTy)
      NameBuffer[NameLen] = 'f';  // floorf
    else
      NameBuffer[NameLen] = 'l';  // floorl
    NameBuffer[NameLen+1] = 0;
    Name = NameBuffer;
  }
  
  Module *M = Caller->getParent();
  Value *Callee = M->getOrInsertFunction(Name, Op->getType(), 
                                         Op->getType(), NULL);
  return B.CreateCall(Callee, Op, Name);
}

/// EmitPutChar - Emit a call to the putchar function.  This assumes that Char
/// is an integer.
void LibCallOptimization::EmitPutChar(Value *Char, IRBuilder<> &B) {
  Module *M = Caller->getParent();
  Value *F = M->getOrInsertFunction("putchar", Type::Int32Ty,
                                    Type::Int32Ty, NULL);
  B.CreateCall(F, B.CreateIntCast(Char, Type::Int32Ty, "chari"), "putchar");
}

/// EmitPutS - Emit a call to the puts function.  This assumes that Str is
/// some pointer.
void LibCallOptimization::EmitPutS(Value *Str, IRBuilder<> &B) {
  Module *M = Caller->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(1, Attribute::NoCapture);
  AWI[1] = AttributeWithIndex::get(~0u, Attribute::NoUnwind);

  Value *F = M->getOrInsertFunction("puts", AttrListPtr::get(AWI, 2),
                                    Type::Int32Ty,
                                    PointerType::getUnqual(Type::Int8Ty), NULL);
  B.CreateCall(F, CastToCStr(Str, B), "puts");
}

/// EmitFPutC - Emit a call to the fputc function.  This assumes that Char is
/// an integer and File is a pointer to FILE.
void LibCallOptimization::EmitFPutC(Value *Char, Value *File, IRBuilder<> &B) {
  Module *M = Caller->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(2, Attribute::NoCapture);
  AWI[1] = AttributeWithIndex::get(~0u, Attribute::NoUnwind);
  Constant *F;
  if (isa<PointerType>(File->getType()))
    F = M->getOrInsertFunction("fputc", AttrListPtr::get(AWI, 2), Type::Int32Ty,
                               Type::Int32Ty, File->getType(), NULL);
                                         
  else
    F = M->getOrInsertFunction("fputc", Type::Int32Ty, Type::Int32Ty,
                               File->getType(), NULL);
  Char = B.CreateIntCast(Char, Type::Int32Ty, "chari");
  B.CreateCall2(F, Char, File, "fputc");
}

/// EmitFPutS - Emit a call to the puts function.  Str is required to be a
/// pointer and File is a pointer to FILE.
void LibCallOptimization::EmitFPutS(Value *Str, Value *File, IRBuilder<> &B) {
  Module *M = Caller->getParent();
  AttributeWithIndex AWI[2];
  AWI[0] = AttributeWithIndex::get(2, Attribute::NoCapture);
  AWI[1] = AttributeWithIndex::get(~0u, Attribute::NoUnwind);
  Constant *F;
  if (isa<PointerType>(File->getType()))
    F = M->getOrInsertFunction("fputs", AttrListPtr::get(AWI, 2), Type::Int32Ty,
                               PointerType::getUnqual(Type::Int8Ty),
                               File->getType(), NULL);
  else
    F = M->getOrInsertFunction("fputs", Type::Int32Ty,
                               PointerType::getUnqual(Type::Int8Ty),
                               File->getType(), NULL);
  B.CreateCall2(F, CastToCStr(Str, B), File, "fputs");
}

/// EmitFWrite - Emit a call to the fwrite function.  This assumes that Ptr is
/// a pointer, Size is an 'intptr_t', and File is a pointer to FILE.
void LibCallOptimization::EmitFWrite(Value *Ptr, Value *Size, Value *File,
                                     IRBuilder<> &B) {
  Module *M = Caller->getParent();
  AttributeWithIndex AWI[3];
  AWI[0] = AttributeWithIndex::get(1, Attribute::NoCapture);
  AWI[1] = AttributeWithIndex::get(4, Attribute::NoCapture);
  AWI[2] = AttributeWithIndex::get(~0u, Attribute::NoUnwind);
  Constant *F;
  if (isa<PointerType>(File->getType()))
    F = M->getOrInsertFunction("fwrite", AttrListPtr::get(AWI, 3),
                               TD->getIntPtrType(),
                               PointerType::getUnqual(Type::Int8Ty),
                               TD->getIntPtrType(), TD->getIntPtrType(),
                               File->getType(), NULL);
  else
    F = M->getOrInsertFunction("fwrite", TD->getIntPtrType(),
                               PointerType::getUnqual(Type::Int8Ty),
                               TD->getIntPtrType(), TD->getIntPtrType(),
                               File->getType(), NULL);
  B.CreateCall4(F, CastToCStr(Ptr, B), Size, 
                ConstantInt::get(TD->getIntPtrType(), 1), File);
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// GetStringLengthH - If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLengthH(Value *V, SmallPtrSet<PHINode*, 32> &PHIs) {
  // Look through noop bitcast instructions.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(V))
    return GetStringLengthH(BCI->getOperand(0), PHIs);
  
  // If this is a PHI node, there are two cases: either we have already seen it
  // or we haven't.
  if (PHINode *PN = dyn_cast<PHINode>(V)) {
    if (!PHIs.insert(PN))
      return ~0ULL;  // already in the set.
    
    // If it was new, see if all the input strings are the same length.
    uint64_t LenSoFar = ~0ULL;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      uint64_t Len = GetStringLengthH(PN->getIncomingValue(i), PHIs);
      if (Len == 0) return 0; // Unknown length -> unknown.
      
      if (Len == ~0ULL) continue;
      
      if (Len != LenSoFar && LenSoFar != ~0ULL)
        return 0;    // Disagree -> unknown.
      LenSoFar = Len;
    }
    
    // Success, all agree.
    return LenSoFar;
  }
  
  // strlen(select(c,x,y)) -> strlen(x) ^ strlen(y)
  if (SelectInst *SI = dyn_cast<SelectInst>(V)) {
    uint64_t Len1 = GetStringLengthH(SI->getTrueValue(), PHIs);
    if (Len1 == 0) return 0;
    uint64_t Len2 = GetStringLengthH(SI->getFalseValue(), PHIs);
    if (Len2 == 0) return 0;
    if (Len1 == ~0ULL) return Len2;
    if (Len2 == ~0ULL) return Len1;
    if (Len1 != Len2) return 0;
    return Len1;
  }
  
  // If the value is not a GEP instruction nor a constant expression with a
  // GEP instruction, then return unknown.
  User *GEP = 0;
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    GEP = GEPI;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() != Instruction::GetElementPtr)
      return 0;
    GEP = CE;
  } else {
    return 0;
  }
  
  // Make sure the GEP has exactly three arguments.
  if (GEP->getNumOperands() != 3)
    return 0;
  
  // Check to make sure that the first operand of the GEP is an integer and
  // has value 0 so that we are sure we're indexing into the initializer.
  if (ConstantInt *Idx = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
    if (!Idx->isZero())
      return 0;
  } else
    return 0;
  
  // If the second index isn't a ConstantInt, then this is a variable index
  // into the array.  If this occurs, we can't say anything meaningful about
  // the string.
  uint64_t StartIdx = 0;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(2)))
    StartIdx = CI->getZExtValue();
  else
    return 0;
  
  // The GEP instruction, constant or instruction, must reference a global
  // variable that is a constant and is initialized. The referenced constant
  // initializer is the array that we'll use for optimization.
  GlobalVariable* GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!GV || !GV->isConstant() || !GV->hasInitializer())
    return 0;
  Constant *GlobalInit = GV->getInitializer();
  
  // Handle the ConstantAggregateZero case, which is a degenerate case. The
  // initializer is constant zero so the length of the string must be zero.
  if (isa<ConstantAggregateZero>(GlobalInit))
    return 1;  // Len = 0 offset by 1.
  
  // Must be a Constant Array
  ConstantArray *Array = dyn_cast<ConstantArray>(GlobalInit);
  if (!Array || Array->getType()->getElementType() != Type::Int8Ty)
    return false;
  
  // Get the number of elements in the array
  uint64_t NumElts = Array->getType()->getNumElements();
  
  // Traverse the constant array from StartIdx (derived above) which is
  // the place the GEP refers to in the array.
  for (unsigned i = StartIdx; i != NumElts; ++i) {
    Constant *Elt = Array->getOperand(i);
    ConstantInt *CI = dyn_cast<ConstantInt>(Elt);
    if (!CI) // This array isn't suitable, non-int initializer.
      return 0;
    if (CI->isZero())
      return i-StartIdx+1; // We found end of string, success!
  }
  
  return 0; // The array isn't null terminated, conservatively return 'unknown'.
}

/// GetStringLength - If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLength(Value *V) {
  if (!isa<PointerType>(V->getType())) return 0;
  
  SmallPtrSet<PHINode*, 32> PHIs;
  uint64_t Len = GetStringLengthH(V, PHIs);
  // If Len is ~0ULL, we had an infinite phi cycle: this is dead code, so return
  // an empty string as a length.
  return Len == ~0ULL ? 1 : Len;
}

/// IsOnlyUsedInZeroEqualityComparison - Return true if it only matters that the
/// value is equal or not-equal to zero. 
static bool IsOnlyUsedInZeroEqualityComparison(Value *V) {
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

//===----------------------------------------------------------------------===//
// Miscellaneous LibCall Optimizations
//===----------------------------------------------------------------------===//

namespace {
//===---------------------------------------===//
// 'exit' Optimizations

/// ExitOpt - int main() { exit(4); } --> int main() { return 4; }
struct VISIBILITY_HIDDEN ExitOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify we have a reasonable prototype for exit.
    if (Callee->arg_size() == 0 || !CI->use_empty())
      return 0;

    // Verify the caller is main, and that the result type of main matches the
    // argument type of exit.
    if (!Caller->isName("main") || !Caller->hasExternalLinkage() ||
        Caller->getReturnType() != CI->getOperand(1)->getType())
      return 0;

    TerminatorInst *OldTI = CI->getParent()->getTerminator();
    
    // Create the return after the call.
    ReturnInst *RI = B.CreateRet(CI->getOperand(1));

    // Drop all successor phi node entries.
    for (unsigned i = 0, e = OldTI->getNumSuccessors(); i != e; ++i)
      OldTI->getSuccessor(i)->removePredecessor(CI->getParent());
    
    // Erase all instructions from after our return instruction until the end of
    // the block.
    BasicBlock::iterator FirstDead = RI; ++FirstDead;
    CI->getParent()->getInstList().erase(FirstDead, CI->getParent()->end());
    return CI;
  }
};

//===----------------------------------------------------------------------===//
// String and Memory LibCall Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'strcat' Optimizations

struct VISIBILITY_HIDDEN StrCatOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strcat" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != PointerType::getUnqual(Type::Int8Ty) ||
        FT->getParamType(0) != FT->getReturnType() ||
        FT->getParamType(1) != FT->getReturnType())
      return 0;
    
    // Extract some information from the instruction
    Value *Dst = CI->getOperand(1);
    Value *Src = CI->getOperand(2);
    
    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;
    --Len;  // Unbias length.
    
    // Handle the simple, do-nothing case: strcat(x, "") -> x
    if (Len == 0)
      return Dst;
    
    // We need to find the end of the destination string.  That's where the
    // memory is to be moved to. We just generate a call to strlen.
    Value *DstLen = EmitStrLen(Dst, B);
    
    // Now that we have the destination's length, we must index into the
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    Dst = B.CreateGEP(Dst, DstLen, "endptr");
    
    // We have enough information to now generate the memcpy call to do the
    // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
    EmitMemCpy(Dst, Src, ConstantInt::get(TD->getIntPtrType(), Len+1), 1, B);
    return Dst;
  }
};

//===---------------------------------------===//
// 'strchr' Optimizations

struct VISIBILITY_HIDDEN StrChrOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strchr" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != PointerType::getUnqual(Type::Int8Ty) ||
        FT->getParamType(0) != FT->getReturnType())
      return 0;
    
    Value *SrcStr = CI->getOperand(1);
    
    // If the second operand is non-constant, see if we can compute the length
    // of the input string and turn this into memchr.
    ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getOperand(2));
    if (CharC == 0) {
      uint64_t Len = GetStringLength(SrcStr);
      if (Len == 0 || FT->getParamType(1) != Type::Int32Ty) // memchr needs i32.
        return 0;
      
      return EmitMemChr(SrcStr, CI->getOperand(2), // include nul.
                        ConstantInt::get(TD->getIntPtrType(), Len), B);
    }

    // Otherwise, the character is a constant, see if the first argument is
    // a string literal.  If so, we can constant fold.
    std::string Str;
    if (!GetConstantStringInfo(SrcStr, Str))
      return 0;
    
    // strchr can find the nul character.
    Str += '\0';
    char CharValue = CharC->getSExtValue();
    
    // Compute the offset.
    uint64_t i = 0;
    while (1) {
      if (i == Str.size())    // Didn't find the char.  strchr returns null.
        return Constant::getNullValue(CI->getType());
      // Did we find our match?
      if (Str[i] == CharValue)
        break;
      ++i;
    }
    
    // strchr(s+n,c)  -> gep(s+n+i,c)
    Value *Idx = ConstantInt::get(Type::Int64Ty, i);
    return B.CreateGEP(SrcStr, Idx, "strchr");
  }
};

//===---------------------------------------===//
// 'strcmp' Optimizations

struct VISIBILITY_HIDDEN StrCmpOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strcmp" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || FT->getReturnType() != Type::Int32Ty ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != PointerType::getUnqual(Type::Int8Ty))
      return 0;
    
    Value *Str1P = CI->getOperand(1), *Str2P = CI->getOperand(2);
    if (Str1P == Str2P)      // strcmp(x,x)  -> 0
      return ConstantInt::get(CI->getType(), 0);
    
    std::string Str1, Str2;
    bool HasStr1 = GetConstantStringInfo(Str1P, Str1);
    bool HasStr2 = GetConstantStringInfo(Str2P, Str2);
    
    if (HasStr1 && Str1.empty()) // strcmp("", x) -> *x
      return B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"), CI->getType());
    
    if (HasStr2 && Str2.empty()) // strcmp(x,"") -> *x
      return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());
    
    // strcmp(x, y)  -> cnst  (if both x and y are constant strings)
    if (HasStr1 && HasStr2)
      return ConstantInt::get(CI->getType(), strcmp(Str1.c_str(),Str2.c_str()));

    // strcmp(P, "x") -> memcmp(P, "x", 2)
    uint64_t Len1 = GetStringLength(Str1P);
    uint64_t Len2 = GetStringLength(Str2P);
    if (Len1 || Len2) {
      // Choose the smallest Len excluding 0 which means 'unknown'.
      if (!Len1 || (Len2 && Len2 < Len1))
        Len1 = Len2;
      return EmitMemCmp(Str1P, Str2P,
                        ConstantInt::get(TD->getIntPtrType(), Len1), B);
    }

    return 0;
  }
};

//===---------------------------------------===//
// 'strncmp' Optimizations

struct VISIBILITY_HIDDEN StrNCmpOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strncmp" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != Type::Int32Ty ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != PointerType::getUnqual(Type::Int8Ty) ||
        !isa<IntegerType>(FT->getParamType(2)))
      return 0;
    
    Value *Str1P = CI->getOperand(1), *Str2P = CI->getOperand(2);
    if (Str1P == Str2P)      // strncmp(x,x,n)  -> 0
      return ConstantInt::get(CI->getType(), 0);
    
    // Get the length argument if it is constant.
    uint64_t Length;
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getOperand(3)))
      Length = LengthArg->getZExtValue();
    else
      return 0;
    
    if (Length == 0) // strncmp(x,y,0)   -> 0
      return ConstantInt::get(CI->getType(), 0);
    
    std::string Str1, Str2;
    bool HasStr1 = GetConstantStringInfo(Str1P, Str1);
    bool HasStr2 = GetConstantStringInfo(Str2P, Str2);
    
    if (HasStr1 && Str1.empty())  // strncmp("", x, n) -> *x
      return B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"), CI->getType());
    
    if (HasStr2 && Str2.empty())  // strncmp(x, "", n) -> *x
      return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());
    
    // strncmp(x, y)  -> cnst  (if both x and y are constant strings)
    if (HasStr1 && HasStr2)
      return ConstantInt::get(CI->getType(),
                              strncmp(Str1.c_str(), Str2.c_str(), Length));
    return 0;
  }
};


//===---------------------------------------===//
// 'strcpy' Optimizations

struct VISIBILITY_HIDDEN StrCpyOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "strcpy" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != PointerType::getUnqual(Type::Int8Ty))
      return 0;
    
    Value *Dst = CI->getOperand(1), *Src = CI->getOperand(2);
    if (Dst == Src)      // strcpy(x,x)  -> x
      return Src;
    
    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;
    
    // We have enough information to now generate the memcpy call to do the
    // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
    EmitMemCpy(Dst, Src, ConstantInt::get(TD->getIntPtrType(), Len), 1, B);
    return Dst;
  }
};



//===---------------------------------------===//
// 'strlen' Optimizations

struct VISIBILITY_HIDDEN StrLenOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 1 ||
        FT->getParamType(0) != PointerType::getUnqual(Type::Int8Ty) ||
        !isa<IntegerType>(FT->getReturnType()))
      return 0;
    
    Value *Src = CI->getOperand(1);

    // Constant folding: strlen("xyz") -> 3
    if (uint64_t Len = GetStringLength(Src))
      return ConstantInt::get(CI->getType(), Len-1);

    // Handle strlen(p) != 0.
    if (!IsOnlyUsedInZeroEqualityComparison(CI)) return 0;

    // strlen(x) != 0 --> *x != 0
    // strlen(x) == 0 --> *x == 0
    return B.CreateZExt(B.CreateLoad(Src, "strlenfirst"), CI->getType());
  }
};

//===---------------------------------------===//
// 'memcmp' Optimizations

struct VISIBILITY_HIDDEN MemCmpOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        FT->getReturnType() != Type::Int32Ty)
      return 0;

    Value *LHS = CI->getOperand(1), *RHS = CI->getOperand(2);

    if (LHS == RHS)  // memcmp(s,s,x) -> 0
      return Constant::getNullValue(CI->getType());

    // Make sure we have a constant length.
    ConstantInt *LenC = dyn_cast<ConstantInt>(CI->getOperand(3));
    if (!LenC) return 0;
    uint64_t Len = LenC->getZExtValue();

    if (Len == 0) // memcmp(s1,s2,0) -> 0
      return Constant::getNullValue(CI->getType());

    if (Len == 1) { // memcmp(S1,S2,1) -> *LHS - *RHS
      Value *LHSV = B.CreateLoad(CastToCStr(LHS, B), "lhsv");
      Value *RHSV = B.CreateLoad(CastToCStr(RHS, B), "rhsv");
      return B.CreateZExt(B.CreateSub(LHSV, RHSV, "chardiff"), CI->getType());
    }

    // memcmp(S1,S2,2) != 0 -> (*(short*)LHS ^ *(short*)RHS)  != 0
    // memcmp(S1,S2,4) != 0 -> (*(int*)LHS ^ *(int*)RHS)  != 0
    if ((Len == 2 || Len == 4) && IsOnlyUsedInZeroEqualityComparison(CI)) {
      const Type *PTy = PointerType::getUnqual(Len == 2 ?
                                               Type::Int16Ty : Type::Int32Ty);
      LHS = B.CreateBitCast(LHS, PTy, "tmp");
      RHS = B.CreateBitCast(RHS, PTy, "tmp");
      LoadInst *LHSV = B.CreateLoad(LHS, "lhsv");
      LoadInst *RHSV = B.CreateLoad(RHS, "rhsv");
      LHSV->setAlignment(1); RHSV->setAlignment(1);  // Unaligned loads.
      return B.CreateZExt(B.CreateXor(LHSV, RHSV, "shortdiff"), CI->getType());
    }

    return 0;
  }
};

//===---------------------------------------===//
// 'memcpy' Optimizations

struct VISIBILITY_HIDDEN MemCpyOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        FT->getParamType(2) != TD->getIntPtrType())
      return 0;

    // memcpy(x, y, n) -> llvm.memcpy(x, y, n, 1)
    EmitMemCpy(CI->getOperand(1), CI->getOperand(2), CI->getOperand(3), 1, B);
    return CI->getOperand(1);
  }
};

//===---------------------------------------===//
// 'memmove' Optimizations

struct VISIBILITY_HIDDEN MemMoveOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        FT->getParamType(2) != TD->getIntPtrType())
      return 0;

    // memmove(x, y, n) -> llvm.memmove(x, y, n, 1)
    Module *M = Caller->getParent();
    Intrinsic::ID IID = Intrinsic::memmove;
    const Type *Tys[1];
    Tys[0] = TD->getIntPtrType();
    Value *MemMove = Intrinsic::getDeclaration(M, IID, Tys, 1);
    Value *Dst = CastToCStr(CI->getOperand(1), B);
    Value *Src = CastToCStr(CI->getOperand(2), B);
    Value *Size = CI->getOperand(3);
    Value *Align = ConstantInt::get(Type::Int32Ty, 1);
    B.CreateCall4(MemMove, Dst, Src, Size, Align);
    return CI->getOperand(1);
  }
};

//===---------------------------------------===//
// 'memset' Optimizations

struct VISIBILITY_HIDDEN MemSetOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !isa<PointerType>(FT->getParamType(0)) ||
        FT->getParamType(1) != TD->getIntPtrType() ||
        FT->getParamType(2) != TD->getIntPtrType())
      return 0;

    // memset(p, v, n) -> llvm.memset(p, v, n, 1)
    Module *M = Caller->getParent();
    Intrinsic::ID IID = Intrinsic::memset;
    const Type *Tys[1];
    Tys[0] = TD->getIntPtrType();
    Value *MemSet = Intrinsic::getDeclaration(M, IID, Tys, 1);
    Value *Dst = CastToCStr(CI->getOperand(1), B);
    Value *Val = B.CreateTrunc(CI->getOperand(2), Type::Int8Ty);
    Value *Size = CI->getOperand(3);
    Value *Align = ConstantInt::get(Type::Int32Ty, 1);
    B.CreateCall4(MemSet, Dst, Val, Size, Align);
    return CI->getOperand(1);
  }
};

//===----------------------------------------------------------------------===//
// Math Library Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'pow*' Optimizations

struct VISIBILITY_HIDDEN PowOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        !FT->getParamType(0)->isFloatingPoint())
      return 0;
    
    Value *Op1 = CI->getOperand(1), *Op2 = CI->getOperand(2);
    if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1)) {
      if (Op1C->isExactlyValue(1.0))  // pow(1.0, x) -> 1.0
        return Op1C;
      if (Op1C->isExactlyValue(2.0))  // pow(2.0, x) -> exp2(x)
        return EmitUnaryFloatFnCall(Op2, "exp2", B);
    }
    
    ConstantFP *Op2C = dyn_cast<ConstantFP>(Op2);
    if (Op2C == 0) return 0;
    
    if (Op2C->getValueAPF().isZero())  // pow(x, 0.0) -> 1.0
      return ConstantFP::get(CI->getType(), 1.0);
    
    if (Op2C->isExactlyValue(0.5)) {
      // FIXME: This is not safe for -0.0 and -inf.  This can only be done when
      // 'unsafe' math optimizations are allowed.
      // x    pow(x, 0.5)  sqrt(x)
      // ---------------------------------------------
      // -0.0    +0.0       -0.0
      // -inf    +inf       NaN
#if 0
      // pow(x, 0.5) -> sqrt(x)
      return B.CreateCall(get_sqrt(), Op1, "sqrt");
#endif
    }
    
    if (Op2C->isExactlyValue(1.0))  // pow(x, 1.0) -> x
      return Op1;
    if (Op2C->isExactlyValue(2.0))  // pow(x, 2.0) -> x*x
      return B.CreateMul(Op1, Op1, "pow2");
    if (Op2C->isExactlyValue(-1.0)) // pow(x, -1.0) -> 1.0/x
      return B.CreateFDiv(ConstantFP::get(CI->getType(), 1.0), Op1, "powrecip");
    return 0;
  }
};

//===---------------------------------------===//
// 'exp2' Optimizations

struct VISIBILITY_HIDDEN Exp2Opt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 1 argument of FP type, which matches the
    // result type.
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isFloatingPoint())
      return 0;
    
    Value *Op = CI->getOperand(1);
    // Turn exp2(sitofp(x)) -> ldexp(1.0, sext(x))  if sizeof(x) <= 32
    // Turn exp2(uitofp(x)) -> ldexp(1.0, zext(x))  if sizeof(x) < 32
    Value *LdExpArg = 0;
    if (SIToFPInst *OpC = dyn_cast<SIToFPInst>(Op)) {
      if (OpC->getOperand(0)->getType()->getPrimitiveSizeInBits() <= 32)
        LdExpArg = B.CreateSExt(OpC->getOperand(0), Type::Int32Ty, "tmp");
    } else if (UIToFPInst *OpC = dyn_cast<UIToFPInst>(Op)) {
      if (OpC->getOperand(0)->getType()->getPrimitiveSizeInBits() < 32)
        LdExpArg = B.CreateZExt(OpC->getOperand(0), Type::Int32Ty, "tmp");
    }
    
    if (LdExpArg) {
      const char *Name;
      if (Op->getType() == Type::FloatTy)
        Name = "ldexpf";
      else if (Op->getType() == Type::DoubleTy)
        Name = "ldexp";
      else
        Name = "ldexpl";

      Constant *One = ConstantFP::get(APFloat(1.0f));
      if (Op->getType() != Type::FloatTy)
        One = ConstantExpr::getFPExtend(One, Op->getType());

      Module *M = Caller->getParent();
      Value *Callee = M->getOrInsertFunction(Name, Op->getType(),
                                             Op->getType(), Type::Int32Ty,NULL);
      return B.CreateCall2(Callee, One, LdExpArg);
    }
    return 0;
  }
};
    

//===---------------------------------------===//
// Double -> Float Shrinking Optimizations for Unary Functions like 'floor'

struct VISIBILITY_HIDDEN UnaryDoubleFPOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 1 || FT->getReturnType() != Type::DoubleTy ||
        FT->getParamType(0) != Type::DoubleTy)
      return 0;
    
    // If this is something like 'floor((double)floatval)', convert to floorf.
    FPExtInst *Cast = dyn_cast<FPExtInst>(CI->getOperand(1));
    if (Cast == 0 || Cast->getOperand(0)->getType() != Type::FloatTy)
      return 0;

    // floor((double)floatval) -> (double)floorf(floatval)
    Value *V = Cast->getOperand(0);
    V = EmitUnaryFloatFnCall(V, Callee->getNameStart(), B);
    return B.CreateFPExt(V, Type::DoubleTy);
  }
};

//===----------------------------------------------------------------------===//
// Integer Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'ffs*' Optimizations

struct VISIBILITY_HIDDEN FFSOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 1 || FT->getReturnType() != Type::Int32Ty ||
        !isa<IntegerType>(FT->getParamType(0)))
      return 0;
    
    Value *Op = CI->getOperand(1);
    
    // Constant fold.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op)) {
      if (CI->getValue() == 0)  // ffs(0) -> 0.
        return Constant::getNullValue(CI->getType());
      return ConstantInt::get(Type::Int32Ty, // ffs(c) -> cttz(c)+1
                              CI->getValue().countTrailingZeros()+1);
    }
    
    // ffs(x) -> x != 0 ? (i32)llvm.cttz(x)+1 : 0
    const Type *ArgType = Op->getType();
    Value *F = Intrinsic::getDeclaration(Callee->getParent(),
                                         Intrinsic::cttz, &ArgType, 1);
    Value *V = B.CreateCall(F, Op, "cttz");
    V = B.CreateAdd(V, ConstantInt::get(Type::Int32Ty, 1), "tmp");
    V = B.CreateIntCast(V, Type::Int32Ty, false, "tmp");
    
    Value *Cond = B.CreateICmpNE(Op, Constant::getNullValue(ArgType), "tmp");
    return B.CreateSelect(Cond, V, ConstantInt::get(Type::Int32Ty, 0));
  }
};

//===---------------------------------------===//
// 'isdigit' Optimizations

struct VISIBILITY_HIDDEN IsDigitOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // We require integer(i32)
    if (FT->getNumParams() != 1 || !isa<IntegerType>(FT->getReturnType()) ||
        FT->getParamType(0) != Type::Int32Ty)
      return 0;
    
    // isdigit(c) -> (c-'0') <u 10
    Value *Op = CI->getOperand(1);
    Op = B.CreateSub(Op, ConstantInt::get(Type::Int32Ty, '0'), "isdigittmp");
    Op = B.CreateICmpULT(Op, ConstantInt::get(Type::Int32Ty, 10), "isdigit");
    return B.CreateZExt(Op, CI->getType());
  }
};

//===---------------------------------------===//
// 'isascii' Optimizations

struct VISIBILITY_HIDDEN IsAsciiOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // We require integer(i32)
    if (FT->getNumParams() != 1 || !isa<IntegerType>(FT->getReturnType()) ||
        FT->getParamType(0) != Type::Int32Ty)
      return 0;
    
    // isascii(c) -> c <u 128
    Value *Op = CI->getOperand(1);
    Op = B.CreateICmpULT(Op, ConstantInt::get(Type::Int32Ty, 128), "isascii");
    return B.CreateZExt(Op, CI->getType());
  }
};
  
//===---------------------------------------===//
// 'abs', 'labs', 'llabs' Optimizations

struct VISIBILITY_HIDDEN AbsOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // We require integer(integer) where the types agree.
    if (FT->getNumParams() != 1 || !isa<IntegerType>(FT->getReturnType()) ||
        FT->getParamType(0) != FT->getReturnType())
      return 0;
    
    // abs(x) -> x >s -1 ? x : -x
    Value *Op = CI->getOperand(1);
    Value *Pos = B.CreateICmpSGT(Op,ConstantInt::getAllOnesValue(Op->getType()),
                                 "ispos");
    Value *Neg = B.CreateNeg(Op, "neg");
    return B.CreateSelect(Pos, Op, Neg);
  }
};
  

//===---------------------------------------===//
// 'toascii' Optimizations

struct VISIBILITY_HIDDEN ToAsciiOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // We require i32(i32)
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != Type::Int32Ty)
      return 0;
    
    // isascii(c) -> c & 0x7f
    return B.CreateAnd(CI->getOperand(1), ConstantInt::get(CI->getType(),0x7F));
  }
};

//===----------------------------------------------------------------------===//
// Formatting and IO Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'printf' Optimizations

struct VISIBILITY_HIDDEN PrintFOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require one fixed pointer argument and an integer/void result.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() < 1 || !isa<PointerType>(FT->getParamType(0)) ||
        !(isa<IntegerType>(FT->getReturnType()) ||
          FT->getReturnType() == Type::VoidTy))
      return 0;
    
    // Check for a fixed format string.
    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(1), FormatStr))
      return 0;

    // Empty format string -> noop.
    if (FormatStr.empty())  // Tolerate printf's declared void.
      return CI->use_empty() ? (Value*)CI : ConstantInt::get(CI->getType(), 0);
    
    // printf("x") -> putchar('x'), even for '%'.
    if (FormatStr.size() == 1) {
      EmitPutChar(ConstantInt::get(Type::Int32Ty, FormatStr[0]), B);
      return CI->use_empty() ? (Value*)CI : ConstantInt::get(CI->getType(), 1);
    }
    
    // printf("foo\n") --> puts("foo")
    if (FormatStr[FormatStr.size()-1] == '\n' &&
        FormatStr.find('%') == std::string::npos) {  // no format characters.
      // Create a string literal with no \n on it.  We expect the constant merge
      // pass to be run after this pass, to merge duplicate strings.
      FormatStr.erase(FormatStr.end()-1);
      Constant *C = ConstantArray::get(FormatStr, true);
      C = new GlobalVariable(C->getType(), true,GlobalVariable::InternalLinkage,
                             C, "str", Callee->getParent());
      EmitPutS(C, B);
      return CI->use_empty() ? (Value*)CI : 
                          ConstantInt::get(CI->getType(), FormatStr.size()+1);
    }
    
    // Optimize specific format strings.
    // printf("%c", chr) --> putchar(*(i8*)dst)
    if (FormatStr == "%c" && CI->getNumOperands() > 2 &&
        isa<IntegerType>(CI->getOperand(2)->getType())) {
      EmitPutChar(CI->getOperand(2), B);
      return CI->use_empty() ? (Value*)CI : ConstantInt::get(CI->getType(), 1);
    }
    
    // printf("%s\n", str) --> puts(str)
    if (FormatStr == "%s\n" && CI->getNumOperands() > 2 &&
        isa<PointerType>(CI->getOperand(2)->getType()) &&
        CI->use_empty()) {
      EmitPutS(CI->getOperand(2), B);
      return CI;
    }
    return 0;
  }
};

//===---------------------------------------===//
// 'sprintf' Optimizations

struct VISIBILITY_HIDDEN SPrintFOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require two fixed pointer arguments and an integer result.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        !isa<IntegerType>(FT->getReturnType()))
      return 0;

    // Check for a fixed format string.
    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(2), FormatStr))
      return 0;
    
    // If we just have a format string (nothing else crazy) transform it.
    if (CI->getNumOperands() == 3) {
      // Make sure there's no % in the constant array.  We could try to handle
      // %% -> % in the future if we cared.
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')
          return 0; // we found a format specifier, bail out.
      
      // sprintf(str, fmt) -> llvm.memcpy(str, fmt, strlen(fmt)+1, 1)
      EmitMemCpy(CI->getOperand(1), CI->getOperand(2), // Copy the nul byte.
                 ConstantInt::get(TD->getIntPtrType(), FormatStr.size()+1),1,B);
      return ConstantInt::get(CI->getType(), FormatStr.size());
    }
    
    // The remaining optimizations require the format string to be "%s" or "%c"
    // and have an extra operand.
    if (FormatStr.size() != 2 || FormatStr[0] != '%' || CI->getNumOperands() <4)
      return 0;
    
    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      // sprintf(dst, "%c", chr) --> *(i8*)dst = chr; *((i8*)dst+1) = 0
      if (!isa<IntegerType>(CI->getOperand(3)->getType())) return 0;
      Value *V = B.CreateTrunc(CI->getOperand(3), Type::Int8Ty, "char");
      Value *Ptr = CastToCStr(CI->getOperand(1), B);
      B.CreateStore(V, Ptr);
      Ptr = B.CreateGEP(Ptr, ConstantInt::get(Type::Int32Ty, 1), "nul");
      B.CreateStore(Constant::getNullValue(Type::Int8Ty), Ptr);
      
      return ConstantInt::get(CI->getType(), 1);
    }
    
    if (FormatStr[1] == 's') {
      // sprintf(dest, "%s", str) -> llvm.memcpy(dest, str, strlen(str)+1, 1)
      if (!isa<PointerType>(CI->getOperand(3)->getType())) return 0;

      Value *Len = EmitStrLen(CI->getOperand(3), B);
      Value *IncLen = B.CreateAdd(Len, ConstantInt::get(Len->getType(), 1),
                                  "leninc");
      EmitMemCpy(CI->getOperand(1), CI->getOperand(3), IncLen, 1, B);
      
      // The sprintf result is the unincremented number of bytes in the string.
      return B.CreateIntCast(Len, CI->getType(), false);
    }
    return 0;
  }
};

//===---------------------------------------===//
// 'fwrite' Optimizations

struct VISIBILITY_HIDDEN FWriteOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require a pointer, an integer, an integer, a pointer, returning integer.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 4 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<IntegerType>(FT->getParamType(1)) ||
        !isa<IntegerType>(FT->getParamType(2)) ||
        !isa<PointerType>(FT->getParamType(3)) ||
        !isa<IntegerType>(FT->getReturnType()))
      return 0;
    
    // Get the element size and count.
    ConstantInt *SizeC = dyn_cast<ConstantInt>(CI->getOperand(2));
    ConstantInt *CountC = dyn_cast<ConstantInt>(CI->getOperand(3));
    if (!SizeC || !CountC) return 0;
    uint64_t Bytes = SizeC->getZExtValue()*CountC->getZExtValue();
    
    // If this is writing zero records, remove the call (it's a noop).
    if (Bytes == 0)
      return ConstantInt::get(CI->getType(), 0);
    
    // If this is writing one byte, turn it into fputc.
    if (Bytes == 1) {  // fwrite(S,1,1,F) -> fputc(S[0],F)
      Value *Char = B.CreateLoad(CastToCStr(CI->getOperand(1), B), "char");
      EmitFPutC(Char, CI->getOperand(4), B);
      return ConstantInt::get(CI->getType(), 1);
    }

    return 0;
  }
};

//===---------------------------------------===//
// 'fputs' Optimizations

struct VISIBILITY_HIDDEN FPutsOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require two pointers.  Also, we can't optimize if return value is used.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        !CI->use_empty())
      return 0;
    
    // fputs(s,F) --> fwrite(s,1,strlen(s),F)
    uint64_t Len = GetStringLength(CI->getOperand(1));
    if (!Len) return 0;
    EmitFWrite(CI->getOperand(1), ConstantInt::get(TD->getIntPtrType(), Len-1),
               CI->getOperand(2), B);
    return CI;  // Known to have no uses (see above).
  }
};

//===---------------------------------------===//
// 'fprintf' Optimizations

struct VISIBILITY_HIDDEN FPrintFOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Require two fixed paramters as pointers and integer result.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        !isa<IntegerType>(FT->getReturnType()))
      return 0;
    
    // All the optimizations depend on the format string.
    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(2), FormatStr))
      return 0;

    // fprintf(F, "foo") --> fwrite("foo", 3, 1, F)
    if (CI->getNumOperands() == 3) {
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')  // Could handle %% -> % if we cared.
          return 0; // We found a format specifier.
      
      EmitFWrite(CI->getOperand(2), ConstantInt::get(TD->getIntPtrType(),
                                                     FormatStr.size()),
                 CI->getOperand(1), B);
      return ConstantInt::get(CI->getType(), FormatStr.size());
    }
    
    // The remaining optimizations require the format string to be "%s" or "%c"
    // and have an extra operand.
    if (FormatStr.size() != 2 || FormatStr[0] != '%' || CI->getNumOperands() <4)
      return 0;
    
    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      // fprintf(F, "%c", chr) --> *(i8*)dst = chr
      if (!isa<IntegerType>(CI->getOperand(3)->getType())) return 0;
      EmitFPutC(CI->getOperand(3), CI->getOperand(1), B);
      return ConstantInt::get(CI->getType(), 1);
    }
    
    if (FormatStr[1] == 's') {
      // fprintf(F, "%s", str) -> fputs(str, F)
      if (!isa<PointerType>(CI->getOperand(3)->getType()) || !CI->use_empty())
        return 0;
      EmitFPutS(CI->getOperand(3), CI->getOperand(1), B);
      return CI;
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
  class VISIBILITY_HIDDEN SimplifyLibCalls : public FunctionPass {
    StringMap<LibCallOptimization*> Optimizations;
    // Miscellaneous LibCall Optimizations
    ExitOpt Exit; 
    // String and Memory LibCall Optimizations
    StrCatOpt StrCat; StrChrOpt StrChr; StrCmpOpt StrCmp; StrNCmpOpt StrNCmp;
    StrCpyOpt StrCpy; StrLenOpt StrLen; MemCmpOpt MemCmp; MemCpyOpt  MemCpy;
    MemMoveOpt MemMove; MemSetOpt MemSet;
    // Math Library Optimizations
    PowOpt Pow; Exp2Opt Exp2; UnaryDoubleFPOpt UnaryDoubleFP;
    // Integer Optimizations
    FFSOpt FFS; AbsOpt Abs; IsDigitOpt IsDigit; IsAsciiOpt IsAscii;
    ToAsciiOpt ToAscii;
    // Formatting and IO Optimizations
    SPrintFOpt SPrintF; PrintFOpt PrintF;
    FWriteOpt FWrite; FPutsOpt FPuts; FPrintFOpt FPrintF;

    bool Modified;  // This is only used by doInitialization.
  public:
    static char ID; // Pass identification
    SimplifyLibCalls() : FunctionPass(&ID) {}

    void InitOptimizations();
    bool runOnFunction(Function &F);

    void setDoesNotAccessMemory(Function &F);
    void setOnlyReadsMemory(Function &F);
    void setDoesNotThrow(Function &F);
    void setDoesNotCapture(Function &F, unsigned n);
    void setDoesNotAlias(Function &F, unsigned n);
    bool doInitialization(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }
  };
  char SimplifyLibCalls::ID = 0;
} // end anonymous namespace.

static RegisterPass<SimplifyLibCalls>
X("simplify-libcalls", "Simplify well-known library calls");

// Public interface to the Simplify LibCalls pass.
FunctionPass *llvm::createSimplifyLibCallsPass() {
  return new SimplifyLibCalls(); 
}

/// Optimizations - Populate the Optimizations map with all the optimizations
/// we know.
void SimplifyLibCalls::InitOptimizations() {
  // Miscellaneous LibCall Optimizations
  Optimizations["exit"] = &Exit;
  
  // String and Memory LibCall Optimizations
  Optimizations["strcat"] = &StrCat;
  Optimizations["strchr"] = &StrChr;
  Optimizations["strcmp"] = &StrCmp;
  Optimizations["strncmp"] = &StrNCmp;
  Optimizations["strcpy"] = &StrCpy;
  Optimizations["strlen"] = &StrLen;
  Optimizations["memcmp"] = &MemCmp;
  Optimizations["memcpy"] = &MemCpy;
  Optimizations["memmove"] = &MemMove;
  Optimizations["memset"] = &MemSet;
  
  // Math Library Optimizations
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
  
#ifdef HAVE_FLOORF
  Optimizations["floor"] = &UnaryDoubleFP;
#endif
#ifdef HAVE_CEILF
  Optimizations["ceil"] = &UnaryDoubleFP;
#endif
#ifdef HAVE_ROUNDF
  Optimizations["round"] = &UnaryDoubleFP;
#endif
#ifdef HAVE_RINTF
  Optimizations["rint"] = &UnaryDoubleFP;
#endif
#ifdef HAVE_NEARBYINTF
  Optimizations["nearbyint"] = &UnaryDoubleFP;
#endif
  
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
  Optimizations["fwrite"] = &FWrite;
  Optimizations["fputs"] = &FPuts;
  Optimizations["fprintf"] = &FPrintF;
}


/// runOnFunction - Top level algorithm.
///
bool SimplifyLibCalls::runOnFunction(Function &F) {
  if (Optimizations.empty())
    InitOptimizations();
  
  const TargetData &TD = getAnalysis<TargetData>();
  
  IRBuilder<> Builder;

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
      const char *CalleeName = Callee->getNameStart();
      StringMap<LibCallOptimization*>::iterator OMI =
        Optimizations.find(CalleeName, CalleeName+Callee->getNameLen());
      if (OMI == Optimizations.end()) continue;
      
      // Set the builder to the instruction after the call.
      Builder.SetInsertPoint(BB, I);
      
      // Try to optimize this call.
      Value *Result = OMI->second->OptimizeCall(CI, TD, Builder);
      if (Result == 0) continue;

      DEBUG(DOUT << "SimplifyLibCalls simplified: " << *CI;
            DOUT << "  into: " << *Result << "\n");
      
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

/// doInitialization - Add attributes to well-known functions.
///
bool SimplifyLibCalls::doInitialization(Module &M) {
  Modified = false;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Function &F = *I;
    if (!F.isDeclaration())
      continue;

    unsigned NameLen = F.getNameLen();
    if (!NameLen)
      continue;

    const FunctionType *FTy = F.getFunctionType();

    const char *NameStr = F.getNameStart();
    switch (NameStr[0]) {
      case 's':
        if (NameLen == 6 && !strcmp(NameStr, "strlen")) {
          if (FTy->getNumParams() != 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setOnlyReadsMemory(F);
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if ((NameLen == 6 && !strcmp(NameStr, "strcpy")) ||
                   (NameLen == 6 && !strcmp(NameStr, "stpcpy")) ||
                   (NameLen == 6 && !strcmp(NameStr, "strcat")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strncat")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strncpy"))) {
          if (FTy->getNumParams() < 2 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 7 && !strcmp(NameStr, "strxfrm")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 6 && !strcmp(NameStr, "strcmp")) ||
                   (NameLen == 6 && !strcmp(NameStr, "strspn")) ||
                   (NameLen == 6 && !strcmp(NameStr, "strtol")) ||
                   (NameLen == 6 && !strcmp(NameStr, "strtod")) ||
                   (NameLen == 6 && !strcmp(NameStr, "strtof")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strtoul")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strtoll")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strtold")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strncmp")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strcspn")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strcoll")) ||
                   (NameLen == 8 && !strcmp(NameStr, "strtoull")) ||
                   (NameLen == 10 && !strcmp(NameStr, "strcasecmp")) ||
                   (NameLen == 11 && !strcmp(NameStr, "strncasecmp"))) {
          if (FTy->getNumParams() < 2 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setOnlyReadsMemory(F);
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 6 && !strcmp(NameStr, "strstr")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strpbrk"))) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setOnlyReadsMemory(F);
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 6 && !strcmp(NameStr, "strtok")) ||
                   (NameLen == 8 && !strcmp(NameStr, "strtok_r"))) {
          if (FTy->getNumParams() < 2 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 5 && !strcmp(NameStr, "scanf")) ||
                   (NameLen == 6 && !strcmp(NameStr, "setbuf")) ||
                   (NameLen == 7 && !strcmp(NameStr, "setvbuf"))) {
          if (FTy->getNumParams() < 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 6 && !strcmp(NameStr, "sscanf")) {
          if (FTy->getNumParams() < 2 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 6 && !strcmp(NameStr, "strdup")) ||
                   (NameLen == 7 && !strcmp(NameStr, "strndup"))) {
          if (FTy->getNumParams() < 1 ||
              !isa<PointerType>(FTy->getReturnType()) ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 7 && !strcmp(NameStr, "sprintf")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 8 && !strcmp(NameStr, "snprintf")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(2)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 3);
        }
        break;
      case 'm':
        if (NameLen == 6 && !strcmp(NameStr, "memcmp")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setOnlyReadsMemory(F);
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 6 && !strcmp(NameStr, "memchr")) ||
                   (NameLen == 7 && !strcmp(NameStr, "memrchr"))) {
          if (FTy->getNumParams() != 3)
            continue;
          setOnlyReadsMemory(F);
          setDoesNotThrow(F);
        } else if ((NameLen == 6 && !strcmp(NameStr, "memcpy")) ||
                   (NameLen == 7 && !strcmp(NameStr, "memccpy")) ||
                   (NameLen == 7 && !strcmp(NameStr, "memmove"))) {
          if (FTy->getNumParams() < 3 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 8 && !strcmp(NameStr, "memalign")) {
          if (!isa<PointerType>(FTy->getReturnType()))
            continue;
          setDoesNotAlias(F, 0);
        }
        break;
      case 'r':
        if (NameLen == 7 && !strcmp(NameStr, "realloc")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getReturnType()))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 4 && !strcmp(NameStr, "read")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          // May throw; "read" is a valid pthread cancellation point.
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 5 && !strcmp(NameStr, "rmdir")) ||
                   (NameLen == 6 && !strcmp(NameStr, "rewind")) ||
                   (NameLen == 6 && !strcmp(NameStr, "remove"))) {
          if (FTy->getNumParams() != 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 6 && !strcmp(NameStr, "rename")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        }
        break;
      case 'w':
        if (NameLen == 5 && !strcmp(NameStr, "write")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          // May throw; "write" is a valid pthread cancellation point.
          setDoesNotCapture(F, 2);
        }
        break;
      case 'b':
        if (NameLen == 5 && !strcmp(NameStr, "bcopy")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 4 && !strcmp(NameStr, "bcmp")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setOnlyReadsMemory(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 5 && !strcmp(NameStr, "bzero")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        }
        break;
      case 'c':
        if (NameLen == 6 && !strcmp(NameStr, "calloc")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getReturnType()))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
        } else if ((NameLen == 5 && !strcmp(NameStr, "chown")) ||
                   (NameLen == 8 && !strcmp(NameStr, "clearerr")) ||
                   (NameLen == 8 && !strcmp(NameStr, "closedir"))) {
          if (FTy->getNumParams() == 0 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        }
        break;
      case 'a':
        if ((NameLen == 4 && !strcmp(NameStr, "atoi")) ||
            (NameLen == 4 && !strcmp(NameStr, "atol")) ||
            (NameLen == 4 && !strcmp(NameStr, "atof")) ||
            (NameLen == 5 && !strcmp(NameStr, "atoll"))) {
          if (FTy->getNumParams() != 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setOnlyReadsMemory(F);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 6 && !strcmp(NameStr, "access")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        }
        break;
      case 'f':
        if (NameLen == 5 && !strcmp(NameStr, "fopen")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getReturnType()) ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 6 && !strcmp(NameStr, "fdopen")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getReturnType()) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 4 && !strcmp(NameStr, "feof")) ||
                   (NameLen == 4 && !strcmp(NameStr, "free")) ||
                   (NameLen == 5 && !strcmp(NameStr, "fseek")) ||
                   (NameLen == 5 && !strcmp(NameStr, "ftell")) ||
                   (NameLen == 5 && !strcmp(NameStr, "fgetc")) ||
                   (NameLen == 6 && !strcmp(NameStr, "fseeko")) ||
                   (NameLen == 6 && !strcmp(NameStr, "ftello")) ||
                   (NameLen == 6 && !strcmp(NameStr, "fileno")) ||
                   (NameLen == 6 && !strcmp(NameStr, "fflush")) ||
                   (NameLen == 6 && !strcmp(NameStr, "fclose")) ||
                   (NameLen == 7 && !strcmp(NameStr, "fsetpos"))) {
          if (FTy->getNumParams() == 0 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 6 && !strcmp(NameStr, "ferror")) {
          if (FTy->getNumParams() != 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setOnlyReadsMemory(F);
        } else if ((NameLen == 5 && !strcmp(NameStr, "fputc")) ||
                   (NameLen == 5 && !strcmp(NameStr, "fputs"))) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 5 && !strcmp(NameStr, "fgets")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(2)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 3);
        } else if ((NameLen == 5 && !strcmp(NameStr, "fread")) ||
                   (NameLen == 6 && !strcmp(NameStr, "fwrite"))) {
          if (FTy->getNumParams() != 4 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(3)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 4);
        } else if (NameLen == 7 && !strcmp(NameStr, "fgetpos")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 6 && !strcmp(NameStr, "fscanf")) {
          if (FTy->getNumParams() < 2 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 7 && !strcmp(NameStr, "fprintf")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        }
        break;
      case 'g':
        if ((NameLen == 4 && !strcmp(NameStr, "getc")) ||
            (NameLen == 10 && !strcmp(NameStr, "getlogin_r"))) {
          if (FTy->getNumParams() == 0 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 6 && !strcmp(NameStr, "getenv")) {
          if (FTy->getNumParams() != 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setOnlyReadsMemory(F);
          setDoesNotCapture(F, 1);
        } else if ((NameLen == 4 && !strcmp(NameStr, "gets")) ||
                   (NameLen == 7 && !strcmp(NameStr, "getchar"))) {
          setDoesNotThrow(F);
        }
        break;
      case 'u':
        if (NameLen == 6 && !strcmp(NameStr, "ungetc")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 6 && !strcmp(NameStr, "unlink")) {
          if (FTy->getNumParams() != 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        }
        break;
      case 'p':
        if (NameLen == 4 && !strcmp(NameStr, "putc")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        } else if ((NameLen == 4 && !strcmp(NameStr, "puts")) ||
                   (NameLen == 6 && !strcmp(NameStr, "printf")) ||
                   (NameLen == 6 && !strcmp(NameStr, "perror"))) {
          if (FTy->getNumParams() != 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if ((NameLen == 5 && !strcmp(NameStr, "pread")) ||
                   (NameLen == 6 && !strcmp(NameStr, "pwrite"))) {
          if (FTy->getNumParams() != 4 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          // May throw; these are valid pthread cancellation points.
          setDoesNotCapture(F, 2);
        } else if (NameLen == 7 && !strcmp(NameStr, "putchar")) {
          setDoesNotThrow(F);
        }
        break;
      case 'v':
        if (NameLen == 6 && !strcmp(NameStr, "vscanf")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if ((NameLen == 7 && !strcmp(NameStr, "vsscanf")) ||
                   (NameLen == 7 && !strcmp(NameStr, "vfscanf"))) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(1)) ||
              !isa<PointerType>(FTy->getParamType(2)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 6 && !strcmp(NameStr, "valloc")) {
          if (!isa<PointerType>(FTy->getReturnType()))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
        } else if (NameLen == 7 && !strcmp(NameStr, "vprintf")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if ((NameLen == 8 && !strcmp(NameStr, "vfprintf")) ||
                   (NameLen == 8 && !strcmp(NameStr, "vsprintf"))) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 9 && !strcmp(NameStr, "vsnprintf")) {
          if (FTy->getNumParams() != 4 ||
              !isa<PointerType>(FTy->getParamType(0)) ||
              !isa<PointerType>(FTy->getParamType(2)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 3);
        }
        break;
      case 'o':
        if (NameLen == 7 && !strcmp(NameStr, "opendir")) {
          // The description of fdopendir sounds like opening the same fd
          // twice might result in the same DIR* !
          if (!isa<PointerType>(FTy->getReturnType()))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
        }
        break;
      case 't':
        if (NameLen == 7 && !strcmp(NameStr, "tmpfile")) {
          if (!isa<PointerType>(FTy->getReturnType()))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
        }
      case 'h':
        if ((NameLen == 5 && !strcmp(NameStr, "htonl")) ||
            (NameLen == 5 && !strcmp(NameStr, "htons"))) {
          setDoesNotThrow(F);
          setDoesNotAccessMemory(F);
        }
        break;
      case 'n':
        if ((NameLen == 5 && !strcmp(NameStr, "ntohl")) ||
            (NameLen == 5 && !strcmp(NameStr, "ntohs"))) {
          setDoesNotThrow(F);
          setDoesNotAccessMemory(F);
        }
      case '_':
        if ((NameLen == 8 && !strcmp(NameStr, "__strdup")) ||
            (NameLen == 9 && !strcmp(NameStr, "__strndup"))) {
          if (FTy->getNumParams() < 1 ||
              !isa<PointerType>(FTy->getReturnType()) ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotAlias(F, 0);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 10 && !strcmp(NameStr, "__strtok_r")) {
          if (FTy->getNumParams() != 3 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        } else if (NameLen == 8 && !strcmp(NameStr, "_IO_getc")) {
          if (FTy->getNumParams() != 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 8 && !strcmp(NameStr, "_IO_putc")) {
          if (FTy->getNumParams() != 2 ||
              !isa<PointerType>(FTy->getParamType(1)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 2);
        }
      case 1:
        if (NameLen == 15 && !strcmp(NameStr, "\1__isoc99_scanf")) {
          if (FTy->getNumParams() < 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
        } else if (NameLen == 16 && !strcmp(NameStr, "\1__isoc99_sscanf")) {
          if (FTy->getNumParams() < 1 ||
              !isa<PointerType>(FTy->getParamType(0)))
            continue;
          setDoesNotThrow(F);
          setDoesNotCapture(F, 1);
          setDoesNotCapture(F, 2);
        }
        break;
    }
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
// cos, cosf, cosl:
//   * cos(-x)  -> cos(x)
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
// memcmp:
//   * memcmp(x,y,l)   -> cnst
//      (if all arguments are constant and strlen(x) <= l and strlen(y) <= l)
//
// pow, powf, powl:
//   * pow(exp(x),y)  -> exp(x*y)
//   * pow(sqrt(x),y) -> pow(x,y*0.5)
//   * pow(pow(x,y),z)-> pow(x,y*z)
//
// puts:
//   * puts("") -> putchar("\n")
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
// stpcpy:
//   * stpcpy(str, "literal") ->
//           llvm.memcpy(str,"literal",strlen("literal")+1,1)
// strrchr:
//   * strrchr(s,c) -> reverse_offset_of_in(c,s)
//      (if c is a constant integer and s is a constant string)
//   * strrchr(s1,0) -> strchr(s1,0)
//
// strncat:
//   * strncat(x,y,0) -> x
//   * strncat(x,y,0) -> x (if strlen(y) = 0)
//   * strncat(x,y,l) -> strcat(x,y) (if y and l are constants an l > strlen(y))
//
// strncpy:
//   * strncpy(d,s,0) -> d
//   * strncpy(d,s,l) -> memcpy(d,s,l,1)
//      (if s and l are constants)
//
// strpbrk:
//   * strpbrk(s,a) -> offset_in_for(s,a)
//      (if s and a are both constant strings)
//   * strpbrk(s,"") -> 0
//   * strpbrk(s,a) -> strchr(s,a[0]) (if a is constant string of length 1)
//
// strspn, strcspn:
//   * strspn(s,a)   -> const_int (if both args are constant)
//   * strspn("",a)  -> 0
//   * strspn(s,"")  -> 0
//   * strcspn(s,a)  -> const_int (if both args are constant)
//   * strcspn("",a) -> 0
//   * strcspn(s,"") -> strlen(a)
//
// strstr:
//   * strstr(x,x)  -> x
//   * strstr(s1,s2) -> offset_of_s2_in(s1)
//       (if s1 and s2 are constant strings)
//
// tan, tanf, tanl:
//   * tan(atan(x)) -> x
//
// trunc, truncf, truncl:
//   * trunc(cnst) -> cnst'
//
//
