//===- AMDGPUEmitPrintf.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility function to lower a printf call into a series of device
// library calls on the AMDGPU target.
//
// WARNING: This file knows about certain library functions. It recognizes them
// by name, and hardwires knowledge of their semantics.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"

#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-emit-printf"

static bool isCString(const Value *Arg) {
  auto Ty = Arg->getType();
  auto PtrTy = dyn_cast<PointerType>(Ty);
  if (!PtrTy)
    return false;

  auto IntTy = dyn_cast<IntegerType>(PtrTy->getElementType());
  if (!IntTy)
    return false;

  return IntTy->getBitWidth() == 8;
}

static Value *fitArgInto64Bits(IRBuilder<> &Builder, Value *Arg) {
  auto Int64Ty = Builder.getInt64Ty();
  auto Ty = Arg->getType();

  if (auto IntTy = dyn_cast<IntegerType>(Ty)) {
    switch (IntTy->getBitWidth()) {
    case 32:
      return Builder.CreateZExt(Arg, Int64Ty);
    case 64:
      return Arg;
    }
  }

  if (Ty->getTypeID() == Type::DoubleTyID) {
    return Builder.CreateBitCast(Arg, Int64Ty);
  }

  if (isa<PointerType>(Ty)) {
    return Builder.CreatePtrToInt(Arg, Int64Ty);
  }

  llvm_unreachable("unexpected type");
}

static Value *callPrintfBegin(IRBuilder<> &Builder, Value *Version) {
  auto Int64Ty = Builder.getInt64Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto Fn = M->getOrInsertFunction("__ockl_printf_begin", Int64Ty, Int64Ty);
  return Builder.CreateCall(Fn, Version);
}

static Value *callAppendArgs(IRBuilder<> &Builder, Value *Desc, int NumArgs,
                             Value *Arg0, Value *Arg1, Value *Arg2, Value *Arg3,
                             Value *Arg4, Value *Arg5, Value *Arg6,
                             bool IsLast) {
  auto Int64Ty = Builder.getInt64Ty();
  auto Int32Ty = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto Fn = M->getOrInsertFunction("__ockl_printf_append_args", Int64Ty,
                                   Int64Ty, Int32Ty, Int64Ty, Int64Ty, Int64Ty,
                                   Int64Ty, Int64Ty, Int64Ty, Int64Ty, Int32Ty);
  auto IsLastValue = Builder.getInt32(IsLast);
  auto NumArgsValue = Builder.getInt32(NumArgs);
  return Builder.CreateCall(Fn, {Desc, NumArgsValue, Arg0, Arg1, Arg2, Arg3,
                                 Arg4, Arg5, Arg6, IsLastValue});
}

static Value *appendArg(IRBuilder<> &Builder, Value *Desc, Value *Arg,
                        bool IsLast) {
  auto Arg0 = fitArgInto64Bits(Builder, Arg);
  auto Zero = Builder.getInt64(0);
  return callAppendArgs(Builder, Desc, 1, Arg0, Zero, Zero, Zero, Zero, Zero,
                        Zero, IsLast);
}

// The device library does not provide strlen, so we build our own loop
// here. While we are at it, we also include the terminating null in the length.
static Value *getStrlenWithNull(IRBuilder<> &Builder, Value *Str) {
  auto *Prev = Builder.GetInsertBlock();
  Module *M = Prev->getModule();

  auto CharZero = Builder.getInt8(0);
  auto One = Builder.getInt64(1);
  auto Zero = Builder.getInt64(0);
  auto Int64Ty = Builder.getInt64Ty();

  // The length is either zero for a null pointer, or the computed value for an
  // actual string. We need a join block for a phi that represents the final
  // value.
  //
  //  Strictly speaking, the zero does not matter since
  // __ockl_printf_append_string_n ignores the length if the pointer is null.
  BasicBlock *Join = nullptr;
  if (Prev->getTerminator()) {
    Join = Prev->splitBasicBlock(Builder.GetInsertPoint(),
                                 "strlen.join");
    Prev->getTerminator()->eraseFromParent();
  } else {
    Join = BasicBlock::Create(M->getContext(), "strlen.join",
                              Prev->getParent());
  }
  BasicBlock *While =
      BasicBlock::Create(M->getContext(), "strlen.while",
                         Prev->getParent(), Join);
  BasicBlock *WhileDone = BasicBlock::Create(
      M->getContext(), "strlen.while.done",
      Prev->getParent(), Join);

  // Emit an early return for when the pointer is null.
  Builder.SetInsertPoint(Prev);
  auto CmpNull =
      Builder.CreateICmpEQ(Str, Constant::getNullValue(Str->getType()));
  BranchInst::Create(Join, While, CmpNull, Prev);

  // Entry to the while loop.
  Builder.SetInsertPoint(While);

  auto PtrPhi = Builder.CreatePHI(Str->getType(), 2);
  PtrPhi->addIncoming(Str, Prev);
  auto PtrNext = Builder.CreateGEP(PtrPhi, One);
  PtrPhi->addIncoming(PtrNext, While);

  // Condition for the while loop.
  auto Data = Builder.CreateLoad(PtrPhi);
  auto Cmp = Builder.CreateICmpEQ(Data, CharZero);
  Builder.CreateCondBr(Cmp, WhileDone, While);

  // Add one to the computed length.
  Builder.SetInsertPoint(WhileDone, WhileDone->begin());
  auto Begin = Builder.CreatePtrToInt(Str, Int64Ty);
  auto End = Builder.CreatePtrToInt(PtrPhi, Int64Ty);
  auto Len = Builder.CreateSub(End, Begin);
  Len = Builder.CreateAdd(Len, One);

  // Final join.
  BranchInst::Create(Join, WhileDone);
  Builder.SetInsertPoint(Join, Join->begin());
  auto LenPhi = Builder.CreatePHI(Len->getType(), 2);
  LenPhi->addIncoming(Len, WhileDone);
  LenPhi->addIncoming(Zero, Prev);

  return LenPhi;
}

static Value *callAppendStringN(IRBuilder<> &Builder, Value *Desc, Value *Str,
                                Value *Length, bool isLast) {
  auto Int64Ty = Builder.getInt64Ty();
  auto CharPtrTy = Builder.getInt8PtrTy();
  auto Int32Ty = Builder.getInt32Ty();
  auto M = Builder.GetInsertBlock()->getModule();
  auto Fn = M->getOrInsertFunction("__ockl_printf_append_string_n", Int64Ty,
                                   Int64Ty, CharPtrTy, Int64Ty, Int32Ty);
  auto IsLastInt32 = Builder.getInt32(isLast);
  return Builder.CreateCall(Fn, {Desc, Str, Length, IsLastInt32});
}

static Value *appendString(IRBuilder<> &Builder, Value *Desc, Value *Arg,
                           bool IsLast) {
  auto Length = getStrlenWithNull(Builder, Arg);
  return callAppendStringN(Builder, Desc, Arg, Length, IsLast);
}

static Value *processArg(IRBuilder<> &Builder, Value *Desc, Value *Arg,
                         bool SpecIsCString, bool IsLast) {
  if (SpecIsCString && isCString(Arg)) {
    return appendString(Builder, Desc, Arg, IsLast);
  }
  // If the format specifies a string but the argument is not, the frontend will
  // have printed a warning. We just rely on undefined behaviour and send the
  // argument anyway.
  return appendArg(Builder, Desc, Arg, IsLast);
}

// Scan the format string to locate all specifiers, and mark the ones that
// specify a string, i.e, the "%s" specifier with optional '*' characters.
static void locateCStrings(SparseBitVector<8> &BV, Value *Fmt) {
  StringRef Str;
  if (!getConstantStringInfo(Fmt, Str) || Str.empty())
    return;

  static const char ConvSpecifiers[] = "diouxXfFeEgGaAcspn";
  size_t SpecPos = 0;
  // Skip the first argument, the format string.
  unsigned ArgIdx = 1;

  while ((SpecPos = Str.find_first_of('%', SpecPos)) != StringRef::npos) {
    if (Str[SpecPos + 1] == '%') {
      SpecPos += 2;
      continue;
    }
    auto SpecEnd = Str.find_first_of(ConvSpecifiers, SpecPos);
    if (SpecEnd == StringRef::npos)
      return;
    auto Spec = Str.slice(SpecPos, SpecEnd + 1);
    ArgIdx += Spec.count('*');
    if (Str[SpecEnd] == 's') {
      BV.set(ArgIdx);
    }
    SpecPos = SpecEnd + 1;
    ++ArgIdx;
  }
}

Value *llvm::emitAMDGPUPrintfCall(IRBuilder<> &Builder,
                                  ArrayRef<Value *> Args) {
  auto NumOps = Args.size();
  assert(NumOps >= 1);

  auto Fmt = Args[0];
  SparseBitVector<8> SpecIsCString;
  locateCStrings(SpecIsCString, Fmt);

  auto Desc = callPrintfBegin(Builder, Builder.getIntN(64, 0));
  Desc = appendString(Builder, Desc, Fmt, NumOps == 1);

  // FIXME: This invokes hostcall once for each argument. We can pack up to
  // seven scalar printf arguments in a single hostcall. See the signature of
  // callAppendArgs().
  for (unsigned int i = 1; i != NumOps; ++i) {
    bool IsLast = i == NumOps - 1;
    bool IsCString = SpecIsCString.test(i);
    Desc = processArg(Builder, Desc, Args[i], IsCString, IsLast);
  }

  return Builder.CreateTrunc(Desc, Builder.getInt32Ty());
}
