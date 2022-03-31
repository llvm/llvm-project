//===- MemoryBuiltinsTest.cpp - Tests for utilities in MemoryBuiltins.h ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// allocsize should not imply that a function is a traditional allocation
// function (e.g. that can be optimized out/...); it just tells us how many
// bytes exist at the pointer handed back by the function.
TEST(AllocSize, AllocationBuiltinsTest) {
  LLVMContext Context;
  Module M("", Context);
  IntegerType *ArgTy = Type::getInt32Ty(Context);

  Function *AllocSizeFn = Function::Create(
      FunctionType::get(Type::getInt8PtrTy(Context), {ArgTy}, false),
      GlobalValue::ExternalLinkage, "F", &M);

  AllocSizeFn->addFnAttr(Attribute::getWithAllocSizeArgs(Context, 1, None));

  // 100 is arbitrary.
  std::unique_ptr<CallInst> Caller(
      CallInst::Create(AllocSizeFn, {ConstantInt::get(ArgTy, 100)}));

  const TargetLibraryInfo *TLI = nullptr;
  EXPECT_FALSE(isAllocLikeFn(Caller.get(), TLI));

  // FIXME: We might be able to treat allocsize functions as general allocation
  // functions.
  EXPECT_FALSE(isAllocationFn(Caller.get(), TLI));
}
}
