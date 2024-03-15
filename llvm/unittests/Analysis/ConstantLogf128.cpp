//===- unittests/CodeGen/BufferSourceTest.cpp - MemoryBuffer source tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ConstantFoldLogf128Fixture, ConstantFoldLogf128) {
#ifdef __FLOAT128__
  LLVMContext Context;
  IRBuilder<> Builder(Context);
  Module MainModule("Logf128TestModule", Context);
  MainModule.setTargetTriple("aarch64-unknown-linux");

  Type *FP128Ty = Type::getFP128Ty(Context);
  FunctionType *FP128Prototype = FunctionType::get(FP128Ty, false);
  Function *Logf128TestFunction = Function::Create(
      FP128Prototype, Function::ExternalLinkage, "logf128test", MainModule);
  BasicBlock *EntryBlock =
      BasicBlock::Create(Context, "entry", Logf128TestFunction);
  Builder.SetInsertPoint(EntryBlock);

  FunctionType *FP128FP128Prototype =
      FunctionType::get(FP128Ty, {FP128Ty}, false);
  Constant *Constant2L = ConstantFP::get128(FP128Ty, 2.0L);
  Function *Logf128 = Function::Create(
      FP128FP128Prototype, Function::ExternalLinkage, "llvm.log.f128", MainModule);
  CallInst *Logf128Call = Builder.CreateCall(Logf128, Constant2L);

  TargetLibraryInfoImpl TLII(Triple(MainModule.getTargetTriple()));
  TargetLibraryInfo TLI(TLII, Logf128TestFunction);
  Constant *FoldResult = ConstantFoldCall(Logf128Call, Logf128, Constant2L, &TLI);

#ifndef HAS_LOGF128
  ASSERT_TRUE(FoldResult == nullptr);
#else
  auto ConstantLog = dyn_cast<ConstantFP>(FoldResult);
  ASSERT_TRUE(ConstantLog);

  APFloat APF = ConstantLog->getValueAPF();
  char LongDoubleHexString[0xFF];
  unsigned Size =
      APF.convertToHexString(LongDoubleHexString, 32, true,
                             APFloatBase::roundingMode::NearestTiesToAway);
  EXPECT_GT(Size, 0U);

  ASSERT_STREQ(LongDoubleHexString,
               std::string("0X1.62E42FEFA39EF000000000000000000P-1").c_str());
#endif //HAS_LOGF128
#else // __FLOAT128__
  ASSERT_TRUE(true);
#endif
}

}