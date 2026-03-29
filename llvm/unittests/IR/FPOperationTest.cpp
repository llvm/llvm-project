//===- llvm/unittest/IR/IntrinsicsTest.cpp - ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class FPOperationTest : public ::testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  Function *Func = nullptr;
  BasicBlock *BB = nullptr;

  void SetUp() override { M = std::make_unique<Module>("Test", Context); }

  void TearDown() override {
    M.reset();
    Func = nullptr;
    BB = nullptr;
  }

  void parseAssembly(const char *Source) {
    SMDiagnostic Err;
    M = parseAssemblyString(Source, Err, Context);
    if (!M) {
      Err.print("FPOperationTests", llvm::errs());
    }
  }

  void createUnaryFunction(Type *DType, StringRef FName) {
    auto *FuncTy = FunctionType::get(DType, DType, false);
    auto F = M->getOrInsertFunction(FName, FuncTy);
    Func = cast<Function>(F.getCallee());
    ASSERT_TRUE(Func);
    BB = BasicBlock::Create(Context, "", Func);
    ASSERT_TRUE(BB);
  }
};

// Build a function with FP operation, then add StrictFP attribute to it.
TEST_F(FPOperationTest, MEBuild) {
  auto *FloatTy = Type::getFloatTy(Context);
  createUnaryFunction(FloatTy, "test_1");
  IRBuilder<> Builder(BB);

  Function *Fn =
      Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::sin, {FloatTy});
  Argument *Arg = Func->getArg(0);
  Value *Call = Builder.CreateCall(Fn, {Arg});
  Builder.CreateRet(Call);

  ASSERT_TRUE(isa<IntrinsicInst>(Call));
  auto *I = cast<IntrinsicInst>(Call);
  EXPECT_EQ(Intrinsic::sin, I->getIntrinsicID());

  {
    MemoryEffects ME = I->getMemoryEffects();
    EXPECT_TRUE(ME.doesNotAccessMemory());
  }
  {
    Func->addFnAttr(Attribute::StrictFP);
    MemoryEffects ME = I->getMemoryEffects();
    EXPECT_TRUE(ME.doesAccessInaccessibleMem());
  }
}

// A StrictFP function is obtained from assembly text.
TEST_F(FPOperationTest, MEParsed) {
  const char *Source = R"*(
  define float @qqq(float %x) strictfp {
    %res = call float @llvm.sin.f32(float %x)
    ret float %res
  }
  )*";
  parseAssembly(Source);
  ASSERT_TRUE(M);
  Function *F = M->getFunction("qqq");
  BasicBlock &BB = *F->begin();
  for (Instruction &I : BB) {
    if (auto *Call = dyn_cast<CallBase>(&I)) {
      ASSERT_TRUE(isa<IntrinsicInst>(Call));
      auto *I = cast<IntrinsicInst>(Call);
      EXPECT_EQ(Intrinsic::sin, I->getIntrinsicID());
      MemoryEffects ME = I->getMemoryEffects();
      EXPECT_TRUE(ME.doesAccessInaccessibleMem());
    }
  }
}

// If a call is outside a function, it is considered as strictfp.
TEST_F(FPOperationTest, MEIsolated) {
  IRBuilder<> Builder(Context);
  auto *FloatTy = Type::getFloatTy(Context);
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::sin, {FloatTy});
  auto *Arg = ConstantFP::get(FloatTy, 1.0);
  Value *Call = Builder.CreateCall(Fn, {Arg});

  ASSERT_TRUE(isa<IntrinsicInst>(Call));
  auto *I = cast<IntrinsicInst>(Call);
  EXPECT_EQ(Intrinsic::sin, I->getIntrinsicID());
  MemoryEffects ME = I->getMemoryEffects();
  EXPECT_TRUE(ME.doesAccessInaccessibleMem());

  createUnaryFunction(FloatTy, "test_1");
  I->insertInto(BB, BB->begin());
}

} // end namespace
