//===- UtilsTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Utils.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct UtilsTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("UtilsTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(UtilsTest, getMemoryLocation) {
  parseIR(C, R"IR(
define void @foo(ptr %arg0) {
  %ld = load i8, ptr %arg0
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  auto *LLVMBB = &*LLVMF->begin();
  auto *LLVMLd = cast<llvm::LoadInst>(&*LLVMBB->begin());
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto *Ld = cast<sandboxir::LoadInst>(&*BB->begin());
  EXPECT_EQ(sandboxir::Utils::memoryLocationGetOrNone(Ld),
            MemoryLocation::getOrNone(LLVMLd));
}

TEST_F(UtilsTest, GetPointerDiffInBytes) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr inbounds float, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds float, ptr %ptr, i64 1
  %gep2 = getelementptr inbounds float, ptr %ptr, i64 2
  %gep3 = getelementptr inbounds float, ptr %ptr, i64 3

  %ld0 = load float, ptr %gep0
  %ld1 = load float, ptr %gep1
  %ld2 = load float, ptr %gep2
  %ld3 = load float, ptr %gep3

  %v2ld0 = load <2 x float>, ptr %gep0
  %v2ld1 = load <2 x float>, ptr %gep1
  %v2ld2 = load <2 x float>, ptr %gep2
  %v2ld3 = load <2 x float>, ptr %gep3

  %v3ld0 = load <3 x float>, ptr %gep0
  %v3ld1 = load <3 x float>, ptr %gep1
  %v3ld2 = load <3 x float>, ptr %gep2
  %v3ld3 = load <3 x float>, ptr %gep3
  ret void
}
)IR");
  llvm::Function &LLVMF = *M->getFunction("foo");
  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  AssumptionCache AC(LLVMF);
  BasicAAResult BAA(DL, LLVMF, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  LoopInfo LI(DT);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);
  sandboxir::Context Ctx(C);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto &BB = *F.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  [[maybe_unused]] auto *L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V2L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L3 = cast<sandboxir::LoadInst>(&*It++);

  [[maybe_unused]] auto *V3L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L1 = cast<sandboxir::LoadInst>(&*It++);
  [[maybe_unused]] auto *V3L2 = cast<sandboxir::LoadInst>(&*It++);
  [[maybe_unused]] auto *V3L3 = cast<sandboxir::LoadInst>(&*It++);

  // getPointerDiffInBytes
  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(L0, L1, SE), 4);
  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(L0, L2, SE), 8);
  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(L1, L0, SE), -4);
  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(L0, V2L0, SE), 0);

  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(L0, V2L1, SE), 4);
  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(L0, V3L1, SE), 4);
  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(V2L0, V2L2, SE), 8);
  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(V2L0, V2L3, SE), 12);
  EXPECT_EQ(*sandboxir::Utils::getPointerDiffInBytes(V2L3, V2L0, SE), -12);

  // atLowerAddress
  EXPECT_TRUE(sandboxir::Utils::atLowerAddress(L0, L1, SE));
  EXPECT_FALSE(sandboxir::Utils::atLowerAddress(L1, L0, SE));
  EXPECT_FALSE(sandboxir::Utils::atLowerAddress(L3, V3L3, SE));
}

TEST_F(UtilsTest, GetExpected) {
  parseIR(C, R"IR(
define float @foo(float %v, ptr %ptr) {
  %add = fadd float %v, %v
  store float %v, ptr %ptr
  ret float %v
}
define void @bar(float %v, ptr %ptr) {
  ret void
}
)IR");
  llvm::Function &Foo = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  Ctx.createFunction(&Foo);
  auto *FooBB = cast<sandboxir::BasicBlock>(Ctx.getValue(&*Foo.begin()));
  auto FooIt = FooBB->begin();
  auto Add = cast<sandboxir::Instruction>(&*FooIt++);
  auto *S0 = cast<sandboxir::Instruction>(&*FooIt++);
  auto *RetF = cast<sandboxir::Instruction>(&*FooIt++);
  // getExpectedValue
  EXPECT_EQ(sandboxir::Utils::getExpectedValue(Add), Add);
  EXPECT_EQ(sandboxir::Utils::getExpectedValue(S0),
            cast<sandboxir::StoreInst>(S0)->getValueOperand());
  EXPECT_EQ(sandboxir::Utils::getExpectedValue(RetF),
            cast<sandboxir::ReturnInst>(RetF)->getReturnValue());
  // getExpectedType
  EXPECT_EQ(sandboxir::Utils::getExpectedType(Add), Add->getType());
  EXPECT_EQ(sandboxir::Utils::getExpectedType(S0),
            cast<sandboxir::StoreInst>(S0)->getValueOperand()->getType());
  EXPECT_EQ(sandboxir::Utils::getExpectedType(RetF),
            cast<sandboxir::ReturnInst>(RetF)->getReturnValue()->getType());

  // getExpectedValue for void returns
  llvm::Function &Bar = *M->getFunction("bar");
  Ctx.createFunction(&Bar);
  auto *BarBB = cast<sandboxir::BasicBlock>(Ctx.getValue(&*Bar.begin()));
  auto BarIt = BarBB->begin();
  auto *RetV = cast<sandboxir::Instruction>(&*BarIt++);
  EXPECT_EQ(sandboxir::Utils::getExpectedValue(RetV), nullptr);
}

TEST_F(UtilsTest, GetNumBits) {
  parseIR(C, R"IR(
define void @foo(float %arg0, double %arg1, i8 %arg2, i64 %arg3, ptr %arg4) {
bb0:
  %ld0 = load float, ptr %arg4
  %ld1 = load double, ptr %arg4
  %ld2 = load i8, ptr %arg4
  %ld3 = load i64, ptr %arg4
  ret void
}
)IR");
  llvm::Function &Foo = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(&Foo);
  const DataLayout &DL = M->getDataLayout();
  // getNumBits for scalars via the Value overload
  EXPECT_EQ(sandboxir::Utils::getNumBits(F->getArg(0), DL),
            DL.getTypeSizeInBits(Type::getFloatTy(C)));
  EXPECT_EQ(sandboxir::Utils::getNumBits(F->getArg(1), DL),
            DL.getTypeSizeInBits(Type::getDoubleTy(C)));
  EXPECT_EQ(sandboxir::Utils::getNumBits(F->getArg(2), DL), 8u);
  EXPECT_EQ(sandboxir::Utils::getNumBits(F->getArg(3), DL), 64u);

  auto &BB = *F->begin();
  auto It = BB.begin();
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L3 = cast<sandboxir::LoadInst>(&*It++);
  // getNumBits for scalars via the Instruction overload
  EXPECT_EQ(sandboxir::Utils::getNumBits(L0),
            DL.getTypeSizeInBits(Type::getFloatTy(C)));
  EXPECT_EQ(sandboxir::Utils::getNumBits(L1),
            DL.getTypeSizeInBits(Type::getDoubleTy(C)));
  EXPECT_EQ(sandboxir::Utils::getNumBits(L2), 8u);
  EXPECT_EQ(sandboxir::Utils::getNumBits(L3), 64u);
}

TEST_F(UtilsTest, GetMemBase) {
  parseIR(C, R"IR(
define void @foo(ptr %ptrA, float %val, ptr %ptrB) {
bb:
  %gepA0 = getelementptr float, ptr %ptrA, i32 0
  %gepA1 = getelementptr float, ptr %ptrA, i32 1
  %gepB0 = getelementptr float, ptr %ptrB, i32 0
  %gepB1 = getelementptr float, ptr %ptrB, i32 1
  store float %val, ptr %gepA0
  store float %val, ptr %gepA1
  store float %val, ptr %gepB0
  store float %val, ptr %gepB1
  ret void
}
)IR");
  llvm::Function &Foo = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(&Foo);

  auto It = std::next(F->begin()->begin(), 4);
  auto *St0 = cast<sandboxir::StoreInst>(&*It++);
  auto *St1 = cast<sandboxir::StoreInst>(&*It++);
  auto *St2 = cast<sandboxir::StoreInst>(&*It++);
  auto *St3 = cast<sandboxir::StoreInst>(&*It++);
  EXPECT_EQ(sandboxir::Utils::getMemInstructionBase(St0),
            sandboxir::Utils::getMemInstructionBase(St1));
  EXPECT_EQ(sandboxir::Utils::getMemInstructionBase(St2),
            sandboxir::Utils::getMemInstructionBase(St3));
  EXPECT_NE(sandboxir::Utils::getMemInstructionBase(St0),
            sandboxir::Utils::getMemInstructionBase(St3));
}
