//===- OperatorTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Operator.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Value.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct OperatorTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("OperatorTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(OperatorTest, Operator) {
  parseIR(C, R"IR(
define void @foo(i8 %v1) {
  %add0 = add i8 %v1, 42
  %add1 = add nuw i8 %v1, 42
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *OperatorI0 = cast<sandboxir::Operator>(&*It++);
  auto *OperatorI1 = cast<sandboxir::Operator>(&*It++);
  EXPECT_FALSE(OperatorI0->hasPoisonGeneratingFlags());
  EXPECT_TRUE(OperatorI1->hasPoisonGeneratingFlags());
}

TEST_F(OperatorTest, OverflowingBinaryOperator) {
  parseIR(C, R"IR(
define void @foo(i8 %v1) {
  %add = add i8 %v1, 42
  %addNSW = add nsw i8 %v1, 42
  %addNUW = add nuw i8 %v1, 42
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Add = cast<sandboxir::OverflowingBinaryOperator>(&*It++);
  auto *AddNSW = cast<sandboxir::OverflowingBinaryOperator>(&*It++);
  auto *AddNUW = cast<sandboxir::OverflowingBinaryOperator>(&*It++);
  EXPECT_FALSE(Add->hasNoUnsignedWrap());
  EXPECT_FALSE(Add->hasNoSignedWrap());
  EXPECT_EQ(Add->getNoWrapKind(), llvm::OverflowingBinaryOperator::AnyWrap);

  EXPECT_FALSE(AddNSW->hasNoUnsignedWrap());
  EXPECT_TRUE(AddNSW->hasNoSignedWrap());
  EXPECT_EQ(AddNSW->getNoWrapKind(),
            llvm::OverflowingBinaryOperator::NoSignedWrap);

  EXPECT_TRUE(AddNUW->hasNoUnsignedWrap());
  EXPECT_FALSE(AddNUW->hasNoSignedWrap());
  EXPECT_EQ(AddNUW->getNoWrapKind(),
            llvm::OverflowingBinaryOperator::NoUnsignedWrap);
}

TEST_F(OperatorTest, FPMathOperator) {
  parseIR(C, R"IR(
define void @foo(float %v1, double %v2) {
  %fadd = fadd float %v1, 42.0
  %Fast = fadd fast float %v1, 42.0
  %Reassoc = fmul reassoc float %v1, 42.0
  %NNAN = fmul nnan float %v1, 42.0
  %NINF = fmul ninf float %v1, 42.0
  %NSZ = fmul nsz float %v1, 42.0
  %ARCP = fmul arcp float %v1, 42.0
  %CONTRACT = fmul contract float %v1, 42.0
  %AFN = fmul afn double %v2, 42.0
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  auto *LLVMBB = &*LLVMF->begin();
  auto LLVMIt = LLVMBB->begin();

  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto TermIt = BB->getTerminator()->getIterator();
  while (It != TermIt) {
    auto *FPM = cast<sandboxir::FPMathOperator>(&*It++);
    auto *LLVMFPM = cast<llvm::FPMathOperator>(&*LLVMIt++);
    EXPECT_EQ(FPM->isFast(), LLVMFPM->isFast());
    EXPECT_EQ(FPM->hasAllowReassoc(), LLVMFPM->hasAllowReassoc());
    EXPECT_EQ(FPM->hasNoNaNs(), LLVMFPM->hasNoNaNs());
    EXPECT_EQ(FPM->hasNoInfs(), LLVMFPM->hasNoInfs());
    EXPECT_EQ(FPM->hasNoSignedZeros(), LLVMFPM->hasNoSignedZeros());
    EXPECT_EQ(FPM->hasAllowReciprocal(), LLVMFPM->hasAllowReciprocal());
    EXPECT_EQ(FPM->hasAllowContract(), LLVMFPM->hasAllowContract());
    EXPECT_EQ(FPM->hasApproxFunc(), LLVMFPM->hasApproxFunc());

    // There doesn't seem to be an operator== for FastMathFlags so let's do a
    // string comparison instead.
    std::string Str1;
    raw_string_ostream SS1(Str1);
    std::string Str2;
    raw_string_ostream SS2(Str2);
    FPM->getFastMathFlags().print(SS1);
    LLVMFPM->getFastMathFlags().print(SS2);
    EXPECT_EQ(Str1, Str2);

    EXPECT_EQ(FPM->getFPAccuracy(), LLVMFPM->getFPAccuracy());
    EXPECT_EQ(
        sandboxir::FPMathOperator::isSupportedFloatingPointType(FPM->getType()),
        llvm::FPMathOperator::isSupportedFloatingPointType(LLVMFPM->getType()));
  }
}
