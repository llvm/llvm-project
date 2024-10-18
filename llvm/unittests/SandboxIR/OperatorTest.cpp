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
