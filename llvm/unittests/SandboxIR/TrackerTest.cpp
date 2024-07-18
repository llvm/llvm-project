//===- TrackerTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct TrackerTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("TrackerTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(TrackerTest, SetOperand) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %gep0
  store float undef, ptr %gep0
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto &Tracker = Ctx.getTracker();
  Tracker.save();
  auto It = BB->begin();
  auto *Gep0 = &*It++;
  auto *Gep1 = &*It++;
  auto *Ld = &*It++;
  auto *St = &*It++;
  St->setOperand(0, Ld);
  St->setOperand(1, Gep1);
  Ld->setOperand(0, Gep1);
  EXPECT_EQ(St->getOperand(0), Ld);
  EXPECT_EQ(St->getOperand(1), Gep1);
  EXPECT_EQ(Ld->getOperand(0), Gep1);

  Ctx.getTracker().revert();
  EXPECT_NE(St->getOperand(0), Ld);
  EXPECT_EQ(St->getOperand(1), Gep0);
  EXPECT_EQ(Ld->getOperand(0), Gep0);
}

TEST_F(TrackerTest, RUWIf_RAUW_RUOW) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load float, ptr %ptr
  %ld1 = load float, ptr %ptr
  store float %ld0, ptr %ptr
  store float %ld0, ptr %ptr
  ret void
}
)IR");
  llvm::Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  llvm::BasicBlock *LLVMBB = &*LLVMF.begin();
  Ctx.createFunction(&LLVMF);
  auto *BB = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMBB));
  auto It = BB->begin();
  sandboxir::Instruction *Ld0 = &*It++;
  sandboxir::Instruction *Ld1 = &*It++;
  sandboxir::Instruction *St0 = &*It++;
  sandboxir::Instruction *St1 = &*It++;
  Ctx.save();
  // Check RUWIf when the lambda returns false.
  Ld0->replaceUsesWithIf(Ld1, [](const sandboxir::Use &Use) { return false; });
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RUWIf when the lambda returns true.
  Ld0->replaceUsesWithIf(Ld1, [](const sandboxir::Use &Use) { return true; });
  EXPECT_EQ(St0->getOperand(0), Ld1);
  EXPECT_EQ(St1->getOperand(0), Ld1);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RUWIf user == St0.
  Ctx.save();
  Ld0->replaceUsesWithIf(
      Ld1, [St0](const sandboxir::Use &Use) { return Use.getUser() == St0; });
  EXPECT_EQ(St0->getOperand(0), Ld1);
  EXPECT_EQ(St1->getOperand(0), Ld0);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RUWIf user == St1.
  Ctx.save();
  Ld0->replaceUsesWithIf(
      Ld1, [St1](const sandboxir::Use &Use) { return Use.getUser() == St1; });
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld1);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RAUW.
  Ctx.save();
  Ld1->replaceAllUsesWith(Ld0);
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RUOW.
  Ctx.save();
  St0->replaceUsesOfWith(Ld0, Ld1);
  EXPECT_EQ(St0->getOperand(0), Ld1);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);

  // Check accept().
  Ctx.save();
  St0->replaceUsesOfWith(Ld0, Ld1);
  EXPECT_EQ(St0->getOperand(0), Ld1);
  Ctx.accept();
  EXPECT_EQ(St0->getOperand(0), Ld1);
}
