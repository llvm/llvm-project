//===- IntrinsicInstTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/IntrinsicInst.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct IntrinsicInstTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxIRTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(IntrinsicInstTest, Basic) {
  parseIR(C, R"IR(
declare void @llvm.sideeffect()
declare void @llvm.assume(i1)
declare i8 @llvm.uadd.sat.i8(i8, i8)
declare i8 @llvm.smax.i8(i8, i8)

define void @foo(i8 %v1, i1 %cond) {
  call void @llvm.sideeffect()
  call void @llvm.assume(i1 %cond)
  call i8 @llvm.uadd.sat.i8(i8 %v1, i8 %v1)
  call i8 @llvm.smax.i8(i8 %v1, i8 %v1)
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
  auto ItE = BB->getTerminator()->getIterator();
  for (; It != ItE; ++It, ++LLVMIt) {
    auto *I = &*It;
    auto *LLVMI = &*LLVMIt;
    // Check classof().
    EXPECT_TRUE(isa<sandboxir::IntrinsicInst>(I));
    // Check getIntrinsicID().
    EXPECT_EQ(cast<sandboxir::IntrinsicInst>(I)->getIntrinsicID(),
              cast<llvm::IntrinsicInst>(LLVMI)->getIntrinsicID());
    // Check isAssociative().
    EXPECT_EQ(cast<sandboxir::IntrinsicInst>(I)->isAssociative(),
              cast<llvm::IntrinsicInst>(LLVMI)->isAssociative());
    // Check isCommutative().
    EXPECT_EQ(cast<sandboxir::IntrinsicInst>(I)->isCommutative(),
              cast<llvm::IntrinsicInst>(LLVMI)->isCommutative());
    // Check isAssumeLikeIntrinsic().
    EXPECT_EQ(cast<sandboxir::IntrinsicInst>(I)->isAssumeLikeIntrinsic(),
              cast<llvm::IntrinsicInst>(LLVMI)->isAssumeLikeIntrinsic());
    // Check mayLowerToFunctionCall().
    auto ID = cast<sandboxir::IntrinsicInst>(I)->getIntrinsicID();
    EXPECT_EQ(sandboxir::IntrinsicInst::mayLowerToFunctionCall(ID),
              llvm::IntrinsicInst::mayLowerToFunctionCall(ID));
  }
}
