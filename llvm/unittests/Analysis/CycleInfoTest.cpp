//===- CycleInfoTest.cpp - CycleInfo unit tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using testing::ElementsAre;

static std::unique_ptr<Module> parseIR(LLVMContext &C, StringRef IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, C);
  if (!M)
    Err.print("CycleInfoTest", errs());
  return M;
}

static BasicBlock *getBlock(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("no such block");
}

// Split the edge Pred->Succ in the IR and tell CI about it.
static BasicBlock *splitEdge(CycleInfo &CI, BasicBlock *Pred,
                             BasicBlock *Succ) {
  BasicBlock *New =
      BasicBlock::Create(Pred->getContext(), "new", Pred->getParent(), Succ);
  UncondBrInst::Create(Succ, New);
  Pred->getTerminator()->replaceSuccessorWith(Succ, New);
  CI.splitCriticalEdge(Pred, Succ, New);
  return New;
}

static SmallVector<BasicBlock *> getExitBlocks(const CycleInfo &CI,
                                               CycleRef C) {
  SmallVector<BasicBlock *> Exits;
  CI.getExitBlocks(C, Exits);
  return Exits;
}

// Splitting an edge that leaves an inner cycle for its parent puts the new
// block in the parent, which changes the inner cycle's exit blocks.
TEST(CycleInfoTest, ExitBlocksAfterSplitIntoParentCycle) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @f(i1 %c) {
entry:
  br label %outer
outer:
  br label %inner
inner:
  br i1 %c, label %inner, label %latch
latch:
  br i1 %c, label %outer, label %exit
exit:
  ret void
}
)");
  ASSERT_TRUE(M);
  Function &F = *M->getFunction("f");
  BasicBlock *Inner = getBlock(F, "inner");
  BasicBlock *Latch = getBlock(F, "latch");

  CycleInfo CI;
  CI.compute(F);
  CycleRef InnerCycle = CI.getCycle(Inner);
  ASSERT_TRUE(InnerCycle);
  CycleRef OuterCycle = CI.getParentCycle(InnerCycle);
  ASSERT_TRUE(OuterCycle);

  // Populate the exit block cache before the split.
  EXPECT_THAT(getExitBlocks(CI, InnerCycle), ElementsAre(Latch));

  BasicBlock *New = splitEdge(CI, Inner, Latch);
  EXPECT_EQ(CI.getCycle(New), OuterCycle);
  EXPECT_THAT(getExitBlocks(CI, InnerCycle), ElementsAre(New));
}

// Splitting an edge that leaves a top-level cycle puts the new block outside
// every cycle, which changes the cycle's exit blocks.
TEST(CycleInfoTest, ExitBlocksAfterSplitOutsideAllCycles) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
define void @f(i1 %c) {
entry:
  br label %loop
loop:
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
)");
  ASSERT_TRUE(M);
  Function &F = *M->getFunction("f");
  BasicBlock *Loop = getBlock(F, "loop");
  BasicBlock *Exit = getBlock(F, "exit");

  CycleInfo CI;
  CI.compute(F);
  CycleRef C = CI.getCycle(Loop);
  ASSERT_TRUE(C);

  // Populate the exit block cache before the split.
  EXPECT_THAT(getExitBlocks(CI, C), ElementsAre(Exit));

  BasicBlock *New = splitEdge(CI, Loop, Exit);
  EXPECT_FALSE(CI.getCycle(New));
  EXPECT_THAT(getExitBlocks(CI, C), ElementsAre(New));
}
