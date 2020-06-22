//===- LoopNestTest.cpp - LoopNestAnalysis unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

/// Build the loop nest analysis for a loop nest and run the given test \p Test.
static void runTest(
    Module &M, StringRef FuncName,
    function_ref<void(Function &F, LoopInfo &LI, ScalarEvolution &SE)> Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;

  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);

  Test(*F, LI, SE);
}

static std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                              const char *ModuleStr) {
  SMDiagnostic Err;
  return parseAssemblyString(ModuleStr, Err, Context);
}

TEST(LoopNestTest, PerfectLoopNest) {
  const char *ModuleStr =
    "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
    "define void @foo(i64 signext %nx, i64 signext %ny) {\n"
    "entry:\n"
    "  br label %for.outer\n"
    "for.outer:\n"
    "  %i = phi i64 [ 0, %entry ], [ %inc13, %for.outer.latch ]\n"
    "  %cmp21 = icmp slt i64 0, %ny\n"
    "  br i1 %cmp21, label %for.inner.preheader, label %for.outer.latch\n"
    "for.inner.preheader:\n"
    "  br label %for.inner\n"
    "for.inner:\n"
    "  %j = phi i64 [ 0, %for.inner.preheader ], [ %inc, %for.inner.latch ]\n"
    "  br label %for.inner.latch\n"
    "for.inner.latch:\n"
    "  %inc = add nsw i64 %j, 1\n"
    "  %cmp2 = icmp slt i64 %inc, %ny\n"
    "  br i1 %cmp2, label %for.inner, label %for.inner.exit\n"
    "for.inner.exit:\n"
    "  br label %for.outer.latch\n"
    "for.outer.latch:\n"
    "  %inc13 = add nsw i64 %i, 1\n"
    "  %cmp = icmp slt i64 %inc13, %nx\n"
    "  br i1 %cmp, label %for.outer, label %for.outer.exit\n"
    "for.outer.exit:\n"
    "  br label %for.end\n"
    "for.end:\n"
    "  ret void\n"
    "}\n";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runTest(*M, "foo", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
    Function::iterator FI = F.begin();
    // Skip the first basic block (entry), get to the outer loop header.
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "for.outer");
    Loop *L = LI.getLoopFor(Header);
    EXPECT_NE(L, nullptr);

    LoopNest LN(*L, SE);
    EXPECT_TRUE(LN.areAllLoopsSimplifyForm());

    // Ensure that we can identify the outermost loop in the nest.
    const Loop &OL = LN.getOutermostLoop();
    EXPECT_EQ(OL.getName(), "for.outer");

    // Ensure that we can identify the innermost loop in the nest.
    const Loop *IL = LN.getInnermostLoop();
    EXPECT_NE(IL, nullptr);
    EXPECT_EQ(IL->getName(), "for.inner");

    // Ensure the loop nest is recognized as having 2 loops.
    const ArrayRef<Loop*> Loops = LN.getLoops();
    EXPECT_EQ(Loops.size(), 2ull);

    // Ensure the loop nest is recognized as perfect in its entirety.
    const SmallVector<LoopVectorTy, 4> &PLV = LN.getPerfectLoops(SE);
    EXPECT_EQ(PLV.size(), 1ull);
    EXPECT_EQ(PLV.front().size(), 2ull);

    // Ensure the nest depth and perfect nest depth are computed correctly.
    EXPECT_EQ(LN.getNestDepth(), 2u);
    EXPECT_EQ(LN.getMaxPerfectDepth(), 2u);
  });
}

TEST(LoopNestTest, ImperfectLoopNest) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo(i32 signext %nx, i32 signext %ny, i32 signext %nk) {\n"
      "entry:\n"
      "  br label %loop.i\n"
      "loop.i:\n"
      "  %i = phi i32 [ 0, %entry ], [ %inci, %for.inci ]\n"
      "  %cmp21 = icmp slt i32 0, %ny\n"
      "  br i1 %cmp21, label %loop.j.preheader, label %for.inci\n"
      "loop.j.preheader:\n"
      "  br label %loop.j\n"
      "loop.j:\n"
      "  %j = phi i32 [ %incj, %for.incj ], [ 0, %loop.j.preheader ]\n"
      "  %cmp22 = icmp slt i32 0, %nk\n"
      "  br i1 %cmp22, label %loop.k.preheader, label %for.incj\n"
      "loop.k.preheader:\n"
      "  call void @bar()\n"
      "  br label %loop.k\n"
      "loop.k:\n"
      "  %k = phi i32 [ %inck, %for.inck ], [ 0, %loop.k.preheader ]\n"
      "  br label %for.inck\n"
      "for.inck:\n"
      "  %inck = add nsw i32 %k, 1\n"
      "  %cmp5 = icmp slt i32 %inck, %nk\n"
      "  br i1 %cmp5, label %loop.k, label %for.incj.loopexit\n"
      "for.incj.loopexit:\n"
      "  br label %for.incj\n"
      "for.incj:\n"
      "  %incj = add nsw i32 %j, 1\n"
      "  %cmp2 = icmp slt i32 %incj, %ny\n"
      "  br i1 %cmp2, label %loop.j, label %for.inci.loopexit\n"
      "for.inci.loopexit:\n"
      "  br label %for.inci\n"
      "for.inci:\n"
      "  %inci = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inci, %nx\n"
      "  br i1 %cmp, label %loop.i, label %loop.i.end\n"
      "loop.i.end:\n"
      "  ret void\n"
      "}\n"
      "declare void @bar()\n";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runTest(*M, "foo", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
    Function::iterator FI = F.begin();
    // Skip the first basic block (entry), get to the outermost loop header.
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "loop.i");
    Loop *L = LI.getLoopFor(Header);
    EXPECT_NE(L, nullptr);

    LoopNest LN(*L, SE);
    EXPECT_TRUE(LN.areAllLoopsSimplifyForm());

    dbgs() << "LN: " << LN << "\n";

    // Ensure that we can identify the outermost loop in the nest.
    const Loop &OL = LN.getOutermostLoop();
    EXPECT_EQ(OL.getName(), "loop.i");

    // Ensure that we can identify the innermost loop in the nest.
    const Loop *IL = LN.getInnermostLoop();
    EXPECT_NE(IL, nullptr);
    EXPECT_EQ(IL->getName(), "loop.k");

    // Ensure the loop nest is recognized as having 3 loops.
    const ArrayRef<Loop*> Loops = LN.getLoops();
    EXPECT_EQ(Loops.size(), 3ull);

    // Ensure the loop nest is recognized as having 2 separate perfect loops groups.
    const SmallVector<LoopVectorTy, 4> &PLV = LN.getPerfectLoops(SE);
    EXPECT_EQ(PLV.size(), 2ull);
    EXPECT_EQ(PLV.front().size(), 2ull);
    EXPECT_EQ(PLV.back().size(), 1ull);

    // Ensure the nest depth and perfect nest depth are computed correctly.
    EXPECT_EQ(LN.getNestDepth(), 3u);
    EXPECT_EQ(LN.getMaxPerfectDepth(), 2u);
  });
}

