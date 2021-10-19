//===- LoopInfoTest.cpp - LoopInfo unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

/// Build the loop info for the function and run the Test.
static void
runWithLoopInfo(Module &M, StringRef FuncName,
                function_ref<void(Function &F, LoopInfo &LI)> Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;
  // Compute the dominator tree and the loop info for the function.
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  Test(*F, LI);
}

/// Build the loop info and scalar evolution for the function and run the Test.
static void runWithLoopInfoPlus(
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

// This tests that for a loop with a single latch, we get the loop id from
// its only latch, even in case the loop may not be in a simplified form.
TEST(LoopInfoTest, LoopWithSingleLatch) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo(i32 %n) {\n"
      "entry:\n"
      "  br i1 undef, label %for.cond, label %for.end\n"
      "for.cond:\n"
      "  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]\n"
      "  %cmp = icmp slt i32 %i.0, %n\n"
      "  br i1 %cmp, label %for.inc, label %for.end\n"
      "for.inc:\n"
      "  %inc = add nsw i32 %i.0, 1\n"
      "  br label %for.cond, !llvm.loop !0\n"
      "for.end:\n"
      "  ret void\n"
      "}\n"
      "!0 = distinct !{!0, !1}\n"
      "!1 = !{!\"llvm.loop.distribute.enable\", i1 true}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfo(*M, "foo", [&](Function &F, LoopInfo &LI) {
    Function::iterator FI = F.begin();
    // First basic block is entry - skip it.
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "for.cond");
    Loop *L = LI.getLoopFor(Header);

    // This loop is not in simplified form.
    EXPECT_FALSE(L->isLoopSimplifyForm());

    // Analyze the loop metadata id.
    bool loopIDFoundAndSet = false;
    // Try to get and set the metadata id for the loop.
    if (MDNode *D = L->getLoopID()) {
      L->setLoopID(D);
      loopIDFoundAndSet = true;
    }

    // We must have successfully found and set the loop id in the
    // only latch the loop has.
    EXPECT_TRUE(loopIDFoundAndSet);
  });
}

// Test loop id handling for a loop with multiple latches.
TEST(LoopInfoTest, LoopWithMultipleLatches) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo(i32 %n) {\n"
      "entry:\n"
      "  br i1 undef, label %for.cond, label %for.end\n"
      "for.cond:\n"
      "  %i.0 = phi i32 [ 0, %entry ], [ %inc, %latch.1 ], [ %inc, %latch.2 ]\n"
      "  %inc = add nsw i32 %i.0, 1\n"
      "  %cmp = icmp slt i32 %i.0, %n\n"
      "  br i1 %cmp, label %latch.1, label %for.end\n"
      "latch.1:\n"
      "  br i1 undef, label %for.cond, label %latch.2, !llvm.loop !0\n"
      "latch.2:\n"
      "  br label %for.cond, !llvm.loop !0\n"
      "for.end:\n"
      "  ret void\n"
      "}\n"
      "!0 = distinct !{!0, !1}\n"
      "!1 = !{!\"llvm.loop.distribute.enable\", i1 true}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfo(*M, "foo", [&](Function &F, LoopInfo &LI) {
    Function::iterator FI = F.begin();
    // First basic block is entry - skip it.
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "for.cond");
    Loop *L = LI.getLoopFor(Header);
    EXPECT_NE(L, nullptr);

    // This loop is not in simplified form.
    EXPECT_FALSE(L->isLoopSimplifyForm());

    // Try to get and set the metadata id for the loop.
    MDNode *OldLoopID = L->getLoopID();
    EXPECT_NE(OldLoopID, nullptr);

    MDNode *NewLoopID = MDNode::get(Context, {nullptr});
    // Set operand 0 to refer to the loop id itself.
    NewLoopID->replaceOperandWith(0, NewLoopID);

    L->setLoopID(NewLoopID);
    EXPECT_EQ(L->getLoopID(), NewLoopID);
    EXPECT_NE(L->getLoopID(), OldLoopID);

    L->setLoopID(OldLoopID);
    EXPECT_EQ(L->getLoopID(), OldLoopID);
    EXPECT_NE(L->getLoopID(), NewLoopID);
  });
}

TEST(LoopInfoTest, PreorderTraversals) {
  const char *ModuleStr = "define void @f() {\n"
                          "entry:\n"
                          "  br label %loop.0\n"
                          "loop.0:\n"
                          "  br i1 undef, label %loop.0.0, label %loop.1\n"
                          "loop.0.0:\n"
                          "  br i1 undef, label %loop.0.0, label %loop.0.1\n"
                          "loop.0.1:\n"
                          "  br i1 undef, label %loop.0.1, label %loop.0.2\n"
                          "loop.0.2:\n"
                          "  br i1 undef, label %loop.0.2, label %loop.0\n"
                          "loop.1:\n"
                          "  br i1 undef, label %loop.1.0, label %end\n"
                          "loop.1.0:\n"
                          "  br i1 undef, label %loop.1.0, label %loop.1.1\n"
                          "loop.1.1:\n"
                          "  br i1 undef, label %loop.1.1, label %loop.1.2\n"
                          "loop.1.2:\n"
                          "  br i1 undef, label %loop.1.2, label %loop.1\n"
                          "end:\n"
                          "  ret void\n"
                          "}\n";
  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);
  Function &F = *M->begin();

  DominatorTree DT(F);
  LoopInfo LI;
  LI.analyze(DT);

  Function::iterator I = F.begin();
  ASSERT_EQ("entry", I->getName());
  ++I;
  Loop &L_0 = *LI.getLoopFor(&*I++);
  ASSERT_EQ("loop.0", L_0.getHeader()->getName());
  Loop &L_0_0 = *LI.getLoopFor(&*I++);
  ASSERT_EQ("loop.0.0", L_0_0.getHeader()->getName());
  Loop &L_0_1 = *LI.getLoopFor(&*I++);
  ASSERT_EQ("loop.0.1", L_0_1.getHeader()->getName());
  Loop &L_0_2 = *LI.getLoopFor(&*I++);
  ASSERT_EQ("loop.0.2", L_0_2.getHeader()->getName());
  Loop &L_1 = *LI.getLoopFor(&*I++);
  ASSERT_EQ("loop.1", L_1.getHeader()->getName());
  Loop &L_1_0 = *LI.getLoopFor(&*I++);
  ASSERT_EQ("loop.1.0", L_1_0.getHeader()->getName());
  Loop &L_1_1 = *LI.getLoopFor(&*I++);
  ASSERT_EQ("loop.1.1", L_1_1.getHeader()->getName());
  Loop &L_1_2 = *LI.getLoopFor(&*I++);
  ASSERT_EQ("loop.1.2", L_1_2.getHeader()->getName());

  auto Preorder = LI.getLoopsInPreorder();
  ASSERT_EQ(8u, Preorder.size());
  EXPECT_EQ(&L_0, Preorder[0]);
  EXPECT_EQ(&L_0_0, Preorder[1]);
  EXPECT_EQ(&L_0_1, Preorder[2]);
  EXPECT_EQ(&L_0_2, Preorder[3]);
  EXPECT_EQ(&L_1, Preorder[4]);
  EXPECT_EQ(&L_1_0, Preorder[5]);
  EXPECT_EQ(&L_1_1, Preorder[6]);
  EXPECT_EQ(&L_1_2, Preorder[7]);

  auto ReverseSiblingPreorder = LI.getLoopsInReverseSiblingPreorder();
  ASSERT_EQ(8u, ReverseSiblingPreorder.size());
  EXPECT_EQ(&L_1, ReverseSiblingPreorder[0]);
  EXPECT_EQ(&L_1_2, ReverseSiblingPreorder[1]);
  EXPECT_EQ(&L_1_1, ReverseSiblingPreorder[2]);
  EXPECT_EQ(&L_1_0, ReverseSiblingPreorder[3]);
  EXPECT_EQ(&L_0, ReverseSiblingPreorder[4]);
  EXPECT_EQ(&L_0_2, ReverseSiblingPreorder[5]);
  EXPECT_EQ(&L_0_1, ReverseSiblingPreorder[6]);
  EXPECT_EQ(&L_0_0, ReverseSiblingPreorder[7]);
}

TEST(LoopInfoTest, CanonicalLoop) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopWithInverseGuardSuccs) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp sge i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.end, label %for.preheader\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopWithSwappedGuardCmp) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp sgt i32 %ub, 0\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp sge i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.exit, label %for.body\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopWithInverseLatchSuccs) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp sge i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.exit, label %for.body\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopWithLatchCmpNE) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp ne i32 %i, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopWithGuardCmpSLE) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %ubPlusOne = add i32 %ub, 1\n"
      "  %guardcmp = icmp sle i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp ne i32 %i, %ubPlusOne\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ubPlusOne");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopNonConstantStep) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub, i32 %step) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = zext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, %step\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        EXPECT_EQ(Bounds->getStepValue()->getName(), "step");
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(), Loop::LoopBounds::Direction::Unknown);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopUnsignedBounds) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp ult i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = zext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add i32 %i, 1\n"
      "  %cmp = icmp ult i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_ULT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, DecreasingLoop) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ %ub, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = sub nsw i32 %i, 1\n"
      "  %cmp = icmp sgt i32 %inc, 0\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        EXPECT_EQ(Bounds->getInitialIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_EQ(StepValue, nullptr);
        ConstantInt *FinalIVValue =
            dyn_cast<ConstantInt>(&Bounds->getFinalIVValue());
        EXPECT_TRUE(FinalIVValue && FinalIVValue->isZero());
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SGT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Decreasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, CannotFindDirection) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub, i32 %step) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, %step\n"
      "  %cmp = icmp ne i32 %i, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader
        // - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        EXPECT_EQ(Bounds->getStepValue()->getName(), "step");
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(),
                  ICmpInst::BAD_ICMP_PREDICATE);
        EXPECT_EQ(Bounds->getDirection(), Loop::LoopBounds::Direction::Unknown);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, ZextIndVar) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %indvars.iv = phi i64 [ 0, %for.preheader ], [ %indvars.iv.next, %for.body ]\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %wide.trip.count = zext i32 %ub to i64\n"
      "  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count\n"
      "  br i1 %exitcond, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "indvars.iv.next");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "wide.trip.count");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_NE);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "indvars.iv");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, MultiExitingLoop) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub, i1 %cond) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body.1 ]\n"
      "  br i1 %cond, label %for.body.1, label %for.exit\n"
      "for.body.1:\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
      });
}

TEST(LoopInfoTest, MultiExitLoop) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub, i1 %cond) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body.1 ]\n"
      "  br i1 %cond, label %for.body.1, label %for.exit\n"
      "for.body.1:\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit.1\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.exit.1:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), nullptr);
        EXPECT_FALSE(L->isGuarded());
      });
}

TEST(LoopInfoTest, UnguardedLoop) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        // First basic block is entry - skip it.
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), nullptr);
        EXPECT_FALSE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, UnguardedLoopWithControlFlow) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub, i1 %cond) {\n"
      "entry:\n"
      "  br i1 %cond, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopNest) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.outer.preheader, label %for.end\n"
      "for.outer.preheader:\n"
      "  br label %for.outer\n"
      "for.outer:\n"
      "  %j = phi i32 [ 0, %for.outer.preheader ], [ %inc.outer, %for.outer.latch ]\n"
      "  br i1 %guardcmp, label %for.inner.preheader, label %for.outer.latch\n"
      "for.inner.preheader:\n"
      "  br label %for.inner\n"
      "for.inner:\n"
      "  %i = phi i32 [ 0, %for.inner.preheader ], [ %inc, %for.inner ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.inner, label %for.inner.exit\n"
      "for.inner.exit:\n"
      "  br label %for.outer.latch\n"
      "for.outer.latch:\n"
      "  %inc.outer = add nsw i32 %j, 1\n"
      "  %cmp.outer = icmp slt i32 %inc.outer, %ub\n"
      "  br i1 %cmp.outer, label %for.outer, label %for.outer.exit\n"
      "for.outer.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *OuterGuard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.outer.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.outer");
        BranchInst *InnerGuard = dyn_cast<BranchInst>(Header->getTerminator());
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc.outer");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "j");
        EXPECT_EQ(L->getLoopGuardBranch(), OuterGuard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());

        // Next two basic blocks are for.outer and for.inner.preheader - skip
        // them.
        ++FI;
        Header = &*(++FI);
        assert(Header->getName() == "for.inner");
        L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> InnerBounds = L->getBounds(SE);
        EXPECT_NE(InnerBounds, None);
        InitialIVValue =
            dyn_cast<ConstantInt>(&InnerBounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(InnerBounds->getStepInst().getName(), "inc");
        StepValue = dyn_cast_or_null<ConstantInt>(InnerBounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(InnerBounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(InnerBounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(InnerBounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        EXPECT_EQ(L->getLoopGuardBranch(), InnerGuard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, AuxiliaryIV) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %aux = phi i32 [ 0, %for.preheader ], [ %auxinc, %for.body ]\n"
      "  %loopvariant = phi i32 [ 0, %for.preheader ], [ %loopvariantinc, %for.body ]\n"
      "  %usedoutside = phi i32 [ 0, %for.preheader ], [ %usedoutsideinc, %for.body ]\n"
      "  %mulopcode = phi i32 [ 0, %for.preheader ], [ %mulopcodeinc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %mulopcodeinc = mul nsw i32 %mulopcode, 5\n"
      "  %usedoutsideinc = add nsw i32 %usedoutside, 5\n"
      "  %loopvariantinc = add nsw i32 %loopvariant, %i\n"
      "  %auxinc = add nsw i32 %aux, 5\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.body, label %for.exit\n"
      "for.exit:\n"
      "  %lcssa = phi i32 [ %usedoutside, %for.body ]\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI);
        BranchInst *Guard = dyn_cast<BranchInst>(Entry->getTerminator());
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);

        Optional<Loop::LoopBounds> Bounds = L->getBounds(SE);
        EXPECT_NE(Bounds, None);
        ConstantInt *InitialIVValue =
            dyn_cast<ConstantInt>(&Bounds->getInitialIVValue());
        EXPECT_TRUE(InitialIVValue && InitialIVValue->isZero());
        EXPECT_EQ(Bounds->getStepInst().getName(), "inc");
        ConstantInt *StepValue =
            dyn_cast_or_null<ConstantInt>(Bounds->getStepValue());
        EXPECT_TRUE(StepValue && StepValue->isOne());
        EXPECT_EQ(Bounds->getFinalIVValue().getName(), "ub");
        EXPECT_EQ(Bounds->getCanonicalPredicate(), ICmpInst::ICMP_SLT);
        EXPECT_EQ(Bounds->getDirection(),
                  Loop::LoopBounds::Direction::Increasing);
        EXPECT_EQ(L->getInductionVariable(SE)->getName(), "i");
        BasicBlock::iterator II = Header->begin();
        PHINode &Instruction_i = cast<PHINode>(*(II));
        EXPECT_TRUE(L->isAuxiliaryInductionVariable(Instruction_i, SE));
        PHINode &Instruction_aux = cast<PHINode>(*(++II));
        EXPECT_TRUE(L->isAuxiliaryInductionVariable(Instruction_aux, SE));
        PHINode &Instruction_loopvariant = cast<PHINode>(*(++II));
        EXPECT_FALSE(
            L->isAuxiliaryInductionVariable(Instruction_loopvariant, SE));
        PHINode &Instruction_usedoutside = cast<PHINode>(*(++II));
        EXPECT_FALSE(
            L->isAuxiliaryInductionVariable(Instruction_usedoutside, SE));
        PHINode &Instruction_mulopcode = cast<PHINode>(*(++II));
        EXPECT_FALSE(
            L->isAuxiliaryInductionVariable(Instruction_mulopcode, SE));
        EXPECT_EQ(L->getLoopGuardBranch(), Guard);
        EXPECT_TRUE(L->isGuarded());
        EXPECT_TRUE(L->isRotatedForm());
      });
}

TEST(LoopInfoTest, LoopNotInSimplifyForm) {
  const char *ModuleStr =
      "define void @foo(i32 %n) {\n"
      "entry:\n"
      "  %guard.cmp = icmp sgt i32 %n, 0\n"
      "  br i1 %guard.cmp, label %for.cond, label %for.end\n"
      "for.cond:\n"
      "  %i.0 = phi i32 [ 0, %entry ], [ %inc, %latch.1 ], [ %inc, %latch.2 ]\n"
      "  %inc = add nsw i32 %i.0, 1\n"
      "  %cmp = icmp slt i32 %i.0, %n\n"
      "  br i1 %cmp, label %latch.1, label %for.end\n"
      "latch.1:\n"
      "  br i1 undef, label %for.cond, label %latch.2\n"
      "latch.2:\n"
      "  br label %for.cond\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfo(*M, "foo", [&](Function &F, LoopInfo &LI) {
    Function::iterator FI = F.begin();
    // First basic block is entry - skip it.
    BasicBlock *Header = &*(++FI);
    assert(Header && "No header");
    Loop *L = LI.getLoopFor(Header);
    EXPECT_NE(L, nullptr);
    EXPECT_FALSE(L->isLoopSimplifyForm());
    // No loop guard because loop in not in simplify form.
    EXPECT_EQ(L->getLoopGuardBranch(), nullptr);
    EXPECT_FALSE(L->isGuarded());
  });
}

TEST(LoopInfoTest, LoopLatchNotExiting) {
  const char *ModuleStr =
      "define void @foo(i32* %A, i32 %ub) {\n"
      "entry:\n"
      "  %guardcmp = icmp slt i32 0, %ub\n"
      "  br i1 %guardcmp, label %for.preheader, label %for.end\n"
      "for.preheader:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]\n"
      "  %idxprom = sext i32 %i to i64\n"
      "  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom\n"
      "  store i32 %i, i32* %arrayidx, align 4\n"
      "  %inc = add nsw i32 %i, 1\n"
      "  %cmp = icmp slt i32 %inc, %ub\n"
      "  br i1 %cmp, label %for.latch, label %for.exit\n"
      "for.latch:\n"
      "  br label %for.body\n"
      "for.exit:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfoPlus(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        // First two basic block are entry and for.preheader - skip them.
        ++FI;
        BasicBlock *Header = &*(++FI);
        BasicBlock *Latch = &*(++FI);
        assert(Header && "No header");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);
        EXPECT_TRUE(L->isLoopSimplifyForm());
        EXPECT_EQ(L->getLoopLatch(), Latch);
        EXPECT_FALSE(L->isLoopExiting(Latch));
        // No loop guard becuase loop is not exiting on latch.
        EXPECT_EQ(L->getLoopGuardBranch(), nullptr);
        EXPECT_FALSE(L->isGuarded());
      });
}

// Examine getUniqueExitBlocks/getUniqueNonLatchExitBlocks functions.
TEST(LoopInfoTest, LoopUniqueExitBlocks) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo(i32 %n, i1 %cond) {\n"
      "entry:\n"
      "  br label %for.cond\n"
      "for.cond:\n"
      "  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]\n"
      "  %cmp = icmp slt i32 %i.0, %n\n"
      "  br i1 %cond, label %for.inc, label %for.end1\n"
      "for.inc:\n"
      "  %inc = add nsw i32 %i.0, 1\n"
      "  br i1 %cmp, label %for.cond, label %for.end2, !llvm.loop !0\n"
      "for.end1:\n"
      "  br label %for.end\n"
      "for.end2:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n"
      "!0 = distinct !{!0, !1}\n"
      "!1 = !{!\"llvm.loop.distribute.enable\", i1 true}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfo(*M, "foo", [&](Function &F, LoopInfo &LI) {
    Function::iterator FI = F.begin();
    // First basic block is entry - skip it.
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "for.cond");
    Loop *L = LI.getLoopFor(Header);

    SmallVector<BasicBlock *, 2> Exits;
    // This loop has 2 unique exits.
    L->getUniqueExitBlocks(Exits);
    EXPECT_TRUE(Exits.size() == 2);
    // And one unique non latch exit.
    Exits.clear();
    L->getUniqueNonLatchExitBlocks(Exits);
    EXPECT_TRUE(Exits.size() == 1);
  });
}

// Regression test for  getUniqueNonLatchExitBlocks functions.
// It should detect the exit if it comes from both latch and non-latch blocks.
TEST(LoopInfoTest, LoopNonLatchUniqueExitBlocks) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo(i32 %n, i1 %cond) {\n"
      "entry:\n"
      "  br label %for.cond\n"
      "for.cond:\n"
      "  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]\n"
      "  %cmp = icmp slt i32 %i.0, %n\n"
      "  br i1 %cond, label %for.inc, label %for.end\n"
      "for.inc:\n"
      "  %inc = add nsw i32 %i.0, 1\n"
      "  br i1 %cmp, label %for.cond, label %for.end, !llvm.loop !0\n"
      "for.end:\n"
      "  ret void\n"
      "}\n"
      "!0 = distinct !{!0, !1}\n"
      "!1 = !{!\"llvm.loop.distribute.enable\", i1 true}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfo(*M, "foo", [&](Function &F, LoopInfo &LI) {
    Function::iterator FI = F.begin();
    // First basic block is entry - skip it.
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "for.cond");
    Loop *L = LI.getLoopFor(Header);

    SmallVector<BasicBlock *, 2> Exits;
    // This loop has 1 unique exit.
    L->getUniqueExitBlocks(Exits);
    EXPECT_TRUE(Exits.size() == 1);
    // And one unique non latch exit.
    Exits.clear();
    L->getUniqueNonLatchExitBlocks(Exits);
    EXPECT_TRUE(Exits.size() == 1);
  });
}

// Test that a pointer-chasing loop is not rotated.
TEST(LoopInfoTest, LoopNotRotated) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo(i32* %elem) {\n"
      "entry:\n"
      "  br label %while.cond\n"
      "while.cond:\n"
      "  %elem.addr.0 = phi i32* [ %elem, %entry ], [ %incdec.ptr, %while.body "
      "]\n"
      "  %tobool = icmp eq i32* %elem.addr.0, null\n"
      "  br i1 %tobool, label %while.end, label %while.body\n"
      "while.body:\n"
      "  %incdec.ptr = getelementptr inbounds i32, i32* %elem.addr.0, i64 1\n"
      "  br label %while.cond\n"
      "while.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfo(*M, "foo", [&](Function &F, LoopInfo &LI) {
    Function::iterator FI = F.begin();
    // First basic block is entry - skip it.
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "while.cond");
    Loop *L = LI.getLoopFor(Header);
    EXPECT_NE(L, nullptr);

    // This loop is in simplified form.
    EXPECT_TRUE(L->isLoopSimplifyForm());

    // This loop is not rotated.
    EXPECT_FALSE(L->isRotatedForm());
  });
}

TEST(LoopInfoTest, LoopUserBranch) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo(i32* %B, i64 signext %nx, i1 %cond) {\n"
      "entry:\n"
      "  br i1 %cond, label %bb, label %guard\n"
      "guard:\n"
      "  %cmp.guard = icmp slt i64 0, %nx\n"
      "  br i1 %cmp.guard, label %for.i.preheader, label %for.end\n"
      "for.i.preheader:\n"
      "  br label %for.i\n"
      "for.i:\n"
      "  %i = phi i64 [ 0, %for.i.preheader ], [ %inc13, %for.i ]\n"
      "  %Bi = getelementptr inbounds i32, i32* %B, i64 %i\n"
      "  store i32 0, i32* %Bi, align 4\n"
      "  %inc13 = add nsw i64 %i, 1\n"
      "  %cmp = icmp slt i64 %inc13, %nx\n"
      "  br i1 %cmp, label %for.i, label %for.i.exit\n"
      "for.i.exit:\n"
      "  br label %bb\n"
      "bb:\n"
      "  br label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runWithLoopInfo(*M, "foo", [&](Function &F, LoopInfo &LI) {
    Function::iterator FI = F.begin();
    FI = ++FI;
    assert(FI->getName() == "guard");

    FI = ++FI;
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "for.i");

    Loop *L = LI.getLoopFor(Header);
    EXPECT_NE(L, nullptr);

    // L should not have a guard branch
    EXPECT_EQ(L->getLoopGuardBranch(), nullptr);
  });
}
