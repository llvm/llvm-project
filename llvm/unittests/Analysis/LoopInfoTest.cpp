//===- LoopInfoTest.cpp - LoopInfo unit tests -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
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
