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
