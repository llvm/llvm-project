//===- LoopUtilsTest.cpp - Unit tests for LoopUtils -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("LoopUtilsTests", errs());
  return Mod;
}

static void run(Module &M, StringRef FuncName,
                function_ref<void(Function &F, DominatorTree &DT,
                                  ScalarEvolution &SE, LoopInfo &LI)>
                    Test) {
  Function *F = M.getFunction(FuncName);
  DominatorTree DT(*F);
  TargetLibraryInfoImpl TLII(M.getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  Test(*F, DT, SE, LI);
}

TEST(LoopUtils, DeleteDeadLoopNest) {
  LLVMContext C;
  std::unique_ptr<Module> M =
      parseIR(C, "define void @foo() {\n"
                 "entry:\n"
                 "  br label %for.i\n"
                 "for.i:\n"
                 "  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]\n"
                 "  br label %for.j\n"
                 "for.j:\n"
                 "  %j = phi i64 [ 0, %for.i ], [ %inc.j, %for.j ]\n"
                 "  %inc.j = add nsw i64 %j, 1\n"
                 "  %cmp.j = icmp slt i64 %inc.j, 100\n"
                 "  br i1 %cmp.j, label %for.j, label %for.k.preheader\n"
                 "for.k.preheader:\n"
                 "  br label %for.k\n"
                 "for.k:\n"
                 "  %k = phi i64 [ %inc.k, %for.k ], [ 0, %for.k.preheader ]\n"
                 "  %inc.k = add nsw i64 %k, 1\n"
                 "  %cmp.k = icmp slt i64 %inc.k, 100\n"
                 "  br i1 %cmp.k, label %for.k, label %for.i.latch\n"
                 "for.i.latch:\n"
                 "  %inc.i = add nsw i64 %i, 1\n"
                 "  %cmp.i = icmp slt i64 %inc.i, 100\n"
                 "  br i1 %cmp.i, label %for.i, label %for.end\n"
                 "for.end:\n"
                 "  ret void\n"
                 "}\n");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, ScalarEvolution &SE, LoopInfo &LI) {
        assert(LI.begin() != LI.end() && "Expecting loops in function F");
        Loop *L = *LI.begin();
        assert(L && L->getName() == "for.i" && "Expecting loop for.i");

        deleteDeadLoop(L, &DT, &SE, &LI);

        assert(DT.verify(DominatorTree::VerificationLevel::Fast) &&
               "Expecting valid dominator tree");
        LI.verify(DT);
        assert(LI.begin() == LI.end() &&
               "Expecting no loops left in function F");
        SE.verify();

        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI++);
        assert(Entry->getName() == "entry" && "Expecting BasicBlock entry");
        const BranchInst *BI = dyn_cast<BranchInst>(Entry->getTerminator());
        assert(BI && "Expecting valid branch instruction");
        EXPECT_EQ(BI->getNumSuccessors(), (unsigned)1);
        EXPECT_EQ(BI->getSuccessor(0)->getName(), "for.end");
      });
}

TEST(LoopUtils, IsKnownPositiveInLoopTest) {
  LLVMContext C;
  std::unique_ptr<Module> M =
      parseIR(C, "define void @foo(i32 %n, i1 %c) {\n"
                 "entry:\n"
                 "  %is.positive = icmp sgt i32 %n, 0\n"
                 "  br i1 %is.positive, label %loop, label %exit\n"
                 "loop:\n"
                 "  br i1 %c, label %loop, label %exit\n"
                 "exit:\n"
                 "  ret void\n"
                 "}\n");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, ScalarEvolution &SE, LoopInfo &LI) {
        assert(LI.begin() != LI.end() && "Expecting loops in function F");
        Loop *L = *LI.begin();
        assert(L && L->getName() == "loop" && "Expecting loop 'loop'");
        auto *Arg = F.getArg(0);
        const SCEV *ArgSCEV = SE.getSCEV(Arg);
        EXPECT_EQ(isKnownPositiveInLoop(ArgSCEV, L, SE), true);
      });
}

TEST(LoopUtils, IsKnownNonPositiveInLoopTest) {
  LLVMContext C;
  std::unique_ptr<Module> M =
      parseIR(C, "define void @foo(i32 %n, i1 %c) {\n"
                 "entry:\n"
                 "  %is.non.positive = icmp sle i32 %n, 0\n"
                 "  br i1 %is.non.positive, label %loop, label %exit\n"
                 "loop:\n"
                 "  br i1 %c, label %loop, label %exit\n"
                 "exit:\n"
                 "  ret void\n"
                 "}\n");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, ScalarEvolution &SE, LoopInfo &LI) {
        assert(LI.begin() != LI.end() && "Expecting loops in function F");
        Loop *L = *LI.begin();
        assert(L && L->getName() == "loop" && "Expecting loop 'loop'");
        auto *Arg = F.getArg(0);
        const SCEV *ArgSCEV = SE.getSCEV(Arg);
        EXPECT_EQ(isKnownNonPositiveInLoop(ArgSCEV, L, SE), true);
      });
}

// The inner and outer loop here share a latch.  Because any loop metadata must
// be attached to that latch, loop metadata cannot distinguish between the two
// loops.  Until that problem is solved (by moving loop metadata to loops'
// header blocks instead), getLoopEstimatedTripCount and
// setLoopEstimatedTripCount must refuse to operate on at least one of the two
// loops.  They choose to reject the outer loop here because the latch does not
// exit it.
TEST(LoopUtils, nestedLoopSharedLatchEstimatedTripCount) {
  LLVMContext C;
  std::unique_ptr<Module> M =
      parseIR(C, "declare i1 @f()\n"
                 "declare i1 @g()\n"
                 "define void @foo() {\n"
                 "entry:\n"
                 "  br label %outer\n"
                 "outer:\n"
                 "  %c0 = call i1 @f()"
                 "  br i1 %c0, label %inner, label %exit, !prof !0\n"
                 "inner:\n"
                 "  %c1 = call i1 @g()"
                 "  br i1 %c1, label %inner, label %outer, !prof !1\n"
                 "exit:\n"
                 "  ret void\n"
                 "}\n"
                 "!0 = !{!\"branch_weights\", i32 100, i32 1}\n"
                 "!1 = !{!\"branch_weights\", i32 4, i32 1}\n"
                 "\n");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, ScalarEvolution &SE, LoopInfo &LI) {
        assert(LI.end() - LI.begin() == 1 && "Expected one outer loop");
        Loop *Outer = *LI.begin();
        assert(Outer->end() - Outer->begin() == 1 && "Expected one inner loop");
        Loop *Inner = *Outer->begin();

        // Even before llvm.loop.estimated_trip_count is added to either loop,
        // getLoopEstimatedTripCount rejects the outer loop.
        EXPECT_EQ(getLoopEstimatedTripCount(Inner), 5);
        EXPECT_EQ(getLoopEstimatedTripCount(Outer), std::nullopt);

        // setLoopEstimatedTripCount for the inner loop does not affect
        // getLoopEstimatedTripCount for the outer loop.
        EXPECT_EQ(setLoopEstimatedTripCount(Inner, 100), true);
        EXPECT_EQ(getLoopEstimatedTripCount(Inner), 100);
        EXPECT_EQ(getLoopEstimatedTripCount(Outer), std::nullopt);

        // setLoopEstimatedTripCount rejects the outer loop.
        EXPECT_EQ(setLoopEstimatedTripCount(Outer, 999), false);
        EXPECT_EQ(getLoopEstimatedTripCount(Inner), 100);
        EXPECT_EQ(getLoopEstimatedTripCount(Outer), std::nullopt);
      });
}
