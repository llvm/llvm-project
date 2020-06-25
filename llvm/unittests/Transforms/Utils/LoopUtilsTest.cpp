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
  TargetLibraryInfoImpl TLII;
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
