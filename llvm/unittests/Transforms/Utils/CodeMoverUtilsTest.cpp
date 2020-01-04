//===- CodeMoverUtils.cpp - Unit tests for CodeMoverUtils ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeMoverUtils.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("CodeMoverUtilsTests", errs());
  return Mod;
}

static void run(Module &M, StringRef FuncName,
                function_ref<void(Function &F, DominatorTree &DT,
                                  PostDominatorTree &PDT, DependenceInfo &DI)>
                    Test) {
  auto *F = M.getFunction(FuncName);
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  AliasAnalysis AA(TLI);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  DependenceInfo DI(F, &AA, &SE, &LI);
  Test(*F, DT, PDT, DI);
}

TEST(CodeMoverUtils, BasicTest) {
  LLVMContext C;

  // void safecall() noexcept willreturn nosync;
  // void unsafecall();
  // void foo(int * __restrict__ A, int * __restrict__ B, int * __restrict__ C,
  //          long N) {
  //   X = N / 1;
  //   safecall();
  //   unsafecall1();
  //   unsafecall2();
  //   for (long i = 0; i < N; ++i) {
  //     A[5] = 5;
  //     A[i] = 0;
  //     B[i] = A[i];
  //     C[i] = A[i];
  //     A[6] = 6;
  //   }
  // }
  std::unique_ptr<Module> M = parseIR(
      C, "define void @foo(i32* noalias %A, i32* noalias %B, i32* noalias %C\n"
         "                  , i64 %N) {\n"
         "entry:\n"
         "  %X = sdiv i64 1, %N\n"
         "  call void @safecall()\n"
         "  %cmp1 = icmp slt i64 0, %N\n"
         "  call void @unsafecall1()\n"
         "  call void @unsafecall2()\n"
         "  br i1 %cmp1, label %for.body, label %for.end\n"
         "for.body:\n"
         "  %i = phi i64 [ 0, %entry ], [ %inc, %for.body ]\n"
         "  %arrayidx_A5 = getelementptr inbounds i32, i32* %A, i64 5\n"
         "  store i32 5, i32* %arrayidx_A5, align 4\n"
         "  %arrayidx_A = getelementptr inbounds i32, i32* %A, i64 %i\n"
         "  store i32 0, i32* %arrayidx_A, align 4\n"
         "  %load1 = load i32, i32* %arrayidx_A, align 4\n"
         "  %arrayidx_B = getelementptr inbounds i32, i32* %B, i64 %i\n"
         "  store i32 %load1, i32* %arrayidx_B, align 4\n"
         "  %load2 = load i32, i32* %arrayidx_A, align 4\n"
         "  %arrayidx_C = getelementptr inbounds i32, i32* %C, i64 %i\n"
         "  store i32 %load2, i32* %arrayidx_C, align 4\n"
         "  %arrayidx_A6 = getelementptr inbounds i32, i32* %A, i64 6\n"
         "  store i32 6, i32* %arrayidx_A6, align 4\n"
         "  %inc = add nsw i64 %i, 1\n"
         "  %cmp = icmp slt i64 %inc, %N\n"
         "  br i1 %cmp, label %for.body, label %for.end\n"
         "for.end:\n"
         "  ret void\n"
         "}\n"
         "declare void @safecall() nounwind nosync willreturn\n"
         "declare void @unsafecall1()\n"
         "declare void @unsafecall2()\n");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        Function::iterator FI = F.begin();
        BasicBlock *Entry = &*(FI++);
        assert(Entry->getName() == "entry" && "Expecting BasicBlock entry");
        Instruction *CI_safecall = Entry->front().getNextNode();
        assert(isa<CallInst>(CI_safecall) &&
               "Expecting CI_safecall to be a CallInst");
        Instruction *CI_unsafecall = CI_safecall->getNextNode()->getNextNode();
        assert(isa<CallInst>(CI_unsafecall) &&
               "Expecting CI_unsafecall to be a CallInst");
        BasicBlock *ForBody = &*(FI++);
        assert(ForBody->getName() == "for.body" &&
               "Expecting BasicBlock for.body");
        Instruction &PN = ForBody->front();
        assert(isa<PHINode>(PN) && "Expecting PN to be a PHINode");
        Instruction *SI_A5 = PN.getNextNode()->getNextNode();
        assert(isa<StoreInst>(SI_A5) &&
               SI_A5->getOperand(1)->getName() == "arrayidx_A5" &&
               "Expecting store to arrayidx_A5");
        Instruction *SI = SI_A5->getNextNode()->getNextNode();
        assert(isa<StoreInst>(SI) &&
               SI->getOperand(1)->getName() == "arrayidx_A" &&
               "Expecting store to arrayidx_A");
        Instruction *LI1 = SI->getNextNode();
        assert(LI1->getName() == "load1" && "Expecting LI1 to be load1");
        Instruction *LI2 = LI1->getNextNode()->getNextNode()->getNextNode();
        assert(LI2->getName() == "load2" && "Expecting LI2 to be load2");
        Instruction *SI_A6 =
            LI2->getNextNode()->getNextNode()->getNextNode()->getNextNode();
        assert(isa<StoreInst>(SI_A6) &&
               SI_A6->getOperand(1)->getName() == "arrayidx_A6" &&
               "Expecting store to arrayidx_A6");

        // Can move after CI_safecall, as it does not throw, not synchronize, or
        // must return.
        EXPECT_TRUE(isSafeToMoveBefore(*CI_safecall->getPrevNode(),
                                       *CI_safecall->getNextNode(), DT, PDT,
                                       DI));

        // Cannot move CI_unsafecall, as it may throw.
        EXPECT_FALSE(isSafeToMoveBefore(*CI_unsafecall->getNextNode(),
                                        *CI_unsafecall, DT, PDT, DI));

        // Moving instruction to non control flow equivalent places are not
        // supported.
        EXPECT_FALSE(
            isSafeToMoveBefore(*SI_A5, *Entry->getTerminator(), DT, PDT, DI));

        // Moving PHINode is not supported.
        EXPECT_FALSE(isSafeToMoveBefore(PN, *PN.getNextNode()->getNextNode(),
                                        DT, PDT, DI));

        // Cannot move non-PHINode before PHINode.
        EXPECT_FALSE(isSafeToMoveBefore(*PN.getNextNode(), PN, DT, PDT, DI));

        // Moving Terminator is not supported.
        EXPECT_FALSE(isSafeToMoveBefore(*Entry->getTerminator(),
                                        *PN.getNextNode(), DT, PDT, DI));

        // Cannot move %arrayidx_A after SI, as SI is its user.
        EXPECT_FALSE(isSafeToMoveBefore(*SI->getPrevNode(), *SI->getNextNode(),
                                        DT, PDT, DI));

        // Cannot move SI before %arrayidx_A, as %arrayidx_A is its operand.
        EXPECT_FALSE(isSafeToMoveBefore(*SI, *SI->getPrevNode(), DT, PDT, DI));

        // Cannot move LI2 after SI_A6, as there is a flow dependence.
        EXPECT_FALSE(
            isSafeToMoveBefore(*LI2, *SI_A6->getNextNode(), DT, PDT, DI));

        // Cannot move SI after LI1, as there is a anti dependence.
        EXPECT_FALSE(isSafeToMoveBefore(*SI, *LI1->getNextNode(), DT, PDT, DI));

        // Cannot move SI_A5 after SI, as there is a output dependence.
        EXPECT_FALSE(isSafeToMoveBefore(*SI_A5, *LI1, DT, PDT, DI));

        // Can move LI2 before LI1, as there is only an input dependence.
        EXPECT_TRUE(isSafeToMoveBefore(*LI2, *LI1, DT, PDT, DI));
      });
}
