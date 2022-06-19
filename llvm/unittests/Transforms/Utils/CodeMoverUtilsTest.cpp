//===- CodeMoverUtils.cpp - Unit tests for CodeMoverUtils ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeMoverUtils.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
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

static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

static Instruction *getInstructionByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (I.getName() == Name)
        return &I;
  llvm_unreachable("Expected to find instruction!");
}

TEST(CodeMoverUtils, IsControlFlowEquivalentSimpleTest) {
  LLVMContext C;

  // void foo(int &i, bool cond1, bool cond2) {
  //   if (cond1)
  //     i = 1;
  //   if (cond1)
  //     i = 2;
  //   if (cond2)
  //     i = 3;
  // }
  std::unique_ptr<Module> M =
      parseIR(C, R"(define void @foo(i32* %i, i1 %cond1, i1 %cond2) {
                 entry:
                   br i1 %cond1, label %if.first, label %if.first.end
                 if.first:
                   store i32 1, i32* %i, align 4
                   br label %if.first.end
                 if.first.end:
                   br i1 %cond1, label %if.second, label %if.second.end
                 if.second:
                   store i32 2, i32* %i, align 4
                   br label %if.second.end
                 if.second.end:
                   br i1 %cond2, label %if.third, label %if.third.end
                 if.third:
                   store i32 3, i32* %i, align 4
                   br label %if.third.end
                 if.third.end:
                   ret void
                 })");
  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        BasicBlock *FirstIfBody = getBasicBlockByName(F, "if.first");
        EXPECT_TRUE(
            isControlFlowEquivalent(*FirstIfBody, *FirstIfBody, DT, PDT));
        BasicBlock *SecondIfBody = getBasicBlockByName(F, "if.second");
        EXPECT_TRUE(
            isControlFlowEquivalent(*FirstIfBody, *SecondIfBody, DT, PDT));

        BasicBlock *ThirdIfBody = getBasicBlockByName(F, "if.third");
        EXPECT_FALSE(
            isControlFlowEquivalent(*FirstIfBody, *ThirdIfBody, DT, PDT));
        EXPECT_FALSE(
            isControlFlowEquivalent(*SecondIfBody, *ThirdIfBody, DT, PDT));
      });
}

TEST(CodeMoverUtils, IsControlFlowEquivalentOppositeCondTest) {
  LLVMContext C;

  // void foo(int &i, unsigned X, unsigned Y) {
  //   if (X < Y)
  //     i = 1;
  //   if (Y > X)
  //     i = 2;
  //   if (X >= Y)
  //     i = 3;
  //   else
  //     i = 4;
  //   if (X == Y)
  //     i = 5;
  //   if (Y == X)
  //     i = 6;
  //   else
  //     i = 7;
  //   if (X != Y)
  //     i = 8;
  //   else
  //     i = 9;
  // }
  std::unique_ptr<Module> M =
      parseIR(C, R"(define void @foo(i32* %i, i32 %X, i32 %Y) {
                 entry:
                   %cmp1 = icmp ult i32 %X, %Y
                   br i1 %cmp1, label %if.first, label %if.first.end
                 if.first:
                   store i32 1, i32* %i, align 4
                   br label %if.first.end
                 if.first.end:
                   %cmp2 = icmp ugt i32 %Y, %X
                   br i1 %cmp2, label %if.second, label %if.second.end
                 if.second:
                   store i32 2, i32* %i, align 4
                   br label %if.second.end
                 if.second.end:
                   %cmp3 = icmp uge i32 %X, %Y
                   br i1 %cmp3, label %if.third, label %if.third.else
                 if.third:
                   store i32 3, i32* %i, align 4
                   br label %if.third.end
                 if.third.else:
                   store i32 4, i32* %i, align 4
                   br label %if.third.end
                 if.third.end:
                   %cmp4 = icmp eq i32 %X, %Y
                   br i1 %cmp4, label %if.fourth, label %if.fourth.end
                 if.fourth:
                   store i32 5, i32* %i, align 4
                   br label %if.fourth.end
                 if.fourth.end:
                   %cmp5 = icmp eq i32 %Y, %X
                   br i1 %cmp5, label %if.fifth, label %if.fifth.else
                 if.fifth:
                   store i32 6, i32* %i, align 4
                   br label %if.fifth.end
                 if.fifth.else:
                   store i32 7, i32* %i, align 4
                   br label %if.fifth.end
                 if.fifth.end:
                   %cmp6 = icmp ne i32 %X, %Y
                   br i1 %cmp6, label %if.sixth, label %if.sixth.else
                 if.sixth:
                   store i32 8, i32* %i, align 4
                   br label %if.sixth.end
                 if.sixth.else:
                   store i32 9, i32* %i, align 4
                   br label %if.sixth.end
                 if.sixth.end:
                   ret void
                 })");
  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        BasicBlock *FirstIfBody = getBasicBlockByName(F, "if.first");
        BasicBlock *SecondIfBody = getBasicBlockByName(F, "if.second");
        BasicBlock *ThirdIfBody = getBasicBlockByName(F, "if.third");
        BasicBlock *ThirdElseBody = getBasicBlockByName(F, "if.third.else");
        EXPECT_TRUE(
            isControlFlowEquivalent(*FirstIfBody, *ThirdElseBody, DT, PDT));
        EXPECT_TRUE(
            isControlFlowEquivalent(*SecondIfBody, *ThirdElseBody, DT, PDT));
        EXPECT_FALSE(
            isControlFlowEquivalent(*ThirdIfBody, *ThirdElseBody, DT, PDT));

        BasicBlock *FourthIfBody = getBasicBlockByName(F, "if.fourth");
        BasicBlock *FifthIfBody = getBasicBlockByName(F, "if.fifth");
        BasicBlock *FifthElseBody = getBasicBlockByName(F, "if.fifth.else");
        EXPECT_FALSE(
            isControlFlowEquivalent(*FifthIfBody, *FifthElseBody, DT, PDT));
        BasicBlock *SixthIfBody = getBasicBlockByName(F, "if.sixth");
        EXPECT_TRUE(
            isControlFlowEquivalent(*FifthElseBody, *SixthIfBody, DT, PDT));
        BasicBlock *SixthElseBody = getBasicBlockByName(F, "if.sixth.else");
        EXPECT_TRUE(
            isControlFlowEquivalent(*FourthIfBody, *SixthElseBody, DT, PDT));
        EXPECT_TRUE(
            isControlFlowEquivalent(*FifthIfBody, *SixthElseBody, DT, PDT));
      });
}

TEST(CodeMoverUtils, IsControlFlowEquivalentCondNestTest) {
  LLVMContext C;

  // void foo(int &i, bool cond1, bool cond2) {
  //   if (cond1)
  //     if (cond2)
  //       i = 1;
  //   if (cond2)
  //     if (cond1)
  //       i = 2;
  // }
  std::unique_ptr<Module> M =
      parseIR(C, R"(define void @foo(i32* %i, i1 %cond1, i1 %cond2) { 
         entry:
           br i1 %cond1, label %if.outer.first, label %if.first.end
         if.outer.first:
           br i1 %cond2, label %if.inner.first, label %if.first.end
         if.inner.first:
           store i32 1, i32* %i, align 4
           br label %if.first.end
         if.first.end:
           br i1 %cond2, label %if.outer.second, label %if.second.end
         if.outer.second:
           br i1 %cond1, label %if.inner.second, label %if.second.end
         if.inner.second:
           store i32 2, i32* %i, align 4
           br label %if.second.end
         if.second.end:
           ret void
         })");
  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        BasicBlock *FirstOuterIfBody = getBasicBlockByName(F, "if.outer.first");
        BasicBlock *FirstInnerIfBody = getBasicBlockByName(F, "if.inner.first");
        BasicBlock *SecondOuterIfBody =
            getBasicBlockByName(F, "if.outer.second");
        BasicBlock *SecondInnerIfBody =
            getBasicBlockByName(F, "if.inner.second");
        EXPECT_TRUE(isControlFlowEquivalent(*FirstInnerIfBody,
                                            *SecondInnerIfBody, DT, PDT));
        EXPECT_FALSE(isControlFlowEquivalent(*FirstOuterIfBody,
                                             *SecondOuterIfBody, DT, PDT));
        EXPECT_FALSE(isControlFlowEquivalent(*FirstOuterIfBody,
                                             *SecondInnerIfBody, DT, PDT));
        EXPECT_FALSE(isControlFlowEquivalent(*FirstInnerIfBody,
                                             *SecondOuterIfBody, DT, PDT));
      });
}

TEST(CodeMoverUtils, IsControlFlowEquivalentImbalanceTest) {
  LLVMContext C;

  // void foo(int &i, bool cond1, bool cond2) {
  //   if (cond1)
  //     if (cond2)
  //       if (cond3)
  //         i = 1;
  //   if (cond2)
  //     if (cond3)
  //       i = 2;
  //   if (cond1)
  //     if (cond1)
  //       i = 3;
  //   if (cond1)
  //     i = 4;
  // }
  std::unique_ptr<Module> M = parseIR(
      C, R"(define void @foo(i32* %i, i1 %cond1, i1 %cond2, i1 %cond3) { 
         entry:
           br i1 %cond1, label %if.outer.first, label %if.first.end
         if.outer.first:
           br i1 %cond2, label %if.middle.first, label %if.first.end
         if.middle.first:
           br i1 %cond3, label %if.inner.first, label %if.first.end
         if.inner.first:
           store i32 1, i32* %i, align 4
           br label %if.first.end
         if.first.end:
           br i1 %cond2, label %if.outer.second, label %if.second.end
         if.outer.second:
           br i1 %cond3, label %if.inner.second, label %if.second.end
         if.inner.second:
           store i32 2, i32* %i, align 4
           br label %if.second.end
         if.second.end:
           br i1 %cond1, label %if.outer.third, label %if.third.end
         if.outer.third:
           br i1 %cond1, label %if.inner.third, label %if.third.end
         if.inner.third:
           store i32 3, i32* %i, align 4
           br label %if.third.end
         if.third.end:
           br i1 %cond1, label %if.fourth, label %if.fourth.end
         if.fourth:
           store i32 4, i32* %i, align 4
           br label %if.fourth.end
         if.fourth.end:
           ret void
         })");
  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        BasicBlock *FirstIfBody = getBasicBlockByName(F, "if.inner.first");
        BasicBlock *SecondIfBody = getBasicBlockByName(F, "if.inner.second");
        EXPECT_FALSE(
            isControlFlowEquivalent(*FirstIfBody, *SecondIfBody, DT, PDT));

        BasicBlock *ThirdIfBody = getBasicBlockByName(F, "if.inner.third");
        BasicBlock *FourthIfBody = getBasicBlockByName(F, "if.fourth");
        EXPECT_TRUE(
            isControlFlowEquivalent(*ThirdIfBody, *FourthIfBody, DT, PDT));
      });
}

TEST(CodeMoverUtils, IsControlFlowEquivalentPointerTest) {
  LLVMContext C;

  // void foo(int &i, int *cond) {
  //   if (*cond)
  //     i = 1;
  //   if (*cond)
  //     i = 2;
  //   *cond = 1;
  //   if (*cond)
  //     i = 3;
  // }
  std::unique_ptr<Module> M =
      parseIR(C, R"(define void @foo(i32* %i, i32* %cond) {
                 entry:
                   %0 = load i32, i32* %cond, align 4
                   %tobool1 = icmp ne i32 %0, 0
                   br i1 %tobool1, label %if.first, label %if.first.end
                 if.first:
                   store i32 1, i32* %i, align 4
                   br label %if.first.end
                 if.first.end:
                   %1 = load i32, i32* %cond, align 4
                   %tobool2 = icmp ne i32 %1, 0
                   br i1 %tobool2, label %if.second, label %if.second.end
                 if.second:
                   store i32 2, i32* %i, align 4
                   br label %if.second.end
                 if.second.end:
                   store i32 1, i32* %cond, align 4
                   %2 = load i32, i32* %cond, align 4
                   %tobool3 = icmp ne i32 %2, 0
                   br i1 %tobool3, label %if.third, label %if.third.end
                 if.third:
                   store i32 3, i32* %i, align 4
                   br label %if.third.end
                 if.third.end:
                   ret void
                 })");
  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        BasicBlock *FirstIfBody = getBasicBlockByName(F, "if.first");
        BasicBlock *SecondIfBody = getBasicBlockByName(F, "if.second");
        // Limitation: if we can prove cond haven't been modify between %0 and
        // %1, then we can prove FirstIfBody and SecondIfBody are control flow
        // equivalent.
        EXPECT_FALSE(
            isControlFlowEquivalent(*FirstIfBody, *SecondIfBody, DT, PDT));

        BasicBlock *ThirdIfBody = getBasicBlockByName(F, "if.third");
        EXPECT_FALSE(
            isControlFlowEquivalent(*FirstIfBody, *ThirdIfBody, DT, PDT));
        EXPECT_FALSE(
            isControlFlowEquivalent(*SecondIfBody, *ThirdIfBody, DT, PDT));
      });
}

TEST(CodeMoverUtils, IsControlFlowEquivalentNotPostdomTest) {
  LLVMContext C;

  // void foo(bool cond1, bool cond2) {
  //   if (cond1) {
  //     if (cond2)
  //       return;
  //   } else
  //     if (cond2)
  //       return;
  //   return;
  // }
  std::unique_ptr<Module> M =
      parseIR(C, R"(define void @foo(i1 %cond1, i1 %cond2) {
                 idom:
                   br i1 %cond1, label %succ0, label %succ1
                 succ0:
                   br i1 %cond2, label %succ0ret, label %succ0succ1
                 succ0ret:
                   ret void
                 succ0succ1:
                   br label %bb
                 succ1:
                   br i1 %cond2, label %succ1ret, label %succ1succ1
                 succ1ret:
                   ret void
                 succ1succ1:
                   br label %bb
                 bb:
                   ret void
                 })");
  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        BasicBlock &Idom = F.front();
        assert(Idom.getName() == "idom" && "Expecting BasicBlock idom");
        BasicBlock &BB = F.back();
        assert(BB.getName() == "bb" && "Expecting BasicBlock bb");
        EXPECT_FALSE(isControlFlowEquivalent(Idom, BB, DT, PDT));
      });
}

TEST(CodeMoverUtils, IsSafeToMoveTest1) {
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
      C, R"(define void @foo(i32* noalias %A, i32* noalias %B, i32* noalias %C
                           , i64 %N) {
         entry:
           %X = sdiv i64 1, %N
           call void @safecall()
           %cmp1 = icmp slt i64 0, %N
           call void @unsafecall1()
           call void @unsafecall2()
           br i1 %cmp1, label %for.body, label %for.end
         for.body:
           %i = phi i64 [ 0, %entry ], [ %inc, %for.body ]
           %arrayidx_A5 = getelementptr inbounds i32, i32* %A, i64 5
           store i32 5, i32* %arrayidx_A5, align 4
           %arrayidx_A = getelementptr inbounds i32, i32* %A, i64 %i
           store i32 0, i32* %arrayidx_A, align 4
           %load1 = load i32, i32* %arrayidx_A, align 4
           %arrayidx_B = getelementptr inbounds i32, i32* %B, i64 %i
           store i32 %load1, i32* %arrayidx_B, align 4
           %load2 = load i32, i32* %arrayidx_A, align 4
           %arrayidx_C = getelementptr inbounds i32, i32* %C, i64 %i
           store i32 %load2, i32* %arrayidx_C, align 4
           %arrayidx_A6 = getelementptr inbounds i32, i32* %A, i64 6
           store i32 6, i32* %arrayidx_A6, align 4
           %inc = add nsw i64 %i, 1
           %cmp = icmp slt i64 %inc, %N
           br i1 %cmp, label %for.body, label %for.end
         for.end:
           ret void
         }
         declare void @safecall() nounwind nosync willreturn
         declare void @unsafecall1()
         declare void @unsafecall2())");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        BasicBlock *Entry = getBasicBlockByName(F, "entry");
        Instruction *CI_safecall = Entry->front().getNextNode();
        assert(isa<CallInst>(CI_safecall) &&
               "Expecting CI_safecall to be a CallInst");
        Instruction *CI_unsafecall = CI_safecall->getNextNode()->getNextNode();
        assert(isa<CallInst>(CI_unsafecall) &&
               "Expecting CI_unsafecall to be a CallInst");
        BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
        Instruction &PN = ForBody->front();
        assert(isa<PHINode>(PN) && "Expecting PN to be a PHINode");
        Instruction *SI_A5 =
            getInstructionByName(F, "arrayidx_A5")->getNextNode();
        Instruction *SI = getInstructionByName(F, "arrayidx_A")->getNextNode();
        Instruction *LI1 = getInstructionByName(F, "load1");
        Instruction *LI2 = getInstructionByName(F, "load2");
        Instruction *SI_A6 =
            getInstructionByName(F, "arrayidx_A6")->getNextNode();

        // Can move after CI_safecall, as it does not throw, not synchronize, or
        // must return.
        EXPECT_TRUE(isSafeToMoveBefore(*CI_safecall->getPrevNode(),
                                       *CI_safecall->getNextNode(), DT, &PDT,
                                       &DI));

        // Cannot move CI_unsafecall, as it may throw.
        EXPECT_FALSE(isSafeToMoveBefore(*CI_unsafecall->getNextNode(),
                                        *CI_unsafecall, DT, &PDT, &DI));

        // Moving instruction to non control flow equivalent places are not
        // supported.
        EXPECT_FALSE(
            isSafeToMoveBefore(*SI_A5, *Entry->getTerminator(), DT, &PDT, &DI));

        // Moving PHINode is not supported.
        EXPECT_FALSE(isSafeToMoveBefore(PN, *PN.getNextNode()->getNextNode(),
                                        DT, &PDT, &DI));

        // Cannot move non-PHINode before PHINode.
        EXPECT_FALSE(isSafeToMoveBefore(*PN.getNextNode(), PN, DT, &PDT, &DI));

        // Moving Terminator is not supported.
        EXPECT_FALSE(isSafeToMoveBefore(*Entry->getTerminator(),
                                        *PN.getNextNode(), DT, &PDT, &DI));

        // Cannot move %arrayidx_A after SI, as SI is its user.
        EXPECT_FALSE(isSafeToMoveBefore(*SI->getPrevNode(), *SI->getNextNode(),
                                        DT, &PDT, &DI));

        // Cannot move SI before %arrayidx_A, as %arrayidx_A is its operand.
        EXPECT_FALSE(
            isSafeToMoveBefore(*SI, *SI->getPrevNode(), DT, &PDT, &DI));

        // Cannot move LI2 after SI_A6, as there is a flow dependence.
        EXPECT_FALSE(
            isSafeToMoveBefore(*LI2, *SI_A6->getNextNode(), DT, &PDT, &DI));

        // Cannot move SI after LI1, as there is a anti dependence.
        EXPECT_FALSE(
            isSafeToMoveBefore(*SI, *LI1->getNextNode(), DT, &PDT, &DI));

        // Cannot move SI_A5 after SI, as there is a output dependence.
        EXPECT_FALSE(isSafeToMoveBefore(*SI_A5, *LI1, DT, &PDT, &DI));

        // Can move LI2 before LI1, as there is only an input dependence.
        EXPECT_TRUE(isSafeToMoveBefore(*LI2, *LI1, DT, &PDT, &DI));
      });
}

TEST(CodeMoverUtils, IsSafeToMoveTest2) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C, R"(define void @foo(i1 %cond, i32 %op0, i32 %op1) {
                 entry:
                   br i1 %cond, label %if.then.first, label %if.end.first
                 if.then.first:
                   %add = add i32 %op0, %op1
                   %user = add i32 %add, 1
                   br label %if.end.first
                 if.end.first:
                   br i1 %cond, label %if.then.second, label %if.end.second
                 if.then.second:
                   %sub_op0 = add i32 %op0, 1
                   %sub = sub i32 %sub_op0, %op1
                   br label %if.end.second
                 if.end.second:
                   ret void
                 })");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        Instruction *AddInst = getInstructionByName(F, "add");
        Instruction *SubInst = getInstructionByName(F, "sub");

        // Cannot move as %user uses %add and %sub doesn't dominates %user.
        EXPECT_FALSE(isSafeToMoveBefore(*AddInst, *SubInst, DT, &PDT, &DI));

        // Cannot move as %sub_op0 is an operand of %sub and %add doesn't
        // dominates %sub_op0.
        EXPECT_FALSE(isSafeToMoveBefore(*SubInst, *AddInst, DT, &PDT, &DI));
      });
}

TEST(CodeMoverUtils, IsSafeToMoveTest3) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(C, R"(define void @foo(i64 %N) {
                 entry:
                   br label %for.body
                 for.body:
                   %i = phi i64 [ 0, %entry ], [ %inc, %for.latch ]
                   %inc = add nsw i64 %i, 1
                   br label %for.latch
                 for.latch:
                   %cmp = icmp slt i64 %inc, %N
                   %add = add i64 100, %N
                   %add2 = add i64 %add, %N
                   br i1 %cmp, label %for.body, label %for.end
                 for.end:
                   ret void
                 })");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        Instruction *IncInst = getInstructionByName(F, "inc");
        Instruction *CmpInst = getInstructionByName(F, "cmp");
        BasicBlock *BB0 = getBasicBlockByName(F, "for.body");
        BasicBlock *BB1 = getBasicBlockByName(F, "for.latch");

        // Can move as the incoming block of %inc for %i (%for.latch) dominated
        // by %cmp.
        EXPECT_TRUE(isSafeToMoveBefore(*IncInst, *CmpInst, DT, &PDT, &DI));

        // Can move as the operands of instructions in BB1 either dominate
        // InsertPoint or appear before that instruction, e.g., %add appears
        // before %add2 although %add does not dominate InsertPoint.
        EXPECT_TRUE(
            isSafeToMoveBefore(*BB1, *BB0->getTerminator(), DT, &PDT, &DI));
      });
}

TEST(CodeMoverUtils, IsSafeToMoveTest4) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C, R"(define void @foo(i1 %cond, i32 %op0, i32 %op1) {
                 entry:
                   br i1 %cond, label %if.end.first, label %if.then.first
                 if.then.first:
                   %add = add i32 %op0, %op1
                   %user = add i32 %add, 1
                   %add2 = add i32 %op0, 1
                   br label %if.end.first
                 if.end.first:
                   br i1 %cond, label %if.end.second, label %if.then.second
                 if.then.second:
                   %sub_op0 = add i32 %op0, 1
                   %sub = sub i32 %sub_op0, %op1
                   %sub2 = sub i32 %op0, 1
                   br label %if.end.second
                 if.end.second:
                   ret void
                 })");

  run(*M, "foo",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        Instruction *AddInst = getInstructionByName(F, "add");
        Instruction *AddInst2 = getInstructionByName(F, "add2");
        Instruction *SubInst = getInstructionByName(F, "sub");
        Instruction *SubInst2 = getInstructionByName(F, "sub2");

        // Cannot move as %user uses %add and %sub doesn't dominates %user.
        EXPECT_FALSE(isSafeToMoveBefore(*AddInst, *SubInst, DT, &PDT, &DI));

        // Cannot move as %sub_op0 is an operand of %sub and %add doesn't
        // dominates %sub_op0.
        EXPECT_FALSE(isSafeToMoveBefore(*SubInst, *AddInst, DT, &PDT, &DI));

        // Can move as %add2 and %sub2 are control flow equivalent,
        // although %add2 does not strictly dominate %sub2.
        EXPECT_TRUE(isSafeToMoveBefore(*AddInst2, *SubInst2, DT, &PDT, &DI));

        // Can move as %add2 and %sub2 are control flow equivalent,
        // although %add2 does not strictly dominate %sub2.
        EXPECT_TRUE(isSafeToMoveBefore(*SubInst2, *AddInst2, DT, &PDT, &DI));
      });
}

TEST(CodeMoverUtils, IsSafeToMoveTest5) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C, R"(define void @dependence(i32* noalias %A, i32* noalias %B){
entry:
    store i32 0, i32* %A, align 4 ; storeA0
    store i32 2, i32* %A, align 4 ; storeA1
    %tmp0 = load i32, i32* %A, align 4 ; loadA0
    store i32 1, i32* %B, align 4 ; storeB0
    %tmp1 = load i32, i32* %A, align 4 ; loadA1
    store i32 2, i32* %A, align 4 ; storeA2 
    store i32 4, i32* %B, align 4 ; StoreB1 
    %tmp2 = load i32, i32* %A, align 4 ; loadA2 
    %tmp3 = load i32, i32* %A, align 4 ; loadA3 
    %tmp4 = load i32, i32* %B, align 4 ; loadB2
    %tmp5 = load i32, i32* %B, align 4 ; loadB3
    ret void
})");

  run(*M, "dependence",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        Instruction *LoadA0 = getInstructionByName(F, "tmp0");
        Instruction *LoadA1 = getInstructionByName(F, "tmp1");
        Instruction *LoadA2 = getInstructionByName(F, "tmp2");
        Instruction *LoadA3 = getInstructionByName(F, "tmp3");
        Instruction *LoadB2 = getInstructionByName(F, "tmp4");
        Instruction *LoadB3 = getInstructionByName(F, "tmp5");
        Instruction *StoreA1 = LoadA0->getPrevNode();
        Instruction *StoreA0 = StoreA1->getPrevNode();
        Instruction *StoreB0 = LoadA0->getNextNode();
        Instruction *StoreB1 = LoadA2->getPrevNode();
        Instruction *StoreA2 = StoreB1->getPrevNode();

        // Input forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA2, *LoadB2, DT, &PDT, &DI));
        // Input backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA3, *LoadA2, DT, &PDT, &DI));

        // Output forward dependency
        EXPECT_FALSE(isSafeToMoveBefore(*StoreA0, *LoadA0, DT, &PDT, &DI));
        // Output backward dependency
        EXPECT_FALSE(isSafeToMoveBefore(*StoreA1, *StoreA0, DT, &PDT, &DI));

        // Flow forward dependency
        EXPECT_FALSE(isSafeToMoveBefore(*StoreA1, *StoreB0, DT, &PDT, &DI));
        // Flow backward dependency
        EXPECT_FALSE(isSafeToMoveBefore(*LoadA0, *StoreA1, DT, &PDT, &DI));

        // Anti forward dependency
        EXPECT_FALSE(isSafeToMoveBefore(*LoadA1, *StoreB1, DT, &PDT, &DI));
        // Anti backward dependency
        EXPECT_FALSE(isSafeToMoveBefore(*StoreA2, *LoadA1, DT, &PDT, &DI));

        // No input backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadB2, *LoadA3, DT, &PDT, &DI));
        // No input forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA3, *LoadB3, DT, &PDT, &DI));

        // No Output forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*StoreA2, *LoadA2, DT, &PDT, &DI));
        // No Output backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*StoreB1, *StoreA2, DT, &PDT, &DI));

        // No flow forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*StoreB0, *StoreA2, DT, &PDT, &DI));
        // No flow backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA1, *StoreB0, DT, &PDT, &DI));

        // No anti backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*StoreB0, *LoadA0, DT, &PDT, &DI));
        // No anti forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA0, *LoadA1, DT, &PDT, &DI));
      });
}

TEST(CodeMoverUtils, IsSafeToMoveTest6) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
      C, R"(define void @dependence(i1 %cond, i32* noalias %A, i32* noalias %B){
   entry:
        br i1 %cond, label %bb0, label %bb1
   bb0:
        br label %bb1
    bb1:
        store i32 0, i32* %A, align 4 ; storeA0
        br i1 %cond, label %bb2, label %bb3
    bb2:
        br label %bb3
    bb3:
        store i32 2, i32* %A, align 4 ; storeA1
        br i1 %cond, label %bb4, label %bb5
    bb4:
        br label %bb5
    bb5:
        %tmp0 = load i32, i32* %A, align 4 ; loadA0
        br i1 %cond, label %bb6, label %bb7
    bb6:
        br label %bb7
    bb7:
        store i32 1, i32* %B, align 4 ; storeB0
        br i1 %cond, label %bb8, label %bb9
    bb8:
        br label %bb9
    bb9:
        %tmp1 = load i32, i32* %A, align 4 ; loadA1
        br i1 %cond, label %bb10, label %bb11
    bb10:
        br label %bb11
    bb11:
        store i32 2, i32* %A, align 4 ; storeA2
        br i1 %cond, label %bb12, label %bb13
    bb12:
        br label %bb13
    bb13:
        store i32 4, i32* %B, align 4 ; StoreB1
        br i1 %cond, label %bb14, label %bb15
    bb14:
        br label %bb15
    bb15:
        %tmp2 = load i32, i32* %A, align 4 ; loadA2
        br i1 %cond, label %bb16, label %bb17
    bb16:
        br label %bb17
    bb17:
        %tmp3 = load i32, i32* %A, align 4 ; loadA3
        br i1 %cond, label %bb18, label %bb19
    bb18:
        br label %bb19
    bb19:
        %tmp4 = load i32, i32* %B, align 4 ; loadB2
        br i1 %cond, label %bb20, label %bb21
    bb20:
       br label %bb21
    bb21:
       %tmp5 = load i32, i32* %B, align 4 ; loadB3
       ret void
   })");
  run(*M, "dependence",
      [&](Function &F, DominatorTree &DT, PostDominatorTree &PDT,
          DependenceInfo &DI) {
        BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
        BasicBlock *BB3 = getBasicBlockByName(F, "bb3");
        BasicBlock *BB7 = getBasicBlockByName(F, "bb7");
        BasicBlock *BB11 = getBasicBlockByName(F, "bb11");
        BasicBlock *BB13 = getBasicBlockByName(F, "bb13");
        Instruction *LoadA0 = getInstructionByName(F, "tmp0");
        Instruction *LoadA1 = getInstructionByName(F, "tmp1");
        Instruction *LoadA2 = getInstructionByName(F, "tmp2");
        Instruction *LoadA3 = getInstructionByName(F, "tmp3");
        Instruction *LoadB2 = getInstructionByName(F, "tmp4");
        Instruction *LoadB3 = getInstructionByName(F, "tmp5");
        Instruction &StoreA1 = BB3->front();
        Instruction &StoreA0 = BB1->front();
        Instruction &StoreB0 = BB7->front();
        Instruction &StoreB1 = BB13->front();
        Instruction &StoreA2 = BB11->front();

        // Input forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA2, *LoadB2, DT, &PDT, &DI));
        // Input backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA3, *LoadA2, DT, &PDT, &DI));

        // Output forward dependency
        EXPECT_FALSE(isSafeToMoveBefore(StoreA0, *LoadA0, DT, &PDT, &DI));
        // Output backward dependency
        EXPECT_FALSE(isSafeToMoveBefore(StoreA1, StoreA0, DT, &PDT, &DI));

        // Flow forward dependency
        EXPECT_FALSE(isSafeToMoveBefore(StoreA1, StoreB0, DT, &PDT, &DI));
        // Flow backward dependency
        EXPECT_FALSE(isSafeToMoveBefore(*LoadA0, StoreA1, DT, &PDT, &DI));

        // Anti forward dependency
        EXPECT_FALSE(isSafeToMoveBefore(*LoadA1, StoreB1, DT, &PDT, &DI));
        // Anti backward dependency
        EXPECT_FALSE(isSafeToMoveBefore(StoreA2, *LoadA1, DT, &PDT, &DI));

        // No input backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadB2, *LoadA3, DT, &PDT, &DI));
        // No input forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA3, *LoadB3, DT, &PDT, &DI));

        // No Output forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(StoreA2, *LoadA2, DT, &PDT, &DI));
        // No Output backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(StoreB1, StoreA2, DT, &PDT, &DI));

        // No flow forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(StoreB0, StoreA2, DT, &PDT, &DI));
        // No flow backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA1, StoreB0, DT, &PDT, &DI));

        // No anti backward dependency
        EXPECT_TRUE(isSafeToMoveBefore(StoreB0, *LoadA0, DT, &PDT, &DI));
        // No anti forward dependency
        EXPECT_TRUE(isSafeToMoveBefore(*LoadA0, *LoadA1, DT, &PDT, &DI));
      });
}
