//===- CFGTest.cpp - CFG tests --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// This fixture assists in running the isPotentiallyReachable utility four ways
// and ensuring it produces the correct answer each time.
class IsPotentiallyReachableTest : public testing::Test {
protected:
  void ParseAssembly(const char *Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);

    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    // A failure here means that the test itself is buggy.
    if (!M)
      report_fatal_error(os.str().c_str());

    Function *F = M->getFunction("test");
    if (F == nullptr)
      report_fatal_error("Test must have a function named @test");

    A = B = nullptr;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (I->hasName()) {
        if (I->getName() == "A")
          A = &*I;
        else if (I->getName() == "B")
          B = &*I;
      }
    }
    if (A == nullptr)
      report_fatal_error("@test must have an instruction %A");
    if (B == nullptr)
      report_fatal_error("@test must have an instruction %B");
  }

  void ExpectPath(bool ExpectedResult) {
    static char ID;
    class IsPotentiallyReachableTestPass : public FunctionPass {
     public:
      IsPotentiallyReachableTestPass(bool ExpectedResult,
                                     Instruction *A, Instruction *B)
          : FunctionPass(ID), ExpectedResult(ExpectedResult), A(A), B(B) {}

      static int initialize() {
        PassInfo *PI = new PassInfo("isPotentiallyReachable testing pass",
                                    "", &ID, nullptr, true, true);
        PassRegistry::getPassRegistry()->registerPass(*PI, false);
        initializeLoopInfoWrapperPassPass(*PassRegistry::getPassRegistry());
        initializeDominatorTreeWrapperPassPass(
            *PassRegistry::getPassRegistry());
        return 0;
      }

      void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll();
        AU.addRequired<LoopInfoWrapperPass>();
        AU.addRequired<DominatorTreeWrapperPass>();
      }

      bool runOnFunction(Function &F) override {
        if (!F.hasName() || F.getName() != "test")
          return false;

        LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
        DominatorTree *DT =
            &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
        EXPECT_EQ(isPotentiallyReachable(A, B, nullptr, nullptr),
                  ExpectedResult);
        EXPECT_EQ(isPotentiallyReachable(A, B, DT, nullptr), ExpectedResult);
        EXPECT_EQ(isPotentiallyReachable(A, B, nullptr, LI), ExpectedResult);
        EXPECT_EQ(isPotentiallyReachable(A, B, DT, LI), ExpectedResult);
        return false;
      }
      bool ExpectedResult;
      Instruction *A, *B;
    };

    static int initialize = IsPotentiallyReachableTestPass::initialize();
    (void)initialize;

    IsPotentiallyReachableTestPass *P =
        new IsPotentiallyReachableTestPass(ExpectedResult, A, B);
    legacy::PassManager PM;
    PM.add(P);
    PM.run(*M);
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
  Instruction *A, *B;
};

}

TEST_F(IsPotentiallyReachableTest, SameBlockNoPath) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  bitcast i8 undef to i8\n"
      "  %B = bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  %A = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}\n");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, SameBlockPath) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  %B = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}\n");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, SameBlockNoLoop) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  br label %middle\n"
      "middle:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  bitcast i8 undef to i8\n"
      "  %A = bitcast i8 undef to i8\n"
      "  br label %nextblock\n"
      "nextblock:\n"
      "  ret void\n"
      "}\n");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, StraightNoPath) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  br label %exit\n"
      "exit:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, StraightPath) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  br label %exit\n"
      "exit:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, DestUnreachable) {
  ParseAssembly(
      "define void @test() {\n"
      "entry:\n"
      "  br label %midblock\n"
      "midblock:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  ret void\n"
      "unreachable:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  br label %midblock\n"
      "}");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, BranchToReturn) {
  ParseAssembly(
      "define void @test(i1 %x) {\n"
      "entry:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  br i1 %x, label %block1, label %block2\n"
      "block1:\n"
      "  ret void\n"
      "block2:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, SimpleLoop1) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop, label %exit\n"
      "exit:\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, SimpleLoop2) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  br label %loop\n"
      "loop:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop, label %exit\n"
      "exit:\n"
      "  ret void\n"
      "}");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, SimpleLoop3) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop, label %exit\n"
      "exit:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  ret void\n"
      "}");
  ExpectPath(false);
}


TEST_F(IsPotentiallyReachableTest, OneLoopAfterTheOther1) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop1\n"
      "loop1:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop1, label %loop1exit\n"
      "loop1exit:\n"
      "  br label %loop2\n"
      "loop2:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %y = call i1 @switch()\n"
      "  br i1 %x, label %loop2, label %loop2exit\n"
      "loop2exit:"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, OneLoopAfterTheOther2) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %loop1\n"
      "loop1:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %loop1, label %loop1exit\n"
      "loop1exit:\n"
      "  br label %loop2\n"
      "loop2:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %y = call i1 @switch()\n"
      "  br i1 %x, label %loop2, label %loop2exit\n"
      "loop2exit:"
      "  ret void\n"
      "}");
  ExpectPath(false);
}

TEST_F(IsPotentiallyReachableTest, OneLoopAfterTheOtherInsideAThirdLoop) {
  ParseAssembly(
      "declare i1 @switch()\n"
      "\n"
      "define void @test() {\n"
      "entry:\n"
      "  br label %outerloop3\n"
      "outerloop3:\n"
      "  br label %innerloop1\n"
      "innerloop1:\n"
      "  %B = bitcast i8 undef to i8\n"
      "  %x = call i1 @switch()\n"
      "  br i1 %x, label %innerloop1, label %innerloop1exit\n"
      "innerloop1exit:\n"
      "  br label %innerloop2\n"
      "innerloop2:\n"
      "  %A = bitcast i8 undef to i8\n"
      "  %y = call i1 @switch()\n"
      "  br i1 %x, label %innerloop2, label %innerloop2exit\n"
      "innerloop2exit:"
      "  ;; In outer loop3 now.\n"
      "  %z = call i1 @switch()\n"
      "  br i1 %z, label %outerloop3, label %exit\n"
      "exit:\n"
      "  ret void\n"
      "}");
  ExpectPath(true);
}

static const char *BranchInsideLoopIR =
    "declare i1 @switch()\n"
    "\n"
    "define void @test() {\n"
    "entry:\n"
    "  br label %loop\n"
    "loop:\n"
    "  %x = call i1 @switch()\n"
    "  br i1 %x, label %nextloopblock, label %exit\n"
    "nextloopblock:\n"
    "  %y = call i1 @switch()\n"
    "  br i1 %y, label %left, label %right\n"
    "left:\n"
    "  %A = bitcast i8 undef to i8\n"
    "  br label %loop\n"
    "right:\n"
    "  %B = bitcast i8 undef to i8\n"
    "  br label %loop\n"
    "exit:\n"
    "  ret void\n"
    "}";

TEST_F(IsPotentiallyReachableTest, BranchInsideLoop) {
  ParseAssembly(BranchInsideLoopIR);
  ExpectPath(true);
}

TEST_F(IsPotentiallyReachableTest, ModifyTest) {
  ParseAssembly(BranchInsideLoopIR);

  succ_iterator S = succ_begin(&*++M->getFunction("test")->begin());
  BasicBlock *OldBB = S[0];
  S[0] = S[1];
  ExpectPath(false);
  S[0] = OldBB;
  ExpectPath(true);
}
