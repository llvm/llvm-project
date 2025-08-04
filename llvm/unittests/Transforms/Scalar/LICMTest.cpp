//===- LICMTest.cpp - LICM unit tests -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "gtest/gtest.h"

namespace llvm {

TEST(LICMTest, TestSCEVInvalidationOnHoisting) {
  LLVMContext Ctx;
  ModulePassManager MPM;
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  StringRef PipelineStr = "require<opt-remark-emit>,loop-mssa(licm)";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(MPM, PipelineStr), Succeeded());

  SMDiagnostic Error;
  StringRef Text = R"(
    define void @foo(i64* %ptr) {
    entry:
      br label %loop

    loop:
      %iv = phi i64 [ 0, %entry ], [ %iv.inc, %loop ]
      %n = load i64, i64* %ptr, !invariant.load !0
      %iv.inc = add i64 %iv, 1
      %cmp = icmp ult i64 %iv.inc, %n
      br i1 %cmp, label %loop, label %exit

    exit:
      ret void
    }

    !0 = !{}
  )";

  std::unique_ptr<Module> M = parseAssemblyString(Text, Error, Ctx);
  ASSERT_TRUE(M);
  Function *F = M->getFunction("foo");
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(*F);
  BasicBlock &EntryBB = F->getEntryBlock();
  BasicBlock *LoopBB = EntryBB.getUniqueSuccessor();

  // Select `load i64, i64* %ptr`.
  Instruction *IBefore = &*LoopBB->getFirstNonPHIIt();
  // Make sure the right instruction was selected.
  ASSERT_TRUE(isa<LoadInst>(IBefore));
  // Upon this query SCEV caches disposition of <load i64, i64* %ptr> SCEV.
  ASSERT_EQ(SE.getBlockDisposition(SE.getSCEV(IBefore), LoopBB),
            ScalarEvolution::BlockDisposition::DominatesBlock);

  MPM.run(*M, MAM);

  // Select `load i64, i64* %ptr` after it was hoisted.
  Instruction *IAfter = &*EntryBB.getFirstNonPHIIt();
  // Make sure the right instruction was selected.
  ASSERT_TRUE(isa<LoadInst>(IAfter));

  ScalarEvolution::BlockDisposition DispositionBeforeInvalidation =
      SE.getBlockDisposition(SE.getSCEV(IAfter), LoopBB);
  SE.forgetValue(IAfter);
  ScalarEvolution::BlockDisposition DispositionAfterInvalidation =
      SE.getBlockDisposition(SE.getSCEV(IAfter), LoopBB);

  // If LICM have properly invalidated SCEV,
  //   1. SCEV of <load i64, i64* %ptr> should properly dominate the "loop" BB,
  //   2. extra invalidation shouldn't change result of the query.
  EXPECT_EQ(DispositionBeforeInvalidation,
            ScalarEvolution::BlockDisposition::ProperlyDominatesBlock);
  EXPECT_EQ(DispositionBeforeInvalidation, DispositionAfterInvalidation);
}
}
