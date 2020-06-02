//===- InlineFeaturesAnalysisTest.cpp - inline features unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ML/InlineFeaturesAnalysis.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("MLAnalysisTests", errs());
  return Mod;
}

TEST(InlineFeaturesTest, BasicTest) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare i32 @f1(i32)
declare i32 @f2(i32)

define i32 @branches(i32) {
  %cond = icmp slt i32 %0, 3
  br i1 %cond, label %then, label %else

then:
  %ret.1 = call i32 @f1(i32 %0)
  br label %last.block

else:
  %ret.2 = call i32 @f2(i32 %0)
  br label %last.block

last.block:
  %ret = phi i32 [%ret.1, %then], [%ret.2, %else]
  ret i32 %ret
}

define internal i32 @top() {
  %1 = call i32 @branches(i32 2)
  %2 = call i32 @f1(i32 %1)
  ret i32 %2
}
)IR");

  FunctionAnalysisManager FAM;
  InlineFeaturesAnalysis FA;

  auto BranchesFeatures = FA.run(*M->getFunction("branches"), FAM);
  EXPECT_EQ(BranchesFeatures.BasicBlockCount, 4);
  EXPECT_EQ(BranchesFeatures.BlocksReachedFromConditionalInstruction, 2);
  EXPECT_EQ(BranchesFeatures.DirectCallsToDefinedFunctions, 0);
  // 2 Users: top is one. The other is added because @branches is not internal,
  // so it may have external callers.
  EXPECT_EQ(BranchesFeatures.Uses, 2);

  auto TopFeatures = FA.run(*M->getFunction("top"), FAM);
  EXPECT_EQ(TopFeatures.BasicBlockCount, 1);
  EXPECT_EQ(TopFeatures.BlocksReachedFromConditionalInstruction, 0);
  EXPECT_EQ(TopFeatures.DirectCallsToDefinedFunctions, 1);
  EXPECT_EQ(TopFeatures.Uses, 0);
}
