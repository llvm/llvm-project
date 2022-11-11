//===----- ProfDataUtils.cpp - Unit tests for ProfDataUtils ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/BreakCriticalEdges.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("ProfDataUtilsTests", errs());
  return Mod;
}

TEST(ProfDataUtils, extractWeights) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i1 %cond0) {
entry:
  br i1 %cond0, label %bb0, label %bb1, !prof !1
bb0:
 %0 = mul i32 1, 2
 br label %bb1
bb1:
  ret void
}

!1 = !{!"branch_weights", i32 1, i32 100000}
)IR");
  Function *F = M->getFunction("foo");
  auto &Entry = F->getEntryBlock();
  auto &I = Entry.front();
  auto *Branch = dyn_cast<BranchInst>(&I);
  EXPECT_NE(nullptr, Branch);
  auto *ProfileData = Branch->getMetadata(LLVMContext::MD_prof);
  EXPECT_NE(ProfileData, nullptr);
  EXPECT_TRUE(hasProfMD(I));
  SmallVector<uint32_t> Weights;
  EXPECT_TRUE(extractBranchWeights(ProfileData, Weights));
  EXPECT_EQ(Weights[0], 1U);
  EXPECT_EQ(Weights[1], 100000U);
  EXPECT_EQ(Weights.size(), 2U);
}

TEST(ProfDataUtils, NoWeights) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i1 %cond0) {
entry:
  br i1 %cond0, label %bb0, label %bb1
bb0:
 %0 = mul i32 1, 2
 br label %bb1
bb1:
  ret void
}
)IR");
  Function *F = M->getFunction("foo");
  auto &Entry = F->getEntryBlock();
  auto &I = Entry.front();
  auto *Branch = dyn_cast<BranchInst>(&I);
  EXPECT_NE(nullptr, Branch);
  auto *ProfileData = Branch->getMetadata(LLVMContext::MD_prof);
  EXPECT_EQ(ProfileData, nullptr);
  EXPECT_FALSE(hasProfMD(I));
  SmallVector<uint32_t> Weights;
  EXPECT_FALSE(extractBranchWeights(ProfileData, Weights));
  EXPECT_EQ(Weights.size(), 0U);
}
