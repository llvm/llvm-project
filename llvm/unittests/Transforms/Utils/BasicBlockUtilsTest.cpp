//===- BasicBlockUtils.cpp - Unit tests for BasicBlockUtils ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("BasicBlockUtilsTests", errs());
  return Mod;
}

TEST(BasicBlockUtils, EliminateUnreachableBlocks) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define i32 @has_unreachable(i1 %cond) {\n"
    "entry:\n"
    "  br i1 %cond, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]"
    "  ret i32 %phi\n"
    "bb2:\n"
    "  ret i32 42\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("has_unreachable");
  DominatorTree DT(*F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

  EXPECT_EQ(F->size(), (size_t)4);
  bool Result = EliminateUnreachableBlocks(*F, &DTU);
  EXPECT_TRUE(Result);
  EXPECT_EQ(F->size(), (size_t)3);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, NoUnreachableBlocksToEliminate) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define i32 @no_unreachable(i1 %cond) {\n"
    "entry:\n"
    "  br i1 %cond, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]"
    "  ret i32 %phi\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("no_unreachable");
  DominatorTree DT(*F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

  EXPECT_EQ(F->size(), (size_t)3);
  bool Result = EliminateUnreachableBlocks(*F, &DTU);
  EXPECT_FALSE(Result);
  EXPECT_EQ(F->size(), (size_t)3);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, SplitBlockPredecessors) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define i32 @basic_func(i1 %cond) {\n"
    "entry:\n"
    "  br i1 %cond, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]"
    "  ret i32 %phi\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("basic_func");
  DominatorTree DT(*F);

  // Make sure the dominator tree is properly updated if calling this on the
  // entry block.
  SplitBlockPredecessors(&F->getEntryBlock(), {}, "split.entry", &DT);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, SplitCriticalEdge) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define void @crit_edge(i1 %cond0, i1 %cond1) {\n"
    "entry:\n"
    "  br i1 %cond0, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  br label %bb2\n"
    "bb2:\n"
    "  ret void\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("crit_edge");
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);

  CriticalEdgeSplittingOptions CESO(&DT, nullptr, nullptr, &PDT);
  EXPECT_EQ(1u, SplitAllCriticalEdges(*F, CESO));
  EXPECT_TRUE(DT.verify());
  EXPECT_TRUE(PDT.verify());
}
