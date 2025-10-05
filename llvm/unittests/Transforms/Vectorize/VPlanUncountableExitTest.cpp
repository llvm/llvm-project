//===- llvm/unittests/Transforms/Vectorize/VPlanUncountableExitTest.cpp ---===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanUtils.h"
#include "VPlanTestBase.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

namespace llvm {

namespace {
class VPUncountableExitTest : public VPlanTestIRBase {};

TEST_F(VPUncountableExitTest, FindUncountableExitRecipes) {
  const char *ModuleString =
      "define void @f(ptr %array, ptr %pred) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]\n"
      "  %st.addr = getelementptr inbounds i16, ptr %array, i64 %iv\n"
      "  %data = load i16, ptr %st.addr, align 2\n"
      "  %inc = add nsw i16 %data, 1\n"
      "  store i16 %inc, ptr %st.addr, align 2\n"
      "  %uncountable.addr = getelementptr inbounds nuw i16, ptr %pred, i64 "
      "%iv\n"
      "  %uncountable.val = load i16, ptr %uncountable.addr, align 2\n"
      "  %uncountable.cond = icmp sgt i16 %uncountable.val, 500\n"
      "  br i1 %uncountable.cond, label %exit, label %for.inc\n"
      "for.inc:\n"
      "  %iv.next = add nuw nsw i64 %iv, 1\n"
      "  %countable.cond = icmp eq i64 %iv.next, 20\n"
      " br i1 %countable.cond, label %exit, label %for.body\n"
      "exit:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildVPlan(LoopHeader, /*HasUncountableExit=*/true);
  VPlanTransforms::tryToConvertVPInstructionsToVPRecipes(
      *Plan, [](PHINode *P) { return nullptr; }, *TLI);
  VPlanTransforms::runPass(VPlanTransforms::optimize, *Plan);

  SmallVector<VPRecipeBase *> Recipes;
  SmallVector<VPRecipeBase *> GEPs;

  std::optional<VPValue *> UncountableCondition =
      vputils::getRecipesForUncountableExit(*Plan, Recipes, GEPs);
  ASSERT_TRUE(UncountableCondition.has_value());
  ASSERT_EQ(GEPs.size(), 1ull);
  ASSERT_EQ(Recipes.size(), 3ull);
}

TEST_F(VPUncountableExitTest, NoUncountableExit) {
  const char *ModuleString =
      "define void @f(ptr %array, ptr %pred) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]\n"
      "  %st.addr = getelementptr inbounds i16, ptr %array, i64 %iv\n"
      "  %data = load i16, ptr %st.addr, align 2\n"
      "  %inc = add nsw i16 %data, 1\n"
      "  store i16 %inc, ptr %st.addr, align 2\n"
      "  %iv.next = add nuw nsw i64 %iv, 1\n"
      "  %countable.cond = icmp eq i64 %iv.next, 20\n"
      " br i1 %countable.cond, label %exit, label %for.body\n"
      "exit:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildVPlan(LoopHeader);
  VPlanTransforms::tryToConvertVPInstructionsToVPRecipes(
      *Plan, [](PHINode *P) { return nullptr; }, *TLI);
  VPlanTransforms::runPass(VPlanTransforms::optimize, *Plan);

  SmallVector<VPRecipeBase *> Recipes;
  SmallVector<VPRecipeBase *> GEPs;

  std::optional<VPValue *> UncountableCondition =
      vputils::getRecipesForUncountableExit(*Plan, Recipes, GEPs);
  ASSERT_FALSE(UncountableCondition.has_value());
  ASSERT_EQ(GEPs.size(), 0ull);
  ASSERT_EQ(Recipes.size(), 0ull);
}

} // namespace
} // namespace llvm
