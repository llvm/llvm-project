//===- llvm/unittests/Transforms/Vectorize/VPlanUncountableExitTest.cpp ---===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPRecipeBuilder.h"
#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanPatternMatch.h"
#include "../lib/Transforms/Vectorize/VPlanUtils.h"
#include "VPlanTestBase.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

namespace llvm {

namespace {
class VPUncountableExitTest : public VPlanTestIRBase {};
using namespace VPlanPatternMatch;

static void combineExitConditions(VPlan &Plan) {
  struct EarlyExitInfo {
    VPBasicBlock *EarlyExitingVPBB;
    VPIRBasicBlock *EarlyExitVPBB;
    VPValue *CondToExit;
  };

  auto *MiddleVPBB = cast<VPBasicBlock>(
      Plan.getScalarHeader()->getSinglePredecessor()->getPredecessors()[0]);
  auto *LatchVPBB = cast<VPBasicBlock>(MiddleVPBB->getSinglePredecessor());

  // Find the single early exit: a non-middle predecessor of an exit block.
  VPBasicBlock *EarlyExitingVPBB = nullptr;
  VPIRBasicBlock *EarlyExitVPBB = nullptr;
  for (VPIRBasicBlock *ExitBlock : Plan.getExitBlocks()) {
    for (VPBlockBase *Pred : ExitBlock->getPredecessors()) {
      if (Pred != MiddleVPBB) {
        EarlyExitingVPBB = cast<VPBasicBlock>(Pred);
        EarlyExitVPBB = ExitBlock;
      }
    }
  }
  assert(EarlyExitingVPBB && "must have an early exit");

  // Wrap the early exit condition in a MaskedCond.
  VPValue *Cond;
  [[maybe_unused]] bool Matched =
      match(EarlyExitingVPBB->getTerminator(), m_BranchOnCond(m_VPValue(Cond)));
  assert(Matched && "Terminator must be BranchOnCond");
  VPBuilder EarlyExitBuilder(EarlyExitingVPBB->getTerminator());
  if (EarlyExitingVPBB->getSuccessors()[0] != EarlyExitVPBB)
    Cond = EarlyExitBuilder.createNot(Cond);
  auto *MaskedCond =
      EarlyExitBuilder.createNaryOp(VPInstruction::MaskedCond, {Cond});

  // Combine the early exit with the latch exit on the latch terminator.
  VPBuilder Builder(LatchVPBB->getTerminator());
  auto *IsAnyExitTaken =
      Builder.createNaryOp(VPInstruction::AnyOf, {MaskedCond});
  auto *LatchBranch = cast<VPInstruction>(LatchVPBB->getTerminator());
  auto *IsLatchExitTaken = Builder.createICmp(
      CmpInst::ICMP_EQ, LatchBranch->getOperand(0), LatchBranch->getOperand(1));
  LatchBranch->eraseFromParent();
  Builder.setInsertPoint(LatchVPBB);
  Builder.createNaryOp(VPInstruction::BranchOnCond,
                       {Builder.createOr(IsAnyExitTaken, IsLatchExitTaken)});

  // Disconnect the early exit edge.
  EarlyExitingVPBB->getTerminator()->eraseFromParent();
  VPBlockUtils::disconnectBlocks(EarlyExitingVPBB, EarlyExitVPBB);
}

TEST_F(VPUncountableExitTest, FindUncountableExitRecipes) {
  const char *ModuleString =
      "target datalayout = "
      "\"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
      "f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:"
      "32:64-S128\"\n"
      "define void @f(ptr dereferenceable(40) align 2 %array, "
      "ptr dereferenceable(40) align 2 %pred) {\n"
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
  VPlanPtr Plan = buildVPlan0(LoopHeader);
  combineExitConditions(*Plan);

  SmallVector<VPInstruction *> Recipes;
  SmallVector<VPInstruction *> GEPs;

  auto *MiddleVPBB = cast<VPBasicBlock>(
      Plan->getScalarHeader()->getSinglePredecessor()->getPredecessors()[0]);
  auto *LatchVPBB = cast<VPBasicBlock>(MiddleVPBB->getSinglePredecessor());

  std::optional<VPValue *> UncountableCondition =
      vputils::getRecipesForUncountableExit(Recipes, GEPs, LatchVPBB);
  ASSERT_TRUE(UncountableCondition.has_value());
  ASSERT_EQ(GEPs.size(), 1ull);
  ASSERT_EQ(Recipes.size(), 4ull);
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
  auto Plan = buildVPlan0(LoopHeader);

  SmallVector<VPInstruction *> Recipes;
  SmallVector<VPInstruction *> GEPs;

  auto *MiddleVPBB = cast<VPBasicBlock>(
      Plan->getScalarHeader()->getSinglePredecessor()->getPredecessors()[0]);
  auto *LatchVPBB = cast<VPBasicBlock>(MiddleVPBB->getSinglePredecessor());

  std::optional<VPValue *> UncountableCondition =
      vputils::getRecipesForUncountableExit(Recipes, GEPs, LatchVPBB);
  ASSERT_FALSE(UncountableCondition.has_value());
  ASSERT_EQ(GEPs.size(), 0ull);
  ASSERT_EQ(Recipes.size(), 0ull);
}

} // namespace
} // namespace llvm
