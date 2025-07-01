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

// TODO: This performs part of VPlanTransforms::handleUncountableEarlyExits,
//       perhaps we could share the code...
static void combineExitConditions(VPlan &Plan) {
  struct EarlyExitInfo {
    VPBasicBlock *EarlyExitingVPBB;
    VPIRBasicBlock *EarlyExitVPBB;
    VPValue *CondToExit;
  };

  auto *MiddleVPBB = cast<VPBasicBlock>(
      Plan.getScalarHeader()->getSinglePredecessor()->getPredecessors()[0]);
  auto *LatchVPBB = cast<VPBasicBlock>(MiddleVPBB->getSinglePredecessor());

  VPDominatorTree VPDT(Plan);
  VPBuilder Builder(LatchVPBB->getTerminator());
  SmallVector<EarlyExitInfo> Exits;
  for (VPIRBasicBlock *ExitBlock : Plan.getExitBlocks()) {
    for (VPBlockBase *Pred : to_vector(ExitBlock->getPredecessors())) {
      if (Pred == MiddleVPBB)
        continue;
      // Collect condition for this early exit.
      auto *EarlyExitingVPBB = cast<VPBasicBlock>(Pred);
      VPBlockBase *TrueSucc = EarlyExitingVPBB->getSuccessors()[0];
      VPValue *CondOfEarlyExitingVPBB;
      [[maybe_unused]] bool Matched =
          match(EarlyExitingVPBB->getTerminator(),
                m_BranchOnCond(m_VPValue(CondOfEarlyExitingVPBB)));
      assert(Matched && "Terminator must be BranchOnCond");
      VPBuilder EarlyExitingBuilder(EarlyExitingVPBB->getTerminator());
      auto *CondToEarlyExit = EarlyExitingBuilder.createNaryOp(
          VPInstruction::MaskedCond,
          TrueSucc == ExitBlock
              ? CondOfEarlyExitingVPBB
              : EarlyExitingBuilder.createNot(CondOfEarlyExitingVPBB));
      assert((isa<VPIRValue>(CondOfEarlyExitingVPBB) ||
              !VPDT.properlyDominates(EarlyExitingVPBB, LatchVPBB) ||
              VPDT.properlyDominates(
                  CondOfEarlyExitingVPBB->getDefiningRecipe()->getParent(),
                  LatchVPBB)) &&
             "exit condition must dominate the latch");
      Exits.push_back({
          EarlyExitingVPBB,
          ExitBlock,
          CondToEarlyExit,
      });
    }
  }

  // For the negative test case.
  if (Exits.empty())
    return;

  // Build the AnyOf condition for the latch terminator using logical OR
  // to avoid poison propagation from later exit conditions when an earlier
  // exit is taken.
  VPValue *Combined = Exits[0].CondToExit;
  for (const EarlyExitInfo &Info : drop_begin(Exits))
    Combined = Builder.createLogicalOr(Combined, Info.CondToExit);

  VPValue *IsAnyExitTaken =
      Builder.createNaryOp(VPInstruction::AnyOf, {Combined});

  auto *LatchExitingBranch = cast<VPInstruction>(LatchVPBB->getTerminator());
  assert(LatchExitingBranch->getOpcode() == VPInstruction::BranchOnCount &&
         "Unexpected terminator");
  auto *IsLatchExitTaken =
      Builder.createICmp(CmpInst::ICMP_EQ, LatchExitingBranch->getOperand(0),
                         LatchExitingBranch->getOperand(1));
  LatchExitingBranch->eraseFromParent();
  Builder.setInsertPoint(LatchVPBB);
  VPValue *CombineAllExits = Builder.createOr(IsAnyExitTaken, IsLatchExitTaken);
  Builder.createNaryOp(VPInstruction::BranchOnCond, {CombineAllExits});

  // Disconnect early exiting blocks from successors, remove branches. We
  // currently don't support multiple uses for recipes involved in creating
  // the uncountable exit condition.
  for (auto &Exit : Exits) {
    if (Exit.EarlyExitingVPBB == LatchVPBB)
      continue;
    Exit.EarlyExitingVPBB->getTerminator()->eraseFromParent();
    VPBlockUtils::disconnectBlocks(Exit.EarlyExitingVPBB, Exit.EarlyExitVPBB);
  }
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
      // TODO: Uncomment store once more support is added for uncountable exits
      //       in loops with stores.
      // "  store i16 %inc, ptr %st.addr, align 2\n"
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
      vputils::getRecipesForUncountableExit(*Plan, Recipes, GEPs, LatchVPBB);
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
  combineExitConditions(*Plan);

  SmallVector<VPInstruction *> Recipes;
  SmallVector<VPInstruction *> GEPs;

  auto *MiddleVPBB = cast<VPBasicBlock>(
      Plan->getScalarHeader()->getSinglePredecessor()->getPredecessors()[0]);
  auto *LatchVPBB = cast<VPBasicBlock>(MiddleVPBB->getSinglePredecessor());

  std::optional<VPValue *> UncountableCondition =
      vputils::getRecipesForUncountableExit(*Plan, Recipes, GEPs, LatchVPBB);
  ASSERT_FALSE(UncountableCondition.has_value());
  ASSERT_EQ(GEPs.size(), 0ull);
  ASSERT_EQ(Recipes.size(), 0ull);
}

} // namespace
} // namespace llvm
