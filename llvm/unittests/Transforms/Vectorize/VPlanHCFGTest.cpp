//===- llvm/unittest/Transforms/Vectorize/VPlanHCFGTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanTransforms.h"
#include "VPlanTestBase.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class VPlanHCFGTest : public VPlanTestBase {};

TEST_F(VPlanHCFGTest, testBuildHCFGInnerLoop) {
  const char *ModuleString =
      "define void @f(i32* %A, i64 %N) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %arr.idx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv\n"
      "  %l1 = load i32, i32* %arr.idx, align 4\n"
      "  %res = add i32 %l1, 10\n"
      "  store i32 %res, i32* %arr.idx, align 4\n"
      "  %indvars.iv.next = add i64 %indvars.iv, 1\n"
      "  %exitcond = icmp ne i64 %indvars.iv.next, %N\n"
      "  br i1 %exitcond, label %for.body, label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPBasicBlock *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  EXPECT_EQ(0u, Entry->getNumPredecessors());
  EXPECT_EQ(1u, Entry->getNumSuccessors());
  EXPECT_EQ(nullptr, Entry->getCondBit());

  VPBasicBlock *VecBB = Entry->getSingleSuccessor()->getEntryBasicBlock();
  EXPECT_EQ(7u, VecBB->size());
  EXPECT_EQ(2u, VecBB->getNumPredecessors());
  EXPECT_EQ(2u, VecBB->getNumSuccessors());
  EXPECT_EQ(&*Plan, VecBB->getPlan());

  auto Iter = VecBB->begin();
  VPInstruction *Phi = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::PHI, Phi->getOpcode());

  VPInstruction *Idx = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::GetElementPtr, Idx->getOpcode());
  EXPECT_EQ(2u, Idx->getNumOperands());
  EXPECT_EQ(Phi, Idx->getOperand(1));

  VPInstruction *Load = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::Load, Load->getOpcode());
  EXPECT_EQ(1u, Load->getNumOperands());
  EXPECT_EQ(Idx, Load->getOperand(0));

  VPInstruction *Add = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::Add, Add->getOpcode());
  EXPECT_EQ(2u, Add->getNumOperands());
  EXPECT_EQ(Load, Add->getOperand(0));

  VPInstruction *Store = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::Store, Store->getOpcode());
  EXPECT_EQ(2u, Store->getNumOperands());
  EXPECT_EQ(Add, Store->getOperand(0));
  EXPECT_EQ(Idx, Store->getOperand(1));

  VPInstruction *IndvarAdd = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::Add, IndvarAdd->getOpcode());
  EXPECT_EQ(2u, IndvarAdd->getNumOperands());
  EXPECT_EQ(Phi, IndvarAdd->getOperand(0));

  VPInstruction *ICmp = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::ICmp, ICmp->getOpcode());
  EXPECT_EQ(2u, ICmp->getNumOperands());
  EXPECT_EQ(IndvarAdd, ICmp->getOperand(0));
  EXPECT_EQ(VecBB->getCondBit(), ICmp);

  LoopVectorizationLegality::InductionList Inductions;
  SmallPtrSet<Instruction *, 1> DeadInstructions;
  VPlanTransforms::VPInstructionsToVPRecipes(LI->getLoopFor(LoopHeader), Plan,
                                             Inductions, DeadInstructions);
}

TEST_F(VPlanHCFGTest, testVPInstructionToVPRecipesInner) {
  const char *ModuleString =
      "define void @f(i32* %A, i64 %N) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %arr.idx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv\n"
      "  %l1 = load i32, i32* %arr.idx, align 4\n"
      "  %res = add i32 %l1, 10\n"
      "  store i32 %res, i32* %arr.idx, align 4\n"
      "  %indvars.iv.next = add i64 %indvars.iv, 1\n"
      "  %exitcond = icmp ne i64 %indvars.iv.next, %N\n"
      "  br i1 %exitcond, label %for.body, label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  LoopVectorizationLegality::InductionList Inductions;
  SmallPtrSet<Instruction *, 1> DeadInstructions;
  VPlanTransforms::VPInstructionsToVPRecipes(LI->getLoopFor(LoopHeader), Plan,
                                             Inductions, DeadInstructions);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  EXPECT_EQ(0u, Entry->getNumPredecessors());
  EXPECT_EQ(1u, Entry->getNumSuccessors());

  VPBasicBlock *VecBB = Entry->getSingleSuccessor()->getEntryBasicBlock();
  EXPECT_EQ(6u, VecBB->size());
  EXPECT_EQ(2u, VecBB->getNumPredecessors());
  EXPECT_EQ(2u, VecBB->getNumSuccessors());

  auto Iter = VecBB->begin();
  auto *Phi = dyn_cast<VPWidenPHIRecipe>(&*Iter++);
  EXPECT_NE(nullptr, Phi);

  auto *Idx = dyn_cast<VPWidenGEPRecipe>(&*Iter++);
  EXPECT_NE(nullptr, Idx);

  auto *Load = dyn_cast<VPWidenMemoryInstructionRecipe>(&*Iter++);
  EXPECT_NE(nullptr, Load);

  auto *Add = dyn_cast<VPWidenRecipe>(&*Iter++);
  EXPECT_NE(nullptr, Add);

  auto *Store = dyn_cast<VPWidenMemoryInstructionRecipe>(&*Iter++);
  EXPECT_NE(nullptr, Store);

  auto *LastWiden = dyn_cast<VPWidenRecipe>(&*Iter++);
  EXPECT_NE(nullptr, LastWiden);
  EXPECT_EQ(VecBB->end(), Iter);
}

} // namespace
} // namespace llvm
