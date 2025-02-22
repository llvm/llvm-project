//===- llvm/unittests/Transforms/Vectorize/VPlanVerifierTest.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanVerifier.h"
#include "../lib/Transforms/Vectorize/VPlan.h"
#include "VPlanTestBase.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"

using namespace llvm;

using VPVerifierTest = VPlanTestBase;

namespace {
TEST_F(VPVerifierTest, VPInstructionUseBeforeDefSameBB) {
  VPlan &Plan = getPlan();
  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(Type::getInt32Ty(C), 0));
  VPInstruction *DefI = new VPInstruction(Instruction::Add, {Zero});
  VPInstruction *UseI = new VPInstruction(Instruction::Sub, {DefI});
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBB1->appendRecipe(UseI);
  VPBB1->appendRecipe(DefI);

  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
  VPBB2->appendRecipe(CanIV);
  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST_F(VPVerifierTest, VPInstructionUseBeforeDefDifferentBB) {
  VPlan &Plan = getPlan();
  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(Type::getInt32Ty(C), 0));
  VPInstruction *DefI = new VPInstruction(Instruction::Add, {Zero});
  VPInstruction *UseI = new VPInstruction(Instruction::Sub, {DefI});
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");

  VPBB1->appendRecipe(UseI);
  VPBB2->appendRecipe(CanIV);
  VPBB2->appendRecipe(DefI);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST_F(VPVerifierTest, VPBlendUseBeforeDefDifferentBB) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *Phi = PHINode::Create(Int32, 1);
  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 0));

  VPInstruction *DefI = new VPInstruction(Instruction::Add, {Zero});
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});
  auto *Blend = new VPBlendRecipe(Phi, {DefI});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
  VPBasicBlock *VPBB3 = Plan.createVPBasicBlock("");
  VPBasicBlock *VPBB4 = Plan.createVPBasicBlock("");

  VPBB2->appendRecipe(CanIV);
  VPBB3->appendRecipe(Blend);
  VPBB4->appendRecipe(DefI);
  VPBB4->appendRecipe(BranchOnCond);

  VPBlockUtils::connectBlocks(VPBB2, VPBB3);
  VPBlockUtils::connectBlocks(VPBB3, VPBB4);
  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB2, VPBB4, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBB3->setParent(R1);

  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif

  delete Phi;
}

TEST_F(VPVerifierTest, DuplicateSuccessorsOutsideRegion) {
  VPlan &Plan = getPlan();
  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(Type::getInt32Ty(C), 0));
  VPInstruction *I1 = new VPInstruction(Instruction::Add, {Zero});
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});
  VPInstruction *BranchOnCond2 =
      new VPInstruction(VPInstruction::BranchOnCond, {I1});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");

  VPBB1->appendRecipe(I1);
  VPBB1->appendRecipe(BranchOnCond2);
  VPBB2->appendRecipe(CanIV);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBlockUtils::connectBlocks(VPBB1, R1);

  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Multiple instances of the same successor.\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST_F(VPVerifierTest, DuplicateSuccessorsInsideRegion) {
  VPlan &Plan = getPlan();
  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(Type::getInt32Ty(C), 0));
  VPInstruction *I1 = new VPInstruction(Instruction::Add, {Zero});
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});
  VPInstruction *BranchOnCond2 =
      new VPInstruction(VPInstruction::BranchOnCond, {I1});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
  VPBasicBlock *VPBB3 = Plan.createVPBasicBlock("");

  VPBB1->appendRecipe(I1);
  VPBB2->appendRecipe(CanIV);
  VPBB2->appendRecipe(BranchOnCond2);
  VPBB3->appendRecipe(BranchOnCond);

  VPBlockUtils::connectBlocks(VPBB2, VPBB3);
  VPBlockUtils::connectBlocks(VPBB2, VPBB3);
  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB2, VPBB3, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBB3->setParent(R1);

  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Multiple instances of the same successor.\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST_F(VPVerifierTest, BlockOutsideRegionWithParent) {
  VPlan &Plan = getPlan();

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");

  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(Type::getInt32Ty(C), 0));
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPBB2->appendRecipe(CanIV);

  VPInstruction *DefI = new VPInstruction(Instruction::Add, {Zero});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {DefI});

  VPBB1->appendRecipe(DefI);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);

  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());
  VPBB1->setParent(R1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Predecessor is not in the same region.\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

} // namespace
