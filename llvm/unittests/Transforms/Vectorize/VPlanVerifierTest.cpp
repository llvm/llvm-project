//===- llvm/unittests/Transforms/Vectorize/VPlanVerifierTest.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanVerifier.h"
#include "../lib/Transforms/Vectorize/VPlan.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
TEST(VPVerifierTest, VPInstructionUseBeforeDefSameBB) {
  VPInstruction *DefI = new VPInstruction(Instruction::Add, {});
  VPInstruction *UseI = new VPInstruction(Instruction::Sub, {DefI});

  VPBasicBlock *VPPH = new VPBasicBlock("ph");
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBB1->appendRecipe(UseI);
  VPBB1->appendRecipe(DefI);

  auto TC = std::make_unique<VPValue>();
  VPlan Plan(VPPH, &*TC, VPBB1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(VPlanVerifier::verifyPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST(VPVerifierTest, VPInstructionUseBeforeDefDifferentBB) {
  VPInstruction *DefI = new VPInstruction(Instruction::Add, {});
  VPInstruction *UseI = new VPInstruction(Instruction::Sub, {DefI});
  auto *CanIV = new VPCanonicalIVPHIRecipe(UseI, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});

  VPBasicBlock *VPPH = new VPBasicBlock("ph");
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBasicBlock *VPBB2 = new VPBasicBlock();

  VPBB1->appendRecipe(UseI);
  VPBB2->appendRecipe(CanIV);
  VPBB2->appendRecipe(DefI);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = new VPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);

  auto TC = std::make_unique<VPValue>();
  VPlan Plan(VPPH, &*TC, VPBB1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(VPlanVerifier::verifyPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST(VPVerifierTest, VPBlendUseBeforeDefDifferentBB) {
  LLVMContext C;
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *Phi = PHINode::Create(Int32, 1);

  VPInstruction *I1 = new VPInstruction(Instruction::Add, {});
  VPInstruction *DefI = new VPInstruction(Instruction::Add, {});
  auto *CanIV = new VPCanonicalIVPHIRecipe(I1, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});
  auto *Blend = new VPBlendRecipe(Phi, {DefI});

  VPBasicBlock *VPPH = new VPBasicBlock("ph");
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBasicBlock *VPBB2 = new VPBasicBlock();
  VPBasicBlock *VPBB3 = new VPBasicBlock();
  VPBasicBlock *VPBB4 = new VPBasicBlock();

  VPBB1->appendRecipe(I1);
  VPBB2->appendRecipe(CanIV);
  VPBB3->appendRecipe(Blend);
  VPBB4->appendRecipe(DefI);
  VPBB4->appendRecipe(BranchOnCond);

  VPBlockUtils::connectBlocks(VPBB2, VPBB3);
  VPBlockUtils::connectBlocks(VPBB3, VPBB4);
  VPRegionBlock *R1 = new VPRegionBlock(VPBB2, VPBB4, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);

  auto TC = std::make_unique<VPValue>();
  VPlan Plan(VPPH, &*TC, VPBB1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(VPlanVerifier::verifyPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif

  delete Phi;
}
} // namespace
