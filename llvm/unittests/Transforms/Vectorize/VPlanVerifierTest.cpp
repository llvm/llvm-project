//===- llvm/unittests/Transforms/Vectorize/VPlanVerifierTest.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanVerifier.h"
#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanTransforms.h"
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
  VPBasicBlock *VPBB2 = new VPBasicBlock();
  VPRegionBlock *R1 = new VPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);

  VPBasicBlock *VPMIDDLE = new VPBasicBlock("middle.block");
  VPBlockUtils::connectBlocks(R1, VPMIDDLE);

  VPlan Plan(VPPH, &*TC, VPBB1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
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

  VPBasicBlock *VPMIDDLE = new VPBasicBlock("middle.block");
  VPBlockUtils::connectBlocks(R1, VPMIDDLE);

  VPBlockUtils::connectBlocks(VPBB1, R1);

  auto TC = std::make_unique<VPValue>();
  VPlan Plan(VPPH, &*TC, VPBB1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
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
  VPBB3->setParent(R1);

  VPBasicBlock *VPMIDDLE = new VPBasicBlock("middle.block");
  VPBlockUtils::connectBlocks(R1, VPMIDDLE);

  auto TC = std::make_unique<VPValue>();
  VPlan Plan(VPPH, &*TC, VPBB1);

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

TEST(VPVerifierTest, DuplicateSuccessorsOutsideRegion) {
  VPInstruction *I1 = new VPInstruction(Instruction::Add, {});
  auto *CanIV = new VPCanonicalIVPHIRecipe(I1, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});
  VPInstruction *BranchOnCond2 =
      new VPInstruction(VPInstruction::BranchOnCond, {I1});

  VPBasicBlock *VPPH = new VPBasicBlock("ph");
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBasicBlock *VPBB2 = new VPBasicBlock();

  VPBB1->appendRecipe(I1);
  VPBB1->appendRecipe(BranchOnCond2);
  VPBB2->appendRecipe(CanIV);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = new VPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBlockUtils::connectBlocks(VPBB1, R1);

  VPBasicBlock *VPMIDDLE = new VPBasicBlock("middle.block");
  VPBlockUtils::connectBlocks(R1, VPMIDDLE);

  auto TC = std::make_unique<VPValue>();
  VPlan Plan(VPPH, &*TC, VPBB1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Multiple instances of the same successor.\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST(VPVerifierTest, DuplicateSuccessorsInsideRegion) {
  VPInstruction *I1 = new VPInstruction(Instruction::Add, {});
  auto *CanIV = new VPCanonicalIVPHIRecipe(I1, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});
  VPInstruction *BranchOnCond2 =
      new VPInstruction(VPInstruction::BranchOnCond, {I1});

  VPBasicBlock *VPPH = new VPBasicBlock("ph");
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBasicBlock *VPBB2 = new VPBasicBlock();
  VPBasicBlock *VPBB3 = new VPBasicBlock();

  VPBB1->appendRecipe(I1);
  VPBB2->appendRecipe(CanIV);
  VPBB2->appendRecipe(BranchOnCond2);
  VPBB3->appendRecipe(BranchOnCond);

  VPBlockUtils::connectBlocks(VPBB2, VPBB3);
  VPBlockUtils::connectBlocks(VPBB2, VPBB3);
  VPRegionBlock *R1 = new VPRegionBlock(VPBB2, VPBB3, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBB3->setParent(R1);

  VPBasicBlock *VPMIDDLE = new VPBasicBlock("middle.block");
  VPBlockUtils::connectBlocks(R1, VPMIDDLE);

  auto TC = std::make_unique<VPValue>();
  VPlan Plan(VPPH, &*TC, VPBB1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Multiple instances of the same successor.\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST(VPVerifierTest, BlockOutsideRegionWithParent) {
  VPBasicBlock *VPPH = new VPBasicBlock("ph");
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBasicBlock *VPBB2 = new VPBasicBlock();

  VPInstruction *DefI = new VPInstruction(Instruction::Add, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {DefI});

  VPBB1->appendRecipe(DefI);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = new VPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBB1->setParent(R1);

  VPBasicBlock *VPMIDDLE = new VPBasicBlock("middle.block");
  VPBlockUtils::connectBlocks(R1, VPMIDDLE);

  auto TC = std::make_unique<VPValue>();
  VPlan Plan(VPPH, &*TC, VPBB1);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Predecessor is not in the same region.\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

TEST(VPVerifierTest, LoopRegionMultipleSuccessors1) {
  VPInstruction *TC = new VPInstruction(Instruction::Add, {});
  VPBasicBlock *VPBBPH = new VPBasicBlock("preheader");
  VPBBPH->appendRecipe(TC);

  VPInstruction *TC2 = new VPInstruction(Instruction::Add, {});
  VPBasicBlock *VPBBENTRY = new VPBasicBlock("entry");
  VPBBENTRY->appendRecipe(TC2);

  // Add a VPCanonicalIVPHIRecipe starting at 0 to the header.
  auto *CanonicalIVPHI = new VPCanonicalIVPHIRecipe(TC2, {});
  VPInstruction *I1 = new VPInstruction(Instruction::Add, {});
  VPInstruction *I2 = new VPInstruction(Instruction::Sub, {I1});
  VPInstruction *I3 = new VPInstruction(VPInstruction::BranchOnCond, {I1});

  VPBasicBlock *RBB1 = new VPBasicBlock();
  RBB1->appendRecipe(CanonicalIVPHI);
  RBB1->appendRecipe(I1);
  RBB1->appendRecipe(I2);
  RBB1->appendRecipe(I3);
  RBB1->setName("bb1");

  VPInstruction *I4 = new VPInstruction(Instruction::Mul, {I2, I1});
  VPInstruction *I5 = new VPInstruction(VPInstruction::BranchOnCond, {I4});
  VPBasicBlock *RBB2 = new VPBasicBlock();
  RBB2->appendRecipe(I4);
  RBB2->appendRecipe(I5);
  RBB2->setName("bb2");

  VPRegionBlock *R1 = new VPRegionBlock(RBB1, RBB2, "R1");
  VPBlockUtils::connectBlocks(RBB1, RBB2);
  VPBlockUtils::connectBlocks(VPBBENTRY, R1);

  VPBasicBlock *VPMIDDLE = new VPBasicBlock("middle.block");
  VPBasicBlock *VPEARLYEXIT = new VPBasicBlock("early.exit");
  VPBlockUtils::connectBlocks(R1, VPMIDDLE);
  VPBlockUtils::connectBlocks(R1, VPEARLYEXIT);

  VPlan Plan(VPBBPH, TC, VPBBENTRY);
  Plan.setName("TestPlan");
  Plan.addVF(ElementCount::getFixed(4));
  Plan.getVectorLoopRegion()->setExiting(RBB2);
  Plan.getVectorLoopRegion()->setEarlyExiting(RBB1);
  Plan.getVectorLoopRegion()->setEarlyExit(VPEARLYEXIT);

  EXPECT_TRUE(verifyVPlanIsValid(Plan));
}

TEST(VPVerifierTest, LoopRegionMultipleSuccessors2) {
  VPInstruction *TC = new VPInstruction(Instruction::Add, {});
  VPBasicBlock *VPBBPH = new VPBasicBlock("preheader");
  VPBBPH->appendRecipe(TC);

  VPInstruction *TC2 = new VPInstruction(Instruction::Add, {});
  VPBasicBlock *VPBBENTRY = new VPBasicBlock("entry");
  VPBBENTRY->appendRecipe(TC2);

  // We can't create a live-in without a VPlan, but we can't create
  // a VPlan without the blocks. So we initialize this to a silly
  // value here, then fix it up later.
  auto *CanonicalIVPHI = new VPCanonicalIVPHIRecipe(TC2, {});
  VPInstruction *I1 = new VPInstruction(Instruction::Add, {});
  VPInstruction *I2 = new VPInstruction(Instruction::Sub, {I1});
  VPInstruction *I3 = new VPInstruction(VPInstruction::BranchOnCond, {I1});

  VPBasicBlock *RBB1 = new VPBasicBlock();
  RBB1->appendRecipe(CanonicalIVPHI);
  RBB1->appendRecipe(I1);
  RBB1->appendRecipe(I2);
  RBB1->appendRecipe(I3);
  RBB1->setName("vector.body");

  // This really is what the vplan cfg looks like before optimising!
  VPBasicBlock *RBB2 = new VPBasicBlock();
  RBB2->setName("loop.inc");
  // A block that inherits the latch name from the original scalar loop.

  VPBasicBlock *RBB3 = new VPBasicBlock();
  // No name

  VPInstruction *I4 = new VPInstruction(Instruction::Mul, {I2, I1});
  VPInstruction *I5 = new VPInstruction(VPInstruction::BranchOnCond, {I4});
  VPBasicBlock *RBB4 = new VPBasicBlock();
  RBB4->appendRecipe(I4);
  RBB4->appendRecipe(I5);
  RBB4->setName("vector.latch");

  VPRegionBlock *R1 = new VPRegionBlock(RBB1, RBB4, "R1");
  VPBlockUtils::insertBlockAfter(RBB2, RBB1);
  VPBlockUtils::insertBlockAfter(RBB3, RBB2);
  VPBlockUtils::insertBlockAfter(RBB4, RBB3);
  VPBlockUtils::connectBlocks(VPBBENTRY, R1);

  VPBasicBlock *VPMIDDLE = new VPBasicBlock("middle.block");
  VPBasicBlock *VPEARLYEXIT = new VPBasicBlock("early.exit");
  VPBlockUtils::connectBlocks(R1, VPMIDDLE);
  VPBlockUtils::connectBlocks(R1, VPEARLYEXIT);

  VPlan Plan(VPBBPH, TC, VPBBENTRY);
  Plan.setName("TestPlan");
  Plan.addVF(ElementCount::getFixed(4));
  Plan.getVectorLoopRegion()->setExiting(RBB4);
  Plan.getVectorLoopRegion()->setEarlyExiting(RBB1);
  Plan.getVectorLoopRegion()->setEarlyExit(VPEARLYEXIT);

  // Update the VPCanonicalIVPHIRecipe to have a live-in IR value.
  LLVMContext C;
  IntegerType *Int32 = IntegerType::get(C, 32);
  Value *StartIV = PoisonValue::get(Int32);
  CanonicalIVPHI->setStartValue(Plan.getOrAddLiveIn(StartIV));

  EXPECT_TRUE(verifyVPlanIsValid(Plan));

  VPlanTransforms::optimize(Plan);
}

} // namespace
