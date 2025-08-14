//===- llvm/unittests/Transforms/Vectorize/VPlanVerifierTest.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanVerifier.h"
#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanCFG.h"
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
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  EXPECT_STREQ("Use before def!\n"
               "  EMIT vp<%1> = sub vp<%2>\n"
               "  before\n"
               "  EMIT vp<%2> = add ir<0>\n",
               ::testing::internal::GetCapturedStderr().c_str());
#else
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
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
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  EXPECT_STREQ("Use before def!\n"
               "  EMIT vp<%1> = sub vp<%3>\n"
               "  before\n"
               "  EMIT vp<%3> = add ir<0>\n",
               ::testing::internal::GetCapturedStderr().c_str());
#else
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
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
  auto *Blend = new VPBlendRecipe(Phi, {DefI}, {});

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
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  EXPECT_STREQ("Use before def!\n"
               "  BLEND ir<<badref>> = vp<%2>\n"
               "  before\n"
               "  EMIT vp<%2> = add ir<0>\n",
               ::testing::internal::GetCapturedStderr().c_str());
#else
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
#endif

  delete Phi;
}

TEST_F(VPVerifierTest, VPPhiIncomingValueDoesntDominateIncomingBlock) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 0));

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
  VPBasicBlock *VPBB3 = Plan.createVPBasicBlock("");
  VPBasicBlock *VPBB4 = Plan.createVPBasicBlock("");

  VPInstruction *DefI = new VPInstruction(Instruction::Add, {Zero});
  VPPhi *Phi = new VPPhi({DefI}, {});
  VPBB2->appendRecipe(Phi);
  VPBB2->appendRecipe(DefI);
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPBB3->appendRecipe(CanIV);

  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB3, VPBB3, "R1");
  VPBlockUtils::connectBlocks(VPBB1, VPBB2);
  VPBlockUtils::connectBlocks(VPBB2, R1);
  VPBlockUtils::connectBlocks(VPBB4, Plan.getScalarHeader());
#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  EXPECT_STREQ("Incoming def does not dominate incoming block!\n"
               "  EMIT vp<%2> = add ir<0>\n"
               "  does not dominate preheader for\n"
               "  EMIT-SCALAR vp<%1> = phi [ vp<%2>, preheader ]",
               ::testing::internal::GetCapturedStderr().c_str());
#else
  EXPECT_STREQ("Incoming def does not dominate incoming block!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
#endif
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

TEST_F(VPVerifierTest, NonHeaderPHIInHeader) {
  VPlan &Plan = getPlan();
  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(Type::getInt32Ty(C), 0));
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  auto *BranchOnCond = new VPInstruction(VPInstruction::BranchOnCond, {CanIV});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("header");

  VPBB2->appendRecipe(CanIV);

  PHINode *PHINode = PHINode::Create(Type::getInt32Ty(C), 2);
  auto *IRPhi = new VPIRPhi(*PHINode);
  VPBB2->appendRecipe(IRPhi);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB2, VPBB2, "R1");
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  EXPECT_STREQ(
      "Found non-header PHI recipe in header VPBB: IR   <badref> = phi i32 \n",
      ::testing::internal::GetCapturedStderr().c_str());
#else
  EXPECT_STREQ("Found non-header PHI recipe in header VPBB",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
#endif

  delete PHINode;
}

class VPIRVerifierTest : public VPlanTestIRBase {};

TEST_F(VPIRVerifierTest, testVerifyIRPhi) {
  const char *ModuleString =
      "define void @f(ptr %A, i64 %N) {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]\n"
      "  %arr.idx = getelementptr inbounds i32, ptr %A, i64 %iv\n"
      "  %l1 = load i32, ptr %arr.idx, align 4\n"
      "  %res = add i32 %l1, 10\n"
      "  store i32 %res, ptr %arr.idx, align 4\n"
      "  %iv.next = add i64 %iv, 1\n"
      "  %exitcond = icmp ne i64 %iv.next, %N\n"
      "  br i1 %exitcond, label %loop, label %for.end\n"
      "for.end:\n"
      "  %p = phi i32 [ %l1, %loop ]\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildVPlan(LoopHeader);

  Plan->getExitBlocks()[0]->front().addOperand(
      Plan->getOrAddLiveIn(ConstantInt::get(Type::getInt32Ty(*Ctx), 0)));

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(*Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ(
      "Phi-like recipe with different number of operands and predecessors.\n",
      ::testing::internal::GetCapturedStderr().c_str());
#endif
}
} // namespace
