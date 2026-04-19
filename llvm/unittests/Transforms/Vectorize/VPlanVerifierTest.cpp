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

namespace llvm {
LLVM_ABI extern cl::opt<bool> VerifyEachVPlan;
} // namespace llvm

using VPVerifierTest = VPlanTestBase;

namespace {
TEST_F(VPVerifierTest, VPInstructionUseBeforeDefSameBB) {
  VPlan &Plan = getPlan();
  VPIRValue *Zero = Plan.getConstantInt(32, 0);
  VPInstruction *DefI =
      new VPInstruction(Instruction::Add, {Zero, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Add));
  VPInstruction *UseI =
      new VPInstruction(Instruction::Sub, {DefI, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Sub));
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBB1->appendRecipe(UseI);
  VPBB1->appendRecipe(DefI);

  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
  VPBB2->appendRecipe(CanIV);
  VPRegionBlock *R1 = Plan.createLoopRegion("R1", VPBB2, VPBB2);
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  EXPECT_STREQ("Use before def!\n"
               "  EMIT vp<%1> = sub vp<%2>, ir<0>\n"
               "  before\n"
               "  EMIT vp<%2> = add ir<0>, ir<0>\n",
               ::testing::internal::GetCapturedStderr().c_str());
#else
  EXPECT_STREQ("Use before def!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
#endif
}

TEST_F(VPVerifierTest, VPInstructionUseBeforeDefDifferentBB) {
  VPlan &Plan = getPlan();
  VPIRValue *Zero = Plan.getConstantInt(32, 0);
  VPInstruction *DefI =
      new VPInstruction(Instruction::Add, {Zero, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Add));
  VPInstruction *UseI =
      new VPInstruction(Instruction::Sub, {DefI, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Sub));
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");

  VPBB1->appendRecipe(UseI);
  VPBB2->appendRecipe(CanIV);
  VPBB2->appendRecipe(DefI);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = Plan.createLoopRegion("R1", VPBB2, VPBB2);
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(Plan));
#if GTEST_HAS_STREAM_REDIRECTION
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  EXPECT_STREQ("Use before def!\n"
               "  EMIT vp<%1> = sub vp<%3>, ir<0>\n"
               "  before\n"
               "  EMIT vp<%3> = add ir<0>, ir<0>\n",
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
  VPIRValue *Zero = Plan.getConstantInt(Int32, 0);

  VPInstruction *DefI =
      new VPInstruction(Instruction::Add, {Zero, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Add));
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {CanIV});
  auto *Blend = new VPBlendRecipe(Phi, {DefI, Plan.getTrue()}, {}, {});

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
  VPRegionBlock *R1 = Plan.createLoopRegion("R1", VPBB2, VPBB4);
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
               "  EMIT vp<%2> = add ir<0>, ir<0>\n",
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
  VPIRValue *Zero = Plan.getConstantInt(Int32, 0);

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
  VPBasicBlock *VPBB3 = Plan.createVPBasicBlock("");
  VPBasicBlock *VPBB4 = Plan.createVPBasicBlock("");

  VPInstruction *DefI =
      new VPInstruction(Instruction::Add, {Zero, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Add));
  VPPhi *Phi = new VPPhi({DefI}, {}, {});
  VPBB2->appendRecipe(Phi);
  VPBB2->appendRecipe(DefI);
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPBB3->appendRecipe(CanIV);

  VPRegionBlock *R1 = Plan.createLoopRegion("R1", VPBB3, VPBB3);
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
               "  EMIT vp<%2> = add ir<0>, ir<0>\n"
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
  VPIRValue *Zero = Plan.getConstantInt(32, 0);
  VPInstruction *I1 =
      new VPInstruction(Instruction::Add, {Zero, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Add));
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

  VPRegionBlock *R1 = Plan.createLoopRegion("R1", VPBB2, VPBB2);
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBlockUtils::connectBlocks(VPBB1, R1);

  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

  // Duplicate successors are allowed for blocks with conditional terminators.
  EXPECT_TRUE(verifyVPlanIsValid(Plan));
}

TEST_F(VPVerifierTest, DuplicateSuccessorsInsideRegion) {
  VPlan &Plan = getPlan();
  VPIRValue *Zero = Plan.getConstantInt(32, 0);
  VPInstruction *I1 =
      new VPInstruction(Instruction::Add, {Zero, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Add));
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
  VPRegionBlock *R1 = Plan.createLoopRegion("R1", VPBB2, VPBB3);
  VPBlockUtils::connectBlocks(VPBB1, R1);
  VPBB3->setParent(R1);

  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

  // Duplicate successors are allowed for blocks with conditional terminators.
  EXPECT_TRUE(verifyVPlanIsValid(Plan));
}

TEST_F(VPVerifierTest, BlockOutsideRegionWithParent) {
  VPlan &Plan = getPlan();

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");

  VPIRValue *Zero = Plan.getConstantInt(32, 0);
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  VPBB2->appendRecipe(CanIV);

  VPInstruction *DefI =
      new VPInstruction(Instruction::Add, {Zero, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Add));
  VPInstruction *BranchOnCond =
      new VPInstruction(VPInstruction::BranchOnCond, {DefI});

  VPBB1->appendRecipe(DefI);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = Plan.createLoopRegion("R1", VPBB2, VPBB2);
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
  VPIRValue *Zero = Plan.getConstantInt(32, 0);
  auto *CanIV = new VPCanonicalIVPHIRecipe(Zero, {});
  auto *BranchOnCond = new VPInstruction(VPInstruction::BranchOnCond, {CanIV});

  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("header");

  VPBB2->appendRecipe(CanIV);

  PHINode *PHINode = PHINode::Create(Type::getInt32Ty(C), 2);
  auto *IRPhi = new VPIRPhi(*PHINode);
  VPBB2->appendRecipe(IRPhi);
  VPBB2->appendRecipe(BranchOnCond);

  VPRegionBlock *R1 = Plan.createLoopRegion("R1", VPBB2, VPBB2);
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

TEST_F(VPVerifierTest, testRUN_VPLAN_PASS) {
  VPlan &Plan = getPlan();
  VPIRValue *Zero = Plan.getConstantInt(32, 0);
  VPInstruction *DefI =
      new VPInstruction(Instruction::Add, {Zero, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Add));
  VPInstruction *UseI =
      new VPInstruction(Instruction::Sub, {DefI, Zero},
                        VPIRFlags::getDefaultFlags(Instruction::Sub));
  VPBasicBlock *VPBB1 = Plan.getEntry();
  VPBB1->appendRecipe(UseI);
  VPBB1->appendRecipe(DefI);

  bool OrigVerifyEachVPlan = VerifyEachVPlan;
  VerifyEachVPlan = true;
  llvm::scope_exit _([&]() { VerifyEachVPlan = OrigVerifyEachVPlan; });
  auto NopPass = [](VPlan &Plan) {};
  EXPECT_DEATH(
      { VPlanTransforms::runPass("simplifyRecipes", NopPass, Plan); },
      "Broken VPlan found, compilation aborted!");
}

class VPIRVerifierTest : public VPlanTestIRBase {};

TEST_F(VPIRVerifierTest, testVerifyIRPhiInScalarHeaderVPIRBB) {
  const char *ModuleString =
      "define void @f(ptr %A, i64 %N) {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]\n"
      "  %iv.next = add i64 %iv, 1\n"
      "  %exitcond = icmp ne i64 %iv.next, %N\n"
      "  br i1 %exitcond, label %loop, label %for.end\n"
      "for.end:\n"
      "  %p = phi i64 [ %iv, %loop ]\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildVPlan(LoopHeader);
  VPValue *Zero = Plan->getConstantInt(32, 0);
  Plan->getScalarHeader()->front().addOperand(Zero);

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

TEST_F(VPIRVerifierTest, testVerifyIRPhiInExitVPIRBB) {
  const char *ModuleString =
      "define void @f(ptr %A, i64 %N) {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]\n"
      "  %iv.next = add i64 %iv, 1\n"
      "  %exitcond = icmp ne i64 %iv.next, %N\n"
      "  br i1 %exitcond, label %loop, label %for.end\n"
      "for.end:\n"
      "  %p = phi i64 [ %iv, %loop ]\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildVPlan(LoopHeader);

  // Create a definition in the vector loop header that will be used by the phi.
  auto *HeaderBlock =
      cast<VPBasicBlock>(Plan->getVectorLoopRegion()->getEntry());
  VPInstruction *DefI =
      new VPInstruction(VPInstruction::ExtractLastLane,
                        {HeaderBlock->front().getVPSingleValue()});
  DefI->insertBefore(Plan->getMiddleBlock()->getTerminator());
  Plan->getExitBlocks()[0]->front().addOperand(DefI);

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

TEST_F(VPIRVerifierTest, BranchOnTwoCondsLatchHeaderVerification) {
  const char *ModuleString =
      "define void @f(ptr dereferenceable(40) align 2 %pred) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]\n"
      "  %uncountable.addr = getelementptr inbounds nuw i16, ptr %pred, i64 "
      "%iv\n"
      "  %uncountable.val = load i16, ptr %uncountable.addr, align 2\n"
      "  %uncountable.cond = icmp sgt i16 %uncountable.val, 500\n"
      "  br i1 %uncountable.cond, label %exit, label %for.inc\n"
      "for.inc:\n"
      "  %iv.next = add nuw nsw i64 %iv, 1\n"
      "  %countable.cond = icmp eq i64 %iv.next, 20\n"
      "  br i1 %countable.cond, label %exit, label %for.body\n"
      "exit:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  // Build a plain CFG VPlan with BranchOnTwoConds as the latch terminator
  // (3 successors), without wrapping blocks in loop regions.
  auto Plan = buildVPlan(LoopHeader, UncountableExitStyle::ReadOnly,
                         /*CreateLoopRegions=*/false);

  auto *MiddleVPBB =
      cast<VPBasicBlock>(Plan->getScalarPreheader()->getPredecessors()[0]);
  auto *Latch = cast<VPBasicBlock>(MiddleVPBB->getSinglePredecessor());
  auto *Header = Latch->getSuccessors().back();
  ASSERT_EQ(Header->getPredecessors()[1], Latch);
  ASSERT_EQ(Latch->getNumSuccessors(), 3u);
  auto *Term = cast<VPInstruction>(&Latch->back());
  EXPECT_EQ(Term->getOpcode(), VPInstruction::BranchOnTwoConds);

  // Verify the plan is valid; this exercises isLatch with a 3-successor latch.
  EXPECT_TRUE(verifyVPlanIsValid(*Plan));

  // Swap the latch's first and last successors, placing the header at index 0
  // instead of the last position. isLatch checks the last successor, so the
  // latch is no longer recognized, triggering the header predecessor check.
  auto &Succs = Latch->getSuccessors();
  std::swap(Succs[0], Succs[2]);

#if GTEST_HAS_STREAM_REDIRECTION
  ::testing::internal::CaptureStderr();
#endif
  EXPECT_FALSE(verifyVPlanIsValid(*Plan));
#if GTEST_HAS_STREAM_REDIRECTION
  EXPECT_STREQ("Header's second predecessor must be the latch!\n",
               ::testing::internal::GetCapturedStderr().c_str());
#endif
}

} // namespace
