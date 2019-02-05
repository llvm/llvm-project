//===- llvm/unittests/Transforms/Vectorize/VPlanPredicatorTest.cpp -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanPredicator.h"
#include "VPlanTestBase.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class VPlanPredicatorTest : public VPlanTestBase {};

TEST_F(VPlanPredicatorTest, BasicPredicatorTest) {
  const char *ModuleString =
      "@arr = common global [8 x [8 x i64]] "
      "zeroinitializer, align 16\n"
      "@arr2 = common global [8 x [8 x i64]] "
      "zeroinitializer, align 16\n"
      "@arr3 = common global [8 x [8 x i64]] "
      "zeroinitializer, align 16\n"
      "define void @f(i64 %n1) {\n"
      "entry:\n"
      "  br label %for.cond1.preheader\n"
      "for.cond1.preheader:                              \n"
      "  %i1.029 = phi i64 [ 0, %entry ], [ %inc14, %for.inc13 ]\n"
      "  br label %for.body3\n"
      "for.body3:                                        \n"
      "  %i2.028 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.inc ]\n"
      "  %arrayidx4 = getelementptr inbounds [8 x [8 x i64]], [8 x [8 x i64]]* "
      "@arr, i64 0, i64 %i2.028, i64 %i1.029\n"
      "  %0 = load i64, i64* %arrayidx4, align 8\n"
      "  %cmp5 = icmp ugt i64 %0, 10\n"
      "  br i1 %cmp5, label %if.then, label %for.inc\n"
      "if.then:                                          \n"
      "  %arrayidx7 = getelementptr inbounds [8 x [8 x i64]], [8 x [8 x i64]]* "
      "@arr2, i64 0, i64 %i2.028, i64 %i1.029\n"
      "  %1 = load i64, i64* %arrayidx7, align 8\n"
      "  %cmp8 = icmp ugt i64 %1, 100\n"
      "  br i1 %cmp8, label %if.then9, label %for.inc\n"
      "if.then9:                                         \n"
      "  %add = add nuw nsw i64 %i2.028, %i1.029\n"
      "  %arrayidx11 = getelementptr inbounds [8 x [8 x i64]], [8 x [8 x "
      "i64]]* @arr3, i64 0, i64 %i2.028, i64 %i1.029\n"
      "  store i64 %add, i64* %arrayidx11, align 8\n"
      "  br label %for.inc\n"
      "for.inc:                                          \n"
      "  %inc = add nuw nsw i64 %i2.028, 1\n"
      "  %exitcond = icmp eq i64 %inc, 8\n"
      "  br i1 %exitcond, label %for.inc13, label %for.body3\n"
      "for.inc13:                                        \n"
      "  %inc14 = add nuw nsw i64 %i1.029, 1\n"
      "  %exitcond30 = icmp eq i64 %inc14, 8\n"
      "  br i1 %exitcond30, label %for.end15, label %for.cond1.preheader\n"
      "for.end15:                                        \n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPRegionBlock *TopRegion = cast<VPRegionBlock>(Plan->getEntry());
  VPBlockBase *PH = TopRegion->getEntry();
  VPBlockBase *H = PH->getSingleSuccessor();
  VPBlockBase *InnerLoopH = H->getSingleSuccessor();
  VPBlockBase *OuterIf = InnerLoopH->getSuccessors()[0];
  VPBlockBase *InnerLoopLatch = InnerLoopH->getSuccessors()[1];
  VPBlockBase *InnerIf = OuterIf->getSuccessors()[0];
  VPValue *CBV1 = InnerLoopH->getCondBit();
  VPValue *CBV2 = OuterIf->getCondBit();

  // Apply predication.
  VPlanPredicator VPP(*Plan);
  VPP.predicate();

  VPBlockBase *InnerLoopLinSucc = InnerLoopH->getSingleSuccessor();
  VPBlockBase *OuterIfLinSucc = OuterIf->getSingleSuccessor();
  VPBlockBase *InnerIfLinSucc = InnerIf->getSingleSuccessor();
  VPValue *OuterIfPred = OuterIf->getPredicate();
  VPInstruction *InnerAnd =
      cast<VPInstruction>(InnerIf->getEntryBasicBlock()->begin());
  VPValue *InnerIfPred = InnerIf->getPredicate();

  // Test block predicates
  EXPECT_NE(nullptr, CBV1);
  EXPECT_NE(nullptr, CBV2);
  EXPECT_NE(nullptr, InnerAnd);
  EXPECT_EQ(CBV1, OuterIfPred);
  EXPECT_EQ(InnerAnd->getOpcode(), Instruction::And);
  EXPECT_EQ(InnerAnd->getOperand(0), CBV1);
  EXPECT_EQ(InnerAnd->getOperand(1), CBV2);
  EXPECT_EQ(InnerIfPred, InnerAnd);

  // Test Linearization
  EXPECT_EQ(InnerLoopLinSucc, OuterIf);
  EXPECT_EQ(OuterIfLinSucc, InnerIf);
  EXPECT_EQ(InnerIfLinSucc, InnerLoopLatch);
}

// Test generation of Not and Or during predication.
TEST_F(VPlanPredicatorTest, PredicatorNegOrTest) {
  const char *ModuleString =
      "@arr = common global [100 x [100 x i32]] zeroinitializer, align 16\n"
      "@arr2 = common global [100 x [100 x i32]] zeroinitializer, align 16\n"
      "@arr3 = common global [100 x [100 x i32]] zeroinitializer, align 16\n"
      "define void @foo() {\n"
      "entry:\n"
      "  br label %for.cond1.preheader\n"
      "for.cond1.preheader:                              \n"
      "  %indvars.iv42 = phi i64 [ 0, %entry ], [ %indvars.iv.next43, "
      "%for.inc22 ]\n"
      "  br label %for.body3\n"
      "for.body3:                                        \n"
      "  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ "
      "%indvars.iv.next, %if.end21 ]\n"
      "  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 "
      "x i32]]* @arr, i64 0, i64 %indvars.iv, i64 %indvars.iv42\n"
      "  %0 = load i32, i32* %arrayidx5, align 4\n"
      "  %cmp6 = icmp slt i32 %0, 100\n"
      "  br i1 %cmp6, label %if.then, label %if.end21\n"
      "if.then:                                          \n"
      "  %cmp7 = icmp sgt i32 %0, 10\n"
      "  br i1 %cmp7, label %if.then8, label %if.else\n"
      "if.then8:                                         \n"
      "  %add = add nsw i32 %0, 10\n"
      "  %arrayidx12 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 "
      "x i32]]* @arr2, i64 0, i64 %indvars.iv, i64 %indvars.iv42\n"
      "  store i32 %add, i32* %arrayidx12, align 4\n"
      "  br label %if.end\n"
      "if.else:                                          \n"
      "  %sub = add nsw i32 %0, -10\n"
      "  %arrayidx16 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 "
      "x i32]]* @arr3, i64 0, i64 %indvars.iv, i64 %indvars.iv42\n"
      "  store i32 %sub, i32* %arrayidx16, align 4\n"
      "  br label %if.end\n"
      "if.end:                                           \n"
      "  store i32 222, i32* %arrayidx5, align 4\n"
      "  br label %if.end21\n"
      "if.end21:                                         \n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 100\n"
      "  br i1 %exitcond, label %for.inc22, label %for.body3\n"
      "for.inc22:                                        \n"
      "  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1\n"
      "  %exitcond44 = icmp eq i64 %indvars.iv.next43, 100\n"
      "  br i1 %exitcond44, label %for.end24, label %for.cond1.preheader\n"
      "for.end24:                                        \n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);
  Function *F = M.getFunction("foo");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPRegionBlock *TopRegion = cast<VPRegionBlock>(Plan->getEntry());
  VPBlockBase *PH = TopRegion->getEntry();
  VPBlockBase *H = PH->getSingleSuccessor();
  VPBlockBase *OuterIfCmpBlk = H->getSingleSuccessor();
  VPBlockBase *InnerIfCmpBlk = OuterIfCmpBlk->getSuccessors()[0];
  VPBlockBase *InnerIfTSucc = InnerIfCmpBlk->getSuccessors()[0];
  VPBlockBase *InnerIfFSucc = InnerIfCmpBlk->getSuccessors()[1];
  VPBlockBase *TSuccSucc = InnerIfTSucc->getSingleSuccessor();
  VPBlockBase *FSuccSucc = InnerIfFSucc->getSingleSuccessor();

  VPValue *OuterCBV = OuterIfCmpBlk->getCondBit();
  VPValue *InnerCBV = InnerIfCmpBlk->getCondBit();

  // Apply predication.
  VPlanPredicator VPP(*Plan);
  VPP.predicate();

  VPInstruction *And =
      cast<VPInstruction>(InnerIfTSucc->getEntryBasicBlock()->begin());
  VPInstruction *Not =
      cast<VPInstruction>(InnerIfFSucc->getEntryBasicBlock()->begin());
  VPInstruction *NotAnd = cast<VPInstruction>(
      &*std::next(InnerIfFSucc->getEntryBasicBlock()->begin(), 1));
  VPInstruction *Or =
      cast<VPInstruction>(TSuccSucc->getEntryBasicBlock()->begin());

  // Test block predicates
  EXPECT_NE(nullptr, OuterCBV);
  EXPECT_NE(nullptr, InnerCBV);
  EXPECT_NE(nullptr, And);
  EXPECT_NE(nullptr, Not);
  EXPECT_NE(nullptr, NotAnd);

  EXPECT_EQ(And->getOpcode(), Instruction::And);
  EXPECT_EQ(NotAnd->getOpcode(), Instruction::And);
  EXPECT_EQ(Not->getOpcode(), VPInstruction::Not);

  EXPECT_EQ(And->getOperand(0), OuterCBV);
  EXPECT_EQ(And->getOperand(1), InnerCBV);

  EXPECT_EQ(Not->getOperand(0), InnerCBV);

  EXPECT_EQ(NotAnd->getOperand(0), OuterCBV);
  EXPECT_EQ(NotAnd->getOperand(1), Not);

  EXPECT_EQ(InnerIfTSucc->getPredicate(), And);
  EXPECT_EQ(InnerIfFSucc->getPredicate(), NotAnd);

  EXPECT_EQ(TSuccSucc, FSuccSucc);
  EXPECT_EQ(Or->getOpcode(), Instruction::Or);
  EXPECT_EQ(TSuccSucc->getPredicate(), Or);

  // Test operands of the Or - account for differences in predecessor block
  // ordering.
  VPInstruction *OrOp0Inst = cast<VPInstruction>(Or->getOperand(0));
  VPInstruction *OrOp1Inst = cast<VPInstruction>(Or->getOperand(1));

  bool ValidOrOperands = false;
  if (((OrOp0Inst == And) && (OrOp1Inst == NotAnd)) ||
      ((OrOp0Inst == NotAnd) && (OrOp1Inst == And)))
    ValidOrOperands = true;

  EXPECT_TRUE(ValidOrOperands);
}

} // namespace
} // namespace llvm
