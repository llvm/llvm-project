//===- llvm/unittest/Transforms/Vectorize/VPlanSlpTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanHCFGBuilder.h"
#include "VPlanTestBase.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class VPlanSlpTest : public VPlanTestBase {
protected:
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  DataLayout DL;

  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<ScalarEvolution> SE;
  std::unique_ptr<AAResults> AARes;
  std::unique_ptr<BasicAAResult> BasicAA;
  std::unique_ptr<LoopAccessInfo> LAI;
  std::unique_ptr<PredicatedScalarEvolution> PSE;
  std::unique_ptr<InterleavedAccessInfo> IAI;

  VPlanSlpTest()
      : TLII(), TLI(TLII),
        DL("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
           "f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:"
           "16:32:64-S128") {}

  VPInterleavedAccessInfo getInterleavedAccessInfo(Function &F, Loop *L,
                                                   VPlan &Plan) {
    AC.reset(new AssumptionCache(F));
    SE.reset(new ScalarEvolution(F, TLI, *AC, *DT, *LI));
    BasicAA.reset(new BasicAAResult(DL, F, TLI, *AC, &*DT));
    AARes.reset(new AAResults(TLI));
    AARes->addAAResult(*BasicAA);
    PSE.reset(new PredicatedScalarEvolution(*SE, *L));
    LAI.reset(new LoopAccessInfo(L, &*SE, &TLI, &*AARes, &*DT, &*LI));
    IAI.reset(new InterleavedAccessInfo(*PSE, L, &*DT, &*LI, &*LAI));
    IAI->analyzeInterleaving(false);
    return {Plan, *IAI};
  }
};

TEST_F(VPlanSlpTest, testSlpSimple_2) {
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "%struct.Test3 = type { i32, i32, i32 }\n"
      "%struct.Test4xi8 = type { i8, i8, i8 }\n"
      "define void @add_x2(%struct.Test* nocapture readonly %A, %struct.Test* "
      "nocapture readonly %B, %struct.Test* nocapture %C)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %add0 = add nsw i32 %vA0, %vB0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %add1 = add nsw i32 %vA1, %vB1\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add0, i32* %C0, align 4\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add1, i32* %C1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x2");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);
  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 12));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 14));

  VPlanSlp Slp(VPIAI, *Body);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  VPInstruction *CombinedStore = Slp.buildGraph(StoreRoot);
  EXPECT_EQ(64u, Slp.getWidestBundleBits());
  EXPECT_EQ(VPInstruction::SLPStore, CombinedStore->getOpcode());

  auto *CombinedAdd = cast<VPInstruction>(CombinedStore->getOperand(0));
  EXPECT_EQ(Instruction::Add, CombinedAdd->getOpcode());

  auto *CombinedLoadA = cast<VPInstruction>(CombinedAdd->getOperand(0));
  auto *CombinedLoadB = cast<VPInstruction>(CombinedAdd->getOperand(1));
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadA->getOpcode());
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadB->getOpcode());

  delete CombinedStore;
  delete CombinedAdd;
  delete CombinedLoadA;
  delete CombinedLoadB;
}

TEST_F(VPlanSlpTest, testSlpSimple_3) {
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "%struct.Test3 = type { i32, i32, i32 }\n"
      "%struct.Test4xi8 = type { i8, i8, i8 }\n"
      "define void @add_x2(%struct.Test* nocapture readonly %A, %struct.Test* "
      "nocapture readonly %B, %struct.Test* nocapture %C)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr %struct.Test, %struct.Test* %A, i64 "
      "                      %indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "                      %indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %add0 = add nsw i32 %vA0, %vB0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "                      %indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "                      %indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %add1 = add nsw i32 %vA1, %vB1\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "                      %indvars.iv, i32 0\n"
      "  store i32 %add0, i32* %C0, align 4\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "                      %indvars.iv, i32 1\n"
      "  store i32 %add1, i32* %C1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x2");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 12));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 14));

  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);

  VPlanSlp Slp(VPIAI, *Body);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  VPInstruction *CombinedStore = Slp.buildGraph(StoreRoot);
  EXPECT_EQ(64u, Slp.getWidestBundleBits());
  EXPECT_EQ(VPInstruction::SLPStore, CombinedStore->getOpcode());

  auto *CombinedAdd = cast<VPInstruction>(CombinedStore->getOperand(0));
  EXPECT_EQ(Instruction::Add, CombinedAdd->getOpcode());

  auto *CombinedLoadA = cast<VPInstruction>(CombinedAdd->getOperand(0));
  auto *CombinedLoadB = cast<VPInstruction>(CombinedAdd->getOperand(1));
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadA->getOpcode());
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadB->getOpcode());

  VPInstruction *GetA = cast<VPInstruction>(&*std::next(Body->begin(), 1));
  VPInstruction *GetB = cast<VPInstruction>(&*std::next(Body->begin(), 3));
  EXPECT_EQ(GetA, CombinedLoadA->getOperand(0));
  EXPECT_EQ(GetB, CombinedLoadB->getOperand(0));

  delete CombinedStore;
  delete CombinedAdd;
  delete CombinedLoadA;
  delete CombinedLoadB;
}

TEST_F(VPlanSlpTest, testSlpReuse_1) {
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "define void @add_x2(%struct.Test* nocapture readonly %A, %struct.Test* "
      "nocapture readonly %B, %struct.Test* nocapture %C)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %add0 = add nsw i32 %vA0, %vA0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %add1 = add nsw i32 %vA1, %vA1\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add0, i32* %C0, align 4\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add1, i32* %C1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x2");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);
  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 8));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 10));

  VPlanSlp Slp(VPIAI, *Body);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  VPInstruction *CombinedStore = Slp.buildGraph(StoreRoot);
  EXPECT_EQ(64u, Slp.getWidestBundleBits());
  EXPECT_EQ(VPInstruction::SLPStore, CombinedStore->getOpcode());

  auto *CombinedAdd = cast<VPInstruction>(CombinedStore->getOperand(0));
  EXPECT_EQ(Instruction::Add, CombinedAdd->getOpcode());

  auto *CombinedLoadA = cast<VPInstruction>(CombinedAdd->getOperand(0));
  EXPECT_EQ(CombinedLoadA, CombinedAdd->getOperand(1));
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadA->getOpcode());

  delete CombinedStore;
  delete CombinedAdd;
  delete CombinedLoadA;
}

TEST_F(VPlanSlpTest, testSlpReuse_2) {
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "define i32 @add_x2(%struct.Test* nocapture readonly %A, %struct.Test* "
      "nocapture readonly %B, %struct.Test* nocapture %C)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %add0 = add nsw i32 %vA0, %vA0\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add0, i32* %C0, align 4\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %add1 = add nsw i32 %vA1, %vA1\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add1, i32* %C1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret i32 %vA1\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x2");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);
  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 5));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 10));

  VPlanSlp Slp(VPIAI, *Body);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  Slp.buildGraph(StoreRoot);
  EXPECT_FALSE(Slp.isCompletelySLP());
}

static void checkReorderExample(VPInstruction *Store1, VPInstruction *Store2,
                                VPBasicBlock *Body,
                                VPInterleavedAccessInfo &&IAI) {
  VPlanSlp Slp(IAI, *Body);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  VPInstruction *CombinedStore = Slp.buildGraph(StoreRoot);

  EXPECT_TRUE(Slp.isCompletelySLP());
  EXPECT_EQ(CombinedStore->getOpcode(), VPInstruction::SLPStore);

  VPInstruction *CombinedAdd =
      cast<VPInstruction>(CombinedStore->getOperand(0));
  EXPECT_EQ(CombinedAdd->getOpcode(), Instruction::Add);

  VPInstruction *CombinedMulAB =
      cast<VPInstruction>(CombinedAdd->getOperand(0));
  VPInstruction *CombinedMulCD =
      cast<VPInstruction>(CombinedAdd->getOperand(1));
  EXPECT_EQ(CombinedMulAB->getOpcode(), Instruction::Mul);

  VPInstruction *CombinedLoadA =
      cast<VPInstruction>(CombinedMulAB->getOperand(0));
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadA->getOpcode());
  VPInstruction *LoadvA0 = cast<VPInstruction>(&*std::next(Body->begin(), 2));
  VPInstruction *LoadvA1 = cast<VPInstruction>(&*std::next(Body->begin(), 12));
  EXPECT_EQ(LoadvA0->getOperand(0), CombinedLoadA->getOperand(0));
  EXPECT_EQ(LoadvA1->getOperand(0), CombinedLoadA->getOperand(1));

  VPInstruction *CombinedLoadB =
      cast<VPInstruction>(CombinedMulAB->getOperand(1));
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadB->getOpcode());
  VPInstruction *LoadvB0 = cast<VPInstruction>(&*std::next(Body->begin(), 4));
  VPInstruction *LoadvB1 = cast<VPInstruction>(&*std::next(Body->begin(), 14));
  EXPECT_EQ(LoadvB0->getOperand(0), CombinedLoadB->getOperand(0));
  EXPECT_EQ(LoadvB1->getOperand(0), CombinedLoadB->getOperand(1));

  EXPECT_EQ(CombinedMulCD->getOpcode(), Instruction::Mul);

  VPInstruction *CombinedLoadC =
      cast<VPInstruction>(CombinedMulCD->getOperand(0));
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadC->getOpcode());
  VPInstruction *LoadvC0 = cast<VPInstruction>(&*std::next(Body->begin(), 7));
  VPInstruction *LoadvC1 = cast<VPInstruction>(&*std::next(Body->begin(), 17));
  EXPECT_EQ(LoadvC0->getOperand(0), CombinedLoadC->getOperand(0));
  EXPECT_EQ(LoadvC1->getOperand(0), CombinedLoadC->getOperand(1));

  VPInstruction *CombinedLoadD =
      cast<VPInstruction>(CombinedMulCD->getOperand(1));
  EXPECT_EQ(VPInstruction::SLPLoad, CombinedLoadD->getOpcode());
  VPInstruction *LoadvD0 = cast<VPInstruction>(&*std::next(Body->begin(), 9));
  VPInstruction *LoadvD1 = cast<VPInstruction>(&*std::next(Body->begin(), 19));
  EXPECT_EQ(LoadvD0->getOperand(0), CombinedLoadD->getOperand(0));
  EXPECT_EQ(LoadvD1->getOperand(0), CombinedLoadD->getOperand(1));

  delete CombinedStore;
  delete CombinedAdd;
  delete CombinedMulAB;
  delete CombinedMulCD;
  delete CombinedLoadA;
  delete CombinedLoadB;
  delete CombinedLoadC;
  delete CombinedLoadD;
}

TEST_F(VPlanSlpTest, testSlpReorder_1) {
  LLVMContext Ctx;
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "define void @add_x3(%struct.Test* %A, %struct.Test* %B, %struct.Test* "
      "%C,  %struct.Test* %D,  %struct.Test* %E)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %mul11 = mul nsw i32 %vA0, %vB0\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  %vC0 = load i32, i32* %C0, align 4\n"
      "  %D0 = getelementptr inbounds %struct.Test, %struct.Test* %D, i64 "
      "%indvars.iv, i32 0\n"
      "  %vD0 = load i32, i32* %D0, align 4\n"
      "  %mul12 = mul nsw i32 %vC0, %vD0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %mul21 = mul nsw i32 %vA1, %vB1\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  %vC1 = load i32, i32* %C1, align 4\n"
      "  %D1 = getelementptr inbounds %struct.Test, %struct.Test* %D, i64 "
      "%indvars.iv, i32 1\n"
      "  %vD1 = load i32, i32* %D1, align 4\n"
      "  %mul22 = mul nsw i32 %vC1, %vD1\n"
      "  %add1 = add nsw i32 %mul11, %mul12\n"
      "  %add2 = add nsw i32 %mul22, %mul21\n"
      "  %E0 = getelementptr inbounds %struct.Test, %struct.Test* %E, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add1, i32* %E0, align 4\n"
      "  %E1 = getelementptr inbounds %struct.Test, %struct.Test* %E, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add2, i32* %E1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x3");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 24));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 26));

  checkReorderExample(
      Store1, Store2, Body,
      getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan));
}

TEST_F(VPlanSlpTest, testSlpReorder_2) {
  LLVMContext Ctx;
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "define void @add_x3(%struct.Test* %A, %struct.Test* %B, %struct.Test* "
      "%C,  %struct.Test* %D,  %struct.Test* %E)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %mul11 = mul nsw i32 %vA0, %vB0\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  %vC0 = load i32, i32* %C0, align 4\n"
      "  %D0 = getelementptr inbounds %struct.Test, %struct.Test* %D, i64 "
      "%indvars.iv, i32 0\n"
      "  %vD0 = load i32, i32* %D0, align 4\n"
      "  %mul12 = mul nsw i32 %vC0, %vD0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %mul21 = mul nsw i32 %vB1, %vA1\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  %vC1 = load i32, i32* %C1, align 4\n"
      "  %D1 = getelementptr inbounds %struct.Test, %struct.Test* %D, i64 "
      "%indvars.iv, i32 1\n"
      "  %vD1 = load i32, i32* %D1, align 4\n"
      "  %mul22 = mul nsw i32 %vD1, %vC1\n"
      "  %add1 = add nsw i32 %mul11, %mul12\n"
      "  %add2 = add nsw i32 %mul22, %mul21\n"
      "  %E0 = getelementptr inbounds %struct.Test, %struct.Test* %E, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add1, i32* %E0, align 4\n"
      "  %E1 = getelementptr inbounds %struct.Test, %struct.Test* %E, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add2, i32* %E1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x3");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 24));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 26));

  checkReorderExample(
      Store1, Store2, Body,
      getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan));
}

TEST_F(VPlanSlpTest, testSlpReorder_3) {
  LLVMContext Ctx;
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "define void @add_x3(%struct.Test* %A, %struct.Test* %B, %struct.Test* "
      "%C,  %struct.Test* %D,  %struct.Test* %E)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %mul11 = mul nsw i32 %vA1, %vB0\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  %vC0 = load i32, i32* %C0, align 4\n"
      "  %D0 = getelementptr inbounds %struct.Test, %struct.Test* %D, i64 "
      "%indvars.iv, i32 0\n"
      "  %vD0 = load i32, i32* %D0, align 4\n"
      "  %mul12 = mul nsw i32 %vC0, %vD0\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %mul21 = mul nsw i32 %vB1, %vA0\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  %vC1 = load i32, i32* %C1, align 4\n"
      "  %D1 = getelementptr inbounds %struct.Test, %struct.Test* %D, i64 "
      "%indvars.iv, i32 1\n"
      "  %vD1 = load i32, i32* %D1, align 4\n"
      "  %mul22 = mul nsw i32 %vD1, %vC1\n"
      "  %add1 = add nsw i32 %mul11, %mul12\n"
      "  %add2 = add nsw i32 %mul22, %mul21\n"
      "  %E0 = getelementptr inbounds %struct.Test, %struct.Test* %E, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add1, i32* %E0, align 4\n"
      "  %E1 = getelementptr inbounds %struct.Test, %struct.Test* %E, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add2, i32* %E1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x3");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 24));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 26));

  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);
  VPlanSlp Slp(VPIAI, *Body);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  EXPECT_EQ(nullptr, Slp.buildGraph(StoreRoot));

  // FIXME Need to select better first value for lane0.
  EXPECT_FALSE(Slp.isCompletelySLP());
}

TEST_F(VPlanSlpTest, testSlpReorder_4) {
  LLVMContext Ctx;
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "define void @add_x3(%struct.Test* %A, %struct.Test* %B, %struct.Test* "
      "%C,  %struct.Test* %D,  %struct.Test* %E)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %mul11 = mul nsw i32 %vA0, %vB0\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  %vC0 = load i32, i32* %C0, align 4\n"
      "  %D0 = getelementptr inbounds %struct.Test, %struct.Test* %D, i64 "
      "%indvars.iv, i32 0\n"
      "  %vD0 = load i32, i32* %D0, align 4\n"
      "  %mul12 = mul nsw i32 %vC0, %vD0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %mul21 = mul nsw i32 %vA1, %vB1\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  %vC1 = load i32, i32* %C1, align 4\n"
      "  %D1 = getelementptr inbounds %struct.Test, %struct.Test* %D, i64 "
      "%indvars.iv, i32 1\n"
      "  %vD1 = load i32, i32* %D1, align 4\n"
      "  %mul22 = mul nsw i32 %vC1, %vD1\n"
      "  %add1 = add nsw i32 %mul11, %mul12\n"
      "  %add2 = add nsw i32 %mul22, %mul21\n"
      "  %E0 = getelementptr inbounds %struct.Test, %struct.Test* %E, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add1, i32* %E0, align 4\n"
      "  %E1 = getelementptr inbounds %struct.Test, %struct.Test* %E, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add2, i32* %E1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x3");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 24));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 26));

  checkReorderExample(
      Store1, Store2, Body,
      getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan));
}

// Make sure we do not combine instructions with operands in different BBs.
TEST_F(VPlanSlpTest, testInstrsInDifferentBBs) {
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "%struct.Test3 = type { i32, i32, i32 }\n"
      "%struct.Test4xi8 = type { i8, i8, i8 }\n"
      "define void @add_x2(%struct.Test* nocapture readonly %A, %struct.Test* "
      "nocapture readonly %B, %struct.Test* nocapture %C)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %add0 = add nsw i32 %vA0, %vB0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  br label %bb2\n"
      "bb2:\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %add1 = add nsw i32 %vA1, %vB1\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add0, i32* %C0, align 4\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add1, i32* %C1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x2");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);
  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();
  VPBasicBlock *BB2 = Body->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(BB2->begin(), 3));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(BB2->begin(), 5));

  VPlanSlp Slp(VPIAI, *BB2);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  EXPECT_EQ(nullptr, Slp.buildGraph(StoreRoot));
  EXPECT_EQ(0u, Slp.getWidestBundleBits());
}

// Make sure we do not combine instructions with operands in different BBs.
TEST_F(VPlanSlpTest, testInstrsInDifferentBBs2) {
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "%struct.Test3 = type { i32, i32, i32 }\n"
      "%struct.Test4xi8 = type { i8, i8, i8 }\n"
      "define void @add_x2(%struct.Test* nocapture readonly %A, %struct.Test* "
      "nocapture readonly %B, %struct.Test* nocapture %C)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %add0 = add nsw i32 %vA0, %vB0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %add1 = add nsw i32 %vA1, %vB1\n"
      "  br label %bb2\n"
      "bb2:\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add0, i32* %C0, align 4\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add1, i32* %C1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x2");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);
  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();
  VPBasicBlock *BB2 = Body->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(BB2->begin(), 1));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(BB2->begin(), 3));

  VPlanSlp Slp(VPIAI, *BB2);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  EXPECT_EQ(nullptr, Slp.buildGraph(StoreRoot));
  EXPECT_EQ(0u, Slp.getWidestBundleBits());
}

TEST_F(VPlanSlpTest, testSlpAtomicLoad) {
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "%struct.Test3 = type { i32, i32, i32 }\n"
      "%struct.Test4xi8 = type { i8, i8, i8 }\n"
      "define void @add_x2(%struct.Test* nocapture readonly %A, %struct.Test* "
      "nocapture readonly %B, %struct.Test* nocapture %C)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load atomic i32, i32* %A0 monotonic, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %add0 = add nsw i32 %vA0, %vB0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %add1 = add nsw i32 %vA1, %vB1\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  store i32 %add0, i32* %C0, align 4\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add1, i32* %C1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x2");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);
  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 12));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 14));

  VPlanSlp Slp(VPIAI, *Body);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  EXPECT_EQ(nullptr, Slp.buildGraph(StoreRoot));
  EXPECT_FALSE(Slp.isCompletelySLP());
}

TEST_F(VPlanSlpTest, testSlpAtomicStore) {
  const char *ModuleString =
      "%struct.Test = type { i32, i32 }\n"
      "%struct.Test3 = type { i32, i32, i32 }\n"
      "%struct.Test4xi8 = type { i8, i8, i8 }\n"
      "define void @add_x2(%struct.Test* nocapture readonly %A, %struct.Test* "
      "nocapture readonly %B, %struct.Test* nocapture %C)  {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:                                         ; preds = %for.body, "
      "%entry\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %A0 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 0\n"
      "  %vA0 = load i32, i32* %A0, align 4\n"
      "  %B0 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 0\n"
      "  %vB0 = load i32, i32* %B0, align 4\n"
      "  %add0 = add nsw i32 %vA0, %vB0\n"
      "  %A1 = getelementptr inbounds %struct.Test, %struct.Test* %A, i64 "
      "%indvars.iv, i32 1\n"
      "  %vA1 = load i32, i32* %A1, align 4\n"
      "  %B1 = getelementptr inbounds %struct.Test, %struct.Test* %B, i64 "
      "%indvars.iv, i32 1\n"
      "  %vB1 = load i32, i32* %B1, align 4\n"
      "  %add1 = add nsw i32 %vA1, %vB1\n"
      "  %C0 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 0\n"
      "  store atomic i32 %add0, i32* %C0 monotonic, align 4\n"
      "  %C1 = getelementptr inbounds %struct.Test, %struct.Test* %C, i64 "
      "%indvars.iv, i32 1\n"
      "  store i32 %add1, i32* %C1, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %indvars.iv.next, 1024\n"
      "  br i1 %exitcond, label %for.cond.cleanup, label %for.body\n"
      "for.cond.cleanup:                                 ; preds = %for.body\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("add_x2");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);
  auto VPIAI = getInterleavedAccessInfo(*F, LI->getLoopFor(LoopHeader), *Plan);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  VPBasicBlock *Body = Entry->getSingleSuccessor()->getEntryBasicBlock();

  VPInstruction *Store1 = cast<VPInstruction>(&*std::next(Body->begin(), 12));
  VPInstruction *Store2 = cast<VPInstruction>(&*std::next(Body->begin(), 14));

  VPlanSlp Slp(VPIAI, *Body);
  SmallVector<VPValue *, 4> StoreRoot = {Store1, Store2};
  Slp.buildGraph(StoreRoot);
  EXPECT_FALSE(Slp.isCompletelySLP());
}

} // namespace
} // namespace llvm
