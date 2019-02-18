//===- llvm/unittests/Transforms/Vectorize/VPlanDominatorTreeTest.cpp -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanHCFGBuilder.h"
#include "VPlanTestBase.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class VPlanDominatorTreeTest : public VPlanTestBase {};

TEST_F(VPlanDominatorTreeTest, BasicVPBBDomination) {
  const char *ModuleString =
      "define void @f(i32* %a, i32* %b, i32* %c, i32 %N, i32 %M, i32 %K) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]\n"
      "  br i1 true, label %if.then, label %if.else\n"
      "if.then:\n"
      "  br label %for.inc\n"
      "if.else:\n"
      "  br label %for.inc\n"
      "for.inc:\n"
      "  %iv.next = add nuw nsw i64 %iv, 1\n"
      "  %exitcond = icmp eq i64 %iv.next, 300\n"
      "  br i1 %exitcond, label %for.end, label %for.body\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildPlainCFG(LoopHeader);

  // Build VPlan domination tree analysis.
  VPRegionBlock *TopRegion = cast<VPRegionBlock>(Plan->getEntry());
  VPDominatorTree VPDT;
  VPDT.recalculate(*TopRegion);

  VPBlockBase *PH = TopRegion->getEntry();
  VPBlockBase *H = PH->getSingleSuccessor();
  VPBlockBase *IfThen = H->getSuccessors()[0];
  VPBlockBase *IfElse = H->getSuccessors()[1];
  VPBlockBase *Latch = IfThen->getSingleSuccessor();
  VPBlockBase *Exit = Latch->getSuccessors()[0] != H
                          ? Latch->getSuccessors()[0]
                          : Latch->getSuccessors()[1];
  // Reachability.
  EXPECT_TRUE(VPDT.isReachableFromEntry(PH));
  EXPECT_TRUE(VPDT.isReachableFromEntry(H));
  EXPECT_TRUE(VPDT.isReachableFromEntry(IfThen));
  EXPECT_TRUE(VPDT.isReachableFromEntry(IfElse));
  EXPECT_TRUE(VPDT.isReachableFromEntry(Latch));
  EXPECT_TRUE(VPDT.isReachableFromEntry(Exit));

  // VPBB dominance.
  EXPECT_TRUE(VPDT.dominates(PH, PH));
  EXPECT_TRUE(VPDT.dominates(PH, H));
  EXPECT_TRUE(VPDT.dominates(PH, IfThen));
  EXPECT_TRUE(VPDT.dominates(PH, IfElse));
  EXPECT_TRUE(VPDT.dominates(PH, Latch));
  EXPECT_TRUE(VPDT.dominates(PH, Exit));

  EXPECT_FALSE(VPDT.dominates(H, PH));
  EXPECT_TRUE(VPDT.dominates(H, H));
  EXPECT_TRUE(VPDT.dominates(H, IfThen));
  EXPECT_TRUE(VPDT.dominates(H, IfElse));
  EXPECT_TRUE(VPDT.dominates(H, Latch));
  EXPECT_TRUE(VPDT.dominates(H, Exit));

  EXPECT_FALSE(VPDT.dominates(IfThen, PH));
  EXPECT_FALSE(VPDT.dominates(IfThen, H));
  EXPECT_TRUE(VPDT.dominates(IfThen, IfThen));
  EXPECT_FALSE(VPDT.dominates(IfThen, IfElse));
  EXPECT_FALSE(VPDT.dominates(IfThen, Latch));
  EXPECT_FALSE(VPDT.dominates(IfThen, Exit));

  EXPECT_FALSE(VPDT.dominates(IfElse, PH));
  EXPECT_FALSE(VPDT.dominates(IfElse, H));
  EXPECT_FALSE(VPDT.dominates(IfElse, IfThen));
  EXPECT_TRUE(VPDT.dominates(IfElse, IfElse));
  EXPECT_FALSE(VPDT.dominates(IfElse, Latch));
  EXPECT_FALSE(VPDT.dominates(IfElse, Exit));

  EXPECT_FALSE(VPDT.dominates(Latch, PH));
  EXPECT_FALSE(VPDT.dominates(Latch, H));
  EXPECT_FALSE(VPDT.dominates(Latch, IfThen));
  EXPECT_FALSE(VPDT.dominates(Latch, IfElse));
  EXPECT_TRUE(VPDT.dominates(Latch, Latch));
  EXPECT_TRUE(VPDT.dominates(Latch, Exit));

  EXPECT_FALSE(VPDT.dominates(Exit, PH));
  EXPECT_FALSE(VPDT.dominates(Exit, H));
  EXPECT_FALSE(VPDT.dominates(Exit, IfThen));
  EXPECT_FALSE(VPDT.dominates(Exit, IfElse));
  EXPECT_FALSE(VPDT.dominates(Exit, Latch));
  EXPECT_TRUE(VPDT.dominates(Exit, Exit));

  // VPBB proper dominance.
  EXPECT_FALSE(VPDT.properlyDominates(PH, PH));
  EXPECT_TRUE(VPDT.properlyDominates(PH, H));
  EXPECT_TRUE(VPDT.properlyDominates(PH, IfThen));
  EXPECT_TRUE(VPDT.properlyDominates(PH, IfElse));
  EXPECT_TRUE(VPDT.properlyDominates(PH, Latch));
  EXPECT_TRUE(VPDT.properlyDominates(PH, Exit));

  EXPECT_FALSE(VPDT.properlyDominates(H, PH));
  EXPECT_FALSE(VPDT.properlyDominates(H, H));
  EXPECT_TRUE(VPDT.properlyDominates(H, IfThen));
  EXPECT_TRUE(VPDT.properlyDominates(H, IfElse));
  EXPECT_TRUE(VPDT.properlyDominates(H, Latch));
  EXPECT_TRUE(VPDT.properlyDominates(H, Exit));

  EXPECT_FALSE(VPDT.properlyDominates(IfThen, PH));
  EXPECT_FALSE(VPDT.properlyDominates(IfThen, H));
  EXPECT_FALSE(VPDT.properlyDominates(IfThen, IfThen));
  EXPECT_FALSE(VPDT.properlyDominates(IfThen, IfElse));
  EXPECT_FALSE(VPDT.properlyDominates(IfThen, Latch));
  EXPECT_FALSE(VPDT.properlyDominates(IfThen, Exit));

  EXPECT_FALSE(VPDT.properlyDominates(IfElse, PH));
  EXPECT_FALSE(VPDT.properlyDominates(IfElse, H));
  EXPECT_FALSE(VPDT.properlyDominates(IfElse, IfThen));
  EXPECT_FALSE(VPDT.properlyDominates(IfElse, IfElse));
  EXPECT_FALSE(VPDT.properlyDominates(IfElse, Latch));
  EXPECT_FALSE(VPDT.properlyDominates(IfElse, Exit));

  EXPECT_FALSE(VPDT.properlyDominates(Latch, PH));
  EXPECT_FALSE(VPDT.properlyDominates(Latch, H));
  EXPECT_FALSE(VPDT.properlyDominates(Latch, IfThen));
  EXPECT_FALSE(VPDT.properlyDominates(Latch, IfElse));
  EXPECT_FALSE(VPDT.properlyDominates(Latch, Latch));
  EXPECT_TRUE(VPDT.properlyDominates(Latch, Exit));

  EXPECT_FALSE(VPDT.properlyDominates(Exit, PH));
  EXPECT_FALSE(VPDT.properlyDominates(Exit, H));
  EXPECT_FALSE(VPDT.properlyDominates(Exit, IfThen));
  EXPECT_FALSE(VPDT.properlyDominates(Exit, IfElse));
  EXPECT_FALSE(VPDT.properlyDominates(Exit, Latch));
  EXPECT_FALSE(VPDT.properlyDominates(Exit, Exit));

  // VPBB nearest common dominator.
  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(PH, PH));
  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(PH, H));
  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(PH, IfThen));
  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(PH, IfElse));
  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(PH, Latch));
  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(PH, Exit));

  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(H, PH));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(H, H));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(H, IfThen));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(H, IfElse));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(H, Latch));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(H, Exit));

  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(IfThen, PH));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(IfThen, H));
  EXPECT_EQ(IfThen, VPDT.findNearestCommonDominator(IfThen, IfThen));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(IfThen, IfElse));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(IfThen, Latch));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(IfThen, Exit));

  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(IfElse, PH));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(IfElse, H));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(IfElse, IfThen));
  EXPECT_EQ(IfElse, VPDT.findNearestCommonDominator(IfElse, IfElse));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(IfElse, Latch));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(IfElse, Exit));

  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(Latch, PH));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(Latch, H));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(Latch, IfThen));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(Latch, IfElse));
  EXPECT_EQ(Latch, VPDT.findNearestCommonDominator(Latch, Latch));
  EXPECT_EQ(Latch, VPDT.findNearestCommonDominator(Latch, Exit));

  EXPECT_EQ(PH, VPDT.findNearestCommonDominator(Exit, PH));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(Exit, H));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(Exit, IfThen));
  EXPECT_EQ(H, VPDT.findNearestCommonDominator(Exit, IfElse));
  EXPECT_EQ(Latch, VPDT.findNearestCommonDominator(Exit, Latch));
  EXPECT_EQ(Exit, VPDT.findNearestCommonDominator(Exit, Exit));
}
} // namespace
} // namespace llvm
