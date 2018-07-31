//===- llvm/unittests/Transforms/Vectorize/VPlanLoopInfoTest.cpp -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanLoopInfo.h"
#include "VPlanTestBase.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class VPlanLoopInfo : public VPlanTestBase {};

TEST_F(VPlanLoopInfo, BasicLoopInfoTest) {
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
  auto Plan = buildHCFG(LoopHeader);

  // Build VPlan domination tree and loop info analyses.
  VPRegionBlock *TopRegion = cast<VPRegionBlock>(Plan->getEntry());
  VPDominatorTree VPDT;
  VPDT.recalculate(*TopRegion);
  VPLoopInfo VPLI;
  VPLI.analyze(VPDT);

  VPBlockBase *PH = TopRegion->getEntry();
  VPBlockBase *H = PH->getSingleSuccessor();
  VPBlockBase *IfThen = H->getSuccessors()[0];
  VPBlockBase *IfElse = H->getSuccessors()[1];
  VPBlockBase *Latch = IfThen->getSingleSuccessor();
  VPBlockBase *Exit = Latch->getSuccessors()[0] != H
                          ? Latch->getSuccessors()[0]
                          : Latch->getSuccessors()[1];

  // Number of loops.
  EXPECT_EQ(1, std::distance(VPLI.begin(), VPLI.end()));
  VPLoop *VPLp = *VPLI.begin();

  // VPBBs contained in VPLoop.
  EXPECT_FALSE(VPLp->contains(PH));
  EXPECT_EQ(nullptr, VPLI.getLoopFor(PH));
  EXPECT_TRUE(VPLp->contains(H));
  EXPECT_EQ(VPLp, VPLI.getLoopFor(H));
  EXPECT_TRUE(VPLp->contains(IfThen));
  EXPECT_EQ(VPLp, VPLI.getLoopFor(IfThen));
  EXPECT_TRUE(VPLp->contains(IfElse));
  EXPECT_EQ(VPLp, VPLI.getLoopFor(IfElse));
  EXPECT_TRUE(VPLp->contains(Latch));
  EXPECT_EQ(VPLp, VPLI.getLoopFor(Latch));
  EXPECT_FALSE(VPLp->contains(Exit));
  EXPECT_EQ(nullptr, VPLI.getLoopFor(Exit));

  // VPLoop's parts.
  EXPECT_EQ(PH, VPLp->getLoopPreheader());
  EXPECT_EQ(H, VPLp->getHeader());
  EXPECT_EQ(Latch, VPLp->getLoopLatch());
  EXPECT_EQ(Latch, VPLp->getExitingBlock());
  EXPECT_EQ(Exit, VPLp->getExitBlock());
}
} // namespace
} // namespace llvm
