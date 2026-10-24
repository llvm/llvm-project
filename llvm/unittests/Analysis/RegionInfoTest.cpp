//===- RegionInfoTest.cpp - Region node cache lifetime tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class RegionCacheTest : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  DominatorTree DT;
  PostDominatorTree PDT;
  DominanceFrontier DF;
  RegionInfo RI;
  BasicBlock *Left = nullptr;
  BasicBlock *Other = nullptr;
  BasicBlock *Exit = nullptr;

  void SetUp() override {
    SMDiagnostic Error;
    M = parseAssemblyString(R"(
      define void @f(i1 %c, i1 %d) {
      entry:
        br i1 %c, label %inner, label %other
      inner:
        br i1 %d, label %left, label %right
      left:
        br label %join
      right:
        br label %join
      join:
        br label %exit
      other:
        br label %exit
      exit:
        ret void
      })", Error, Context);
    ASSERT_TRUE(M);
    Function &F = *M->getFunction("f");
    for (BasicBlock &BB : F) {
      if (BB.getName() == "left") Left = &BB;
      if (BB.getName() == "other") Other = &BB;
      if (BB.getName() == "exit") Exit = &BB;
    }
    DT.recalculate(F);
    PDT.recalculate(F);
    DF.analyze(DT);
    RI.recalculate(F, &DT, &PDT, &DF);
  }

  void populate(RegionInfo &Info) {
    auto *Inner = Info.getRegionFor(Left);
    auto *Outer = Info.getRegionFor(Other);
    EXPECT_NE(Inner, Outer);
    EXPECT_EQ(Inner->getBBNode(Left)->getParent(), Inner);
    EXPECT_EQ(Outer->getBBNode(Other)->getParent(), Outer);
    auto *Root = Info.getTopLevelRegion();
    EXPECT_EQ(Root->getBBNode(Exit)->getEntry(), Exit);
  }
};

TEST_F(RegionCacheTest, RepeatedGlobalInvalidation) {
  for (unsigned I = 0; I != 8; ++I) {
    populate(RI);
    RI.clearNodeCache();
    RI.clearNodeCache();
  }
  populate(RI);
}

TEST_F(RegionCacheTest, LocalInvalidationPreservesOtherCaches) {
  populate(RI);
  auto *Root = RI.getTopLevelRegion();
  auto *ExitNode = Root->getBBNode(Exit);
  auto *Inner = RI.getRegionFor(Left);
  Inner->clearNodeCache();
  EXPECT_EQ(Root->getBBNode(Exit), ExitNode);
  EXPECT_EQ(Inner->getBBNode(Left)->getEntry(), Left);
  RI.clearNodeCache();
  populate(RI);
}

TEST_F(RegionCacheTest, DestroyCachedDetachedRegion) {
  // Refined or detached regions may be destroyed before the owning analysis.
  {
    Region Detached(Left, Exit, &RI, &DT);
    EXPECT_EQ(Detached.getBBNode(Left)->getParent(), &Detached);
  }
  RI.clearNodeCache();
  populate(RI);
}

TEST_F(RegionCacheTest, ClearThenDestroyDetachedRegion) {
  {
    Region Detached(Left, Exit, &RI, &DT);
    Detached.getBBNode(Left);
    Detached.clearNodeCache();
    RI.clearNodeCache();
    Detached.getBBNode(Left);
    RI.clearNodeCache();
  }
  RI.clearNodeCache();
}

TEST_F(RegionCacheTest, MovePopulatedAnalysis) {
  populate(RI);
  RegionInfo Moved(std::move(RI));
  RI.clearNodeCache();
  Moved.clearNodeCache();
  populate(Moved);
  Moved.clearNodeCache();
}

TEST_F(RegionCacheTest, MoveAssignmentReleasesPreviousCaches) {
  populate(RI);
  RegionInfo Moved;
  Moved.recalculate(*M->getFunction("f"), &DT, &PDT, &DF);
  populate(Moved);
  Moved = std::move(RI);
  RI.clearNodeCache();
  Moved.clearNodeCache();
  populate(Moved);
}

TEST_F(RegionCacheTest, MoveWithDetachedCachedRegion) {
  auto Detached = std::make_unique<Region>(Left, Exit, &RI, &DT);
  Detached->getBBNode(Left);
  RegionInfo Moved(std::move(RI));
  Moved.clearNodeCache();
  Detached->getBBNode(Left);
  Detached.reset();
  Moved.clearNodeCache();
}

TEST_F(RegionCacheTest, ReleaseAndRecalculate) {
  populate(RI);
  RI.releaseMemory();
  RI.clearNodeCache();
  RI.recalculate(*M->getFunction("f"), &DT, &PDT, &DF);
  populate(RI);
  RI.clearNodeCache();
}
} // namespace
