//===- SizeOptsTest.cpp - SizeOpts unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SizeOpts.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class SizeOptsTest : public testing::Test {
protected:
  static const char* IRString;
  LLVMContext C;
  std::unique_ptr<Module> M;
  struct BFIData {
    std::unique_ptr<DominatorTree> DT;
    std::unique_ptr<LoopInfo> LI;
    std::unique_ptr<BranchProbabilityInfo> BPI;
    std::unique_ptr<BlockFrequencyInfo> BFI;
    BFIData(Function &F) {
      DT.reset(new DominatorTree(F));
      LI.reset(new LoopInfo(*DT));
      BPI.reset(new BranchProbabilityInfo(F, *LI));
      BFI.reset(new BlockFrequencyInfo(F, *BPI, *LI));
    }
    BlockFrequencyInfo *get() { return BFI.get(); }
  };

  void SetUp() override {
    SMDiagnostic Err;
    M = parseAssemblyString(IRString, Err, C);
  }
};

TEST_F(SizeOptsTest, Test) {
  Function *F = M->getFunction("f");
  Function *G = M->getFunction("g");
  Function *H = M->getFunction("h");

  ProfileSummaryInfo PSI(*M);
  BFIData BFID_F(*F);
  BFIData BFID_G(*G);
  BFIData BFID_H(*H);
  BlockFrequencyInfo *BFI_F = BFID_F.get();
  BlockFrequencyInfo *BFI_G = BFID_G.get();
  BlockFrequencyInfo *BFI_H = BFID_H.get();
  BasicBlock &BB0 = F->getEntryBlock();
  BasicBlock *BB1 = BB0.getTerminator()->getSuccessor(0);
  BasicBlock *BB2 = BB0.getTerminator()->getSuccessor(1);
  BasicBlock *BB3 = BB1->getSingleSuccessor();

  EXPECT_TRUE(PSI.hasProfileSummary());
  EXPECT_FALSE(shouldOptimizeForSize(F, &PSI, BFI_F, PGSOQueryType::Test));
  EXPECT_TRUE(shouldOptimizeForSize(G, &PSI, BFI_G, PGSOQueryType::Test));
  EXPECT_FALSE(shouldOptimizeForSize(H, &PSI, BFI_H, PGSOQueryType::Test));
  EXPECT_FALSE(shouldOptimizeForSize(&BB0, &PSI, BFI_F, PGSOQueryType::Test));
  EXPECT_FALSE(shouldOptimizeForSize(BB1, &PSI, BFI_F, PGSOQueryType::Test));
  EXPECT_TRUE(shouldOptimizeForSize(BB2, &PSI, BFI_F, PGSOQueryType::Test));
  EXPECT_FALSE(shouldOptimizeForSize(BB3, &PSI, BFI_F, PGSOQueryType::Test));
}

const char* SizeOptsTest::IRString = R"IR(
  define i32 @g(i32 %x) !prof !14 {
    ret i32 0
  }

  define i32 @h(i32 %x) !prof !15 {
    ret i32 0
  }

  define i32 @f(i32 %x) !prof !16 {
  bb0:
    %y1 = icmp eq i32 %x, 0
    br i1 %y1, label %bb1, label %bb2, !prof !17

  bb1:                                              ; preds = %bb0
    %z1 = call i32 @g(i32 %x)
    br label %bb3

  bb2:                                              ; preds = %bb0
    %z2 = call i32 @h(i32 %x)
    br label %bb3

  bb3:                                              ; preds = %bb2, %bb1
    %y2 = phi i32 [ 0, %bb1 ], [ 1, %bb2 ]
    ret i32 %y2
  }

  !llvm.module.flags = !{!0}

  !0 = !{i32 1, !"ProfileSummary", !1}
  !1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
  !2 = !{!"ProfileFormat", !"InstrProf"}
  !3 = !{!"TotalCount", i64 10000}
  !4 = !{!"MaxCount", i64 10}
  !5 = !{!"MaxInternalCount", i64 1}
  !6 = !{!"MaxFunctionCount", i64 1000}
  !7 = !{!"NumCounts", i64 3}
  !8 = !{!"NumFunctions", i64 3}
  !9 = !{!"DetailedSummary", !10}
  !10 = !{!11, !12, !13}
  !11 = !{i32 10000, i64 1000, i32 1}
  !12 = !{i32 999000, i64 300, i32 3}
  !13 = !{i32 999999, i64 5, i32 10}
  !14 = !{!"function_entry_count", i64 1}
  !15 = !{!"function_entry_count", i64 100}
  !16 = !{!"function_entry_count", i64 400}
  !17 = !{!"branch_weights", i32 100, i32 1}
)IR";

} // end anonymous namespace
