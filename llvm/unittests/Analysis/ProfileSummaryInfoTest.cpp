//===- ProfileSummaryInfoTest.cpp - ProfileSummaryInfo unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

extern llvm::cl::opt<bool> ScalePartialSampleProfileWorkingSetSize;

namespace llvm {
namespace {

class ProfileSummaryInfoTest : public testing::Test {
protected:
  LLVMContext C;
  std::unique_ptr<BranchProbabilityInfo> BPI;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;

  ProfileSummaryInfo buildPSI(Module *M) {
    return ProfileSummaryInfo(*M);
  }
  BlockFrequencyInfo buildBFI(Function &F) {
    DT.reset(new DominatorTree(F));
    LI.reset(new LoopInfo(*DT));
    BPI.reset(new BranchProbabilityInfo(F, *LI));
    return BlockFrequencyInfo(F, *BPI, *LI);
  }
  std::unique_ptr<Module> makeLLVMModule(const char *ProfKind = nullptr,
                                         uint64_t NumCounts = 3,
                                         uint64_t IsPartialProfile = 0,
                                         double PartialProfileRatio = 0.0,
                                         uint64_t HotNumCounts = 3,
                                         uint64_t ColdNumCounts = 10) {
    const char *ModuleString =
        "define i32 @g(i32 %x) !prof !21 {{\n"
        "  ret i32 0\n"
        "}\n"
        "define i32 @h(i32 %x) !prof !22 {{\n"
        "  ret i32 0\n"
        "}\n"
        "define i32 @f(i32 %x) !prof !20 {{\n"
        "bb0:\n"
        "  %y1 = icmp eq i32 %x, 0 \n"
        "  br i1 %y1, label %bb1, label %bb2, !prof !23 \n"
        "bb1:\n"
        "  %z1 = call i32 @g(i32 %x)\n"
        "  br label %bb3\n"
        "bb2:\n"
        "  %z2 = call i32 @h(i32 %x)\n"
        "  br label %bb3\n"
        "bb3:\n"
        "  %y2 = phi i32 [0, %bb1], [1, %bb2] \n"
        "  ret i32 %y2\n"
        "}\n"
        "define i32 @l(i32 %x) {{\n"
        "bb0:\n"
        "  %y1 = icmp eq i32 %x, 0 \n"
        "  br i1 %y1, label %bb1, label %bb2, !prof !23 \n"
        "bb1:\n"
        "  %z1 = call i32 @g(i32 %x)\n"
        "  br label %bb3\n"
        "bb2:\n"
        "  %z2 = call i32 @h(i32 %x)\n"
        "  br label %bb3\n"
        "bb3:\n"
        "  %y2 = phi i32 [0, %bb1], [1, %bb2] \n"
        "  ret i32 %y2\n"
        "}\n"
        "!20 = !{{!\"function_entry_count\", i64 400}\n"
        "!21 = !{{!\"function_entry_count\", i64 1}\n"
        "!22 = !{{!\"function_entry_count\", i64 100}\n"
        "!23 = !{{!\"branch_weights\", i32 64, i32 4}\n"
        "{0}";
    const char *SummaryString =
        "!llvm.module.flags = !{{!1}\n"
        "!1 = !{{i32 1, !\"ProfileSummary\", !2}\n"
        "!2 = !{{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}\n"
        "!3 = !{{!\"ProfileFormat\", !\"{0}\"}\n"
        "!4 = !{{!\"TotalCount\", i64 10000}\n"
        "!5 = !{{!\"MaxCount\", i64 10}\n"
        "!6 = !{{!\"MaxInternalCount\", i64 1}\n"
        "!7 = !{{!\"MaxFunctionCount\", i64 1000}\n"
        "!8 = !{{!\"NumCounts\", i64 {1}}\n"
        "!9 = !{{!\"NumFunctions\", i64 3}\n"
        "!10 = !{{!\"IsPartialProfile\", i64 {2}}\n"
        "!11 = !{{!\"PartialProfileRatio\", double {3}}\n"
        "!12 = !{{!\"DetailedSummary\", !13}\n"
        "!13 = !{{!14, !15, !16}\n"
        "!14 = !{{i32 10000, i64 1000, i32 1}\n"
        "!15 = !{{i32 990000, i64 300, i32 {4}}\n"
        "!16 = !{{i32 999999, i64 5, i32 {5}}\n";
    SMDiagnostic Err;
    if (ProfKind) {
      auto Summary =
          formatv(SummaryString, ProfKind, NumCounts, IsPartialProfile,
                  PartialProfileRatio, HotNumCounts, ColdNumCounts)
              .str();
      return parseAssemblyString(formatv(ModuleString, Summary).str(), Err, C);
    } else
      return parseAssemblyString(formatv(ModuleString, "").str(), Err, C);
  }
};

TEST_F(ProfileSummaryInfoTest, TestNoProfile) {
  auto M = makeLLVMModule(/*ProfKind=*/nullptr);
  Function *F = M->getFunction("f");

  ProfileSummaryInfo PSI = buildPSI(M.get());
  EXPECT_FALSE(PSI.hasProfileSummary());
  EXPECT_FALSE(PSI.hasSampleProfile());
  EXPECT_FALSE(PSI.hasInstrumentationProfile());
  // In the absence of profiles, is{Hot|Cold}X methods should always return
  // false.
  EXPECT_FALSE(PSI.isHotCount(1000));
  EXPECT_FALSE(PSI.isHotCount(0));
  EXPECT_FALSE(PSI.isColdCount(1000));
  EXPECT_FALSE(PSI.isColdCount(0));

  EXPECT_FALSE(PSI.isFunctionEntryHot(F));
  EXPECT_FALSE(PSI.isFunctionEntryCold(F));

  BasicBlock &BB0 = F->getEntryBlock();
  BasicBlock *BB1 = BB0.getTerminator()->getSuccessor(0);

  BlockFrequencyInfo BFI = buildBFI(*F);
  EXPECT_FALSE(PSI.isHotBlock(&BB0, &BFI));
  EXPECT_FALSE(PSI.isColdBlock(&BB0, &BFI));

  CallBase &CS1 = cast<CallBase>(*BB1->getFirstNonPHIIt());
  EXPECT_FALSE(PSI.isHotCallSite(CS1, &BFI));
  EXPECT_FALSE(PSI.isColdCallSite(CS1, &BFI));
}
TEST_F(ProfileSummaryInfoTest, TestCommon) {
  auto M = makeLLVMModule("InstrProf");
  Function *F = M->getFunction("f");
  Function *G = M->getFunction("g");
  Function *H = M->getFunction("h");

  ProfileSummaryInfo PSI = buildPSI(M.get());
  EXPECT_TRUE(PSI.hasProfileSummary());
  EXPECT_TRUE(PSI.isHotCount(400));
  EXPECT_TRUE(PSI.isColdCount(2));
  EXPECT_FALSE(PSI.isColdCount(100));
  EXPECT_FALSE(PSI.isHotCount(100));

  EXPECT_TRUE(PSI.isHotCountNthPercentile(990000, 400));
  EXPECT_FALSE(PSI.isHotCountNthPercentile(990000, 100));
  EXPECT_FALSE(PSI.isHotCountNthPercentile(990000, 2));

  EXPECT_FALSE(PSI.isColdCountNthPercentile(990000, 400));
  EXPECT_TRUE(PSI.isColdCountNthPercentile(990000, 100));
  EXPECT_TRUE(PSI.isColdCountNthPercentile(990000, 2));

  EXPECT_TRUE(PSI.isHotCountNthPercentile(999999, 400));
  EXPECT_TRUE(PSI.isHotCountNthPercentile(999999, 100));
  EXPECT_FALSE(PSI.isHotCountNthPercentile(999999, 2));

  EXPECT_FALSE(PSI.isColdCountNthPercentile(999999, 400));
  EXPECT_FALSE(PSI.isColdCountNthPercentile(999999, 100));
  EXPECT_TRUE(PSI.isColdCountNthPercentile(999999, 2));

  EXPECT_FALSE(PSI.isHotCountNthPercentile(10000, 400));
  EXPECT_FALSE(PSI.isHotCountNthPercentile(10000, 100));
  EXPECT_FALSE(PSI.isHotCountNthPercentile(10000, 2));

  EXPECT_TRUE(PSI.isColdCountNthPercentile(10000, 400));
  EXPECT_TRUE(PSI.isColdCountNthPercentile(10000, 100));
  EXPECT_TRUE(PSI.isColdCountNthPercentile(10000, 2));

  EXPECT_TRUE(PSI.isFunctionEntryHot(F));
  EXPECT_FALSE(PSI.isFunctionEntryHot(G));
  EXPECT_FALSE(PSI.isFunctionEntryHot(H));
}

TEST_F(ProfileSummaryInfoTest, InstrProf) {
  auto M = makeLLVMModule("InstrProf");
  Function *F = M->getFunction("f");
  ProfileSummaryInfo PSI = buildPSI(M.get());
  EXPECT_TRUE(PSI.hasProfileSummary());
  EXPECT_TRUE(PSI.hasInstrumentationProfile());

  BasicBlock &BB0 = F->getEntryBlock();
  BasicBlock *BB1 = BB0.getTerminator()->getSuccessor(0);
  BasicBlock *BB2 = BB0.getTerminator()->getSuccessor(1);
  BasicBlock *BB3 = BB1->getSingleSuccessor();

  BlockFrequencyInfo BFI = buildBFI(*F);
  EXPECT_TRUE(PSI.isHotBlock(&BB0, &BFI));
  EXPECT_TRUE(PSI.isHotBlock(BB1, &BFI));
  EXPECT_FALSE(PSI.isHotBlock(BB2, &BFI));
  EXPECT_TRUE(PSI.isHotBlock(BB3, &BFI));

  EXPECT_TRUE(PSI.isHotBlockNthPercentile(990000, &BB0, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(990000, BB1, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, BB2, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(990000, BB3, &BFI));

  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, &BB0, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB1, &BFI));
  EXPECT_TRUE(PSI.isColdBlockNthPercentile(990000, BB2, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB3, &BFI));

  EXPECT_TRUE(PSI.isHotBlockNthPercentile(999900, &BB0, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(999900, BB1, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(999900, BB2, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(999900, BB3, &BFI));

  EXPECT_FALSE(PSI.isColdBlockNthPercentile(999900, &BB0, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(999900, BB1, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(999900, BB2, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(999900, BB3, &BFI));

  EXPECT_FALSE(PSI.isHotBlockNthPercentile(10000, &BB0, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(10000, BB1, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(10000, BB2, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(10000, BB3, &BFI));

  EXPECT_TRUE(PSI.isColdBlockNthPercentile(10000, &BB0, &BFI));
  EXPECT_TRUE(PSI.isColdBlockNthPercentile(10000, BB1, &BFI));
  EXPECT_TRUE(PSI.isColdBlockNthPercentile(10000, BB2, &BFI));
  EXPECT_TRUE(PSI.isColdBlockNthPercentile(10000, BB3, &BFI));

  CallBase &CS1 = cast<CallBase>(*BB1->getFirstNonPHIIt());
  BasicBlock::iterator CI2 = BB2->getFirstNonPHIIt();
  CallBase &CS2 = cast<CallBase>(*CI2);

  EXPECT_TRUE(PSI.isHotCallSite(CS1, &BFI));
  EXPECT_FALSE(PSI.isHotCallSite(CS2, &BFI));

  // Test that adding an MD_prof metadata with a hot count on CS2 does not
  // change its hotness as it has no effect in instrumented profiling.
  MDBuilder MDB(M->getContext());
  CI2->setMetadata(llvm::LLVMContext::MD_prof, MDB.createBranchWeights({400}));
  EXPECT_FALSE(PSI.isHotCallSite(CS2, &BFI));

  EXPECT_TRUE(PSI.isFunctionHotInCallGraphNthPercentile(990000, F, BFI));
  EXPECT_FALSE(PSI.isFunctionColdInCallGraphNthPercentile(990000, F, BFI));
  EXPECT_FALSE(PSI.isFunctionHotInCallGraphNthPercentile(10000, F, BFI));
  EXPECT_TRUE(PSI.isFunctionColdInCallGraphNthPercentile(10000, F, BFI));
}

TEST_F(ProfileSummaryInfoTest, InstrProfNoFuncEntryCount) {
  auto M = makeLLVMModule("InstrProf");
  Function *F = M->getFunction("l");
  ProfileSummaryInfo PSI = buildPSI(M.get());
  EXPECT_TRUE(PSI.hasProfileSummary());
  EXPECT_TRUE(PSI.hasInstrumentationProfile());

  BasicBlock &BB0 = F->getEntryBlock();
  BasicBlock *BB1 = BB0.getTerminator()->getSuccessor(0);
  BasicBlock *BB2 = BB0.getTerminator()->getSuccessor(1);
  BasicBlock *BB3 = BB1->getSingleSuccessor();

  BlockFrequencyInfo BFI = buildBFI(*F);

  // Without the entry count, all should return false.
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, &BB0, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, BB1, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, BB2, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, BB3, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, &BB0, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB1, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB2, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB3, &BFI));

  EXPECT_FALSE(PSI.isFunctionHotInCallGraphNthPercentile(990000, F, BFI));
  EXPECT_FALSE(PSI.isFunctionColdInCallGraphNthPercentile(990000, F, BFI));
}

TEST_F(ProfileSummaryInfoTest, SampleProf) {
  auto M = makeLLVMModule("SampleProfile");
  Function *F = M->getFunction("f");
  ProfileSummaryInfo PSI = buildPSI(M.get());
  EXPECT_TRUE(PSI.hasProfileSummary());
  EXPECT_TRUE(PSI.hasSampleProfile());
  EXPECT_FALSE(PSI.hasPartialSampleProfile());

  BasicBlock &BB0 = F->getEntryBlock();
  BasicBlock *BB1 = BB0.getTerminator()->getSuccessor(0);
  BasicBlock *BB2 = BB0.getTerminator()->getSuccessor(1);
  BasicBlock *BB3 = BB1->getSingleSuccessor();

  BlockFrequencyInfo BFI = buildBFI(*F);
  EXPECT_TRUE(PSI.isHotBlock(&BB0, &BFI));
  EXPECT_TRUE(PSI.isHotBlock(BB1, &BFI));
  EXPECT_FALSE(PSI.isHotBlock(BB2, &BFI));
  EXPECT_TRUE(PSI.isHotBlock(BB3, &BFI));

  EXPECT_TRUE(PSI.isHotBlockNthPercentile(990000, &BB0, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(990000, BB1, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, BB2, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(990000, BB3, &BFI));

  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, &BB0, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB1, &BFI));
  EXPECT_TRUE(PSI.isColdBlockNthPercentile(990000, BB2, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB3, &BFI));

  EXPECT_TRUE(PSI.isHotBlockNthPercentile(999900, &BB0, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(999900, BB1, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(999900, BB2, &BFI));
  EXPECT_TRUE(PSI.isHotBlockNthPercentile(999900, BB3, &BFI));

  EXPECT_FALSE(PSI.isColdBlockNthPercentile(999900, &BB0, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(999900, BB1, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(999900, BB2, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(999900, BB3, &BFI));

  EXPECT_FALSE(PSI.isHotBlockNthPercentile(10000, &BB0, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(10000, BB1, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(10000, BB2, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(10000, BB3, &BFI));

  EXPECT_TRUE(PSI.isColdBlockNthPercentile(10000, &BB0, &BFI));
  EXPECT_TRUE(PSI.isColdBlockNthPercentile(10000, BB1, &BFI));
  EXPECT_TRUE(PSI.isColdBlockNthPercentile(10000, BB2, &BFI));
  EXPECT_TRUE(PSI.isColdBlockNthPercentile(10000, BB3, &BFI));

  CallBase &CS1 = cast<CallBase>(*BB1->getFirstNonPHIIt());
  BasicBlock::iterator CI2 = BB2->getFirstNonPHIIt();
  // Manually attach branch weights metadata to the call instruction.
  SmallVector<uint32_t, 1> Weights;
  Weights.push_back(1000);
  MDBuilder MDB(M->getContext());
  CI2->setMetadata(LLVMContext::MD_prof, MDB.createBranchWeights(Weights));
  CallBase &CS2 = cast<CallBase>(*CI2);

  EXPECT_FALSE(PSI.isHotCallSite(CS1, &BFI));
  EXPECT_TRUE(PSI.isHotCallSite(CS2, &BFI));

  // Test that CS2 is considered hot when it gets an MD_prof metadata with
  // weights that exceed the hot count threshold.
  CI2->setMetadata(llvm::LLVMContext::MD_prof, MDB.createBranchWeights({400}));
  EXPECT_TRUE(PSI.isHotCallSite(CS2, &BFI));

  EXPECT_TRUE(PSI.isFunctionHotInCallGraphNthPercentile(990000, F, BFI));
  EXPECT_FALSE(PSI.isFunctionColdInCallGraphNthPercentile(990000, F, BFI));
  EXPECT_FALSE(PSI.isFunctionHotInCallGraphNthPercentile(10000, F, BFI));
  EXPECT_TRUE(PSI.isFunctionColdInCallGraphNthPercentile(10000, F, BFI));
}

TEST_F(ProfileSummaryInfoTest, SampleProfNoFuncEntryCount) {
  auto M = makeLLVMModule("SampleProfile");
  Function *F = M->getFunction("l");
  ProfileSummaryInfo PSI = buildPSI(M.get());
  EXPECT_TRUE(PSI.hasProfileSummary());
  EXPECT_TRUE(PSI.hasSampleProfile());

  BasicBlock &BB0 = F->getEntryBlock();
  BasicBlock *BB1 = BB0.getTerminator()->getSuccessor(0);
  BasicBlock *BB2 = BB0.getTerminator()->getSuccessor(1);
  BasicBlock *BB3 = BB1->getSingleSuccessor();

  BlockFrequencyInfo BFI = buildBFI(*F);

  // Without the entry count, all should return false.
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, &BB0, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, BB1, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, BB2, &BFI));
  EXPECT_FALSE(PSI.isHotBlockNthPercentile(990000, BB3, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, &BB0, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB1, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB2, &BFI));
  EXPECT_FALSE(PSI.isColdBlockNthPercentile(990000, BB3, &BFI));

  EXPECT_FALSE(PSI.isFunctionHotInCallGraphNthPercentile(990000, F, BFI));
  EXPECT_FALSE(PSI.isFunctionColdInCallGraphNthPercentile(990000, F, BFI));
}

TEST_F(ProfileSummaryInfoTest, PartialSampleProfWorkingSetSize) {
  ScalePartialSampleProfileWorkingSetSize.setValue(true);

  // With PartialProfileRatio unset (zero.)
  auto M1 = makeLLVMModule("SampleProfile", /*NumCounts*/ 3,
                           /*IsPartialProfile*/ 1,
                           /*PartialProfileRatio*/ 0.0,
                           /*HotNumCounts*/ 3, /*ColdNumCounts*/ 10);
  ProfileSummaryInfo PSI1 = buildPSI(M1.get());
  EXPECT_TRUE(PSI1.hasProfileSummary());
  EXPECT_TRUE(PSI1.hasSampleProfile());
  EXPECT_TRUE(PSI1.hasPartialSampleProfile());
  EXPECT_FALSE(PSI1.hasHugeWorkingSetSize());
  EXPECT_FALSE(PSI1.hasLargeWorkingSetSize());

  // With PartialProfileRatio set (non-zero) and a small working set size.
  auto M2 = makeLLVMModule("SampleProfile", /*NumCounts*/ 27493235,
                           /*IsPartialProfile*/ 1,
                           /*PartialProfileRatio*/ 0.00000012,
                           /*HotNumCounts*/ 3102082,
                           /*ColdNumCounts*/ 18306149);
  ProfileSummaryInfo PSI2 = buildPSI(M2.get());
  EXPECT_TRUE(PSI2.hasProfileSummary());
  EXPECT_TRUE(PSI2.hasSampleProfile());
  EXPECT_TRUE(PSI2.hasPartialSampleProfile());
  EXPECT_FALSE(PSI2.hasHugeWorkingSetSize());
  EXPECT_FALSE(PSI2.hasLargeWorkingSetSize());

  // With PartialProfileRatio is set (non-zero) and a large working set size.
  auto M3 = makeLLVMModule("SampleProfile", /*NumCounts*/ 27493235,
                           /*IsPartialProfile*/ 1,
                           /*PartialProfileRatio*/ 0.9,
                           /*HotNumCounts*/ 3102082,
                           /*ColdNumCounts*/ 18306149);
  ProfileSummaryInfo PSI3 = buildPSI(M3.get());
  EXPECT_TRUE(PSI3.hasProfileSummary());
  EXPECT_TRUE(PSI3.hasSampleProfile());
  EXPECT_TRUE(PSI3.hasPartialSampleProfile());
  EXPECT_TRUE(PSI3.hasHugeWorkingSetSize());
  EXPECT_TRUE(PSI3.hasLargeWorkingSetSize());
}

} // end anonymous namespace
} // end namespace llvm
