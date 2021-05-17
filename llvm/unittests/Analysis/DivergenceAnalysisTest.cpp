//===- DivergenceAnalysisTest.cpp - DivergenceAnalysis unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/SyncDependenceAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

BasicBlock *GetBlockByName(StringRef BlockName, Function &F) {
  for (auto &BB : F) {
    if (BB.getName() != BlockName)
      continue;
    return &BB;
  }
  return nullptr;
}

// We use this fixture to ensure that we clean up DivergenceAnalysisImpl before
// deleting the PassManager.
class DivergenceAnalysisTest : public testing::Test {
protected:
  LLVMContext Context;
  Module M;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;

  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<PostDominatorTree> PDT;
  std::unique_ptr<LoopInfo> LI;
  std::unique_ptr<SyncDependenceAnalysis> SDA;

  DivergenceAnalysisTest() : M("", Context), TLII(), TLI(TLII) {}

  DivergenceAnalysisImpl buildDA(Function &F, bool IsLCSSA) {
    DT.reset(new DominatorTree(F));
    PDT.reset(new PostDominatorTree(F));
    LI.reset(new LoopInfo(*DT));
    SDA.reset(new SyncDependenceAnalysis(*DT, *PDT, *LI));
    return DivergenceAnalysisImpl(F, nullptr, *DT, *LI, *SDA, IsLCSSA);
  }

  void runWithDA(
      Module &M, StringRef FuncName, bool IsLCSSA,
      function_ref<void(Function &F, LoopInfo &LI, DivergenceAnalysisImpl &DA)>
          Test) {
    auto *F = M.getFunction(FuncName);
    ASSERT_NE(F, nullptr) << "Could not find " << FuncName;
    DivergenceAnalysisImpl DA = buildDA(*F, IsLCSSA);
    Test(*F, *LI, DA);
  }
};

// Simple initial state test
TEST_F(DivergenceAnalysisTest, DAInitialState) {
  IntegerType *IntTy = IntegerType::getInt32Ty(Context);
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), {IntTy}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);
  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  ReturnInst::Create(Context, nullptr, BB);

  DivergenceAnalysisImpl DA = buildDA(*F, false);

  // Whole function region
  EXPECT_EQ(DA.getRegionLoop(), nullptr);

  // No divergence in initial state
  EXPECT_FALSE(DA.hasDetectedDivergence());

  // No spurious divergence
  DA.compute();
  EXPECT_FALSE(DA.hasDetectedDivergence());

  // Detected divergence after marking
  Argument &arg = *F->arg_begin();
  DA.markDivergent(arg);

  EXPECT_TRUE(DA.hasDetectedDivergence());
  EXPECT_TRUE(DA.isDivergent(arg));

  DA.compute();
  EXPECT_TRUE(DA.hasDetectedDivergence());
  EXPECT_TRUE(DA.isDivergent(arg));
}

TEST_F(DivergenceAnalysisTest, DANoLCSSA) {
  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> M = parseAssemblyString(
      "target datalayout = \"e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128\" "
      " "
      "define i32 @f_1(i8* nocapture %arr, i32 %n, i32* %A, i32* %B) "
      "    local_unnamed_addr { "
      "entry: "
      "  br label %loop.ph "
      " "
      "loop.ph: "
      "  br label %loop "
      " "
      "loop: "
      "  %iv0 = phi i32 [ %iv0.inc, %loop ], [ 0, %loop.ph ] "
      "  %iv1 = phi i32 [ %iv1.inc, %loop ], [ -2147483648, %loop.ph ] "
      "  %iv0.inc = add i32 %iv0, 1 "
      "  %iv1.inc = add i32 %iv1, 3 "
      "  %cond.cont = icmp slt i32 %iv0, %n "
      "  br i1 %cond.cont, label %loop, label %for.end.loopexit "
      " "
      "for.end.loopexit: "
      "  ret i32 %iv0 "
      "} ",
      Err, C);

  Function *F = M->getFunction("f_1");
  DivergenceAnalysisImpl DA = buildDA(*F, false);
  EXPECT_FALSE(DA.hasDetectedDivergence());

  auto ItArg = F->arg_begin();
  ItArg++;
  auto &NArg = *ItArg;

  // Seed divergence in argument %n
  DA.markDivergent(NArg);

  DA.compute();
  EXPECT_TRUE(DA.hasDetectedDivergence());

  // Verify that "ret %iv.0" is divergent
  auto ItBlock = F->begin();
  std::advance(ItBlock, 3);
  auto &ExitBlock = *GetBlockByName("for.end.loopexit", *F);
  auto &RetInst = *cast<ReturnInst>(ExitBlock.begin());
  EXPECT_TRUE(DA.isDivergent(RetInst));
}

TEST_F(DivergenceAnalysisTest, DALCSSA) {
  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> M = parseAssemblyString(
      "target datalayout = \"e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128\" "
      " "
      "define i32 @f_lcssa(i8* nocapture %arr, i32 %n, i32* %A, i32* %B) "
      "    local_unnamed_addr { "
      "entry: "
      "  br label %loop.ph "
      " "
      "loop.ph: "
      "  br label %loop "
      " "
      "loop: "
      "  %iv0 = phi i32 [ %iv0.inc, %loop ], [ 0, %loop.ph ] "
      "  %iv1 = phi i32 [ %iv1.inc, %loop ], [ -2147483648, %loop.ph ] "
      "  %iv0.inc = add i32 %iv0, 1 "
      "  %iv1.inc = add i32 %iv1, 3 "
      "  %cond.cont = icmp slt i32 %iv0, %n "
      "  br i1 %cond.cont, label %loop, label %for.end.loopexit "
      " "
      "for.end.loopexit: "
      "  %val.ret = phi i32 [ %iv0, %loop ] "
      "  br label %detached.return "
      " "
      "detached.return: "
      "  ret i32 %val.ret "
      "} ",
      Err, C);

  Function *F = M->getFunction("f_lcssa");
  DivergenceAnalysisImpl DA = buildDA(*F, true);
  EXPECT_FALSE(DA.hasDetectedDivergence());

  auto ItArg = F->arg_begin();
  ItArg++;
  auto &NArg = *ItArg;

  // Seed divergence in argument %n
  DA.markDivergent(NArg);

  DA.compute();
  EXPECT_TRUE(DA.hasDetectedDivergence());

  // Verify that "ret %iv.0" is divergent
  auto ItBlock = F->begin();
  std::advance(ItBlock, 4);
  auto &ExitBlock = *GetBlockByName("detached.return", *F);
  auto &RetInst = *cast<ReturnInst>(ExitBlock.begin());
  EXPECT_TRUE(DA.isDivergent(RetInst));
}

TEST_F(DivergenceAnalysisTest, DAJoinDivergence) {
  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> M = parseAssemblyString(
      "target datalayout = \"e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128\" "
      " "
      "define void @f_1(i1 %a, i1 %b, i1 %c) "
      "    local_unnamed_addr { "
      "A: "
      "  br i1 %a, label %B, label %C "
      " "
      "B: "
      "  br i1 %b, label %C, label %D "
      " "
      "C: "
      "  %c.join = phi i32 [ 0, %A ], [ 1, %B ] "
      "  br i1 %c, label %D, label %E "
      " "
      "D: "
      "  %d.join = phi i32 [ 0, %B ], [ 1, %C ] "
      "  br label %E "
      " "
      "E: "
      "  %e.join = phi i32 [ 0, %C ], [ 1, %D ] "
      "  ret void "
      "} "
      " "
      "define void @f_2(i1 %a, i1 %b, i1 %c) "
      "    local_unnamed_addr { "
      "A: "
      "  br i1 %a, label %B, label %E "
      " "
      "B: "
      "  br i1 %b, label %C, label %D "
      " "
      "C: "
      "  br label %D "
      " "
      "D: "
      "  %d.join = phi i32 [ 0, %B ], [ 1, %C ] "
      "  br label %E "
      " "
      "E: "
      "  %e.join = phi i32 [ 0, %A ], [ 1, %D ] "
      "  ret void "
      "} "
      " "
      "define void @f_3(i1 %a, i1 %b, i1 %c)"
      "    local_unnamed_addr { "
      "A: "
      "  br i1 %a, label %B, label %C "
      " "
      "B: "
      "  br label %C "
      " "
      "C: "
      "  %c.join = phi i32 [ 0, %A ], [ 1, %B ] "
      "  br i1 %c, label %D, label %E "
      " "
      "D: "
      "  br label %E "
      " "
      "E: "
      "  %e.join = phi i32 [ 0, %C ], [ 1, %D ] "
      "  ret void "
      "} ",
      Err, C);

  // Maps divergent conditions to the basic blocks whose Phi nodes become
  // divergent. Blocks need to be listed in IR order.
  using SmallBlockVec = SmallVector<const BasicBlock *, 4>;
  using InducedDivJoinMap = std::map<const Value *, SmallBlockVec>;

  // Actual function performing the checks.
  auto CheckDivergenceFunc = [this](Function &F,
                                    InducedDivJoinMap &ExpectedDivJoins) {
    for (auto &ItCase : ExpectedDivJoins) {
      auto *DivVal = ItCase.first;
      auto DA = buildDA(F, false);
      DA.markDivergent(*DivVal);
      DA.compute();

      // List of basic blocks that shall host divergent Phi nodes.
      auto ItDivJoins = ItCase.second.begin();

      for (auto &BB : F) {
        auto *Phi = dyn_cast<PHINode>(BB.begin());
        if (!Phi)
          continue;

        if (ItDivJoins != ItCase.second.end() && &BB == *ItDivJoins) {
          EXPECT_TRUE(DA.isDivergent(*Phi));
          // Advance to next block with expected divergent PHI node.
          ++ItDivJoins;
        } else {
          EXPECT_FALSE(DA.isDivergent(*Phi));
        }
      }
    }
  };

  {
    auto *F = M->getFunction("f_1");
    auto ItBlocks = F->begin();
    ItBlocks++; // Skip A
    ItBlocks++; // Skip B
    auto *C = &*ItBlocks++;
    auto *D = &*ItBlocks++;
    auto *E = &*ItBlocks;

    auto ItArg = F->arg_begin();
    auto *AArg = &*ItArg++;
    auto *BArg = &*ItArg++;
    auto *CArg = &*ItArg;

    InducedDivJoinMap DivJoins;
    DivJoins.emplace(AArg, SmallBlockVec({C, D, E}));
    DivJoins.emplace(BArg, SmallBlockVec({D, E}));
    DivJoins.emplace(CArg, SmallBlockVec({E}));

    CheckDivergenceFunc(*F, DivJoins);
  }

  {
    auto *F = M->getFunction("f_2");
    auto ItBlocks = F->begin();
    ItBlocks++; // Skip A
    ItBlocks++; // Skip B
    ItBlocks++; // Skip C
    auto *D = &*ItBlocks++;
    auto *E = &*ItBlocks;

    auto ItArg = F->arg_begin();
    auto *AArg = &*ItArg++;
    auto *BArg = &*ItArg++;
    auto *CArg = &*ItArg;

    InducedDivJoinMap DivJoins;
    DivJoins.emplace(AArg, SmallBlockVec({E}));
    DivJoins.emplace(BArg, SmallBlockVec({D}));
    DivJoins.emplace(CArg, SmallBlockVec({}));

    CheckDivergenceFunc(*F, DivJoins);
  }

  {
    auto *F = M->getFunction("f_3");
    auto ItBlocks = F->begin();
    ItBlocks++; // Skip A
    ItBlocks++; // Skip B
    auto *C = &*ItBlocks++;
    ItBlocks++; // Skip D
    auto *E = &*ItBlocks;

    auto ItArg = F->arg_begin();
    auto *AArg = &*ItArg++;
    auto *BArg = &*ItArg++;
    auto *CArg = &*ItArg;

    InducedDivJoinMap DivJoins;
    DivJoins.emplace(AArg, SmallBlockVec({C}));
    DivJoins.emplace(BArg, SmallBlockVec({}));
    DivJoins.emplace(CArg, SmallBlockVec({E}));

    CheckDivergenceFunc(*F, DivJoins);
  }
}

TEST_F(DivergenceAnalysisTest, DASwitchUnreachableDefault) {
  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> M = parseAssemblyString(
      "target datalayout = \"e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128\" "
      " "
      "define void @switch_unreachable_default(i32 %cond) local_unnamed_addr { "
      "entry: "
      "  switch i32 %cond, label %sw.default [ "
      "    i32 0, label %sw.bb0 "
      "    i32 1, label %sw.bb1 "
      "  ] "
      " "
      "sw.bb0: "
      "  br label %sw.epilog "
      " "
      "sw.bb1: "
      "  br label %sw.epilog "
      " "
      "sw.default: "
      "  unreachable "
      " "
      "sw.epilog: "
      "  %div.dbl = phi double [ 0.0, %sw.bb0], [ -1.0, %sw.bb1 ] "
      "  ret void "
      "}",
      Err, C);

  auto *F = M->getFunction("switch_unreachable_default");
  auto &CondArg = *F->arg_begin();
  auto DA = buildDA(*F, false);

  EXPECT_FALSE(DA.hasDetectedDivergence());

  DA.markDivergent(CondArg);
  DA.compute();

  // Still %CondArg is divergent.
  EXPECT_TRUE(DA.hasDetectedDivergence());

  // The join uni.dbl is not divergent (see D52221)
  auto &ExitBlock = *GetBlockByName("sw.epilog", *F);
  auto &DivDblPhi = *cast<PHINode>(ExitBlock.begin());
  EXPECT_TRUE(DA.isDivergent(DivDblPhi));
}

} // end anonymous namespace
} // end namespace llvm
