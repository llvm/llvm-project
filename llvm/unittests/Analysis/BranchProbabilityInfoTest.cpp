//===- BranchProbabilityInfoTest.cpp - BranchProbabilityInfo unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

struct BranchProbabilityInfoTest : public testing::Test {
  std::unique_ptr<BranchProbabilityInfo> BPI;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;
  LLVMContext C;

  BranchProbabilityInfo &buildBPI(Function &F) {
    DT.reset(new DominatorTree(F));
    LI.reset(new LoopInfo(*DT));
    BPI.reset(new BranchProbabilityInfo(F, *LI));
    return *BPI;
  }

  std::unique_ptr<Module> makeLLVMModule() {
    const char *ModuleString = "define void @f() { exit: ret void }\n";
    SMDiagnostic Err;
    return parseAssemblyString(ModuleString, Err, C);
  }
};

TEST_F(BranchProbabilityInfoTest, StressUnreachableHeuristic) {
  auto M = makeLLVMModule();
  Function *F = M->getFunction("f");

  // define void @f() {
  // entry:
  //   switch i32 undef, label %exit, [
  //      i32 0, label %preexit
  //      ...                   ;;< Add lots of cases to stress the heuristic.
  //   ]
  // preexit:
  //   unreachable
  // exit:
  //   ret void
  // }

  auto *ExitBB = &F->back();
  auto *EntryBB = BasicBlock::Create(C, "entry", F, /*insertBefore=*/ExitBB);

  auto *PreExitBB =
      BasicBlock::Create(C, "preexit", F, /*insertBefore=*/ExitBB);
  new UnreachableInst(C, PreExitBB);

  unsigned NumCases = 4096;
  auto *I32 = IntegerType::get(C, 32);
  auto *Undef = UndefValue::get(I32);
  auto *Switch = SwitchInst::Create(Undef, ExitBB, NumCases, EntryBB);
  for (unsigned I = 0; I < NumCases; ++I)
    Switch->addCase(ConstantInt::get(I32, I), PreExitBB);

  BranchProbabilityInfo &BPI = buildBPI(*F);

  // FIXME: This doesn't seem optimal. Since all of the cases handled by the
  // switch have the *same* destination block ("preexit"), shouldn't it be the
  // hot one? I'd expect the results to be reversed here...
  EXPECT_FALSE(BPI.isEdgeHot(EntryBB, PreExitBB));
  EXPECT_TRUE(BPI.isEdgeHot(EntryBB, ExitBB));
}

TEST_F(BranchProbabilityInfoTest, SwapProbabilities) {
  StringRef Assembly = R"(
    define void @f() {
    entry:
      br label %loop

    loop:
      %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
      %iv.next = add i32 %iv, 1
      %cond = icmp slt i32 %iv.next, 10
      br i1 %cond, label %exit, label %loop

    exit:
      ret void
    }
  )";
  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  Function *F = M->getFunction("f");
  auto *LoopHeaderBB = F->front().getSingleSuccessor();
  ASSERT_TRUE(LoopHeaderBB != nullptr);
  BranchInst *Branch = dyn_cast<BranchInst>(LoopHeaderBB->getTerminator());
  ASSERT_TRUE(Branch != nullptr);
  // Save the probabilities before successors swapping
  BranchProbabilityInfo *BPI = &buildBPI(*F);
  auto ProbEdge0 = BPI->getEdgeProbability(LoopHeaderBB, 0U);
  auto ProbEdge1 = BPI->getEdgeProbability(LoopHeaderBB, 1U);
  EXPECT_LT(ProbEdge0, ProbEdge1);

  Branch->swapSuccessors();
  BPI->swapSuccEdgesProbabilities(LoopHeaderBB);
  // TODO: Check the probabilities are swapped as well as the edges
  EXPECT_EQ(ProbEdge0, BPI->getEdgeProbability(LoopHeaderBB, 1U));
  EXPECT_EQ(ProbEdge1, BPI->getEdgeProbability(LoopHeaderBB, 0U));
}

} // end anonymous namespace
} // end namespace llvm
