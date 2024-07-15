//===- BasicBlockUtils.cpp - Unit tests for BasicBlockUtils ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("BasicBlockUtilsTests", errs());
  return Mod;
}

static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

TEST(BasicBlockUtils, EliminateUnreachableBlocks) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define i32 @has_unreachable(i1 %cond) {
entry:
  br i1 %cond, label %bb0, label %bb1
bb0:
  br label %bb1
bb1:
  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]
  ret i32 %phi
bb2:
  ret i32 42
}
)IR");
  Function *F = M->getFunction("has_unreachable");
  DominatorTree DT(*F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

  EXPECT_EQ(F->size(), (size_t)4);
  bool Result = EliminateUnreachableBlocks(*F, &DTU);
  EXPECT_TRUE(Result);
  EXPECT_EQ(F->size(), (size_t)3);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, SplitEdge_ex1) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i1 %cond0) {
entry:
  br i1 %cond0, label %bb0, label %bb1
bb0:
 %0 = mul i32 1, 2
  br label %bb1
bb1:
  br label %bb2
bb2:
  ret void
}
)IR");
  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);
  BasicBlock *SrcBlock;
  BasicBlock *DestBlock;
  BasicBlock *NewBB;

  SrcBlock = getBasicBlockByName(*F, "entry");
  DestBlock = getBasicBlockByName(*F, "bb0");
  NewBB = SplitEdge(SrcBlock, DestBlock, &DT, nullptr, nullptr);

  EXPECT_TRUE(DT.verify());
  EXPECT_EQ(NewBB->getSinglePredecessor(), SrcBlock);
  EXPECT_EQ(NewBB->getSingleSuccessor(), DestBlock);
  EXPECT_EQ(NewBB->getParent(), F);

  bool BBFlag = false;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == NewBB->getName()) {
      BBFlag = true;
    }
  }
  EXPECT_TRUE(BBFlag);
}

TEST(BasicBlockUtils, SplitEdge_ex2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo() {
bb0:
  br label %bb2
bb1:
  br label %bb2
bb2:
  ret void
}
)IR");
  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);

  BasicBlock *SrcBlock;
  BasicBlock *DestBlock;
  BasicBlock *NewBB;

  SrcBlock = getBasicBlockByName(*F, "bb0");
  DestBlock = getBasicBlockByName(*F, "bb2");
  NewBB = SplitEdge(SrcBlock, DestBlock, &DT, nullptr, nullptr);

  EXPECT_TRUE(DT.verify());
  EXPECT_EQ(NewBB->getSinglePredecessor(), SrcBlock);
  EXPECT_EQ(NewBB->getSingleSuccessor(), DestBlock);
  EXPECT_EQ(NewBB->getParent(), F);

  bool BBFlag = false;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == NewBB->getName()) {
      BBFlag = true;
    }
  }
  EXPECT_TRUE(BBFlag);
}

TEST(BasicBlockUtils, SplitEdge_ex3) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define i32 @foo(i32 %n) {
entry:
 br label %header
header:
 %sum.02 = phi i32 [ 0, %entry ], [ %sum.1, %bb3 ]
 %0 = phi i32 [ 0, %entry ], [ %4, %bb3 ]
 %1 = icmp slt i32 %0, %n
 br i1 %1, label %bb0, label %bb1
bb0:
  %2 = add nsw i32 %sum.02, 2
  br label %bb2
bb1:
  %3 = add nsw i32 %sum.02, 1
  br label %bb2
bb2:
  %sum.1 = phi i32 [ %2, %bb0 ], [ %3, %bb1 ]
  br label %bb3
bb3:
  %4 = add nsw i32 %0, 1
  %5 = icmp slt i32 %4, 100
  br i1 %5, label %header, label %bb4
bb4:
 %sum.0.lcssa = phi i32 [ %sum.1, %bb3 ]
 ret i32 %sum.0.lcssa
}
)IR");
  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);

  LoopInfo LI(DT);

  DataLayout DL("e-i64:64-f80:128-n8:16:32:64-S128");
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  AAResults AA(TLI);

  BasicAAResult BAA(DL, *F, TLI, AC, &DT);
  AA.addAAResult(BAA);

  MemorySSA MSSA(*F, &AA, &DT);
  MemorySSAUpdater Updater(&MSSA);

  BasicBlock *SrcBlock;
  BasicBlock *DestBlock;
  BasicBlock *NewBB;

  SrcBlock = getBasicBlockByName(*F, "header");
  DestBlock = getBasicBlockByName(*F, "bb0");
  NewBB = SplitEdge(SrcBlock, DestBlock, &DT, &LI, &Updater);

  Updater.getMemorySSA()->verifyMemorySSA();
  EXPECT_TRUE(DT.verify());
  EXPECT_NE(LI.getLoopFor(SrcBlock), nullptr);
  EXPECT_NE(LI.getLoopFor(DestBlock), nullptr);
  EXPECT_NE(LI.getLoopFor(NewBB), nullptr);
  EXPECT_EQ(NewBB->getSinglePredecessor(), SrcBlock);
  EXPECT_EQ(NewBB->getSingleSuccessor(), DestBlock);
  EXPECT_EQ(NewBB->getParent(), F);

  bool BBFlag = false;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == NewBB->getName()) {
      BBFlag = true;
    }
  }
  EXPECT_TRUE(BBFlag);
}

TEST(BasicBlockUtils, SplitEdge_ex4) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @bar(i32 %cond) personality i8 0 {
entry:
  switch i32 %cond, label %exit [
    i32 -1, label %continue
    i32 0, label %continue
    i32 1, label %continue_alt
    i32 2, label %continue_alt
  ]
exit:
  ret void
continue:
  invoke void @sink() to label %normal unwind label %exception
continue_alt:
  invoke void @sink_alt() to label %normal unwind label %exception
exception:
  %cleanup = landingpad i8 cleanup
  br label %trivial-eh-handler
trivial-eh-handler:
  call void @sideeffect(i32 1)
  br label %normal
normal:
  call void @sideeffect(i32 0)
  ret void
}

declare void @sideeffect(i32)
declare void @sink() cold
declare void @sink_alt() cold
)IR");
  Function *F = M->getFunction("bar");

  DominatorTree DT(*F);

  LoopInfo LI(DT);

  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);

  AAResults AA(TLI);

  MemorySSA MSSA(*F, &AA, &DT);
  MemorySSAUpdater MSSAU(&MSSA);

  BasicBlock *SrcBlock;
  BasicBlock *DestBlock;

  SrcBlock = getBasicBlockByName(*F, "continue");
  DestBlock = getBasicBlockByName(*F, "exception");

  unsigned SuccNum = GetSuccessorNumber(SrcBlock, DestBlock);
  Instruction *LatchTerm = SrcBlock->getTerminator();

  const CriticalEdgeSplittingOptions Options =
      CriticalEdgeSplittingOptions(&DT, &LI, &MSSAU);

  // Check that the following edge is both critical and the destination block is
  // an exception block. These must be handled differently by SplitEdge
  bool CriticalEdge =
      isCriticalEdge(LatchTerm, SuccNum, Options.MergeIdenticalEdges);
  EXPECT_TRUE(CriticalEdge);

  bool Ehpad = DestBlock->isEHPad();
  EXPECT_TRUE(Ehpad);

  BasicBlock *NewBB = SplitEdge(SrcBlock, DestBlock, &DT, &LI, &MSSAU, "");

  MSSA.verifyMemorySSA();
  EXPECT_TRUE(DT.verify());
  EXPECT_NE(NewBB, nullptr);
  EXPECT_EQ(NewBB->getSinglePredecessor(), SrcBlock);
  EXPECT_EQ(NewBB, SrcBlock->getTerminator()->getSuccessor(SuccNum));
  EXPECT_EQ(NewBB->getParent(), F);

  bool BBFlag = false;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == NewBB->getName()) {
      BBFlag = true;
      break;
    }
  }
  EXPECT_TRUE(BBFlag);
}

TEST(BasicBlockUtils, splitBasicBlockBefore_ex1) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo() {
bb0:
 %0 = mul i32 1, 2
  br label %bb2
bb1:
  br label %bb3
bb2:
  %1 = phi  i32 [ %0, %bb0 ]
  br label %bb3
bb3:
  ret void
}
)IR");
  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);

  BasicBlock *DestBlock;
  BasicBlock *NewBB;

  DestBlock = getBasicBlockByName(*F, "bb2");

  NewBB = DestBlock->splitBasicBlockBefore(DestBlock->front().getIterator(),
                                           "test");

  PHINode *PN = dyn_cast<PHINode>(&(DestBlock->front()));
  EXPECT_EQ(PN->getIncomingBlock(0), NewBB);
  EXPECT_EQ(NewBB->getName(), "test");
  EXPECT_EQ(NewBB->getSingleSuccessor(), DestBlock);
  EXPECT_EQ(DestBlock->getSinglePredecessor(), NewBB);
}

#ifndef NDEBUG
TEST(BasicBlockUtils, splitBasicBlockBefore_ex2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo() {
bb0:
 %0 = mul i32 1, 2
  br label %bb2
bb1:
  br label %bb2
bb2:
  %1 = phi  i32 [ %0, %bb0 ], [ 1, %bb1 ]
  br label %bb3
bb3:
  ret void
}
)IR");
  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);

  BasicBlock *DestBlock = getBasicBlockByName(*F, "bb2");

  ASSERT_DEATH(
      {
        DestBlock->splitBasicBlockBefore(DestBlock->front().getIterator(),
                                         "test");
      },
      "cannot split on multi incoming phis");
}
#endif

TEST(BasicBlockUtils, NoUnreachableBlocksToEliminate) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define i32 @no_unreachable(i1 %cond) {
entry:
  br i1 %cond, label %bb0, label %bb1
bb0:
  br label %bb1
bb1:
  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]
  ret i32 %phi
}
)IR");
  Function *F = M->getFunction("no_unreachable");
  DominatorTree DT(*F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

  EXPECT_EQ(F->size(), (size_t)3);
  bool Result = EliminateUnreachableBlocks(*F, &DTU);
  EXPECT_FALSE(Result);
  EXPECT_EQ(F->size(), (size_t)3);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, SplitBlockPredecessors) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define i32 @basic_func(i1 %cond) {
entry:
  br i1 %cond, label %bb0, label %bb1
bb0:
  br label %bb1
bb1:
  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]
  ret i32 %phi
}
)IR");
  Function *F = M->getFunction("basic_func");
  DominatorTree DT(*F);

  // Make sure the dominator tree is properly updated if calling this on the
  // entry block.
  SplitBlockPredecessors(&F->getEntryBlock(), {}, "split.entry", &DT);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, SplitCriticalEdge) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @crit_edge(i1 %cond0, i1 %cond1) {
entry:
  br i1 %cond0, label %bb0, label %bb1
bb0:
  br label %bb1
bb1:
  br label %bb2
bb2:
  ret void
}
)IR");
  Function *F = M->getFunction("crit_edge");
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);

  CriticalEdgeSplittingOptions CESO(&DT, nullptr, nullptr, &PDT);
  EXPECT_EQ(1u, SplitAllCriticalEdges(*F, CESO));
  EXPECT_TRUE(DT.verify());
  EXPECT_TRUE(PDT.verify());
}

TEST(BasicBlockUtils, SplitIndirectBrCriticalEdgesIgnorePHIs) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @crit_edge(i8* %tgt, i1 %cond0, i1 %cond1) {
entry:
  indirectbr i8* %tgt, [label %bb0, label %bb1, label %bb2]
bb0:
  br i1 %cond0, label %bb1, label %bb2
bb1:
  %p = phi i32 [0, %bb0], [0, %entry]
  br i1 %cond1, label %bb3, label %bb4
bb2:
  ret void
bb3:
  ret void
bb4:
  ret void
}
)IR");
  Function *F = M->getFunction("crit_edge");
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  BranchProbabilityInfo BPI(*F, LI);
  BlockFrequencyInfo BFI(*F, BPI, LI);

  ASSERT_TRUE(SplitIndirectBrCriticalEdges(*F, /*IgnoreBlocksWithoutPHI=*/true,
                                           &BPI, &BFI));

  // Check that successors of the split block get their probability correct.
  BasicBlock *BB1 = getBasicBlockByName(*F, "bb1");
  BasicBlock *SplitBB = BB1->getTerminator()->getSuccessor(0);
  ASSERT_EQ(2u, SplitBB->getTerminator()->getNumSuccessors());
  EXPECT_EQ(BranchProbability(1, 2), BPI.getEdgeProbability(SplitBB, 0u));
  EXPECT_EQ(BranchProbability(1, 2), BPI.getEdgeProbability(SplitBB, 1u));

  // bb2 has no PHI, so we shouldn't split bb0 -> bb2
  BasicBlock *BB0 = getBasicBlockByName(*F, "bb0");
  ASSERT_EQ(2u, BB0->getTerminator()->getNumSuccessors());
  EXPECT_EQ(BB0->getTerminator()->getSuccessor(1),
            getBasicBlockByName(*F, "bb2"));
}

TEST(BasicBlockUtils, SplitIndirectBrCriticalEdges) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @crit_edge(i8* %tgt, i1 %cond0, i1 %cond1) {
entry:
  indirectbr i8* %tgt, [label %bb0, label %bb1, label %bb2]
bb0:
  br i1 %cond0, label %bb1, label %bb2
bb1:
  %p = phi i32 [0, %bb0], [0, %entry]
  br i1 %cond1, label %bb3, label %bb4
bb2:
  ret void
bb3:
  ret void
bb4:
  ret void
}
)IR");
  Function *F = M->getFunction("crit_edge");
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  BranchProbabilityInfo BPI(*F, LI);
  BlockFrequencyInfo BFI(*F, BPI, LI);

  ASSERT_TRUE(SplitIndirectBrCriticalEdges(*F, /*IgnoreBlocksWithoutPHI=*/false,
                                           &BPI, &BFI));

  // Check that successors of the split block get their probability correct.
  BasicBlock *BB1 = getBasicBlockByName(*F, "bb1");
  BasicBlock *SplitBB = BB1->getTerminator()->getSuccessor(0);
  ASSERT_EQ(2u, SplitBB->getTerminator()->getNumSuccessors());
  EXPECT_EQ(BranchProbability(1, 2), BPI.getEdgeProbability(SplitBB, 0u));
  EXPECT_EQ(BranchProbability(1, 2), BPI.getEdgeProbability(SplitBB, 1u));

  // Should split, resulting in:
  //   bb0 -> bb2.clone; bb2 -> split1; bb2.clone -> split,
  BasicBlock *BB0 = getBasicBlockByName(*F, "bb0");
  ASSERT_EQ(2u, BB0->getTerminator()->getNumSuccessors());
  BasicBlock *BB2Clone = BB0->getTerminator()->getSuccessor(1);
  BasicBlock *BB2 = getBasicBlockByName(*F, "bb2");
  EXPECT_NE(BB2Clone, BB2);
  ASSERT_EQ(1u, BB2->getTerminator()->getNumSuccessors());
  ASSERT_EQ(1u, BB2Clone->getTerminator()->getNumSuccessors());
  EXPECT_EQ(BB2->getTerminator()->getSuccessor(0),
            BB2Clone->getTerminator()->getSuccessor(0));
}

TEST(BasicBlockUtils, SetEdgeProbability) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @edge_probability(i32 %0) {
entry:
switch i32 %0, label %LD [
  i32 700, label %L0
  i32 701, label %L1
  i32 702, label %L2
  i32 703, label %L3
  i32 704, label %L4
  i32 705, label %L5
  i32 706, label %L6
  i32 707, label %L7
  i32 708, label %L8
  i32 709, label %L9
  i32 710, label %L10
  i32 711, label %L11
  i32 712, label %L12
  i32 713, label %L13
  i32 714, label %L14
  i32 715, label %L15
  i32 716, label %L16
  i32 717, label %L17
  i32 718, label %L18
  i32 719, label %L19
], !prof !{!"branch_weights", i32 1, i32 1, i32 1, i32 1, i32 1, i32 451, i32 1,
           i32 12, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
           i32 1, i32 1, i32 1, i32 1, i32 1}
LD:
  unreachable
L0:
  ret void
L1:
  ret void
L2:
  ret void
L3:
  ret void
L4:
  ret void
L5:
  ret void
L6:
  ret void
L7:
  ret void
L8:
  ret void
L9:
  ret void
L10:
  ret void
L11:
  ret void
L12:
  ret void
L13:
  ret void
L14:
  ret void
L15:
  ret void
L16:
  ret void
L17:
  ret void
L18:
  ret void
L19:
  ret void
}
)IR");
  Function *F = M->getFunction("edge_probability");
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  BranchProbabilityInfo BPI(*F, LI);

  // Check that the unreachable block has the minimal probability.
  const BasicBlock *EntryBB = getBasicBlockByName(*F, "entry");
  const BasicBlock *UnreachableBB = getBasicBlockByName(*F, "LD");
  EXPECT_EQ(BranchProbability::getRaw(1),
            BPI.getEdgeProbability(EntryBB, UnreachableBB));
}

TEST(BasicBlockUtils, IsPresplitCoroSuspendExitTest) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @positive_case(i32 %0) #0 {
entry:
  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %resume
    i8 1, label %destroy
  ]
resume:
  ret void
destroy:
  ret void
exit:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

define void @notpresplit(i32 %0) {
entry:
  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %resume
    i8 1, label %destroy
  ]
resume:
  ret void
destroy:
  ret void
exit:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

declare token @llvm.coro.save(ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare i1 @llvm.coro.end(ptr, i1, token)

attributes #0 = { presplitcoroutine }
)IR");

  auto FindExit = [](const Function &F) -> const BasicBlock * {
    for (const auto &BB : F)
      if (BB.getName() == "exit")
        return &BB;
    return nullptr;
  };
  Function *P = M->getFunction("positive_case");
  const auto &ExitP = *FindExit(*P);
  EXPECT_TRUE(llvm::isPresplitCoroSuspendExitEdge(*ExitP.getSinglePredecessor(),
                                                  ExitP));

  Function *N = M->getFunction("notpresplit");
  const auto &ExitN = *FindExit(*N);
  EXPECT_FALSE(llvm::isPresplitCoroSuspendExitEdge(
      *ExitN.getSinglePredecessor(), ExitN));
}
