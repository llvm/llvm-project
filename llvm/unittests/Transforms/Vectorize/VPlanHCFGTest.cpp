//===- llvm/unittest/Transforms/Vectorize/VPlanHCFGTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanTransforms.h"
#include "VPlanTestBase.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"
#include <string>

namespace llvm {
namespace {

class VPlanHCFGTest : public VPlanTestIRBase {};

TEST_F(VPlanHCFGTest, testBuildHCFGInnerLoop) {
  const char *ModuleString =
      "define void @f(ptr %A, i64 %N) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %arr.idx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv\n"
      "  %l1 = load i32, ptr %arr.idx, align 4\n"
      "  %res = add i32 %l1, 10\n"
      "  store i32 %res, ptr %arr.idx, align 4\n"
      "  %indvars.iv.next = add i64 %indvars.iv, 1\n"
      "  %exitcond = icmp ne i64 %indvars.iv.next, %N\n"
      "  br i1 %exitcond, label %for.body, label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  VPBasicBlock *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  EXPECT_EQ(0u, Entry->getNumPredecessors());
  EXPECT_EQ(1u, Entry->getNumSuccessors());

  // Check that the region following the preheader consists of a block for the
  // original header and a separate latch.
  VPBasicBlock *VecBB = Plan->getVectorLoopRegion()->getEntryBasicBlock();
  EXPECT_EQ(7u, VecBB->size());
  EXPECT_EQ(0u, VecBB->getNumPredecessors());
  EXPECT_EQ(1u, VecBB->getNumSuccessors());
  EXPECT_EQ(VecBB->getParent()->getEntryBasicBlock(), VecBB);
  EXPECT_EQ(&*Plan, VecBB->getPlan());

  VPBlockBase *VecLatch = VecBB->getSingleSuccessor();
  EXPECT_EQ(VecLatch->getParent()->getExitingBasicBlock(), VecLatch);
  EXPECT_EQ(0u, VecLatch->getNumSuccessors());

  auto Iter = VecBB->begin();
  VPWidenPHIRecipe *Phi = dyn_cast<VPWidenPHIRecipe>(&*Iter++);
  EXPECT_NE(nullptr, Phi);

  VPInstruction *Idx = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::GetElementPtr, Idx->getOpcode());
  EXPECT_EQ(2u, Idx->getNumOperands());
  EXPECT_EQ(Phi, Idx->getOperand(1));

  VPInstruction *Load = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::Load, Load->getOpcode());
  EXPECT_EQ(1u, Load->getNumOperands());
  EXPECT_EQ(Idx, Load->getOperand(0));

  VPInstruction *Add = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::Add, Add->getOpcode());
  EXPECT_EQ(2u, Add->getNumOperands());
  EXPECT_EQ(Load, Add->getOperand(0));

  VPInstruction *Store = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::Store, Store->getOpcode());
  EXPECT_EQ(2u, Store->getNumOperands());
  EXPECT_EQ(Add, Store->getOperand(0));
  EXPECT_EQ(Idx, Store->getOperand(1));

  VPInstruction *IndvarAdd = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::Add, IndvarAdd->getOpcode());
  EXPECT_EQ(2u, IndvarAdd->getNumOperands());
  EXPECT_EQ(Phi, IndvarAdd->getOperand(0));

  VPInstruction *ICmp = dyn_cast<VPInstruction>(&*Iter++);
  EXPECT_EQ(Instruction::ICmp, ICmp->getOpcode());
  EXPECT_EQ(2u, ICmp->getNumOperands());
  EXPECT_EQ(IndvarAdd, ICmp->getOperand(0));

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  // Add an external value to check we do not print the list of external values,
  // as this is not required with the new printing.
  Plan->getOrAddLiveIn(&*F->arg_begin());
  std::string FullDump;
  raw_string_ostream OS(FullDump);
  Plan->printDOT(OS);
  const char *ExpectedStr = R"(digraph VPlan {
graph [labelloc=t, fontsize=30; label="Vectorization Plan\n for UF\>=1\nLive-in vp\<%0\> = vector-trip-count\nLive-in ir\<%N\> = original trip-count\n"]
node [shape=rect, fontname=Courier, fontsize=30]
edge [fontname=Courier, fontsize=30]
compound=true
  N0 [label =
    "ir-bb\<entry\>:\l" +
    "Successor(s): vector.ph\l"
  ]
  N0 -> N1 [ label=""]
  N1 [label =
    "vector.ph:\l" +
    "Successor(s): vector loop\l"
  ]
  N1 -> N2 [ label="" lhead=cluster_N3]
  subgraph cluster_N3 {
    fontname=Courier
    label="\<x1\> vector loop"
    N2 [label =
      "vector.body:\l" +
      "  WIDEN-PHI ir\<%indvars.iv\> = phi ir\<0\>, ir\<%indvars.iv.next\>\l" +
      "  EMIT ir\<%arr.idx\> = getelementptr ir\<%A\>, ir\<%indvars.iv\>\l" +
      "  EMIT ir\<%l1\> = load ir\<%arr.idx\>\l" +
      "  EMIT ir\<%res\> = add ir\<%l1\>, ir\<10\>\l" +
      "  EMIT store ir\<%res\>, ir\<%arr.idx\>\l" +
      "  EMIT ir\<%indvars.iv.next\> = add ir\<%indvars.iv\>, ir\<1\>\l" +
      "  EMIT ir\<%exitcond\> = icmp ir\<%indvars.iv.next\>, ir\<%N\>\l" +
      "Successor(s): vector.latch\l"
    ]
    N2 -> N4 [ label=""]
    N4 [label =
      "vector.latch:\l" +
      "No successors\l"
    ]
  }
  N4 -> N5 [ label="" ltail=cluster_N3]
  N5 [label =
    "middle.block:\l" +
    "  EMIT vp\<%cmp.n\> = icmp eq ir\<%N\>, vp\<%0\>\l" +
    "  EMIT branch-on-cond vp\<%cmp.n\>\l" +
    "Successor(s): ir-bb\<for.end\>, scalar.ph\l"
  ]
  N5 -> N6 [ label="T"]
  N5 -> N7 [ label="F"]
  N6 [label =
    "ir-bb\<for.end\>:\l" +
    "No successors\l"
  ]
  N7 [label =
    "scalar.ph:\l" +
    "Successor(s): ir-bb\<for.body\>\l"
  ]
  N7 -> N8 [ label=""]
  N8 [label =
    "ir-bb\<for.body\>:\l" +
    "  IR   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\l" +
    "  IR   %arr.idx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv\l" +
    "  IR   %l1 = load i32, ptr %arr.idx, align 4\l" +
    "  IR   %res = add i32 %l1, 10\l" +
    "  IR   store i32 %res, ptr %arr.idx, align 4\l" +
    "  IR   %indvars.iv.next = add i64 %indvars.iv, 1\l" +
    "  IR   %exitcond = icmp ne i64 %indvars.iv.next, %N\l" +
    "No successors\l"
  ]
}
)";
  EXPECT_EQ(ExpectedStr, FullDump);
#endif
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  TargetLibraryInfo TLI(TLII);
  VPlanTransforms::VPInstructionsToVPRecipes(
      Plan, [](PHINode *P) { return nullptr; }, *SE, TLI);
}

TEST_F(VPlanHCFGTest, testVPInstructionToVPRecipesInner) {
  const char *ModuleString =
      "define void @f(ptr %A, i64 %N) {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]\n"
      "  %arr.idx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv\n"
      "  %l1 = load i32, ptr %arr.idx, align 4\n"
      "  %res = add i32 %l1, 10\n"
      "  store i32 %res, ptr %arr.idx, align 4\n"
      "  %indvars.iv.next = add i64 %indvars.iv, 1\n"
      "  %exitcond = icmp ne i64 %indvars.iv.next, %N\n"
      "  br i1 %exitcond, label %for.body, label %for.end\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  TargetLibraryInfo TLI(TLII);
  VPlanTransforms::VPInstructionsToVPRecipes(
      Plan, [](PHINode *P) { return nullptr; }, *SE, TLI);

  VPBlockBase *Entry = Plan->getEntry()->getEntryBasicBlock();
  EXPECT_NE(nullptr, Entry->getSingleSuccessor());
  EXPECT_EQ(0u, Entry->getNumPredecessors());
  EXPECT_EQ(1u, Entry->getNumSuccessors());

  // Check that the region following the preheader consists of a block for the
  // original header and a separate latch.
  VPBasicBlock *VecBB = Plan->getVectorLoopRegion()->getEntryBasicBlock();
  EXPECT_EQ(7u, VecBB->size());
  EXPECT_EQ(0u, VecBB->getNumPredecessors());
  EXPECT_EQ(1u, VecBB->getNumSuccessors());
  EXPECT_EQ(VecBB->getParent()->getEntryBasicBlock(), VecBB);

  VPBlockBase *VecLatch = VecBB->getSingleSuccessor();
  EXPECT_EQ(VecLatch->getParent()->getExitingBasicBlock(), VecLatch);
  EXPECT_EQ(0u, VecLatch->getNumSuccessors());

  auto Iter = VecBB->begin();
  EXPECT_NE(nullptr, dyn_cast<VPWidenPHIRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenGEPRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenMemoryRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenMemoryRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenRecipe>(&*Iter++));
  EXPECT_EQ(VecBB->end(), Iter);
}

TEST_F(VPlanHCFGTest, testBuildHCFGInnerLoopMultiExit) {
  const char *ModuleString =
      "define void @f(ptr %A, i64 %N) {\n"
      "entry:\n"
      "  br label %loop.header\n"
      "loop.header:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]\n"
      "  %arr.idx = getelementptr inbounds i32, ptr %A, i64 %iv\n"
      "  %l1 = load i32, ptr %arr.idx, align 4\n"
      "  %c = icmp eq i32 %l1, 0\n"
      "  br i1 %c, label %exit.1, label %loop.latch\n"
      "loop.latch:\n"
      "  %res = add i32 %l1, 10\n"
      "  store i32 %res, ptr %arr.idx, align 4\n"
      "  %iv.next = add i64 %iv, 1\n"
      "  %exitcond = icmp ne i64 %iv.next, %N\n"
      "  br i1 %exitcond, label %loop.header, label %exit.2\n"
      "exit.1:\n"
      "  ret void\n"
      "exit.2:\n"
      "  ret void\n"
      "}\n";

  Module &M = parseModule(ModuleString);

  Function *F = M.getFunction("f");
  BasicBlock *LoopHeader = F->getEntryBlock().getSingleSuccessor();
  auto Plan = buildHCFG(LoopHeader);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  // Add an external value to check we do not print the list of external values,
  // as this is not required with the new printing.
  Plan->getOrAddLiveIn(&*F->arg_begin());
  std::string FullDump;
  raw_string_ostream OS(FullDump);
  Plan->printDOT(OS);
  const char *ExpectedStr = R"(digraph VPlan {
graph [labelloc=t, fontsize=30; label="Vectorization Plan\n for UF\>=1\nLive-in vp\<%0\> = vector-trip-count\nLive-in ir\<%N\> = original trip-count\n"]
node [shape=rect, fontname=Courier, fontsize=30]
edge [fontname=Courier, fontsize=30]
compound=true
  N0 [label =
    "ir-bb\<entry\>:\l" +
    "Successor(s): vector.ph\l"
  ]
  N0 -> N1 [ label=""]
  N1 [label =
    "vector.ph:\l" +
    "Successor(s): vector loop\l"
  ]
  N1 -> N2 [ label="" lhead=cluster_N3]
  subgraph cluster_N3 {
    fontname=Courier
    label="\<x1\> vector loop"
    N2 [label =
      "vector.body:\l" +
      "  WIDEN-PHI ir\<%iv\> = phi ir\<0\>, ir\<%iv.next\>\l" +
      "  EMIT ir\<%arr.idx\> = getelementptr ir\<%A\>, ir\<%iv\>\l" +
      "  EMIT ir\<%l1\> = load ir\<%arr.idx\>\l" +
      "  EMIT ir\<%c\> = icmp ir\<%l1\>, ir\<0\>\l" +
      "Successor(s): loop.latch\l"
    ]
    N2 -> N4 [ label=""]
    N4 [label =
      "loop.latch:\l" +
      "  EMIT ir\<%res\> = add ir\<%l1\>, ir\<10\>\l" +
      "  EMIT store ir\<%res\>, ir\<%arr.idx\>\l" +
      "  EMIT ir\<%iv.next\> = add ir\<%iv\>, ir\<1\>\l" +
      "  EMIT ir\<%exitcond\> = icmp ir\<%iv.next\>, ir\<%N\>\l" +
      "Successor(s): vector.latch\l"
    ]
    N4 -> N5 [ label=""]
    N5 [label =
      "vector.latch:\l" +
      "No successors\l"
    ]
  }
  N5 -> N6 [ label="" ltail=cluster_N3]
  N6 [label =
    "middle.block:\l" +
    "  EMIT vp\<%cmp.n\> = icmp eq ir\<%N\>, vp\<%0\>\l" +
    "  EMIT branch-on-cond vp\<%cmp.n\>\l" +
    "Successor(s): ir-bb\<exit.2\>, scalar.ph\l"
  ]
  N6 -> N7 [ label="T"]
  N6 -> N8 [ label="F"]
  N7 [label =
    "ir-bb\<exit.2\>:\l" +
    "No successors\l"
  ]
  N8 [label =
    "scalar.ph:\l" +
    "Successor(s): ir-bb\<loop.header\>\l"
  ]
  N8 -> N9 [ label=""]
  N9 [label =
    "ir-bb\<loop.header\>:\l" +
    "  IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]\l" +
    "  IR   %arr.idx = getelementptr inbounds i32, ptr %A, i64 %iv\l" +
    "  IR   %l1 = load i32, ptr %arr.idx, align 4\l" +
    "  IR   %c = icmp eq i32 %l1, 0\l" +
    "No successors\l"
  ]
}
)";
  EXPECT_EQ(ExpectedStr, FullDump);
#endif
}

} // namespace
} // namespace llvm
