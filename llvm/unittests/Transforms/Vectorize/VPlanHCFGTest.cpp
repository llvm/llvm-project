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

class VPlanHCFGTest : public VPlanTestBase {};

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

  // Check that the region following the preheader is a single basic-block
  // region (loop).
  VPBasicBlock *VecBB = Entry->getSingleSuccessor()->getEntryBasicBlock();
  EXPECT_EQ(8u, VecBB->size());
  EXPECT_EQ(0u, VecBB->getNumPredecessors());
  EXPECT_EQ(0u, VecBB->getNumSuccessors());
  EXPECT_EQ(VecBB->getParent()->getEntryBasicBlock(), VecBB);
  EXPECT_EQ(VecBB->getParent()->getExitingBasicBlock(), VecBB);
  EXPECT_EQ(&*Plan, VecBB->getPlan());

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
    "No successors\l"
  ]
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
      "  EMIT branch-on-cond ir\<%exitcond\>\l" +
      "No successors\l"
    ]
  }
  N2 -> N4 [ label="" ltail=cluster_N3]
  N4 [label =
    "middle.block:\l" +
    "  EMIT vp\<%1\> = icmp eq ir\<%N\>, vp\<%0\>\l" +
    "  EMIT branch-on-cond vp\<%1\>\l" +
    "Successor(s): ir-bb\<for.end\>, scalar.ph\l"
  ]
  N4 -> N5 [ label="T"]
  N4 -> N6 [ label="F"]
  N5 [label =
    "ir-bb\<for.end\>:\l" +
    "No successors\l"
  ]
  N6 [label =
    "scalar.ph:\l" +
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

  // Check that the region following the preheader is a single basic-block
  // region (loop).
  VPBasicBlock *VecBB = Entry->getSingleSuccessor()->getEntryBasicBlock();
  EXPECT_EQ(8u, VecBB->size());
  EXPECT_EQ(0u, VecBB->getNumPredecessors());
  EXPECT_EQ(0u, VecBB->getNumSuccessors());
  EXPECT_EQ(VecBB->getParent()->getEntryBasicBlock(), VecBB);
  EXPECT_EQ(VecBB->getParent()->getExitingBasicBlock(), VecBB);

  auto Iter = VecBB->begin();
  EXPECT_NE(nullptr, dyn_cast<VPWidenPHIRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenGEPRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenMemoryRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenMemoryRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPWidenRecipe>(&*Iter++));
  EXPECT_NE(nullptr, dyn_cast<VPInstruction>(&*Iter++));
  EXPECT_EQ(VecBB->end(), Iter);
}

} // namespace
} // namespace llvm
