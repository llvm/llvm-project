//===- llvm/unittests/IR/DominatorTreeTest.cpp - Constants unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <random>
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "CFGBuilder.h"
#include "gtest/gtest.h"

using namespace llvm;


/// Build the dominator tree for the function and run the Test.
static void runWithDomTree(
    Module &M, StringRef FuncName,
    function_ref<void(Function &F, DominatorTree *DT, PostDominatorTree *PDT)>
        Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;
  // Compute the dominator tree for the function.
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);
  Test(*F, &DT, &PDT);
}

static std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                              StringRef ModuleStr) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad assembly?");
  return M;
}

TEST(DominatorTree, PHIs) {
  StringRef ModuleString = R"(
      define void @f() {
      bb1:
        br label %bb1
      bb2:
        %a = phi i32 [0, %bb1], [1, %bb2]
        %b = phi i32 [2, %bb1], [%a, %bb2]
        br label %bb2
      };
  )";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(*M, "f",
                 [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
                   auto FI = F.begin();
                   ++FI;
                   BasicBlock *BB2 = &*FI;
                   auto BI = BB2->begin();
                   Instruction *PhiA = &*BI++;
                   Instruction *PhiB = &*BI;

                   // Phis are thought to execute "instantly, together".
                   EXPECT_TRUE(DT->dominates(PhiA, PhiB));
                   EXPECT_TRUE(DT->dominates(PhiB, PhiA));
                 });
}

TEST(DominatorTree, Unreachable) {
  StringRef ModuleString =
      "declare i32 @g()\n"
      "define void @f(i32 %x) personality i32 ()* @g {\n"
      "bb0:\n"
      "  %y1 = add i32 %x, 1\n"
      "  %y2 = add i32 %x, 1\n"
      "  %y3 = invoke i32 @g() to label %bb1 unwind label %bb2\n"
      "bb1:\n"
      "  %y4 = add i32 %x, 1\n"
      "  br label %bb4\n"
      "bb2:\n"
      "  %y5 = landingpad i32\n"
      "          cleanup\n"
      "  br label %bb4\n"
      "bb3:\n"
      "  %y6 = add i32 %x, 1\n"
      "  %y7 = add i32 %x, 1\n"
      "  ret void\n"
      "bb4:\n"
      "  %y8 = phi i32 [0, %bb2], [%y4, %bb1]\n"
      "  %y9 = phi i32 [0, %bb2], [%y4, %bb1]\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(
      *M, "f", [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
        Function::iterator FI = F.begin();

        BasicBlock *BB0 = &*FI++;
        BasicBlock::iterator BBI = BB0->begin();
        Instruction *Y1 = &*BBI++;
        Instruction *Y2 = &*BBI++;
        Instruction *Y3 = &*BBI++;

        BasicBlock *BB1 = &*FI++;
        BBI = BB1->begin();
        Instruction *Y4 = &*BBI++;

        BasicBlock *BB2 = &*FI++;
        BBI = BB2->begin();
        Instruction *Y5 = &*BBI++;

        BasicBlock *BB3 = &*FI++;
        BBI = BB3->begin();
        Instruction *Y6 = &*BBI++;
        Instruction *Y7 = &*BBI++;

        BasicBlock *BB4 = &*FI++;
        BBI = BB4->begin();
        Instruction *Y8 = &*BBI++;
        Instruction *Y9 = &*BBI++;

        // Reachability
        EXPECT_TRUE(DT->isReachableFromEntry(BB0));
        EXPECT_TRUE(DT->isReachableFromEntry(BB1));
        EXPECT_TRUE(DT->isReachableFromEntry(BB2));
        EXPECT_FALSE(DT->isReachableFromEntry(BB3));
        EXPECT_TRUE(DT->isReachableFromEntry(BB4));

        // BB dominance
        EXPECT_TRUE(DT->dominates(BB0, BB0));
        EXPECT_TRUE(DT->dominates(BB0, BB1));
        EXPECT_TRUE(DT->dominates(BB0, BB2));
        EXPECT_TRUE(DT->dominates(BB0, BB3));
        EXPECT_TRUE(DT->dominates(BB0, BB4));

        EXPECT_FALSE(DT->dominates(BB1, BB0));
        EXPECT_TRUE(DT->dominates(BB1, BB1));
        EXPECT_FALSE(DT->dominates(BB1, BB2));
        EXPECT_TRUE(DT->dominates(BB1, BB3));
        EXPECT_FALSE(DT->dominates(BB1, BB4));

        EXPECT_FALSE(DT->dominates(BB2, BB0));
        EXPECT_FALSE(DT->dominates(BB2, BB1));
        EXPECT_TRUE(DT->dominates(BB2, BB2));
        EXPECT_TRUE(DT->dominates(BB2, BB3));
        EXPECT_FALSE(DT->dominates(BB2, BB4));

        EXPECT_FALSE(DT->dominates(BB3, BB0));
        EXPECT_FALSE(DT->dominates(BB3, BB1));
        EXPECT_FALSE(DT->dominates(BB3, BB2));
        EXPECT_TRUE(DT->dominates(BB3, BB3));
        EXPECT_FALSE(DT->dominates(BB3, BB4));

        // BB proper dominance
        EXPECT_FALSE(DT->properlyDominates(BB0, BB0));
        EXPECT_TRUE(DT->properlyDominates(BB0, BB1));
        EXPECT_TRUE(DT->properlyDominates(BB0, BB2));
        EXPECT_TRUE(DT->properlyDominates(BB0, BB3));

        EXPECT_FALSE(DT->properlyDominates(BB1, BB0));
        EXPECT_FALSE(DT->properlyDominates(BB1, BB1));
        EXPECT_FALSE(DT->properlyDominates(BB1, BB2));
        EXPECT_TRUE(DT->properlyDominates(BB1, BB3));

        EXPECT_FALSE(DT->properlyDominates(BB2, BB0));
        EXPECT_FALSE(DT->properlyDominates(BB2, BB1));
        EXPECT_FALSE(DT->properlyDominates(BB2, BB2));
        EXPECT_TRUE(DT->properlyDominates(BB2, BB3));

        EXPECT_FALSE(DT->properlyDominates(BB3, BB0));
        EXPECT_FALSE(DT->properlyDominates(BB3, BB1));
        EXPECT_FALSE(DT->properlyDominates(BB3, BB2));
        EXPECT_FALSE(DT->properlyDominates(BB3, BB3));

        // Instruction dominance in the same reachable BB
        EXPECT_FALSE(DT->dominates(Y1, Y1));
        EXPECT_TRUE(DT->dominates(Y1, Y2));
        EXPECT_FALSE(DT->dominates(Y2, Y1));
        EXPECT_FALSE(DT->dominates(Y2, Y2));

        // Instruction dominance in the same unreachable BB
        EXPECT_TRUE(DT->dominates(Y6, Y6));
        EXPECT_TRUE(DT->dominates(Y6, Y7));
        EXPECT_TRUE(DT->dominates(Y7, Y6));
        EXPECT_TRUE(DT->dominates(Y7, Y7));

        // Invoke
        EXPECT_TRUE(DT->dominates(Y3, Y4));
        EXPECT_FALSE(DT->dominates(Y3, Y5));

        // Phi
        EXPECT_TRUE(DT->dominates(Y2, Y9));
        EXPECT_FALSE(DT->dominates(Y3, Y9));
        EXPECT_FALSE(DT->dominates(Y8, Y9));

        // Anything dominates unreachable
        EXPECT_TRUE(DT->dominates(Y1, Y6));
        EXPECT_TRUE(DT->dominates(Y3, Y6));

        // Unreachable doesn't dominate reachable
        EXPECT_FALSE(DT->dominates(Y6, Y1));

        // Instruction, BB dominance
        EXPECT_FALSE(DT->dominates(Y1, BB0));
        EXPECT_TRUE(DT->dominates(Y1, BB1));
        EXPECT_TRUE(DT->dominates(Y1, BB2));
        EXPECT_TRUE(DT->dominates(Y1, BB3));
        EXPECT_TRUE(DT->dominates(Y1, BB4));

        EXPECT_FALSE(DT->dominates(Y3, BB0));
        EXPECT_TRUE(DT->dominates(Y3, BB1));
        EXPECT_FALSE(DT->dominates(Y3, BB2));
        EXPECT_TRUE(DT->dominates(Y3, BB3));
        EXPECT_FALSE(DT->dominates(Y3, BB4));

        EXPECT_TRUE(DT->dominates(Y6, BB3));

        // Post dominance.
        EXPECT_TRUE(PDT->dominates(BB0, BB0));
        EXPECT_FALSE(PDT->dominates(BB1, BB0));
        EXPECT_FALSE(PDT->dominates(BB2, BB0));
        EXPECT_FALSE(PDT->dominates(BB3, BB0));
        EXPECT_TRUE(PDT->dominates(BB4, BB1));

        // Dominance descendants.
        SmallVector<BasicBlock *, 8> DominatedBBs, PostDominatedBBs;

        DT->getDescendants(BB0, DominatedBBs);
        PDT->getDescendants(BB0, PostDominatedBBs);
        EXPECT_EQ(DominatedBBs.size(), 4UL);
        EXPECT_EQ(PostDominatedBBs.size(), 1UL);

        // BB3 is unreachable. It should have no dominators nor postdominators.
        DominatedBBs.clear();
        PostDominatedBBs.clear();
        DT->getDescendants(BB3, DominatedBBs);
        DT->getDescendants(BB3, PostDominatedBBs);
        EXPECT_EQ(DominatedBBs.size(), 0UL);
        EXPECT_EQ(PostDominatedBBs.size(), 0UL);

        // Check DFS Numbers before
        DT->updateDFSNumbers();
        EXPECT_EQ(DT->getNode(BB0)->getDFSNumIn(), 0UL);
        EXPECT_EQ(DT->getNode(BB0)->getDFSNumOut(), 7UL);
        EXPECT_EQ(DT->getNode(BB1)->getDFSNumIn(), 1UL);
        EXPECT_EQ(DT->getNode(BB1)->getDFSNumOut(), 2UL);
        EXPECT_EQ(DT->getNode(BB2)->getDFSNumIn(), 5UL);
        EXPECT_EQ(DT->getNode(BB2)->getDFSNumOut(), 6UL);
        EXPECT_EQ(DT->getNode(BB4)->getDFSNumIn(), 3UL);
        EXPECT_EQ(DT->getNode(BB4)->getDFSNumOut(), 4UL);

        // Check levels before
        EXPECT_EQ(DT->getNode(BB0)->getLevel(), 0U);
        EXPECT_EQ(DT->getNode(BB1)->getLevel(), 1U);
        EXPECT_EQ(DT->getNode(BB2)->getLevel(), 1U);
        EXPECT_EQ(DT->getNode(BB4)->getLevel(), 1U);

        // Reattach block 3 to block 1 and recalculate
        BB1->getTerminator()->eraseFromParent();
        BranchInst::Create(BB4, BB3, ConstantInt::getTrue(F.getContext()), BB1);
        DT->recalculate(F);

        // Check DFS Numbers after
        DT->updateDFSNumbers();
        EXPECT_EQ(DT->getNode(BB0)->getDFSNumIn(), 0UL);
        EXPECT_EQ(DT->getNode(BB0)->getDFSNumOut(), 9UL);
        EXPECT_EQ(DT->getNode(BB1)->getDFSNumIn(), 1UL);
        EXPECT_EQ(DT->getNode(BB1)->getDFSNumOut(), 4UL);
        EXPECT_EQ(DT->getNode(BB2)->getDFSNumIn(), 7UL);
        EXPECT_EQ(DT->getNode(BB2)->getDFSNumOut(), 8UL);
        EXPECT_EQ(DT->getNode(BB3)->getDFSNumIn(), 2UL);
        EXPECT_EQ(DT->getNode(BB3)->getDFSNumOut(), 3UL);
        EXPECT_EQ(DT->getNode(BB4)->getDFSNumIn(), 5UL);
        EXPECT_EQ(DT->getNode(BB4)->getDFSNumOut(), 6UL);

        // Check levels after
        EXPECT_EQ(DT->getNode(BB0)->getLevel(), 0U);
        EXPECT_EQ(DT->getNode(BB1)->getLevel(), 1U);
        EXPECT_EQ(DT->getNode(BB2)->getLevel(), 1U);
        EXPECT_EQ(DT->getNode(BB3)->getLevel(), 2U);
        EXPECT_EQ(DT->getNode(BB4)->getLevel(), 1U);

        // Change root node
        EXPECT_TRUE(DT->verify());
        BasicBlock *NewEntry =
            BasicBlock::Create(F.getContext(), "new_entry", &F, BB0);
        BranchInst::Create(BB0, NewEntry);
        EXPECT_EQ(F.begin()->getName(), NewEntry->getName());
        EXPECT_TRUE(&F.getEntryBlock() == NewEntry);
        DT->setNewRoot(NewEntry);
        EXPECT_TRUE(DT->verify());
      });
}

TEST(DominatorTree, NonUniqueEdges) {
  StringRef ModuleString =
      "define i32 @f(i32 %i, i32 *%p) {\n"
      "bb0:\n"
      "   store i32 %i, i32 *%p\n"
      "   switch i32 %i, label %bb2 [\n"
      "     i32 0, label %bb1\n"
      "     i32 1, label %bb1\n"
      "   ]\n"
      " bb1:\n"
      "   ret i32 1\n"
      " bb2:\n"
      "   ret i32 4\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(
      *M, "f", [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
        Function::iterator FI = F.begin();

        BasicBlock *BB0 = &*FI++;
        BasicBlock *BB1 = &*FI++;
        BasicBlock *BB2 = &*FI++;

        const Instruction *TI = BB0->getTerminator();
        assert(TI->getNumSuccessors() == 3 && "Switch has three successors");

        BasicBlockEdge Edge_BB0_BB2(BB0, TI->getSuccessor(0));
        assert(Edge_BB0_BB2.getEnd() == BB2 &&
               "Default label is the 1st successor");

        BasicBlockEdge Edge_BB0_BB1_a(BB0, TI->getSuccessor(1));
        assert(Edge_BB0_BB1_a.getEnd() == BB1 && "BB1 is the 2nd successor");

        BasicBlockEdge Edge_BB0_BB1_b(BB0, TI->getSuccessor(2));
        assert(Edge_BB0_BB1_b.getEnd() == BB1 && "BB1 is the 3rd successor");

        EXPECT_TRUE(DT->dominates(Edge_BB0_BB2, BB2));
        EXPECT_FALSE(DT->dominates(Edge_BB0_BB2, BB1));

        EXPECT_FALSE(DT->dominates(Edge_BB0_BB1_a, BB1));
        EXPECT_FALSE(DT->dominates(Edge_BB0_BB1_b, BB1));

        EXPECT_FALSE(DT->dominates(Edge_BB0_BB1_a, BB2));
        EXPECT_FALSE(DT->dominates(Edge_BB0_BB1_b, BB2));
      });
}

// Verify that the PDT is correctly updated in case an edge removal results
// in a new unreachable CFG node. Also make sure that the updated PDT is the
// same as a freshly recalculated one.
//
// For the following input code and initial PDT:
//
//          CFG                   PDT
//
//           A                    Exit
//           |                     |
//          _B                     D
//         / | \                   |
//        ^  v  \                  B
//        \ /    D                / \
//         C      \              C   A
//                v
//                Exit
//
// we verify that CFG' and PDT-updated is obtained after removal of edge C -> B.
//
//          CFG'               PDT-updated
//
//           A                    Exit
//           |                   / | \
//           B                  C  B  D
//           | \                   |
//           v  \                  A
//          /    D
//         C      \
//         |       \
// unreachable    Exit
//
// Both the blocks that end with ret and with unreachable become trivial
// PostDominatorTree roots, as they have no successors.
//
TEST(DominatorTree, DeletingEdgesIntroducesUnreachables) {
  StringRef ModuleString =
      "define void @f() {\n"
      "A:\n"
      "  br label %B\n"
      "B:\n"
      "  br i1 undef, label %D, label %C\n"
      "C:\n"
      "  br label %B\n"
      "D:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(
      *M, "f", [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
        Function::iterator FI = F.begin();

        FI++;
        BasicBlock *B = &*FI++;
        BasicBlock *C = &*FI++;
        BasicBlock *D = &*FI++;

        ASSERT_TRUE(PDT->dominates(PDT->getNode(D), PDT->getNode(B)));
        EXPECT_TRUE(DT->verify());
        EXPECT_TRUE(PDT->verify());

        C->getTerminator()->eraseFromParent();
        new UnreachableInst(C->getContext(), C);

        DT->deleteEdge(C, B);
        PDT->deleteEdge(C, B);

        EXPECT_TRUE(DT->verify());
        EXPECT_TRUE(PDT->verify());

        EXPECT_FALSE(PDT->dominates(PDT->getNode(D), PDT->getNode(B)));
        EXPECT_NE(PDT->getNode(C), nullptr);

        DominatorTree NDT(F);
        EXPECT_EQ(DT->compare(NDT), 0);

        PostDominatorTree NPDT(F);
        EXPECT_EQ(PDT->compare(NPDT), 0);
      });
}

// Verify that the PDT is correctly updated in case an edge removal results
// in an infinite loop. Also make sure that the updated PDT is the
// same as a freshly recalculated one.
//
// Test case:
//
//          CFG                   PDT
//
//           A                    Exit
//           |                     |
//          _B                     D
//         / | \                   |
//        ^  v  \                  B
//        \ /    D                / \
//         C      \              C   A
//        / \      v
//       ^  v      Exit
//        \_/
//
// After deleting the edge C->B, C is part of an infinite reverse-unreachable
// loop:
//
//          CFG'                  PDT'
//
//           A                    Exit
//           |                   / | \
//           B                  C  B  D
//           | \                   |
//           v  \                  A
//          /    D
//         C      \
//        / \      v
//       ^  v      Exit
//        \_/
//
// As C now becomes reverse-unreachable, it forms a new non-trivial root and
// gets connected to the virtual exit.
// D does not postdominate B anymore, because there are two forward paths from
// B to the virtual exit:
//  - B -> C -> VirtualExit
//  - B -> D -> VirtualExit.
//
TEST(DominatorTree, DeletingEdgesIntroducesInfiniteLoop) {
  StringRef ModuleString =
      "define void @f() {\n"
      "A:\n"
      "  br label %B\n"
      "B:\n"
      "  br i1 undef, label %D, label %C\n"
      "C:\n"
      "  switch i32 undef, label %C [\n"
      "    i32 0, label %B\n"
      "  ]\n"
      "D:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(
      *M, "f", [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
        Function::iterator FI = F.begin();

        FI++;
        BasicBlock *B = &*FI++;
        BasicBlock *C = &*FI++;
        BasicBlock *D = &*FI++;

        ASSERT_TRUE(PDT->dominates(PDT->getNode(D), PDT->getNode(B)));
        EXPECT_TRUE(DT->verify());
        EXPECT_TRUE(PDT->verify());

        auto SwitchC = cast<SwitchInst>(C->getTerminator());
        SwitchC->removeCase(SwitchC->case_begin());
        DT->deleteEdge(C, B);
        EXPECT_TRUE(DT->verify());
        PDT->deleteEdge(C, B);
        EXPECT_TRUE(PDT->verify());

        EXPECT_FALSE(PDT->dominates(PDT->getNode(D), PDT->getNode(B)));
        EXPECT_NE(PDT->getNode(C), nullptr);

        DominatorTree NDT(F);
        EXPECT_EQ(DT->compare(NDT), 0);

        PostDominatorTree NPDT(F);
        EXPECT_EQ(PDT->compare(NPDT), 0);
      });
}

// Verify that the PDT is correctly updated in case an edge removal results
// in an infinite loop.
//
// Test case:
//
//          CFG                   PDT
//
//           A                    Exit
//           |                   / | \
//           B--               C2  B  D
//           |  \              /   |
//           v   \            C    A
//          /     D
//         C--C2   \
//        / \  \    v
//       ^  v  --Exit
//        \_/
//
// After deleting the edge C->E, C is part of an infinite reverse-unreachable
// loop:
//
//          CFG'                  PDT'
//
//           A                    Exit
//           |                   / | \
//           B                  C  B  D
//           | \                   |
//           v  \                  A
//          /    D
//         C      \
//        / \      v
//       ^  v      Exit
//        \_/
//
// In PDT, D does not post-dominate B. After the edge C -> C2 is removed,
// C becomes a new nontrivial PDT root.
//
TEST(DominatorTree, DeletingEdgesIntroducesInfiniteLoop2) {
  StringRef ModuleString =
      "define void @f() {\n"
      "A:\n"
      "  br label %B\n"
      "B:\n"
      "  br i1 undef, label %D, label %C\n"
      "C:\n"
      "  switch i32 undef, label %C [\n"
      "    i32 0, label %C2\n"
      "  ]\n"
      "C2:\n"
      "  ret void\n"
      "D:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(
      *M, "f", [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
        Function::iterator FI = F.begin();

        FI++;
        BasicBlock *B = &*FI++;
        BasicBlock *C = &*FI++;
        BasicBlock *C2 = &*FI++;
        BasicBlock *D = &*FI++;

        EXPECT_TRUE(DT->verify());
        EXPECT_TRUE(PDT->verify());

        auto SwitchC = cast<SwitchInst>(C->getTerminator());
        SwitchC->removeCase(SwitchC->case_begin());
        DT->deleteEdge(C, C2);
        PDT->deleteEdge(C, C2);
        C2->removeFromParent();

        EXPECT_EQ(DT->getNode(C2), nullptr);
        PDT->eraseNode(C2);
        delete C2;

        EXPECT_TRUE(DT->verify());
        EXPECT_TRUE(PDT->verify());

        EXPECT_FALSE(PDT->dominates(PDT->getNode(D), PDT->getNode(B)));
        EXPECT_NE(PDT->getNode(C), nullptr);

        DominatorTree NDT(F);
        EXPECT_EQ(DT->compare(NDT), 0);

        PostDominatorTree NPDT(F);
        EXPECT_EQ(PDT->compare(NPDT), 0);
      });
}

// Verify that the IDF returns blocks in a deterministic way.
//
// Test case:
//
//          CFG
//
//          (A)
//          / \
//         /   \
//       (B)   (C)
//        |\   /|
//        |  X  |
//        |/   \|
//       (D)   (E)
//
// IDF for block B is {D, E}, and the order of blocks in this list is defined by
// their 1) level in dom-tree and 2) DFSIn number if the level is the same.
//
TEST(DominatorTree, IDFDeterminismTest) {
  StringRef ModuleString =
      "define void @f() {\n"
      "A:\n"
      "  br i1 undef, label %B, label %C\n"
      "B:\n"
      "  br i1 undef, label %D, label %E\n"
      "C:\n"
      "  br i1 undef, label %D, label %E\n"
      "D:\n"
      "  ret void\n"
      "E:\n"
      "  ret void\n"
      "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(
      *M, "f", [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
        Function::iterator FI = F.begin();

        BasicBlock *A = &*FI++;
        BasicBlock *B = &*FI++;
        BasicBlock *C = &*FI++;
        BasicBlock *D = &*FI++;
        BasicBlock *E = &*FI++;
        (void)C;

        DT->updateDFSNumbers();
        ForwardIDFCalculator IDF(*DT);
        SmallPtrSet<BasicBlock *, 1> DefBlocks;
        DefBlocks.insert(B);
        IDF.setDefiningBlocks(DefBlocks);

        SmallVector<BasicBlock *, 32> IDFBlocks;
        SmallPtrSet<BasicBlock *, 32> LiveInBlocks;
        IDF.resetLiveInBlocks();
        IDF.calculate(IDFBlocks);


        EXPECT_EQ(IDFBlocks.size(), 2UL);
        EXPECT_EQ(DT->getNode(A)->getDFSNumIn(), 0UL);
        EXPECT_EQ(IDFBlocks[0], D);
        EXPECT_EQ(IDFBlocks[1], E);
        EXPECT_TRUE(DT->getNode(IDFBlocks[0])->getDFSNumIn() <
                    DT->getNode(IDFBlocks[1])->getDFSNumIn());
      });
}

namespace {
const auto Insert = CFGBuilder::ActionKind::Insert;
const auto Delete = CFGBuilder::ActionKind::Delete;

bool CompUpdates(const CFGBuilder::Update &A, const CFGBuilder::Update &B) {
  return std::tie(A.Action, A.Edge.From, A.Edge.To) <
         std::tie(B.Action, B.Edge.From, B.Edge.To);
}
}  // namespace

TEST(DominatorTree, InsertReachable) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"}, {"3", "4"},  {"4", "5"},  {"5", "6"},  {"5", "7"},
      {"3", "8"}, {"8", "9"}, {"9", "10"}, {"8", "11"}, {"11", "12"}};

  std::vector<CFGBuilder::Update> Updates = {{Insert, {"12", "10"}},
                                             {Insert, {"10", "9"}},
                                             {Insert, {"7", "6"}},
                                             {Insert, {"7", "5"}}};
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDominatorTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  std::optional<CFGBuilder::Update> LastUpdate;
  while ((LastUpdate = B.applyUpdate())) {
    EXPECT_EQ(LastUpdate->Action, Insert);
    BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
    BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
    DT.insertEdge(From, To);
    EXPECT_TRUE(DT.verify());
    PDT.insertEdge(From, To);
    EXPECT_TRUE(PDT.verify());
  }
}

TEST(DominatorTree, InsertReachable2) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"}, {"3", "4"},  {"4", "5"},  {"5", "6"},  {"5", "7"},
      {"7", "5"}, {"2", "8"}, {"8", "11"}, {"11", "12"}, {"12", "10"},
      {"10", "9"}, {"9", "10"}};

  std::vector<CFGBuilder::Update> Updates = {{Insert, {"10", "7"}}};
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDominatorTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  std::optional<CFGBuilder::Update> LastUpdate = B.applyUpdate();
  EXPECT_TRUE(LastUpdate);

  EXPECT_EQ(LastUpdate->Action, Insert);
  BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
  BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
  DT.insertEdge(From, To);
  EXPECT_TRUE(DT.verify());
  PDT.insertEdge(From, To);
  EXPECT_TRUE(PDT.verify());
}

TEST(DominatorTree, InsertUnreachable) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {{"1", "2"},  {"2", "3"},  {"3", "4"},
                                       {"5", "6"},  {"5", "7"},  {"3", "8"},
                                       {"9", "10"}, {"11", "12"}};

  std::vector<CFGBuilder::Update> Updates = {{Insert, {"4", "5"}},
                                             {Insert, {"8", "9"}},
                                             {Insert, {"10", "12"}},
                                             {Insert, {"10", "11"}}};
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDominatorTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  std::optional<CFGBuilder::Update> LastUpdate;
  while ((LastUpdate = B.applyUpdate())) {
    EXPECT_EQ(LastUpdate->Action, Insert);
    BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
    BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
    DT.insertEdge(From, To);
    EXPECT_TRUE(DT.verify());
    PDT.insertEdge(From, To);
    EXPECT_TRUE(PDT.verify());
  }
}

TEST(DominatorTree, InsertFromUnreachable) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {{"1", "2"}, {"2", "3"}, {"3", "4"}};

  std::vector<CFGBuilder::Update> Updates = {{Insert, {"3", "5"}}};
  CFGBuilder B(Holder.F, Arcs, Updates);
  PostDominatorTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  std::optional<CFGBuilder::Update> LastUpdate = B.applyUpdate();
  EXPECT_TRUE(LastUpdate);

  EXPECT_EQ(LastUpdate->Action, Insert);
  BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
  BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
  PDT.insertEdge(From, To);
  EXPECT_TRUE(PDT.verify());
  EXPECT_EQ(PDT.root_size(), 2UL);
  // Make sure we can use a const pointer with getNode.
  const BasicBlock *BB5 = B.getOrAddBlock("5");
  EXPECT_NE(PDT.getNode(BB5), nullptr);
}

TEST(DominatorTree, InsertMixed) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"},  {"3", "4"},  {"5", "6"},   {"5", "7"},
      {"8", "9"}, {"9", "10"}, {"8", "11"}, {"11", "12"}, {"7", "3"}};

  std::vector<CFGBuilder::Update> Updates = {
      {Insert, {"4", "5"}},   {Insert, {"2", "5"}},   {Insert, {"10", "9"}},
      {Insert, {"12", "10"}}, {Insert, {"12", "10"}}, {Insert, {"7", "8"}},
      {Insert, {"7", "5"}}};
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDominatorTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  std::optional<CFGBuilder::Update> LastUpdate;
  while ((LastUpdate = B.applyUpdate())) {
    EXPECT_EQ(LastUpdate->Action, Insert);
    BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
    BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
    DT.insertEdge(From, To);
    EXPECT_TRUE(DT.verify());
    PDT.insertEdge(From, To);
    EXPECT_TRUE(PDT.verify());
  }
}

TEST(DominatorTree, InsertPermut) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"},  {"3", "4"},  {"5", "6"},   {"5", "7"},
      {"8", "9"}, {"9", "10"}, {"8", "11"}, {"11", "12"}, {"7", "3"}};

  std::vector<CFGBuilder::Update> Updates = {{Insert, {"4", "5"}},
                                             {Insert, {"2", "5"}},
                                             {Insert, {"10", "9"}},
                                             {Insert, {"12", "10"}}};

  while (std::next_permutation(Updates.begin(), Updates.end(), CompUpdates)) {
    CFGHolder Holder;
    CFGBuilder B(Holder.F, Arcs, Updates);
    DominatorTree DT(*Holder.F);
    EXPECT_TRUE(DT.verify());
    PostDominatorTree PDT(*Holder.F);
    EXPECT_TRUE(PDT.verify());

    std::optional<CFGBuilder::Update> LastUpdate;
    while ((LastUpdate = B.applyUpdate())) {
      EXPECT_EQ(LastUpdate->Action, Insert);
      BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
      BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
      DT.insertEdge(From, To);
      EXPECT_TRUE(DT.verify());
      PDT.insertEdge(From, To);
      EXPECT_TRUE(PDT.verify());
    }
  }
}

TEST(DominatorTree, DeleteReachable) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"}, {"2", "4"}, {"3", "4"}, {"4", "5"},  {"5", "6"},
      {"5", "7"}, {"7", "8"}, {"3", "8"}, {"8", "9"}, {"9", "10"}, {"10", "2"}};

  std::vector<CFGBuilder::Update> Updates = {
      {Delete, {"2", "4"}}, {Delete, {"7", "8"}}, {Delete, {"10", "2"}}};
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDominatorTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  std::optional<CFGBuilder::Update> LastUpdate;
  while ((LastUpdate = B.applyUpdate())) {
    EXPECT_EQ(LastUpdate->Action, Delete);
    BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
    BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
    DT.deleteEdge(From, To);
    EXPECT_TRUE(DT.verify());
    PDT.deleteEdge(From, To);
    EXPECT_TRUE(PDT.verify());
  }
}

TEST(DominatorTree, DeleteUnreachable) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"}, {"3", "4"}, {"4", "5"},  {"5", "6"}, {"5", "7"},
      {"7", "8"}, {"3", "8"}, {"8", "9"}, {"9", "10"}, {"10", "2"}};

  std::vector<CFGBuilder::Update> Updates = {
      {Delete, {"8", "9"}}, {Delete, {"7", "8"}}, {Delete, {"3", "4"}}};
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDominatorTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  std::optional<CFGBuilder::Update> LastUpdate;
  while ((LastUpdate = B.applyUpdate())) {
    EXPECT_EQ(LastUpdate->Action, Delete);
    BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
    BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
    DT.deleteEdge(From, To);
    EXPECT_TRUE(DT.verify());
    PDT.deleteEdge(From, To);
    EXPECT_TRUE(PDT.verify());
  }
}

TEST(DominatorTree, InsertDelete) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"}, {"3", "4"},  {"4", "5"},  {"5", "6"},  {"5", "7"},
      {"3", "8"}, {"8", "9"}, {"9", "10"}, {"8", "11"}, {"11", "12"}};

  std::vector<CFGBuilder::Update> Updates = {
      {Insert, {"2", "4"}},  {Insert, {"12", "10"}}, {Insert, {"10", "9"}},
      {Insert, {"7", "6"}},  {Insert, {"7", "5"}},   {Delete, {"3", "8"}},
      {Insert, {"10", "7"}}, {Insert, {"2", "8"}},   {Delete, {"3", "4"}},
      {Delete, {"8", "9"}},  {Delete, {"11", "12"}}};

  CFGHolder Holder;
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDominatorTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  std::optional<CFGBuilder::Update> LastUpdate;
  while ((LastUpdate = B.applyUpdate())) {
    BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
    BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
    if (LastUpdate->Action == Insert) {
      DT.insertEdge(From, To);
      PDT.insertEdge(From, To);
    } else {
      DT.deleteEdge(From, To);
      PDT.deleteEdge(From, To);
    }

    EXPECT_TRUE(DT.verify());
    EXPECT_TRUE(PDT.verify());
  }
}

TEST(DominatorTree, InsertDeleteExhaustive) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"}, {"3", "4"},  {"4", "5"},  {"5", "6"},  {"5", "7"},
      {"3", "8"}, {"8", "9"}, {"9", "10"}, {"8", "11"}, {"11", "12"}};

  std::vector<CFGBuilder::Update> Updates = {
      {Insert, {"2", "4"}},  {Insert, {"12", "10"}}, {Insert, {"10", "9"}},
      {Insert, {"7", "6"}},  {Insert, {"7", "5"}},   {Delete, {"3", "8"}},
      {Insert, {"10", "7"}}, {Insert, {"2", "8"}},   {Delete, {"3", "4"}},
      {Delete, {"8", "9"}},  {Delete, {"11", "12"}}};

  std::mt19937 Generator(0);
  for (unsigned i = 0; i < 16; ++i) {
    std::shuffle(Updates.begin(), Updates.end(), Generator);
    CFGHolder Holder;
    CFGBuilder B(Holder.F, Arcs, Updates);
    DominatorTree DT(*Holder.F);
    EXPECT_TRUE(DT.verify());
    PostDominatorTree PDT(*Holder.F);
    EXPECT_TRUE(PDT.verify());

    std::optional<CFGBuilder::Update> LastUpdate;
    while ((LastUpdate = B.applyUpdate())) {
      BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
      BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
      if (LastUpdate->Action == Insert) {
        DT.insertEdge(From, To);
        PDT.insertEdge(From, To);
      } else {
        DT.deleteEdge(From, To);
        PDT.deleteEdge(From, To);
      }

      EXPECT_TRUE(DT.verify());
      EXPECT_TRUE(PDT.verify());
    }
  }
}

TEST(DominatorTree, InsertIntoIrreducible) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"0", "1"},
      {"1", "27"}, {"1", "7"},
      {"10", "18"},
      {"13", "10"},
      {"18", "13"}, {"18", "23"},
      {"23", "13"}, {"23", "24"},
      {"24", "1"}, {"24", "18"},
      {"27", "24"}};

  CFGHolder Holder;
  CFGBuilder B(Holder.F, Arcs, {{Insert, {"7", "23"}}});
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());

  B.applyUpdate();
  BasicBlock *From = B.getOrAddBlock("7");
  BasicBlock *To = B.getOrAddBlock("23");
  DT.insertEdge(From, To);

  EXPECT_TRUE(DT.verify());
}

TEST(DominatorTree, EdgeDomination) {
  StringRef ModuleString = "define i32 @f(i1 %cond) {\n"
                           " bb0:\n"
                           "   br i1 %cond, label %bb1, label %bb2\n"
                           " bb1:\n"
                           "   br label %bb3\n"
                           " bb2:\n"
                           "   br label %bb3\n"
                           " bb3:\n"
                           "   ret i32 4"
                           "}\n";

  // Parse the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(*M, "f",
                 [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
    Function::iterator FI = F.begin();

    BasicBlock *BB0 = &*FI++;
    BasicBlock *BB1 = &*FI++;
    BasicBlock *BB2 = &*FI++;
    BasicBlock *BB3 = &*FI++;

    BasicBlockEdge E01(BB0, BB1);
    BasicBlockEdge E02(BB0, BB2);
    BasicBlockEdge E13(BB1, BB3);
    BasicBlockEdge E23(BB2, BB3);

    EXPECT_TRUE(DT->dominates(E01, E01));
    EXPECT_FALSE(DT->dominates(E01, E02));
    EXPECT_TRUE(DT->dominates(E01, E13));
    EXPECT_FALSE(DT->dominates(E01, E23));

    EXPECT_FALSE(DT->dominates(E02, E01));
    EXPECT_TRUE(DT->dominates(E02, E02));
    EXPECT_FALSE(DT->dominates(E02, E13));
    EXPECT_TRUE(DT->dominates(E02, E23));

    EXPECT_FALSE(DT->dominates(E13, E01));
    EXPECT_FALSE(DT->dominates(E13, E02));
    EXPECT_TRUE(DT->dominates(E13, E13));
    EXPECT_FALSE(DT->dominates(E13, E23));

    EXPECT_FALSE(DT->dominates(E23, E01));
    EXPECT_FALSE(DT->dominates(E23, E02));
    EXPECT_FALSE(DT->dominates(E23, E13));
    EXPECT_TRUE(DT->dominates(E23, E23));
  });
}

TEST(DominatorTree, ValueDomination) {
  StringRef ModuleString = R"(
    @foo = global i8 0
    define i8 @f(i8 %arg) {
      ret i8 %arg
    }
  )";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(*M, "f",
                 [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
    Argument *A = F.getArg(0);
    GlobalValue *G = M->getNamedValue("foo");
    Constant *C = ConstantInt::getNullValue(Type::getInt8Ty(Context));

    Instruction *I = F.getEntryBlock().getTerminator();
    EXPECT_TRUE(DT->dominates(A, I));
    EXPECT_TRUE(DT->dominates(G, I));
    EXPECT_TRUE(DT->dominates(C, I));

    const Use &U = I->getOperandUse(0);
    EXPECT_TRUE(DT->dominates(A, U));
    EXPECT_TRUE(DT->dominates(G, U));
    EXPECT_TRUE(DT->dominates(C, U));
  });
}
TEST(DominatorTree, CallBrDomination) {
  StringRef ModuleString = R"(
define void @x() {
  %y = alloca i32
  %w = callbr i32 asm "", "=r,!i"()
          to label %asm.fallthrough [label %z]

asm.fallthrough:
  br label %cleanup

z:
  store i32 %w, ptr %y
  br label %cleanup

cleanup:
  ret void
})";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  runWithDomTree(
      *M, "x", [&](Function &F, DominatorTree *DT, PostDominatorTree *PDT) {
        Function::iterator FI = F.begin();

        BasicBlock *Entry = &*FI++;
        BasicBlock *ASMFallthrough = &*FI++;
        BasicBlock *Z = &*FI++;

        EXPECT_TRUE(DT->dominates(Entry, ASMFallthrough));
        EXPECT_TRUE(DT->dominates(Entry, Z));

        BasicBlock::iterator BBI = Entry->begin();
        ++BBI;
        Instruction &I = *BBI;
        EXPECT_TRUE(isa<CallBrInst>(I));
        EXPECT_TRUE(isa<Value>(I));
        for (const User *U : I.users()) {
          EXPECT_TRUE(isa<Instruction>(U));
          EXPECT_TRUE(DT->dominates(cast<Value>(&I), cast<Instruction>(U)));
        }
      });
}
