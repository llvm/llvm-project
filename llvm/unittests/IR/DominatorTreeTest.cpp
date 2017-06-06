//===- llvm/unittests/IR/DominatorTreeTest.cpp - Constants unit tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PostDominators.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

/// Build the dominator tree for the function and run the Test.
static void
runWithDomTree(Module &M, StringRef FuncName,
               function_ref<void(Function &F, DominatorTree *DT,
                                 DominatorTreeBase<BasicBlock> *PDT)>
                   Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;
  // Compute the dominator tree for the function.
  DominatorTree DT(*F);
  DominatorTreeBase<BasicBlock> PDT(/*isPostDom*/ true);
  PDT.recalculate(*F);
  Test(*F, &DT, &PDT);
}

static std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                              StringRef ModuleStr) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad assembly?");
  return M;
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
      *M, "f",
      [&](Function &F, DominatorTree *DT, DominatorTreeBase<BasicBlock> *PDT) {
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
        EXPECT_EQ(DT->getNode(BB0)->getDFSNumIn(), 0UL);
        EXPECT_EQ(DT->getNode(BB0)->getDFSNumOut(), 7UL);
        EXPECT_EQ(DT->getNode(BB1)->getDFSNumIn(), 1UL);
        EXPECT_EQ(DT->getNode(BB1)->getDFSNumOut(), 2UL);
        EXPECT_EQ(DT->getNode(BB2)->getDFSNumIn(), 5UL);
        EXPECT_EQ(DT->getNode(BB2)->getDFSNumOut(), 6UL);
        EXPECT_EQ(DT->getNode(BB4)->getDFSNumIn(), 3UL);
        EXPECT_EQ(DT->getNode(BB4)->getDFSNumOut(), 4UL);

        // Reattach block 3 to block 1 and recalculate
        BB1->getTerminator()->eraseFromParent();
        BranchInst::Create(BB4, BB3, ConstantInt::getTrue(F.getContext()), BB1);
        DT->recalculate(F);

        // Check DFS Numbers after
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

        // Change root node
        DT->verifyDomTree();
        BasicBlock *NewEntry =
            BasicBlock::Create(F.getContext(), "new_entry", &F, BB0);
        BranchInst::Create(BB0, NewEntry);
        EXPECT_EQ(F.begin()->getName(), NewEntry->getName());
        EXPECT_TRUE(&F.getEntryBlock() == NewEntry);
        DT->setNewRoot(NewEntry);
        DT->verifyDomTree();
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
      *M, "f",
      [&](Function &F, DominatorTree *DT, DominatorTreeBase<BasicBlock> *PDT) {
        Function::iterator FI = F.begin();

        BasicBlock *BB0 = &*FI++;
        BasicBlock *BB1 = &*FI++;
        BasicBlock *BB2 = &*FI++;

        const TerminatorInst *TI = BB0->getTerminator();
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
