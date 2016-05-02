//===- UnrollAnalyzerTest.cpp - UnrollAnalyzer unit tests -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Analysis/LoopUnrollAnalyzer.h"
#include "llvm/IR/Dominators.h"
#include "gtest/gtest.h"

using namespace llvm;
namespace llvm {
void initializeUnrollAnalyzerTestPass(PassRegistry &);

static SmallVector<DenseMap<Value *, Constant *>, 16> SimplifiedValuesVector;
static unsigned TripCount = 0;

namespace {
struct UnrollAnalyzerTest : public FunctionPass {
  static char ID;
  bool runOnFunction(Function &F) override {
    LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    ScalarEvolution *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();

    Function::iterator FI = F.begin();
    FI++; // First basic block is entry - skip it.
    BasicBlock *Header = &*FI++;
    Loop *L = LI->getLoopFor(Header);
    BasicBlock *Exiting = L->getExitingBlock();

    SimplifiedValuesVector.clear();
    TripCount = SE->getSmallConstantTripCount(L, Exiting);
    for (unsigned Iteration = 0; Iteration < TripCount; Iteration++) {
      DenseMap<Value *, Constant *> SimplifiedValues;
      UnrolledInstAnalyzer Analyzer(Iteration, SimplifiedValues, *SE, L);
      for (auto *BB : L->getBlocks())
        for (Instruction &I : *BB)
          Analyzer.visit(I);
      SimplifiedValuesVector.push_back(SimplifiedValues);
    }
    return false;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.setPreservesAll();
  }
  UnrollAnalyzerTest() : FunctionPass(ID) {
    initializeUnrollAnalyzerTestPass(*PassRegistry::getPassRegistry());
  }
};
}

char UnrollAnalyzerTest::ID = 0;

std::unique_ptr<Module> makeLLVMModule(UnrollAnalyzerTest *P,
                                       const char *ModuleStr) {
  LLVMContext &C = getGlobalContext();
  SMDiagnostic Err;
  return parseAssemblyString(ModuleStr, Err, C);
}

TEST(UnrollAnalyzerTest, BasicSimplifications) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define i64 @propagate_loop_phis() {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]\n"
      "  %x0 = phi i64 [ 0, %entry ], [ %x2, %loop ]\n"
      "  %x1 = or i64 %x0, 1\n"
      "  %x2 = or i64 %x1, 2\n"
      "  %inc = add nuw nsw i64 %iv, 1\n"
      "  %cond = icmp sge i64 %inc, 8\n"
      "  br i1 %cond, label %loop.end, label %loop\n"
      "loop.end:\n"
      "  %x.lcssa = phi i64 [ %x2, %loop ]\n"
      "  ret i64 %x.lcssa\n"
      "}\n";
  UnrollAnalyzerTest *P = new UnrollAnalyzerTest();
  std::unique_ptr<Module> M = makeLLVMModule(P, ModuleStr);
  legacy::PassManager Passes;
  Passes.add(P);
  Passes.run(*M);

  // Perform checks
  Module::iterator MI = M->begin();
  Function *F = &*MI++;
  Function::iterator FI = F->begin();
  FI++; // First basic block is entry - skip it.
  BasicBlock *Header = &*FI++;

  BasicBlock::iterator BBI = Header->begin();
  std::advance(BBI, 4);
  Instruction *Y1 = &*BBI++;
  Instruction *Y2 = &*BBI++;
  // Check simplification expected on the 1st iteration.
  // Check that "%inc = add nuw nsw i64 %iv, 1" is simplified to 1
  auto I1 = SimplifiedValuesVector[0].find(Y1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[0].end());
  EXPECT_EQ(dyn_cast<ConstantInt>((*I1).second)->getZExtValue(), 1U);

  // Check that "%cond = icmp sge i64 %inc, 10" is simplified to false
  auto I2 = SimplifiedValuesVector[0].find(Y2);
  EXPECT_TRUE(I2 != SimplifiedValuesVector[0].end());
  EXPECT_FALSE(dyn_cast<ConstantInt>((*I2).second)->getZExtValue());

  // Check simplification expected on the last iteration.
  // Check that "%inc = add nuw nsw i64 %iv, 1" is simplified to 8
  I1 = SimplifiedValuesVector[TripCount - 1].find(Y1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[TripCount - 1].end());
  EXPECT_EQ(dyn_cast<ConstantInt>((*I1).second)->getZExtValue(), TripCount);

  // Check that "%cond = icmp sge i64 %inc, 10" is simplified to false
  I2 = SimplifiedValuesVector[TripCount - 1].find(Y2);
  EXPECT_TRUE(I2 != SimplifiedValuesVector[TripCount - 1].end());
  EXPECT_TRUE(dyn_cast<ConstantInt>((*I2).second)->getZExtValue());
}

TEST(UnrollAnalyzerTest, OuterLoopSimplification) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo() {\n"
      "entry:\n"
      "  br label %outer.loop\n"
      "outer.loop:\n"
      "  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.loop.latch ]\n"
      "  br label %inner.loop\n"
      "inner.loop:\n"
      "  %iv.inner = phi i64 [ 0, %outer.loop ], [ %iv.inner.next, %inner.loop ]\n"
      "  %iv.inner.next = add nuw nsw i64 %iv.inner, 1\n"
      "  %exitcond.inner = icmp eq i64 %iv.inner.next, 1000\n"
      "  br i1 %exitcond.inner, label %outer.loop.latch, label %inner.loop\n"
      "outer.loop.latch:\n"
      "  %iv.outer.next = add nuw nsw i64 %iv.outer, 1\n"
      "  %exitcond.outer = icmp eq i64 %iv.outer.next, 40\n"
      "  br i1 %exitcond.outer, label %exit, label %outer.loop\n"
      "exit:\n"
      "  ret void\n"
      "}\n";

  UnrollAnalyzerTest *P = new UnrollAnalyzerTest();
  std::unique_ptr<Module> M = makeLLVMModule(P, ModuleStr);
  legacy::PassManager Passes;
  Passes.add(P);
  Passes.run(*M);

  Module::iterator MI = M->begin();
  Function *F = &*MI++;
  Function::iterator FI = F->begin();
  FI++;
  BasicBlock *Header = &*FI++;
  BasicBlock *InnerBody = &*FI++;

  BasicBlock::iterator BBI = Header->begin();
  Instruction *Y1 = &*BBI++;
  BBI = InnerBody->begin();
  Instruction *Y2 = &*BBI++;
  // Check that we can simplify IV of the outer loop, but can't simplify the IV
  // of the inner loop if we only know the iteration number of the outer loop.
  auto I1 = SimplifiedValuesVector[0].find(Y1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[0].end());
  auto I2 = SimplifiedValuesVector[0].find(Y2);
  EXPECT_TRUE(I2 == SimplifiedValuesVector[0].end());
}
} // end namespace llvm

INITIALIZE_PASS_BEGIN(UnrollAnalyzerTest, "unrollanalyzertestpass",
                      "unrollanalyzertestpass", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(UnrollAnalyzerTest, "unrollanalyzertestpass",
                    "unrollanalyzertestpass", false, false)
