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

    SimplifiedValuesVector.clear();
    TripCount = SE->getSmallConstantTripCount(L, Header);
    for (unsigned Iteration = 0; Iteration < TripCount; Iteration++) {
      DenseMap<Value *, Constant *> SimplifiedValues;
      UnrolledInstAnalyzer Analyzer(Iteration, SimplifiedValues, *SE);
      for (Instruction &I : *Header)
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
} // end namespace llvm

INITIALIZE_PASS_BEGIN(UnrollAnalyzerTest, "unrollanalyzertestpass",
                      "unrollanalyzertestpass", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(UnrollAnalyzerTest, "unrollanalyzertestpass",
                    "unrollanalyzertestpass", false, false)
