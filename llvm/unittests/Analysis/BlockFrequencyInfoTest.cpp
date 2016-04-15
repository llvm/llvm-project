//===- BlockFrequencyInfoTest.cpp - BlockFrequencyInfo unit tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class BlockFrequencyInfoTest : public testing::Test {
protected:
  std::unique_ptr<BranchProbabilityInfo> BPI;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;
  LLVMContext C;

  BlockFrequencyInfo buildBFI(Function &F) {
    DT.reset(new DominatorTree(F));
    LI.reset(new LoopInfo(*DT));
    BPI.reset(new BranchProbabilityInfo(F, *LI));
    return BlockFrequencyInfo(F, *BPI, *LI);
  }
  std::unique_ptr<Module> makeLLVMModule() {
    const char *ModuleStrig = "define i32 @f(i32 %x) {\n"
                              "bb0:\n"
                              "  %y1 = icmp eq i32 %x, 0 \n"
                              "  br i1 %y1, label %bb1, label %bb2 \n"
                              "bb1:\n"
                              "  br label %bb3\n"
                              "bb2:\n"
                              "  br label %bb3\n"
                              "bb3:\n"
                              "  %y2 = phi i32 [0, %bb1], [1, %bb2] \n"
                              "  ret i32 %y2\n"
                              "}\n";
    SMDiagnostic Err;
    return parseAssemblyString(ModuleStrig, Err, C);
  }
};

TEST_F(BlockFrequencyInfoTest, Basic) {
  auto M = makeLLVMModule();
  Function *F = M->getFunction("f");
  F->setEntryCount(100);

  BlockFrequencyInfo BFI = buildBFI(*F);
  BasicBlock &BB0 = F->getEntryBlock();
  BasicBlock *BB1 = BB0.getTerminator()->getSuccessor(0);
  BasicBlock *BB2 = BB0.getTerminator()->getSuccessor(1);
  BasicBlock *BB3 = BB1->getSingleSuccessor();

  uint64_t BB0Freq = BFI.getBlockFreq(&BB0).getFrequency();
  uint64_t BB1Freq = BFI.getBlockFreq(BB1).getFrequency();
  uint64_t BB2Freq = BFI.getBlockFreq(BB2).getFrequency();
  uint64_t BB3Freq = BFI.getBlockFreq(BB3).getFrequency();

  EXPECT_EQ(BB0Freq, BB3Freq);
  EXPECT_EQ(BB0Freq, BB1Freq + BB2Freq);
  EXPECT_EQ(BB0Freq, BB3Freq);

  EXPECT_EQ(BFI.getBlockProfileCount(&BB0).getValue(), UINT64_C(100));
  EXPECT_EQ(BFI.getBlockProfileCount(BB3).getValue(), UINT64_C(100));
  EXPECT_EQ(BFI.getBlockProfileCount(BB1).getValue(), 100 * BB1Freq / BB0Freq);
  EXPECT_EQ(BFI.getBlockProfileCount(BB2).getValue(), 100 * BB2Freq / BB0Freq);
}

} // end anonymous namespace
} // end namespace llvm
