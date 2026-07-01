//===- llvm/unittests/Transforms/IPO/PGOVerifyTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if AOCC_BUILD
#include "llvm/Transforms/IPO/PGOVerify.h"

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class PGOVerifyTest : public ::testing::Test {
protected:
  using AllBlockFreqInfo = IPGOVerifier::AllBlockFreqInfo;

  LLVMContext Context;
  std::unique_ptr<Module> M;
  IPGOVerifier Verifier;

  void SetUp() override {
    M = std::make_unique<Module>("test_module", Context);
  }

  void expectFunctionIsValid(const Function *F) {
    ASSERT_NE(F, nullptr);
    std::string Error;
    raw_string_ostream OS(Error);
    EXPECT_FALSE(verifyFunction(*F, &OS)) << OS.str();
  }

  Function *createIfElseReturnFunction(StringRef FuncName) {
    auto *F =
        Function::Create(FunctionType::get(Type::getInt32Ty(Context),
                                           {Type::getInt1Ty(Context)}, false),
                         Function::ExternalLinkage, FuncName, M.get());

    IRBuilder<> Builder(Context);
    BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
    BasicBlock *ThenBB = BasicBlock::Create(Context, "then", F);
    BasicBlock *ElseBB = BasicBlock::Create(Context, "else", F);

    Builder.SetInsertPoint(Entry);
    Value *Cond = F->arg_begin();
    Builder.CreateCondBr(Cond, ThenBB, ElseBB);

    Builder.SetInsertPoint(ThenBB);
    Builder.CreateRet(ConstantInt::get(Type::getInt32Ty(Context), 1));

    Builder.SetInsertPoint(ElseBB);
    Builder.CreateRet(ConstantInt::get(Type::getInt32Ty(Context), 0));

    return F;
  }

  void setBranchWeights(Instruction *Term, ArrayRef<uint32_t> Weights) {
    ASSERT_NE(Term, nullptr);
    MDBuilder MDB(Context);
    Term->setMetadata(LLVMContext::MD_prof, MDB.createBranchWeights(Weights));
  }

  AllBlockFreqInfo computeFrequenciesFor(Function *F) {
    DominatorTree DT(*F);
    LoopInfo LI(DT);
    BranchProbabilityInfo BPI(*F, LI, nullptr, &DT, nullptr);
    BlockFrequencyInfo BFI(*F, BPI, LI);
    Verifier.computeBlockFrequencies(F, BFI);
    const AllBlockFreqInfo *Info = Verifier.getCachedBlockFreqInfo(F);
    return Info ? *Info : AllBlockFreqInfo();
  }
};

TEST_F(PGOVerifyTest, ComputeBlockFrequenciesSingleReturnBlock) {
  auto *F =
      Function::Create(FunctionType::get(Type::getInt32Ty(Context), {}, false),
                       Function::ExternalLinkage, "single_ret", M.get());

  BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
  IRBuilder<> Builder(Entry);
  Builder.CreateRet(ConstantInt::get(Type::getInt32Ty(Context), 0));
  expectFunctionIsValid(F);

  F->setEntryCount(42);
  AllBlockFreqInfo Info = computeFrequenciesFor(F);

  auto It = Info.find(Entry);
  ASSERT_NE(It, Info.end());
  EXPECT_EQ(It->second.numUnknownIn, 0u);
  EXPECT_EQ(It->second.numUnknownOut, 0u);
  EXPECT_EQ(It->second.sumIn, 42u);
  EXPECT_EQ(It->second.sumOut, 42u);
}

TEST_F(PGOVerifyTest, ComputeBlockFrequenciesWeightedBranchToReturns) {
  Function *F = createIfElseReturnFunction("ifelse_ret");
  ASSERT_NE(F, nullptr);
  expectFunctionIsValid(F);

  BasicBlock *Entry = nullptr;
  BasicBlock *ThenBB = nullptr;
  BasicBlock *ElseBB = nullptr;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == "entry")
      Entry = &BB;
    else if (BB.getName() == "then")
      ThenBB = &BB;
    else if (BB.getName() == "else")
      ElseBB = &BB;
  }

  ASSERT_NE(Entry, nullptr);
  ASSERT_NE(ThenBB, nullptr);
  ASSERT_NE(ElseBB, nullptr);

  setBranchWeights(Entry->getTerminator(), {30, 70});
  F->setEntryCount(100);

  AllBlockFreqInfo Info = computeFrequenciesFor(F);

  auto EntryIt = Info.find(Entry);
  auto ThenIt = Info.find(ThenBB);
  auto ElseIt = Info.find(ElseBB);
  ASSERT_NE(EntryIt, Info.end());
  ASSERT_NE(ThenIt, Info.end());
  ASSERT_NE(ElseIt, Info.end());

  EXPECT_EQ(EntryIt->second.sumOut, 100u);
  EXPECT_EQ(ThenIt->second.sumIn, 30u);
  EXPECT_EQ(ThenIt->second.sumOut, 30u);
  EXPECT_EQ(ElseIt->second.sumIn, 70u);
  EXPECT_EQ(ElseIt->second.sumOut, 70u);
  EXPECT_EQ(ThenIt->second.numUnknownIn, 0u);
  EXPECT_EQ(ThenIt->second.numUnknownOut, 0u);
  EXPECT_EQ(ElseIt->second.numUnknownIn, 0u);
  EXPECT_EQ(ElseIt->second.numUnknownOut, 0u);
}

} // namespace
#endif // AOCC_BUILD
