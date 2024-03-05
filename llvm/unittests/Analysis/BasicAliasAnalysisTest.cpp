//===- BasicAliasAnalysisTest.cpp - Unit tests for BasicAA ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Targeted tests that are hard/convoluted to make happen with just `opt`.
//

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

// FIXME: This is duplicated between this file and MemorySSATest. Refactor.
const static char DLString[] = "e-i64:64-f80:128-n8:16:32:64-S128";

/// There's a lot of common setup between these tests. This fixture helps reduce
/// that. Tests should mock up a function, store it in F, and then call
/// setupAnalyses().
class BasicAATest : public testing::Test {
protected:
  // N.B. Many of these members depend on each other (e.g. the Module depends on
  // the Context, etc.). So, order matters here (and in TestAnalyses).
  LLVMContext C;
  Module M;
  IRBuilder<> B;
  DataLayout DL;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  Function *F;

  // Things that we need to build after the function is created.
  struct TestAnalyses {
    DominatorTree DT;
    AssumptionCache AC;
    BasicAAResult BAA;
    AAResults AAR;
    SimpleAAQueryInfo AAQI;

    TestAnalyses(BasicAATest &Test)
        : DT(*Test.F), AC(*Test.F), BAA(Test.DL, *Test.F, Test.TLI, AC, &DT),
          AAR(Test.TLI), AAQI(AAR) {
      AAR.addAAResult(BAA);
    }
  };

  std::optional<TestAnalyses> Analyses;

  TestAnalyses &setupAnalyses() {
    assert(F);
    Analyses.emplace(*this);
    return *Analyses;
  }

public:
  BasicAATest()
      : M("BasicAATest", C), B(C), DL(DLString), TLI(TLII), F(nullptr) {}
};

// Check that a function arg can't trivially alias a global when we're accessing
// >sizeof(global) bytes through that arg, unless the access size is just an
// upper-bound.
TEST_F(BasicAATest, AliasInstWithObjectOfImpreciseSize) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);

  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);

  Value *IncomingI32Ptr = F->arg_begin();

  auto *GlobalPtr =
      cast<GlobalVariable>(M.getOrInsertGlobal("some_global", B.getInt8Ty()));

  // Without sufficiently restricted linkage/an init, some of the object size
  // checking bits get more conservative.
  GlobalPtr->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
  GlobalPtr->setInitializer(B.getInt8(0));

  auto &AllAnalyses = setupAnalyses();
  BasicAAResult &BasicAA = AllAnalyses.BAA;
  AAQueryInfo &AAQI = AllAnalyses.AAQI;
  ASSERT_EQ(
      BasicAA.alias(MemoryLocation(IncomingI32Ptr, LocationSize::precise(4)),
                    MemoryLocation(GlobalPtr, LocationSize::precise(1)), AAQI,
                    nullptr),
      AliasResult::NoAlias);

  ASSERT_EQ(
      BasicAA.alias(MemoryLocation(IncomingI32Ptr, LocationSize::upperBound(4)),
                    MemoryLocation(GlobalPtr, LocationSize::precise(1)), AAQI,
                    nullptr),
      AliasResult::MayAlias);
}

// Check that we fall back to MayAlias if we see an access of an entire object
// that's just an upper-bound.
TEST_F(BasicAATest, AliasInstWithFullObjectOfImpreciseSize) {
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt64Ty()}, false),
      GlobalValue::ExternalLinkage, "F", &M);

  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);

  Value *ArbitraryI32 = F->arg_begin();
  AllocaInst *I8 = B.CreateAlloca(B.getInt8Ty(), B.getInt32(2));
  auto *I8AtUncertainOffset =
      cast<GetElementPtrInst>(B.CreatePtrAdd(I8, ArbitraryI32));

  auto &AllAnalyses = setupAnalyses();
  BasicAAResult &BasicAA = AllAnalyses.BAA;
  AAQueryInfo &AAQI = AllAnalyses.AAQI;
  ASSERT_EQ(BasicAA.alias(
                MemoryLocation(I8, LocationSize::precise(2)),
                MemoryLocation(I8AtUncertainOffset, LocationSize::precise(1)),
                AAQI, nullptr),
            AliasResult::PartialAlias);

  ASSERT_EQ(BasicAA.alias(
                MemoryLocation(I8, LocationSize::upperBound(2)),
                MemoryLocation(I8AtUncertainOffset, LocationSize::precise(1)),
                AAQI, nullptr),
            AliasResult::MayAlias);
}

TEST_F(BasicAATest, PartialAliasOffsetPhi) {
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getPtrTy(), B.getInt1Ty()}, false),
      GlobalValue::ExternalLinkage, "F", &M);

  Value *Ptr = F->arg_begin();
  Value *I = F->arg_begin() + 1;

  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *B1(BasicBlock::Create(C, "", F));
  BasicBlock *B2(BasicBlock::Create(C, "", F));
  BasicBlock *End(BasicBlock::Create(C, "", F));

  B.SetInsertPoint(Entry);
  B.CreateCondBr(I, B1, B2);

  B.SetInsertPoint(B1);
  auto *Ptr1 = cast<GetElementPtrInst>(B.CreatePtrAdd(Ptr, B.getInt32(1)));
  B.CreateBr(End);

  B.SetInsertPoint(B2);
  auto *Ptr2 = cast<GetElementPtrInst>(B.CreatePtrAdd(Ptr, B.getInt32(1)));
  B.CreateBr(End);

  B.SetInsertPoint(End);
  auto *Phi = B.CreatePHI(B.getPtrTy(), 2);
  Phi->addIncoming(Ptr1, B1);
  Phi->addIncoming(Ptr2, B2);
  B.CreateRetVoid();

  auto &AllAnalyses = setupAnalyses();
  BasicAAResult &BasicAA = AllAnalyses.BAA;
  AAQueryInfo &AAQI = AllAnalyses.AAQI;
  AliasResult AR = BasicAA.alias(MemoryLocation(Ptr, LocationSize::precise(2)),
                                 MemoryLocation(Phi, LocationSize::precise(1)),
                                 AAQI, nullptr);
  ASSERT_EQ(AR.getOffset(), 1);
}

TEST_F(BasicAATest, PartialAliasOffsetSelect) {
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getPtrTy(), B.getInt1Ty()}, false),
      GlobalValue::ExternalLinkage, "F", &M);

  Value *Ptr = F->arg_begin();
  Value *I = F->arg_begin() + 1;

  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);

  auto *Ptr1 = cast<GetElementPtrInst>(B.CreatePtrAdd(Ptr, B.getInt32(1)));
  auto *Ptr2 = cast<GetElementPtrInst>(B.CreatePtrAdd(Ptr, B.getInt32(1)));
  auto *Select = B.CreateSelect(I, Ptr1, Ptr2);
  B.CreateRetVoid();

  auto &AllAnalyses = setupAnalyses();
  BasicAAResult &BasicAA = AllAnalyses.BAA;
  AAQueryInfo &AAQI = AllAnalyses.AAQI;
  AliasResult AR = BasicAA.alias(
      MemoryLocation(Ptr, LocationSize::precise(2)),
      MemoryLocation(Select, LocationSize::precise(1)), AAQI, nullptr);
  ASSERT_EQ(AR.getOffset(), 1);
}
