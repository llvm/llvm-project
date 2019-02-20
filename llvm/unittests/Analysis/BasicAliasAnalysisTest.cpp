//===- BasicAliasAnalysisTest.cpp - Unit tests for BasicAA ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Targeted tests that are hard/convoluted to make happen with just `opt`.
//

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/AliasAnalysis.h"
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

    TestAnalyses(BasicAATest &Test)
        : DT(*Test.F), AC(*Test.F), BAA(Test.DL, *Test.F, Test.TLI, AC, &DT) {}
  };

  llvm::Optional<TestAnalyses> Analyses;

  BasicAAResult &setupAnalyses() {
    assert(F);
    Analyses.emplace(*this);
    return Analyses->BAA;
  }

public:
  BasicAATest()
      : M("BasicAATest", C), B(C), DL(DLString), TLI(TLII), F(nullptr) {}
};

// Check that a function arg can't trivially alias a global when we're accessing
// >sizeof(global) bytes through that arg, unless the access size is just an
// upper-bound.
TEST_F(BasicAATest, AliasInstWithObjectOfImpreciseSize) {
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt32Ty()->getPointerTo()}, false),
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

  BasicAAResult &BasicAA = setupAnalyses();
  ASSERT_EQ(
      BasicAA.alias(MemoryLocation(IncomingI32Ptr, LocationSize::precise(4)),
                    MemoryLocation(GlobalPtr, LocationSize::precise(1))),
      AliasResult::NoAlias);

  ASSERT_EQ(
      BasicAA.alias(MemoryLocation(IncomingI32Ptr, LocationSize::upperBound(4)),
                    MemoryLocation(GlobalPtr, LocationSize::precise(1))),
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
      cast<GetElementPtrInst>(B.CreateGEP(I8, ArbitraryI32));

  BasicAAResult &BasicAA = setupAnalyses();
  ASSERT_EQ(BasicAA.alias(
                MemoryLocation(I8, LocationSize::precise(2)),
                MemoryLocation(I8AtUncertainOffset, LocationSize::precise(1))),
            AliasResult::PartialAlias);

  ASSERT_EQ(BasicAA.alias(
                MemoryLocation(I8, LocationSize::upperBound(2)),
                MemoryLocation(I8AtUncertainOffset, LocationSize::precise(1))),
            AliasResult::MayAlias);
}
