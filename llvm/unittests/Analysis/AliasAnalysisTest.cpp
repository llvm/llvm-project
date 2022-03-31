//===--- AliasAnalysisTest.cpp - Mixed TBAA unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

// Set up some test passes.
namespace llvm {
void initializeAATestPassPass(PassRegistry&);
void initializeTestCustomAAWrapperPassPass(PassRegistry&);
}

namespace {
struct AATestPass : FunctionPass {
  static char ID;
  AATestPass() : FunctionPass(ID) {
    initializeAATestPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.setPreservesAll();
  }

  bool runOnFunction(Function &F) override {
    AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();

    SetVector<Value *> Pointers;
    for (Argument &A : F.args())
      if (A.getType()->isPointerTy())
        Pointers.insert(&A);
    for (Instruction &I : instructions(F))
      if (I.getType()->isPointerTy())
        Pointers.insert(&I);

    for (Value *P1 : Pointers)
      for (Value *P2 : Pointers)
        (void)AA.alias(P1, LocationSize::beforeOrAfterPointer(), P2,
                       LocationSize::beforeOrAfterPointer());

    return false;
  }
};
}

char AATestPass::ID = 0;
INITIALIZE_PASS_BEGIN(AATestPass, "aa-test-pas", "Alias Analysis Test Pass",
                      false, true)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(AATestPass, "aa-test-pass", "Alias Analysis Test Pass",
                    false, true)

namespace {
/// A test customizable AA result. It merely accepts a callback to run whenever
/// it receives an alias query. Useful for testing that a particular AA result
/// is reached.
struct TestCustomAAResult : AAResultBase<TestCustomAAResult> {
  friend AAResultBase<TestCustomAAResult>;

  std::function<void()> CB;

  explicit TestCustomAAResult(std::function<void()> CB)
      : AAResultBase(), CB(std::move(CB)) {}
  TestCustomAAResult(TestCustomAAResult &&Arg)
      : AAResultBase(std::move(Arg)), CB(std::move(Arg.CB)) {}

  bool invalidate(Function &, const PreservedAnalyses &) { return false; }

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB,
                    AAQueryInfo &AAQI) {
    CB();
    return AliasResult::MayAlias;
  }
};
}

namespace {
/// A wrapper pass for the legacy pass manager to use with the above custom AA
/// result.
class TestCustomAAWrapperPass : public ImmutablePass {
  std::function<void()> CB;
  std::unique_ptr<TestCustomAAResult> Result;

public:
  static char ID;

  explicit TestCustomAAWrapperPass(
      std::function<void()> CB = std::function<void()>())
      : ImmutablePass(ID), CB(std::move(CB)) {
    initializeTestCustomAAWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool doInitialization(Module &M) override {
    Result.reset(new TestCustomAAResult(std::move(CB)));
    return true;
  }

  bool doFinalization(Module &M) override {
    Result.reset();
    return true;
  }

  TestCustomAAResult &getResult() { return *Result; }
  const TestCustomAAResult &getResult() const { return *Result; }
};
}

char TestCustomAAWrapperPass::ID = 0;
INITIALIZE_PASS_BEGIN(TestCustomAAWrapperPass, "test-custom-aa",
                "Test Custom AA Wrapper Pass", false, true)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(TestCustomAAWrapperPass, "test-custom-aa",
                "Test Custom AA Wrapper Pass", false, true)

namespace {

class AliasAnalysisTest : public testing::Test {
protected:
  LLVMContext C;
  Module M;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<BasicAAResult> BAR;
  std::unique_ptr<AAResults> AAR;

  AliasAnalysisTest() : M("AliasAnalysisTest", C), TLI(TLII) {}

  AAResults &getAAResults(Function &F) {
    // Reset the Function AA results first to clear out any references.
    AAR.reset(new AAResults(TLI));

    // Build the various AA results and register them.
    AC.reset(new AssumptionCache(F));
    BAR.reset(new BasicAAResult(M.getDataLayout(), F, TLI, *AC));
    AAR->addAAResult(*BAR);

    return *AAR;
  }
};

TEST_F(AliasAnalysisTest, getModRefInfo) {
  // Setup function.
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(C), std::vector<Type *>(), false);
  auto *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);
  auto *BB = BasicBlock::Create(C, "entry", F);
  auto IntType = Type::getInt32Ty(C);
  auto PtrType = Type::getInt32PtrTy(C);
  auto *Value = ConstantInt::get(IntType, 42);
  auto *Addr = ConstantPointerNull::get(PtrType);
  auto Alignment = Align(IntType->getBitWidth() / 8);

  auto *Store1 = new StoreInst(Value, Addr, BB);
  auto *Load1 = new LoadInst(IntType, Addr, "load", BB);
  auto *Add1 = BinaryOperator::CreateAdd(Value, Value, "add", BB);
  auto *VAArg1 = new VAArgInst(Addr, PtrType, "vaarg", BB);
  auto *CmpXChg1 = new AtomicCmpXchgInst(
      Addr, ConstantInt::get(IntType, 0), ConstantInt::get(IntType, 1),
      Alignment, AtomicOrdering::Monotonic, AtomicOrdering::Monotonic,
      SyncScope::System, BB);
  auto *AtomicRMW = new AtomicRMWInst(
      AtomicRMWInst::Xchg, Addr, ConstantInt::get(IntType, 1), Alignment,
      AtomicOrdering::Monotonic, SyncScope::System, BB);

  ReturnInst::Create(C, nullptr, BB);

  auto &AA = getAAResults(*F);

  // Check basic results
  EXPECT_EQ(AA.getModRefInfo(Store1, MemoryLocation()), ModRefInfo::Mod);
  EXPECT_EQ(AA.getModRefInfo(Store1, None), ModRefInfo::Mod);
  EXPECT_EQ(AA.getModRefInfo(Load1, MemoryLocation()), ModRefInfo::Ref);
  EXPECT_EQ(AA.getModRefInfo(Load1, None), ModRefInfo::Ref);
  EXPECT_EQ(AA.getModRefInfo(Add1, MemoryLocation()), ModRefInfo::NoModRef);
  EXPECT_EQ(AA.getModRefInfo(Add1, None), ModRefInfo::NoModRef);
  EXPECT_EQ(AA.getModRefInfo(VAArg1, MemoryLocation()), ModRefInfo::ModRef);
  EXPECT_EQ(AA.getModRefInfo(VAArg1, None), ModRefInfo::ModRef);
  EXPECT_EQ(AA.getModRefInfo(CmpXChg1, MemoryLocation()), ModRefInfo::ModRef);
  EXPECT_EQ(AA.getModRefInfo(CmpXChg1, None), ModRefInfo::ModRef);
  EXPECT_EQ(AA.getModRefInfo(AtomicRMW, MemoryLocation()), ModRefInfo::ModRef);
  EXPECT_EQ(AA.getModRefInfo(AtomicRMW, None), ModRefInfo::ModRef);
}

static Instruction *getInstructionByName(Function &F, StringRef Name) {
  for (auto &I : instructions(F))
    if (I.getName() == Name)
      return &I;
  llvm_unreachable("Expected to find instruction!");
}

TEST_F(AliasAnalysisTest, BatchAAPhiCycles) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
    define void @f(i8* noalias %a, i1 %c) {
    entry:
      br label %loop

    loop:
      %phi = phi i8* [ null, %entry ], [ %a2, %loop ]
      %offset1 = phi i64 [ 0, %entry ], [ %offset2, %loop]
      %offset2 = add i64 %offset1, 1
      %a1 = getelementptr i8, i8* %a, i64 %offset1
      %a2 = getelementptr i8, i8* %a, i64 %offset2
      %s1 = select i1 %c, i8* %a1, i8* %phi
      %s2 = select i1 %c, i8* %a2, i8* %a1
      br label %loop
    }
  )", Err, C);

  Function *F = M->getFunction("f");
  Instruction *Phi = getInstructionByName(*F, "phi");
  Instruction *A1 = getInstructionByName(*F, "a1");
  Instruction *A2 = getInstructionByName(*F, "a2");
  Instruction *S1 = getInstructionByName(*F, "s1");
  Instruction *S2 = getInstructionByName(*F, "s2");
  MemoryLocation PhiLoc(Phi, LocationSize::precise(1));
  MemoryLocation A1Loc(A1, LocationSize::precise(1));
  MemoryLocation A2Loc(A2, LocationSize::precise(1));
  MemoryLocation S1Loc(S1, LocationSize::precise(1));
  MemoryLocation S2Loc(S2, LocationSize::precise(1));

  auto &AA = getAAResults(*F);
  EXPECT_EQ(AliasResult::NoAlias, AA.alias(A1Loc, A2Loc));
  EXPECT_EQ(AliasResult::MayAlias, AA.alias(PhiLoc, A1Loc));
  EXPECT_EQ(AliasResult::MayAlias, AA.alias(S1Loc, S2Loc));

  BatchAAResults BatchAA(AA);
  EXPECT_EQ(AliasResult::NoAlias, BatchAA.alias(A1Loc, A2Loc));
  EXPECT_EQ(AliasResult::MayAlias, BatchAA.alias(PhiLoc, A1Loc));
  EXPECT_EQ(AliasResult::MayAlias, BatchAA.alias(S1Loc, S2Loc));

  BatchAAResults BatchAA2(AA);
  EXPECT_EQ(AliasResult::NoAlias, BatchAA2.alias(A1Loc, A2Loc));
  EXPECT_EQ(AliasResult::MayAlias, BatchAA2.alias(S1Loc, S2Loc));
  EXPECT_EQ(AliasResult::MayAlias, BatchAA2.alias(PhiLoc, A1Loc));
}

TEST_F(AliasAnalysisTest, BatchAAPhiAssumption) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
    define void @f(i8* %a.base, i8* %b.base, i1 %c) {
    entry:
      br label %loop

    loop:
      %a = phi i8* [ %a.next, %loop ], [ %a.base, %entry ]
      %b = phi i8* [ %b.next, %loop ], [ %b.base, %entry ]
      %a.next = getelementptr i8, i8* %a, i64 1
      %b.next = getelementptr i8, i8* %b, i64 1
      br label %loop
    }
  )", Err, C);

  Function *F = M->getFunction("f");
  Instruction *A = getInstructionByName(*F, "a");
  Instruction *B = getInstructionByName(*F, "b");
  Instruction *ANext = getInstructionByName(*F, "a.next");
  Instruction *BNext = getInstructionByName(*F, "b.next");
  MemoryLocation ALoc(A, LocationSize::precise(1));
  MemoryLocation BLoc(B, LocationSize::precise(1));
  MemoryLocation ANextLoc(ANext, LocationSize::precise(1));
  MemoryLocation BNextLoc(BNext, LocationSize::precise(1));

  auto &AA = getAAResults(*F);
  EXPECT_EQ(AliasResult::MayAlias, AA.alias(ALoc, BLoc));
  EXPECT_EQ(AliasResult::MayAlias, AA.alias(ANextLoc, BNextLoc));

  BatchAAResults BatchAA(AA);
  EXPECT_EQ(AliasResult::MayAlias, BatchAA.alias(ALoc, BLoc));
  EXPECT_EQ(AliasResult::MayAlias, BatchAA.alias(ANextLoc, BNextLoc));
}

// Check that two aliased GEPs with non-constant offsets are correctly
// analyzed and their relative offset can be requested from AA.
TEST_F(AliasAnalysisTest, PartialAliasOffset) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
    define void @foo(float* %arg, i32 %i) {
    bb:
      %i2 = zext i32 %i to i64
      %i3 = getelementptr inbounds float, float* %arg, i64 %i2
      %i4 = bitcast float* %i3 to <2 x float>*
      %L1 = load <2 x float>, <2 x float>* %i4, align 16
      %i7 = add nuw nsw i32 %i, 1
      %i8 = zext i32 %i7 to i64
      %i9 = getelementptr inbounds float, float* %arg, i64 %i8
      %L2 = load float, float* %i9, align 4
      ret void
    }
  )",
                                                  Err, C);

  if (!M)
    Err.print("PartialAliasOffset", errs());

  Function *F = M->getFunction("foo");
  const auto Loc1 = MemoryLocation::get(getInstructionByName(*F, "L1"));
  const auto Loc2 = MemoryLocation::get(getInstructionByName(*F, "L2"));

  auto &AA = getAAResults(*F);

  const auto AR = AA.alias(Loc1, Loc2);
  EXPECT_EQ(AR, AliasResult::PartialAlias);
  EXPECT_EQ(4, AR.getOffset());
}

// Check that swapping the order of parameters to `AA.alias()` changes offset
// sign and that the sign is such that FirstLoc + Offset == SecondLoc.
TEST_F(AliasAnalysisTest, PartialAliasOffsetSign) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
    define void @f(i64* %p) {
      %L1 = load i64, i64* %p
      %p.i8 = bitcast i64* %p to i8*
      %q = getelementptr i8,  i8* %p.i8, i32 1
      %L2 = load i8, i8* %q
      ret void
    }
  )",
                                                  Err, C);

  if (!M)
    Err.print("PartialAliasOffsetSign", errs());

  Function *F = M->getFunction("f");
  const auto Loc1 = MemoryLocation::get(getInstructionByName(*F, "L1"));
  const auto Loc2 = MemoryLocation::get(getInstructionByName(*F, "L2"));

  auto &AA = getAAResults(*F);

  auto AR = AA.alias(Loc1, Loc2);
  EXPECT_EQ(AR, AliasResult::PartialAlias);
  EXPECT_EQ(1, AR.getOffset());

  AR = AA.alias(Loc2, Loc1);
  EXPECT_EQ(AR, AliasResult::PartialAlias);
  EXPECT_EQ(-1, AR.getOffset());
}
class AAPassInfraTest : public testing::Test {
protected:
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M;

public:
  AAPassInfraTest()
      : M(parseAssemblyString("define i32 @f(i32* %x, i32* %y) {\n"
                              "entry:\n"
                              "  %lx = load i32, i32* %x\n"
                              "  %ly = load i32, i32* %y\n"
                              "  %sum = add i32 %lx, %ly\n"
                              "  ret i32 %sum\n"
                              "}\n",
                              Err, C)) {
    assert(M && "Failed to build the module!");
  }
};

TEST_F(AAPassInfraTest, injectExternalAA) {
  legacy::PassManager PM;

  // Register our custom AA's wrapper pass manually.
  bool IsCustomAAQueried = false;
  PM.add(new TestCustomAAWrapperPass([&] { IsCustomAAQueried = true; }));

  // Now add the external AA wrapper with a lambda which queries for the
  // wrapper around our custom AA and adds it to the results.
  PM.add(createExternalAAWrapperPass([](Pass &P, Function &, AAResults &AAR) {
    if (auto *WrapperPass = P.getAnalysisIfAvailable<TestCustomAAWrapperPass>())
      AAR.addAAResult(WrapperPass->getResult());
  }));

  // And run a pass that will make some alias queries. This will automatically
  // trigger the rest of the alias analysis stack to be run. It is analagous to
  // building a full pass pipeline with any of the existing pass manager
  // builders.
  PM.add(new AATestPass());
  PM.run(*M);

  // Finally, ensure that our custom AA was indeed queried.
  EXPECT_TRUE(IsCustomAAQueried);
}

} // end anonymous namspace
