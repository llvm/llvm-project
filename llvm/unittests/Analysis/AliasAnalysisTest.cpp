//===--- AliasAnalysisTest.cpp - Mixed TBAA unit tests --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
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
        (void)AA.alias(P1, MemoryLocation::UnknownSize, P2,
                       MemoryLocation::UnknownSize);

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

  explicit TestCustomAAResult(const TargetLibraryInfo &TLI,
                              std::function<void()> CB)
      : AAResultBase(TLI), CB(std::move(CB)) {}
  TestCustomAAResult(TestCustomAAResult &&Arg)
      : AAResultBase(std::move(Arg)), CB(std::move(Arg.CB)) {}

  bool invalidate(Function &, const PreservedAnalyses &) { return false; }

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB) {
    CB();
    return MayAlias;
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
    Result.reset(new TestCustomAAResult(
        getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(), std::move(CB)));
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
    AAR.reset(new AAResults());

    // Build the various AA results and register them.
    AC.reset(new AssumptionCache(F));
    BAR.reset(new BasicAAResult(M.getDataLayout(), TLI, *AC));
    AAR->addAAResult(*BAR);

    return *AAR;
  }
};

TEST_F(AliasAnalysisTest, getModRefInfo) {
  // Setup function.
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(C), std::vector<Type *>(), false);
  auto *F = cast<Function>(M.getOrInsertFunction("f", FTy));
  auto *BB = BasicBlock::Create(C, "entry", F);
  auto IntType = Type::getInt32Ty(C);
  auto PtrType = Type::getInt32PtrTy(C);
  auto *Value = ConstantInt::get(IntType, 42);
  auto *Addr = ConstantPointerNull::get(PtrType);

  auto *Store1 = new StoreInst(Value, Addr, BB);
  auto *Load1 = new LoadInst(Addr, "load", BB);
  auto *Add1 = BinaryOperator::CreateAdd(Value, Value, "add", BB);
  auto *VAArg1 = new VAArgInst(Addr, PtrType, "vaarg", BB);
  auto *CmpXChg1 = new AtomicCmpXchgInst(Addr, ConstantInt::get(IntType, 0),
                                         ConstantInt::get(IntType, 1),
                                         Monotonic, Monotonic, CrossThread, BB);
  auto *AtomicRMW =
      new AtomicRMWInst(AtomicRMWInst::Xchg, Addr, ConstantInt::get(IntType, 1),
                        Monotonic, CrossThread, BB);

  ReturnInst::Create(C, nullptr, BB);

  auto &AA = getAAResults(*F);

  // Check basic results
  EXPECT_EQ(AA.getModRefInfo(Store1, MemoryLocation()), MRI_Mod);
  EXPECT_EQ(AA.getModRefInfo(Store1), MRI_Mod);
  EXPECT_EQ(AA.getModRefInfo(Load1, MemoryLocation()), MRI_Ref);
  EXPECT_EQ(AA.getModRefInfo(Load1), MRI_Ref);
  EXPECT_EQ(AA.getModRefInfo(Add1, MemoryLocation()), MRI_NoModRef);
  EXPECT_EQ(AA.getModRefInfo(Add1), MRI_NoModRef);
  EXPECT_EQ(AA.getModRefInfo(VAArg1, MemoryLocation()), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(VAArg1), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(CmpXChg1, MemoryLocation()), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(CmpXChg1), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(AtomicRMW, MemoryLocation()), MRI_ModRef);
  EXPECT_EQ(AA.getModRefInfo(AtomicRMW), MRI_ModRef);
}

class AAPassInfraTest : public testing::Test {
protected:
  LLVMContext &C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M;

public:
  AAPassInfraTest()
      : C(getGlobalContext()),
        M(parseAssemblyString("define i32 @f(i32* %x, i32* %y) {\n"
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
