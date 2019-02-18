//===- llvm/unittest/IR/PassManager.cpp - PassManager tests ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TestFunctionAnalysis : public AnalysisInfoMixin<TestFunctionAnalysis> {
public:
  struct Result {
    Result(int Count) : InstructionCount(Count) {}
    int InstructionCount;
  };

  TestFunctionAnalysis(int &Runs) : Runs(Runs) {}

  /// Run the analysis pass over the function and return a result.
  Result run(Function &F, FunctionAnalysisManager &AM) {
    ++Runs;
    int Count = 0;
    for (Function::iterator BBI = F.begin(), BBE = F.end(); BBI != BBE; ++BBI)
      for (BasicBlock::iterator II = BBI->begin(), IE = BBI->end(); II != IE;
           ++II)
        ++Count;
    return Result(Count);
  }

private:
  friend AnalysisInfoMixin<TestFunctionAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestFunctionAnalysis::Key;

class TestModuleAnalysis : public AnalysisInfoMixin<TestModuleAnalysis> {
public:
  struct Result {
    Result(int Count) : FunctionCount(Count) {}
    int FunctionCount;
  };

  TestModuleAnalysis(int &Runs) : Runs(Runs) {}

  Result run(Module &M, ModuleAnalysisManager &AM) {
    ++Runs;
    int Count = 0;
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      ++Count;
    return Result(Count);
  }

private:
  friend AnalysisInfoMixin<TestModuleAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestModuleAnalysis::Key;

struct TestModulePass : PassInfoMixin<TestModulePass> {
  TestModulePass(int &RunCount) : RunCount(RunCount) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    ++RunCount;
    return PreservedAnalyses::none();
  }

  int &RunCount;
};

struct TestPreservingModulePass : PassInfoMixin<TestPreservingModulePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    return PreservedAnalyses::all();
  }
};

struct TestFunctionPass : PassInfoMixin<TestFunctionPass> {
  TestFunctionPass(int &RunCount, int &AnalyzedInstrCount,
                   int &AnalyzedFunctionCount,
                   bool OnlyUseCachedResults = false)
      : RunCount(RunCount), AnalyzedInstrCount(AnalyzedInstrCount),
        AnalyzedFunctionCount(AnalyzedFunctionCount),
        OnlyUseCachedResults(OnlyUseCachedResults) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    ++RunCount;

    const ModuleAnalysisManager &MAM =
        AM.getResult<ModuleAnalysisManagerFunctionProxy>(F).getManager();
    if (TestModuleAnalysis::Result *TMA =
            MAM.getCachedResult<TestModuleAnalysis>(*F.getParent()))
      AnalyzedFunctionCount += TMA->FunctionCount;

    if (OnlyUseCachedResults) {
      // Hack to force the use of the cached interface.
      if (TestFunctionAnalysis::Result *AR =
              AM.getCachedResult<TestFunctionAnalysis>(F))
        AnalyzedInstrCount += AR->InstructionCount;
    } else {
      // Typical path just runs the analysis as needed.
      TestFunctionAnalysis::Result &AR = AM.getResult<TestFunctionAnalysis>(F);
      AnalyzedInstrCount += AR.InstructionCount;
    }

    return PreservedAnalyses::all();
  }

  int &RunCount;
  int &AnalyzedInstrCount;
  int &AnalyzedFunctionCount;
  bool OnlyUseCachedResults;
};

// A test function pass that invalidates all function analyses for a function
// with a specific name.
struct TestInvalidationFunctionPass
    : PassInfoMixin<TestInvalidationFunctionPass> {
  TestInvalidationFunctionPass(StringRef FunctionName) : Name(FunctionName) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    return F.getName() == Name ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
  }

  StringRef Name;
};

std::unique_ptr<Module> parseIR(LLVMContext &Context, const char *IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, Context);
}

class PassManagerTest : public ::testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;

public:
  PassManagerTest()
      : M(parseIR(Context, "define void @f() {\n"
                           "entry:\n"
                           "  call void @g()\n"
                           "  call void @h()\n"
                           "  ret void\n"
                           "}\n"
                           "define void @g() {\n"
                           "  ret void\n"
                           "}\n"
                           "define void @h() {\n"
                           "  ret void\n"
                           "}\n")) {}
};

TEST(PreservedAnalysesTest, Basic) {
  PreservedAnalyses PA1 = PreservedAnalyses();
  {
    auto PAC = PA1.getChecker<TestFunctionAnalysis>();
    EXPECT_FALSE(PAC.preserved());
    EXPECT_FALSE(PAC.preservedSet<AllAnalysesOn<Function>>());
  }
  {
    auto PAC = PA1.getChecker<TestModuleAnalysis>();
    EXPECT_FALSE(PAC.preserved());
    EXPECT_FALSE(PAC.preservedSet<AllAnalysesOn<Module>>());
  }
  auto PA2 = PreservedAnalyses::none();
  {
    auto PAC = PA2.getChecker<TestFunctionAnalysis>();
    EXPECT_FALSE(PAC.preserved());
    EXPECT_FALSE(PAC.preservedSet<AllAnalysesOn<Function>>());
  }
  auto PA3 = PreservedAnalyses::all();
  {
    auto PAC = PA3.getChecker<TestFunctionAnalysis>();
    EXPECT_TRUE(PAC.preserved());
    EXPECT_TRUE(PAC.preservedSet<AllAnalysesOn<Function>>());
  }
  PreservedAnalyses PA4 = PA1;
  {
    auto PAC = PA4.getChecker<TestFunctionAnalysis>();
    EXPECT_FALSE(PAC.preserved());
    EXPECT_FALSE(PAC.preservedSet<AllAnalysesOn<Function>>());
  }
  PA4 = PA3;
  {
    auto PAC = PA4.getChecker<TestFunctionAnalysis>();
    EXPECT_TRUE(PAC.preserved());
    EXPECT_TRUE(PAC.preservedSet<AllAnalysesOn<Function>>());
  }
  PA4 = std::move(PA2);
  {
    auto PAC = PA4.getChecker<TestFunctionAnalysis>();
    EXPECT_FALSE(PAC.preserved());
    EXPECT_FALSE(PAC.preservedSet<AllAnalysesOn<Function>>());
  }
  auto PA5 = PreservedAnalyses::allInSet<AllAnalysesOn<Function>>();
  {
    auto PAC = PA5.getChecker<TestFunctionAnalysis>();
    EXPECT_FALSE(PAC.preserved());
    EXPECT_TRUE(PAC.preservedSet<AllAnalysesOn<Function>>());
    EXPECT_FALSE(PAC.preservedSet<AllAnalysesOn<Module>>());
  }
}

TEST(PreservedAnalysesTest, Preserve) {
  auto PA = PreservedAnalyses::none();
  PA.preserve<TestFunctionAnalysis>();
  EXPECT_TRUE(PA.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(PA.getChecker<TestModuleAnalysis>().preserved());
  PA.preserve<TestModuleAnalysis>();
  EXPECT_TRUE(PA.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_TRUE(PA.getChecker<TestModuleAnalysis>().preserved());

  // Redundant calls are fine.
  PA.preserve<TestFunctionAnalysis>();
  EXPECT_TRUE(PA.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_TRUE(PA.getChecker<TestModuleAnalysis>().preserved());
}

TEST(PreservedAnalysesTest, PreserveSets) {
  auto PA = PreservedAnalyses::none();
  PA.preserveSet<AllAnalysesOn<Function>>();
  EXPECT_TRUE(PA.getChecker<TestFunctionAnalysis>()
                  .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(PA.getChecker<TestModuleAnalysis>()
                   .preservedSet<AllAnalysesOn<Module>>());
  PA.preserveSet<AllAnalysesOn<Module>>();
  EXPECT_TRUE(PA.getChecker<TestFunctionAnalysis>()
                  .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_TRUE(PA.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());

  // Mixing is fine.
  PA.preserve<TestFunctionAnalysis>();
  EXPECT_TRUE(PA.getChecker<TestFunctionAnalysis>()
                  .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_TRUE(PA.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());

  // Redundant calls are fine.
  PA.preserveSet<AllAnalysesOn<Module>>();
  EXPECT_TRUE(PA.getChecker<TestFunctionAnalysis>()
                  .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_TRUE(PA.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());
}

TEST(PreservedAnalysisTest, Intersect) {
  // Setup the initial sets.
  auto PA1 = PreservedAnalyses::none();
  PA1.preserve<TestFunctionAnalysis>();
  PA1.preserveSet<AllAnalysesOn<Module>>();
  auto PA2 = PreservedAnalyses::none();
  PA2.preserve<TestFunctionAnalysis>();
  PA2.preserveSet<AllAnalysesOn<Function>>();
  PA2.preserve<TestModuleAnalysis>();
  PA2.preserveSet<AllAnalysesOn<Module>>();
  auto PA3 = PreservedAnalyses::none();
  PA3.preserve<TestModuleAnalysis>();
  PA3.preserveSet<AllAnalysesOn<Function>>();

  // Self intersection is a no-op.
  auto Intersected = PA1;
  Intersected.intersect(PA1);
  EXPECT_TRUE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_TRUE(Intersected.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());

  // Intersecting with all is a no-op.
  Intersected.intersect(PreservedAnalyses::all());
  EXPECT_TRUE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_TRUE(Intersected.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());

  // Intersecting a narrow set with a more broad set is the narrow set.
  Intersected.intersect(PA2);
  EXPECT_TRUE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_TRUE(Intersected.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());

  // Intersecting a broad set with a more narrow set is the narrow set.
  Intersected = PA2;
  Intersected.intersect(PA1);
  EXPECT_TRUE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_TRUE(Intersected.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());

  // Intersecting with empty clears.
  Intersected.intersect(PreservedAnalyses::none());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>()
                   .preservedSet<AllAnalysesOn<Module>>());

  // Intersecting non-overlapping clears.
  Intersected = PA1;
  Intersected.intersect(PA3);
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>()
                   .preservedSet<AllAnalysesOn<Module>>());

  // Intersecting with moves works in when there is storage on both sides.
  Intersected = PA1;
  auto Tmp = PA2;
  Intersected.intersect(std::move(Tmp));
  EXPECT_TRUE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_TRUE(Intersected.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());

  // Intersecting with move works for incoming all and existing all.
  auto Tmp2 = PreservedAnalyses::all();
  Intersected.intersect(std::move(Tmp2));
  EXPECT_TRUE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_TRUE(Intersected.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());
  Intersected = PreservedAnalyses::all();
  auto Tmp3 = PA1;
  Intersected.intersect(std::move(Tmp3));
  EXPECT_TRUE(Intersected.getChecker<TestFunctionAnalysis>().preserved());
  EXPECT_FALSE(Intersected.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(Intersected.getChecker<TestModuleAnalysis>().preserved());
  EXPECT_TRUE(Intersected.getChecker<TestModuleAnalysis>()
                  .preservedSet<AllAnalysesOn<Module>>());
}

TEST(PreservedAnalysisTest, Abandon) {
  auto PA = PreservedAnalyses::none();

  // We can abandon things after they are preserved.
  PA.preserve<TestFunctionAnalysis>();
  PA.abandon<TestFunctionAnalysis>();
  EXPECT_FALSE(PA.getChecker<TestFunctionAnalysis>().preserved());

  // Repeated is fine, and abandoning if they were never preserved is fine.
  PA.abandon<TestFunctionAnalysis>();
  EXPECT_FALSE(PA.getChecker<TestFunctionAnalysis>().preserved());
  PA.abandon<TestModuleAnalysis>();
  EXPECT_FALSE(PA.getChecker<TestModuleAnalysis>().preserved());

  // Even if the sets are preserved, the abandoned analyses' checker won't
  // return true for those sets.
  PA.preserveSet<AllAnalysesOn<Function>>();
  PA.preserveSet<AllAnalysesOn<Module>>();
  EXPECT_FALSE(PA.getChecker<TestFunctionAnalysis>()
                   .preservedSet<AllAnalysesOn<Function>>());
  EXPECT_FALSE(PA.getChecker<TestModuleAnalysis>()
                   .preservedSet<AllAnalysesOn<Module>>());

  // But an arbitrary (opaque) analysis will still observe the sets as
  // preserved. This also checks that we can use an explicit ID rather than
  // a type.
  AnalysisKey FakeKey, *FakeID = &FakeKey;
  EXPECT_TRUE(PA.getChecker(FakeID).preservedSet<AllAnalysesOn<Function>>());
  EXPECT_TRUE(PA.getChecker(FakeID).preservedSet<AllAnalysesOn<Module>>());
}

TEST_F(PassManagerTest, Basic) {
  FunctionAnalysisManager FAM(/*DebugLogging*/ true);
  int FunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });

  ModuleAnalysisManager MAM(/*DebugLogging*/ true);
  int ModuleAnalysisRuns = 0;
  MAM.registerPass([&] { return TestModuleAnalysis(ModuleAnalysisRuns); });
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });

  ModulePassManager MPM;

  // Count the runs over a Function.
  int FunctionPassRunCount1 = 0;
  int AnalyzedInstrCount1 = 0;
  int AnalyzedFunctionCount1 = 0;
  {
    // Pointless scoped copy to test move assignment.
    ModulePassManager NestedMPM(/*DebugLogging*/ true);
    FunctionPassManager FPM;
    {
      // Pointless scope to test move assignment.
      FunctionPassManager NestedFPM(/*DebugLogging*/ true);
      NestedFPM.addPass(TestFunctionPass(
          FunctionPassRunCount1, AnalyzedInstrCount1, AnalyzedFunctionCount1));
      FPM = std::move(NestedFPM);
    }
    NestedMPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM = std::move(NestedMPM);
  }

  // Count the runs over a module.
  int ModulePassRunCount = 0;
  MPM.addPass(TestModulePass(ModulePassRunCount));

  // Count the runs over a Function in a separate manager.
  int FunctionPassRunCount2 = 0;
  int AnalyzedInstrCount2 = 0;
  int AnalyzedFunctionCount2 = 0;
  {
    FunctionPassManager FPM(/*DebugLogging*/ true);
    FPM.addPass(TestFunctionPass(FunctionPassRunCount2, AnalyzedInstrCount2,
                                 AnalyzedFunctionCount2));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  // A third function pass manager but with only preserving intervening passes
  // and with a function pass that invalidates exactly one analysis.
  MPM.addPass(TestPreservingModulePass());
  int FunctionPassRunCount3 = 0;
  int AnalyzedInstrCount3 = 0;
  int AnalyzedFunctionCount3 = 0;
  {
    FunctionPassManager FPM(/*DebugLogging*/ true);
    FPM.addPass(TestFunctionPass(FunctionPassRunCount3, AnalyzedInstrCount3,
                                 AnalyzedFunctionCount3));
    FPM.addPass(TestInvalidationFunctionPass("f"));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  // A fourth function pass manager but with only preserving intervening
  // passes but triggering the module analysis.
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  int FunctionPassRunCount4 = 0;
  int AnalyzedInstrCount4 = 0;
  int AnalyzedFunctionCount4 = 0;
  {
    FunctionPassManager FPM;
    FPM.addPass(TestFunctionPass(FunctionPassRunCount4, AnalyzedInstrCount4,
                                 AnalyzedFunctionCount4));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  // A fifth function pass manager which invalidates one function first but
  // uses only cached results.
  int FunctionPassRunCount5 = 0;
  int AnalyzedInstrCount5 = 0;
  int AnalyzedFunctionCount5 = 0;
  {
    FunctionPassManager FPM(/*DebugLogging*/ true);
    FPM.addPass(TestInvalidationFunctionPass("f"));
    FPM.addPass(TestFunctionPass(FunctionPassRunCount5, AnalyzedInstrCount5,
                                 AnalyzedFunctionCount5,
                                 /*OnlyUseCachedResults=*/true));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  MPM.run(*M, MAM);

  // Validate module pass counters.
  EXPECT_EQ(1, ModulePassRunCount);

  // Validate all function pass counter sets are the same.
  EXPECT_EQ(3, FunctionPassRunCount1);
  EXPECT_EQ(5, AnalyzedInstrCount1);
  EXPECT_EQ(0, AnalyzedFunctionCount1);
  EXPECT_EQ(3, FunctionPassRunCount2);
  EXPECT_EQ(5, AnalyzedInstrCount2);
  EXPECT_EQ(0, AnalyzedFunctionCount2);
  EXPECT_EQ(3, FunctionPassRunCount3);
  EXPECT_EQ(5, AnalyzedInstrCount3);
  EXPECT_EQ(0, AnalyzedFunctionCount3);
  EXPECT_EQ(3, FunctionPassRunCount4);
  EXPECT_EQ(5, AnalyzedInstrCount4);
  EXPECT_EQ(9, AnalyzedFunctionCount4);
  EXPECT_EQ(3, FunctionPassRunCount5);
  EXPECT_EQ(2, AnalyzedInstrCount5); // Only 'g' and 'h' were cached.
  EXPECT_EQ(9, AnalyzedFunctionCount5);

  // Validate the analysis counters:
  //   first run over 3 functions, then module pass invalidates
  //   second run over 3 functions, nothing invalidates
  //   third run over 0 functions, but 1 function invalidated
  //   fourth run over 1 function
  //   fifth run invalidates 1 function first, but runs over 0 functions
  EXPECT_EQ(7, FunctionAnalysisRuns);

  EXPECT_EQ(1, ModuleAnalysisRuns);
}

// A customized pass manager that passes extra arguments through the
// infrastructure.
typedef AnalysisManager<Function, int> CustomizedAnalysisManager;
typedef PassManager<Function, CustomizedAnalysisManager, int, int &>
    CustomizedPassManager;

class CustomizedAnalysis : public AnalysisInfoMixin<CustomizedAnalysis> {
public:
  struct Result {
    Result(int I) : I(I) {}
    int I;
  };

  Result run(Function &F, CustomizedAnalysisManager &AM, int I) {
    return Result(I);
  }

private:
  friend AnalysisInfoMixin<CustomizedAnalysis>;
  static AnalysisKey Key;
};

AnalysisKey CustomizedAnalysis::Key;

struct CustomizedPass : PassInfoMixin<CustomizedPass> {
  std::function<void(CustomizedAnalysis::Result &, int &)> Callback;

  template <typename CallbackT>
  CustomizedPass(CallbackT Callback) : Callback(Callback) {}

  PreservedAnalyses run(Function &F, CustomizedAnalysisManager &AM, int I,
                        int &O) {
    Callback(AM.getResult<CustomizedAnalysis>(F, I), O);
    return PreservedAnalyses::none();
  }
};

TEST_F(PassManagerTest, CustomizedPassManagerArgs) {
  CustomizedAnalysisManager AM;
  AM.registerPass([&] { return CustomizedAnalysis(); });
  PassInstrumentationCallbacks PIC;
  AM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });

  CustomizedPassManager PM;

  // Add an instance of the customized pass that just accumulates the input
  // after it is round-tripped through the analysis.
  int Result = 0;
  PM.addPass(
      CustomizedPass([](CustomizedAnalysis::Result &R, int &O) { O += R.I; }));

  // Run this over every function with the input of 42.
  for (Function &F : *M)
    PM.run(F, AM, 42, Result);

  // And ensure that we accumulated the correct result.
  EXPECT_EQ(42 * (int)M->size(), Result);
}

/// A test analysis pass which caches in its result another analysis pass and
/// uses it to serve queries. This requires the result to invalidate itself
/// when its dependency is invalidated.
struct TestIndirectFunctionAnalysis
    : public AnalysisInfoMixin<TestIndirectFunctionAnalysis> {
  struct Result {
    Result(TestFunctionAnalysis::Result &FDep, TestModuleAnalysis::Result &MDep)
        : FDep(FDep), MDep(MDep) {}
    TestFunctionAnalysis::Result &FDep;
    TestModuleAnalysis::Result &MDep;

    bool invalidate(Function &F, const PreservedAnalyses &PA,
                    FunctionAnalysisManager::Invalidator &Inv) {
      auto PAC = PA.getChecker<TestIndirectFunctionAnalysis>();
      return !(PAC.preserved() ||
               PAC.preservedSet<AllAnalysesOn<Function>>()) ||
             Inv.invalidate<TestFunctionAnalysis>(F, PA);
    }
  };

  TestIndirectFunctionAnalysis(int &Runs) : Runs(Runs) {}

  /// Run the analysis pass over the function and return a result.
  Result run(Function &F, FunctionAnalysisManager &AM) {
    ++Runs;
    auto &FDep = AM.getResult<TestFunctionAnalysis>(F);
    auto &Proxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
    const ModuleAnalysisManager &MAM = Proxy.getManager();
    // For the test, we insist that the module analysis starts off in the
    // cache.
    auto &MDep = *MAM.getCachedResult<TestModuleAnalysis>(*F.getParent());
    // And register the dependency as module analysis dependencies have to be
    // pre-registered on the proxy.
    Proxy.registerOuterAnalysisInvalidation<TestModuleAnalysis,
                                            TestIndirectFunctionAnalysis>();
    return Result(FDep, MDep);
  }

private:
  friend AnalysisInfoMixin<TestIndirectFunctionAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestIndirectFunctionAnalysis::Key;

/// A test analysis pass which chaches in its result the result from the above
/// indirect analysis pass.
///
/// This allows us to ensure that whenever an analysis pass is invalidated due
/// to dependencies (especially dependencies across IR units that trigger
/// asynchronous invalidation) we correctly detect that this may in turn cause
/// other analysis to be invalidated.
struct TestDoublyIndirectFunctionAnalysis
    : public AnalysisInfoMixin<TestDoublyIndirectFunctionAnalysis> {
  struct Result {
    Result(TestIndirectFunctionAnalysis::Result &IDep) : IDep(IDep) {}
    TestIndirectFunctionAnalysis::Result &IDep;

    bool invalidate(Function &F, const PreservedAnalyses &PA,
                    FunctionAnalysisManager::Invalidator &Inv) {
      auto PAC = PA.getChecker<TestDoublyIndirectFunctionAnalysis>();
      return !(PAC.preserved() ||
               PAC.preservedSet<AllAnalysesOn<Function>>()) ||
             Inv.invalidate<TestIndirectFunctionAnalysis>(F, PA);
    }
  };

  TestDoublyIndirectFunctionAnalysis(int &Runs) : Runs(Runs) {}

  /// Run the analysis pass over the function and return a result.
  Result run(Function &F, FunctionAnalysisManager &AM) {
    ++Runs;
    auto &IDep = AM.getResult<TestIndirectFunctionAnalysis>(F);
    return Result(IDep);
  }

private:
  friend AnalysisInfoMixin<TestDoublyIndirectFunctionAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestDoublyIndirectFunctionAnalysis::Key;

struct LambdaPass : public PassInfoMixin<LambdaPass> {
  using FuncT = std::function<PreservedAnalyses(Function &, FunctionAnalysisManager &)>;

  LambdaPass(FuncT Func) : Func(std::move(Func)) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    return Func(F, AM);
  }

  FuncT Func;
};

TEST_F(PassManagerTest, IndirectAnalysisInvalidation) {
  FunctionAnalysisManager FAM(/*DebugLogging*/ true);
  int FunctionAnalysisRuns = 0, ModuleAnalysisRuns = 0,
      IndirectAnalysisRuns = 0, DoublyIndirectAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });
  FAM.registerPass(
      [&] { return TestIndirectFunctionAnalysis(IndirectAnalysisRuns); });
  FAM.registerPass([&] {
    return TestDoublyIndirectFunctionAnalysis(DoublyIndirectAnalysisRuns);
  });

  ModuleAnalysisManager MAM(/*DebugLogging*/ true);
  MAM.registerPass([&] { return TestModuleAnalysis(ModuleAnalysisRuns); });
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

  PassInstrumentationCallbacks PIC;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });

  int InstrCount = 0, FunctionCount = 0;
  ModulePassManager MPM(/*DebugLogging*/ true);
  FunctionPassManager FPM(/*DebugLogging*/ true);
  // First just use the analysis to get the instruction count, and preserve
  // everything.
  FPM.addPass(LambdaPass([&](Function &F, FunctionAnalysisManager &AM) {
    auto &DoublyIndirectResult =
        AM.getResult<TestDoublyIndirectFunctionAnalysis>(F);
    auto &IndirectResult = DoublyIndirectResult.IDep;
    InstrCount += IndirectResult.FDep.InstructionCount;
    FunctionCount += IndirectResult.MDep.FunctionCount;
    return PreservedAnalyses::all();
  }));
  // Next, invalidate
  //   - both analyses for "f",
  //   - just the underlying (indirect) analysis for "g", and
  //   - just the direct analysis for "h".
  FPM.addPass(LambdaPass([&](Function &F, FunctionAnalysisManager &AM) {
    auto &DoublyIndirectResult =
        AM.getResult<TestDoublyIndirectFunctionAnalysis>(F);
    auto &IndirectResult = DoublyIndirectResult.IDep;
    InstrCount += IndirectResult.FDep.InstructionCount;
    FunctionCount += IndirectResult.MDep.FunctionCount;
    auto PA = PreservedAnalyses::none();
    if (F.getName() == "g")
      PA.preserve<TestFunctionAnalysis>();
    else if (F.getName() == "h")
      PA.preserve<TestIndirectFunctionAnalysis>();
    return PA;
  }));
  // Finally, use the analysis again on each function, forcing re-computation
  // for all of them.
  FPM.addPass(LambdaPass([&](Function &F, FunctionAnalysisManager &AM) {
    auto &DoublyIndirectResult =
        AM.getResult<TestDoublyIndirectFunctionAnalysis>(F);
    auto &IndirectResult = DoublyIndirectResult.IDep;
    InstrCount += IndirectResult.FDep.InstructionCount;
    FunctionCount += IndirectResult.MDep.FunctionCount;
    return PreservedAnalyses::all();
  }));

  // Create a second function pass manager. This will cause the module-level
  // invalidation to occur, which will force yet another invalidation of the
  // indirect function-level analysis as the module analysis it depends on gets
  // invalidated.
  FunctionPassManager FPM2(/*DebugLogging*/ true);
  FPM2.addPass(LambdaPass([&](Function &F, FunctionAnalysisManager &AM) {
    auto &DoublyIndirectResult =
        AM.getResult<TestDoublyIndirectFunctionAnalysis>(F);
    auto &IndirectResult = DoublyIndirectResult.IDep;
    InstrCount += IndirectResult.FDep.InstructionCount;
    FunctionCount += IndirectResult.MDep.FunctionCount;
    return PreservedAnalyses::all();
  }));

  // Add a requires pass to populate the module analysis and then our function
  // pass pipeline.
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  // Now require the module analysis again (it will have been invalidated once)
  // and then use it again from a function pass manager.
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM2)));
  MPM.run(*M, MAM);

  // There are generally two possible runs for each of the three functions. But
  // for one function, we only invalidate the indirect analysis so the base one
  // only gets run five times.
  EXPECT_EQ(5, FunctionAnalysisRuns);
  // The module analysis pass should be run twice here.
  EXPECT_EQ(2, ModuleAnalysisRuns);
  // The indirect analysis is invalidated for each function (either directly or
  // indirectly) and run twice for each.
  EXPECT_EQ(9, IndirectAnalysisRuns);
  EXPECT_EQ(9, DoublyIndirectAnalysisRuns);

  // There are five instructions in the module and we add the count four
  // times.
  EXPECT_EQ(5 * 4, InstrCount);

  // There are three functions and we count them four times for each of the
  // three functions.
  EXPECT_EQ(3 * 4 * 3, FunctionCount);
}
}
