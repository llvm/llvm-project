//===- CGSCCPassManagerTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TestModuleAnalysis : public AnalysisInfoMixin<TestModuleAnalysis> {
public:
  struct Result {
    Result(int Count) : FunctionCount(Count) {}
    int FunctionCount;
    bool invalidate(Module &, const PreservedAnalyses &PA,
                    ModuleAnalysisManager::Invalidator &) {
      // Check whether the analysis or all analyses on modules have been
      // preserved.
      auto PAC = PA.getChecker<TestModuleAnalysis>();
      return !(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Module>>());
    }
  };

  TestModuleAnalysis(int &Runs) : Runs(Runs) {}

  Result run(Module &M, ModuleAnalysisManager &AM) {
    ++Runs;
    return Result(M.size());
  }

private:
  friend AnalysisInfoMixin<TestModuleAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestModuleAnalysis::Key;

class TestSCCAnalysis : public AnalysisInfoMixin<TestSCCAnalysis> {
public:
  struct Result {
    Result(int Count) : FunctionCount(Count) {}
    int FunctionCount;
    bool invalidate(LazyCallGraph::SCC &, const PreservedAnalyses &PA,
                    CGSCCAnalysisManager::Invalidator &) {
      // Check whether the analysis or all analyses on SCCs have been
      // preserved.
      auto PAC = PA.getChecker<TestSCCAnalysis>();
      return !(PAC.preserved() ||
               PAC.preservedSet<AllAnalysesOn<LazyCallGraph::SCC>>());
    }
  };

  TestSCCAnalysis(int &Runs) : Runs(Runs) {}

  Result run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &) {
    ++Runs;
    return Result(C.size());
  }

private:
  friend AnalysisInfoMixin<TestSCCAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestSCCAnalysis::Key;

class TestFunctionAnalysis : public AnalysisInfoMixin<TestFunctionAnalysis> {
public:
  struct Result {
    Result(int Count) : InstructionCount(Count) {}
    int InstructionCount;
    bool invalidate(Function &, const PreservedAnalyses &PA,
                    FunctionAnalysisManager::Invalidator &) {
      // Check whether the analysis or all analyses on functions have been
      // preserved.
      auto PAC = PA.getChecker<TestFunctionAnalysis>();
      return !(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>());
    }
  };

  TestFunctionAnalysis(int &Runs) : Runs(Runs) {}

  Result run(Function &F, FunctionAnalysisManager &AM) {
    ++Runs;
    int Count = 0;
    for (Instruction &I : instructions(F)) {
      (void)I;
      ++Count;
    }
    return Result(Count);
  }

private:
  friend AnalysisInfoMixin<TestFunctionAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestFunctionAnalysis::Key;

class TestImmutableFunctionAnalysis
    : public AnalysisInfoMixin<TestImmutableFunctionAnalysis> {
public:
  struct Result {
    bool invalidate(Function &, const PreservedAnalyses &,
                    FunctionAnalysisManager::Invalidator &) {
      return false;
    }
  };

  TestImmutableFunctionAnalysis(int &Runs) : Runs(Runs) {}

  Result run(Function &F, FunctionAnalysisManager &AM) {
    ++Runs;
    return Result();
  }

private:
  friend AnalysisInfoMixin<TestImmutableFunctionAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestImmutableFunctionAnalysis::Key;

struct LambdaModulePass : public PassInfoMixin<LambdaModulePass> {
  template <typename T>
  LambdaModulePass(T &&Arg) : Func(std::forward<T>(Arg)) {}

  PreservedAnalyses run(Module &F, ModuleAnalysisManager &AM) {
    return Func(F, AM);
  }

  std::function<PreservedAnalyses(Module &, ModuleAnalysisManager &)> Func;
};

struct LambdaSCCPass : public PassInfoMixin<LambdaSCCPass> {
  template <typename T> LambdaSCCPass(T &&Arg) : Func(std::forward<T>(Arg)) {}

  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR) {
    return Func(C, AM, CG, UR);
  }

  std::function<PreservedAnalyses(LazyCallGraph::SCC &, CGSCCAnalysisManager &,
                                  LazyCallGraph &, CGSCCUpdateResult &)>
      Func;
};

struct LambdaFunctionPass : public PassInfoMixin<LambdaFunctionPass> {
  template <typename T>
  LambdaFunctionPass(T &&Arg) : Func(std::forward<T>(Arg)) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    return Func(F, AM);
  }

  std::function<PreservedAnalyses(Function &, FunctionAnalysisManager &)> Func;
};

std::unique_ptr<Module> parseIR(const char *IR) {
  // We just use a static context here. This is never called from multiple
  // threads so it is harmless no matter how it is implemented. We just need
  // the context to outlive the module which it does.
  static LLVMContext C;
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, C);
}

class CGSCCPassManagerTest : public ::testing::Test {
protected:
  LLVMContext Context;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  std::unique_ptr<Module> M;

public:
  CGSCCPassManagerTest()
      : FAM(), CGAM(), MAM(),
        M(parseIR(
            // Define a module with the following call graph, where calls go
            // out the bottom of nodes and enter the top:
            //
            // f
            // |\   _
            // | \ / |
            // g  h1 |
            // |  |  |
            // |  h2 |
            // |  |  |
            // |  h3 |
            // | / \_/
            // |/
            // x
            //
            "define void @x() {\n"
            "entry:\n"
            "  ret void\n"
            "}\n"
            "define void @h3() {\n"
            "entry:\n"
            "  call void @h1()\n"
            "  ret void\n"
            "}\n"
            "define void @h2() {\n"
            "entry:\n"
            "  call void @h3()\n"
            "  call void @x()\n"
            "  ret void\n"
            "}\n"
            "define void @h1() {\n"
            "entry:\n"
            "  call void @h2()\n"
            "  ret void\n"
            "}\n"
            "define void @g() {\n"
            "entry:\n"
            "  call void @g()\n"
            "  call void @x()\n"
            "  ret void\n"
            "}\n"
            "define void @f() {\n"
            "entry:\n"
            "  call void @g()\n"
            "  call void @h1()\n"
            "  ret void\n"
            "}\n")) {
    FAM.registerPass([&] { return TargetLibraryAnalysis(); });
    MAM.registerPass([&] { return LazyCallGraphAnalysis(); });
    MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });

    // Register required pass instrumentation analysis.
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    CGAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });

    // Cross-register proxies.
    MAM.registerPass([&] { return CGSCCAnalysisManagerModuleProxy(CGAM); });
    CGAM.registerPass([&] { return FunctionAnalysisManagerCGSCCProxy(); });
    CGAM.registerPass([&] { return ModuleAnalysisManagerCGSCCProxy(MAM); });
    FAM.registerPass([&] { return CGSCCAnalysisManagerFunctionProxy(CGAM); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  }
};

TEST_F(CGSCCPassManagerTest, Basic) {
  int FunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });
  int ImmutableFunctionAnalysisRuns = 0;
  FAM.registerPass([&] {
    return TestImmutableFunctionAnalysis(ImmutableFunctionAnalysisRuns);
  });

  int SCCAnalysisRuns = 0;
  CGAM.registerPass([&] { return TestSCCAnalysis(SCCAnalysisRuns); });

  int ModuleAnalysisRuns = 0;
  MAM.registerPass([&] { return TestModuleAnalysis(ModuleAnalysisRuns); });

  ModulePassManager MPM;
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());

  CGSCCPassManager CGPM1;
  FunctionPassManager FPM1;
  int FunctionPassRunCount1 = 0;
  FPM1.addPass(LambdaFunctionPass([&](Function &, FunctionAnalysisManager &) {
    ++FunctionPassRunCount1;
    return PreservedAnalyses::none();
  }));
  CGPM1.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM1)));

  int SCCPassRunCount1 = 0;
  int AnalyzedInstrCount1 = 0;
  int AnalyzedSCCFunctionCount1 = 0;
  int AnalyzedModuleFunctionCount1 = 0;
  CGPM1.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR) {
        ++SCCPassRunCount1;

        // Note: The proper way to get to a module pass from a CGSCC pass is
        // through the ModuleAnalysisManagerCGSCCProxy:
        // ```
        // const auto &MAMProxy =
        //    AM.getResult<ModuleAnalysisManagerCGSCCProxy>(C, CG);
        // ```
        // However getting a stateful analysis is incorrect usage, and the call
        // to getCachedResult below asserts:
        // ```
        // if (TestModuleAnalysis::Result *TMA =
        //        MAMProxy.getCachedResult<TestModuleAnalysis>(
        //            *C.begin()->getFunction().getParent()))
        //   AnalyzedModuleFunctionCount1 += TMA->FunctionCount;
        // ```
        // For the purposes of this unittest, use the above MAM directly.
        if (TestModuleAnalysis::Result *TMA =
                MAM.getCachedResult<TestModuleAnalysis>(
                    *C.begin()->getFunction().getParent()))
          AnalyzedModuleFunctionCount1 += TMA->FunctionCount;

        FunctionAnalysisManager &FAM =
            AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
        TestSCCAnalysis::Result &AR = AM.getResult<TestSCCAnalysis>(C, CG);
        AnalyzedSCCFunctionCount1 += AR.FunctionCount;
        for (LazyCallGraph::Node &N : C) {
          TestFunctionAnalysis::Result &FAR =
              FAM.getResult<TestFunctionAnalysis>(N.getFunction());
          AnalyzedInstrCount1 += FAR.InstructionCount;

          // Just ensure we get the immutable results.
          (void)FAM.getResult<TestImmutableFunctionAnalysis>(N.getFunction());
        }

        return PreservedAnalyses::all();
      }));

  FunctionPassManager FPM2;
  int FunctionPassRunCount2 = 0;
  FPM2.addPass(LambdaFunctionPass([&](Function &, FunctionAnalysisManager &) {
    ++FunctionPassRunCount2;
    return PreservedAnalyses::none();
  }));
  CGPM1.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM2)));

  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  FunctionPassManager FPM3;
  int FunctionPassRunCount3 = 0;
  FPM3.addPass(LambdaFunctionPass([&](Function &, FunctionAnalysisManager &) {
    ++FunctionPassRunCount3;
    return PreservedAnalyses::none();
  }));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM3)));

  MPM.run(*M, MAM);

  EXPECT_EQ(4, SCCPassRunCount1);
  EXPECT_EQ(6, FunctionPassRunCount1);
  EXPECT_EQ(6, FunctionPassRunCount2);
  EXPECT_EQ(6, FunctionPassRunCount3);

  EXPECT_EQ(1, ModuleAnalysisRuns);
  EXPECT_EQ(4, SCCAnalysisRuns);
  EXPECT_EQ(6, FunctionAnalysisRuns);
  EXPECT_EQ(6, ImmutableFunctionAnalysisRuns);

  EXPECT_EQ(14, AnalyzedInstrCount1);
  EXPECT_EQ(6, AnalyzedSCCFunctionCount1);
  EXPECT_EQ(4 * 6, AnalyzedModuleFunctionCount1);
}

// Test that an SCC pass which fails to preserve a module analysis does in fact
// invalidate that module analysis.
TEST_F(CGSCCPassManagerTest, TestSCCPassInvalidatesModuleAnalysis) {
  int ModuleAnalysisRuns = 0;
  MAM.registerPass([&] { return TestModuleAnalysis(ModuleAnalysisRuns); });

  ModulePassManager MPM;
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());

  // The first CGSCC run we preserve everything and make sure that works and
  // the module analysis is available in the second CGSCC run from the one
  // required module pass above.
  CGSCCPassManager CGPM1;
  int CountFoundModuleAnalysis1 = 0;
  CGPM1.addPass(LambdaSCCPass([&](LazyCallGraph::SCC &C,
                                  CGSCCAnalysisManager &AM, LazyCallGraph &CG,
                                  CGSCCUpdateResult &UR) {
    const auto &MAMProxy = AM.getResult<ModuleAnalysisManagerCGSCCProxy>(C, CG);
    if (MAMProxy.cachedResultExists<TestModuleAnalysis>(
            *C.begin()->getFunction().getParent()))
      ++CountFoundModuleAnalysis1;

    return PreservedAnalyses::all();
  }));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  // The second CGSCC run checks that the module analysis got preserved the
  // previous time and in one SCC fails to preserve it.
  CGSCCPassManager CGPM2;
  int CountFoundModuleAnalysis2 = 0;
  CGPM2.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR) {
        const auto &MAMProxy =
            AM.getResult<ModuleAnalysisManagerCGSCCProxy>(C, CG);
        if (MAMProxy.cachedResultExists<TestModuleAnalysis>(
                *C.begin()->getFunction().getParent()))
          ++CountFoundModuleAnalysis2;

        // Only fail to preserve analyses on one SCC and make sure that gets
        // propagated.
        return C.getName() == "(g)" ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
      }));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));

  // The third CGSCC run should fail to find a cached module analysis as it
  // should have been invalidated by the above CGSCC run.
  CGSCCPassManager CGPM3;
  int CountFoundModuleAnalysis3 = 0;
  CGPM3.addPass(LambdaSCCPass([&](LazyCallGraph::SCC &C,
                                  CGSCCAnalysisManager &AM, LazyCallGraph &CG,
                                  CGSCCUpdateResult &UR) {
    const auto &MAMProxy = AM.getResult<ModuleAnalysisManagerCGSCCProxy>(C, CG);
    if (MAMProxy.cachedResultExists<TestModuleAnalysis>(
            *C.begin()->getFunction().getParent()))
      ++CountFoundModuleAnalysis3;

    return PreservedAnalyses::none();
  }));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM3)));

  MPM.run(*M, MAM);

  EXPECT_EQ(1, ModuleAnalysisRuns);
  EXPECT_EQ(4, CountFoundModuleAnalysis1);
  EXPECT_EQ(4, CountFoundModuleAnalysis2);
  EXPECT_EQ(0, CountFoundModuleAnalysis3);
}

// Similar to the above, but test that this works for function passes embedded
// *within* a CGSCC layer.
TEST_F(CGSCCPassManagerTest, TestFunctionPassInsideCGSCCInvalidatesModuleAnalysis) {
  int ModuleAnalysisRuns = 0;
  MAM.registerPass([&] { return TestModuleAnalysis(ModuleAnalysisRuns); });

  ModulePassManager MPM;
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());

  // The first run we preserve everything and make sure that works and the
  // module analysis is available in the second run from the one required
  // module pass above.
  FunctionPassManager FPM1;
  // Start true and mark false if we ever failed to find a module analysis
  // because we expect this to succeed for each SCC.
  bool FoundModuleAnalysis1 = true;
  FPM1.addPass(LambdaFunctionPass([&](Function &F,
                                      FunctionAnalysisManager &AM) {
    const auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
    if (!MAMProxy.cachedResultExists<TestModuleAnalysis>(*F.getParent()))
      FoundModuleAnalysis1 = false;

    return PreservedAnalyses::all();
  }));
  CGSCCPassManager CGPM1;
  CGPM1.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM1)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  // The second run checks that the module analysis got preserved the previous
  // time and in one function fails to preserve it.
  FunctionPassManager FPM2;
  // Again, start true and mark false if we ever failed to find a module analysis
  // because we expect this to succeed for each SCC.
  bool FoundModuleAnalysis2 = true;
  FPM2.addPass(LambdaFunctionPass([&](Function &F,
                                      FunctionAnalysisManager &AM) {
    const auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
    if (!MAMProxy.cachedResultExists<TestModuleAnalysis>(*F.getParent()))
      FoundModuleAnalysis2 = false;

    // Only fail to preserve analyses on one SCC and make sure that gets
    // propagated.
    return F.getName() == "h2" ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
  }));
  CGSCCPassManager CGPM2;
  CGPM2.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM2)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));

  // The third run should fail to find a cached module analysis as it should
  // have been invalidated by the above run.
  FunctionPassManager FPM3;
  // Start false and mark true if we ever *succeeded* to find a module
  // analysis, as we expect this to fail for every function.
  bool FoundModuleAnalysis3 = false;
  FPM3.addPass(LambdaFunctionPass([&](Function &F,
                                      FunctionAnalysisManager &AM) {
    const auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
    if (MAMProxy.cachedResultExists<TestModuleAnalysis>(*F.getParent()))
      FoundModuleAnalysis3 = true;

    return PreservedAnalyses::none();
  }));
  CGSCCPassManager CGPM3;
  CGPM3.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM3)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM3)));

  MPM.run(*M, MAM);

  EXPECT_EQ(1, ModuleAnalysisRuns);
  EXPECT_TRUE(FoundModuleAnalysis1);
  EXPECT_TRUE(FoundModuleAnalysis2);
  EXPECT_FALSE(FoundModuleAnalysis3);
}

// Test that a Module pass which fails to preserve an SCC analysis in fact
// invalidates that analysis.
TEST_F(CGSCCPassManagerTest, TestModulePassInvalidatesSCCAnalysis) {
  int SCCAnalysisRuns = 0;
  CGAM.registerPass([&] { return TestSCCAnalysis(SCCAnalysisRuns); });

  ModulePassManager MPM;

  // First force the analysis to be run.
  CGSCCPassManager CGPM1;
  CGPM1.addPass(RequireAnalysisPass<TestSCCAnalysis, LazyCallGraph::SCC,
                                    CGSCCAnalysisManager, LazyCallGraph &,
                                    CGSCCUpdateResult &>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  // Now run a module pass that preserves the LazyCallGraph and the proxy but
  // not the SCC analysis.
  MPM.addPass(LambdaModulePass([&](Module &M, ModuleAnalysisManager &) {
    PreservedAnalyses PA;
    PA.preserve<LazyCallGraphAnalysis>();
    PA.preserve<CGSCCAnalysisManagerModuleProxy>();
    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    return PA;
  }));

  // And now a second CGSCC run which requires the SCC analysis again. This
  // will trigger re-running it.
  CGSCCPassManager CGPM2;
  CGPM2.addPass(RequireAnalysisPass<TestSCCAnalysis, LazyCallGraph::SCC,
                                    CGSCCAnalysisManager, LazyCallGraph &,
                                    CGSCCUpdateResult &>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));

  MPM.run(*M, MAM);
  // Two runs and four SCCs.
  EXPECT_EQ(2 * 4, SCCAnalysisRuns);
}

// Check that marking the SCC analysis preserved is sufficient to avoid
// invaliadtion. This should only run the analysis once for each SCC.
TEST_F(CGSCCPassManagerTest, TestModulePassCanPreserveSCCAnalysis) {
  int SCCAnalysisRuns = 0;
  CGAM.registerPass([&] { return TestSCCAnalysis(SCCAnalysisRuns); });

  ModulePassManager MPM;

  // First force the analysis to be run.
  CGSCCPassManager CGPM1;
  CGPM1.addPass(RequireAnalysisPass<TestSCCAnalysis, LazyCallGraph::SCC,
                                    CGSCCAnalysisManager, LazyCallGraph &,
                                    CGSCCUpdateResult &>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  // Now run a module pass that preserves each of the necessary components
  // (but not everything).
  MPM.addPass(LambdaModulePass([&](Module &M, ModuleAnalysisManager &) {
    PreservedAnalyses PA;
    PA.preserve<LazyCallGraphAnalysis>();
    PA.preserve<CGSCCAnalysisManagerModuleProxy>();
    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    PA.preserve<TestSCCAnalysis>();
    return PA;
  }));

  // And now a second CGSCC run which requires the SCC analysis again but find
  // it in the cache.
  CGSCCPassManager CGPM2;
  CGPM2.addPass(RequireAnalysisPass<TestSCCAnalysis, LazyCallGraph::SCC,
                                    CGSCCAnalysisManager, LazyCallGraph &,
                                    CGSCCUpdateResult &>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));

  MPM.run(*M, MAM);
  // Four SCCs
  EXPECT_EQ(4, SCCAnalysisRuns);
}

// Check that even when the analysis is preserved, if the SCC information isn't
// we still nuke things because the SCC keys could change.
TEST_F(CGSCCPassManagerTest, TestModulePassInvalidatesSCCAnalysisOnCGChange) {
  int SCCAnalysisRuns = 0;
  CGAM.registerPass([&] { return TestSCCAnalysis(SCCAnalysisRuns); });

  ModulePassManager MPM;

  // First force the analysis to be run.
  CGSCCPassManager CGPM1;
  CGPM1.addPass(RequireAnalysisPass<TestSCCAnalysis, LazyCallGraph::SCC,
                                    CGSCCAnalysisManager, LazyCallGraph &,
                                    CGSCCUpdateResult &>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  // Now run a module pass that preserves the analysis but not the call
  // graph or proxy.
  MPM.addPass(LambdaModulePass([&](Module &M, ModuleAnalysisManager &) {
    PreservedAnalyses PA;
    PA.preserve<TestSCCAnalysis>();
    return PA;
  }));

  // And now a second CGSCC run which requires the SCC analysis again.
  CGSCCPassManager CGPM2;
  CGPM2.addPass(RequireAnalysisPass<TestSCCAnalysis, LazyCallGraph::SCC,
                                    CGSCCAnalysisManager, LazyCallGraph &,
                                    CGSCCUpdateResult &>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));

  MPM.run(*M, MAM);
  // Two runs and four SCCs.
  EXPECT_EQ(2 * 4, SCCAnalysisRuns);
}

// Test that an SCC pass which fails to preserve a Function analysis in fact
// invalidates that analysis.
TEST_F(CGSCCPassManagerTest, TestSCCPassInvalidatesFunctionAnalysis) {
  int FunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });

  // Create a very simple module with a single function and SCC to make testing
  // these issues much easier.
  std::unique_ptr<Module> M = parseIR("declare void @g()\n"
                                      "declare void @h()\n"
                                      "define void @f() {\n"
                                      "entry:\n"
                                      "  call void @g()\n"
                                      "  call void @h()\n"
                                      "  ret void\n"
                                      "}\n");

  CGSCCPassManager CGPM;

  // First force the analysis to be run.
  FunctionPassManager FPM1;
  FPM1.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM1)));

  // Now run a module pass that preserves the LazyCallGraph and proxy but not
  // the SCC analysis.
  CGPM.addPass(LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &,
                                 LazyCallGraph &, CGSCCUpdateResult &) {
    PreservedAnalyses PA;
    PA.preserve<LazyCallGraphAnalysis>();
    return PA;
  }));

  // And now a second CGSCC run which requires the SCC analysis again. This
  // will trigger re-running it.
  FunctionPassManager FPM2;
  FPM2.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM2)));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
  EXPECT_EQ(2, FunctionAnalysisRuns);
}

// Check that marking the SCC analysis preserved is sufficient. This should
// only run the analysis once the SCC.
TEST_F(CGSCCPassManagerTest, TestSCCPassCanPreserveFunctionAnalysis) {
  int FunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });

  // Create a very simple module with a single function and SCC to make testing
  // these issues much easier.
  std::unique_ptr<Module> M = parseIR("declare void @g()\n"
                                      "declare void @h()\n"
                                      "define void @f() {\n"
                                      "entry:\n"
                                      "  call void @g()\n"
                                      "  call void @h()\n"
                                      "  ret void\n"
                                      "}\n");

  CGSCCPassManager CGPM;

  // First force the analysis to be run.
  FunctionPassManager FPM1;
  FPM1.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM1)));

  // Now run a module pass that preserves each of the necessary components
  // (but
  // not everything).
  CGPM.addPass(LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &,
                                 LazyCallGraph &, CGSCCUpdateResult &) {
    PreservedAnalyses PA;
    PA.preserve<LazyCallGraphAnalysis>();
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
    PA.preserve<TestFunctionAnalysis>();
    return PA;
  }));

  // And now a second CGSCC run which requires the SCC analysis again but find
  // it in the cache.
  FunctionPassManager FPM2;
  FPM2.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM2)));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
  EXPECT_EQ(1, FunctionAnalysisRuns);
}

// Note that there is no test for invalidating the call graph or other
// structure with an SCC pass because there is no mechanism to do that from
// withinsuch a pass. Instead, such a pass has to directly update the call
// graph structure.

// Test that a madule pass invalidates function analyses when the CGSCC proxies
// and pass manager.
TEST_F(CGSCCPassManagerTest,
       TestModulePassInvalidatesFunctionAnalysisNestedInCGSCC) {
  MAM.registerPass([&] { return LazyCallGraphAnalysis(); });

  int FunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });

  ModulePassManager MPM;

  // First force the analysis to be run.
  FunctionPassManager FPM1;
  FPM1.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGSCCPassManager CGPM1;
  CGPM1.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM1)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  // Now run a module pass that preserves the LazyCallGraph and proxies but not
  // the Function analysis.
  MPM.addPass(LambdaModulePass([&](Module &M, ModuleAnalysisManager &) {
    PreservedAnalyses PA;
    PA.preserve<LazyCallGraphAnalysis>();
    PA.preserve<CGSCCAnalysisManagerModuleProxy>();
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    return PA;
  }));

  // And now a second CGSCC run which requires the SCC analysis again. This
  // will trigger re-running it.
  FunctionPassManager FPM2;
  FPM2.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGSCCPassManager CGPM2;
  CGPM2.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM2)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));

  MPM.run(*M, MAM);
  // Two runs and 6 functions.
  EXPECT_EQ(2 * 6, FunctionAnalysisRuns);
}

// Check that by marking the function pass and proxies as preserved, this
// propagates all the way through.
TEST_F(CGSCCPassManagerTest,
       TestModulePassCanPreserveFunctionAnalysisNestedInCGSCC) {
  MAM.registerPass([&] { return LazyCallGraphAnalysis(); });

  int FunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });

  ModulePassManager MPM;

  // First force the analysis to be run.
  FunctionPassManager FPM1;
  FPM1.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGSCCPassManager CGPM1;
  CGPM1.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM1)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  // Now run a module pass that preserves the LazyCallGraph, the proxy, and
  // the Function analysis.
  MPM.addPass(LambdaModulePass([&](Module &M, ModuleAnalysisManager &) {
    PreservedAnalyses PA;
    PA.preserve<LazyCallGraphAnalysis>();
    PA.preserve<CGSCCAnalysisManagerModuleProxy>();
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    PA.preserve<TestFunctionAnalysis>();
    return PA;
  }));

  // And now a second CGSCC run which requires the SCC analysis again. This
  // will trigger re-running it.
  FunctionPassManager FPM2;
  FPM2.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGSCCPassManager CGPM2;
  CGPM2.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM2)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));

  MPM.run(*M, MAM);
  // One run and 6 functions.
  EXPECT_EQ(6, FunctionAnalysisRuns);
}

// Check that if the lazy call graph itself isn't preserved we still manage to
// invalidate everything.
TEST_F(CGSCCPassManagerTest,
       TestModulePassInvalidatesFunctionAnalysisNestedInCGSCCOnCGChange) {
  MAM.registerPass([&] { return LazyCallGraphAnalysis(); });

  int FunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });

  ModulePassManager MPM;

  // First force the analysis to be run.
  FunctionPassManager FPM1;
  FPM1.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGSCCPassManager CGPM1;
  CGPM1.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM1)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM1)));

  // Now run a module pass that preserves the LazyCallGraph but not the
  // Function analysis.
  MPM.addPass(LambdaModulePass([&](Module &M, ModuleAnalysisManager &) {
    PreservedAnalyses PA;
    return PA;
  }));

  // And now a second CGSCC run which requires the SCC analysis again. This
  // will trigger re-running it.
  FunctionPassManager FPM2;
  FPM2.addPass(RequireAnalysisPass<TestFunctionAnalysis, Function>());
  CGSCCPassManager CGPM2;
  CGPM2.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM2)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));

  MPM.run(*M, MAM);
  // Two runs and 6 functions.
  EXPECT_EQ(2 * 6, FunctionAnalysisRuns);
}

/// A test CGSCC-level analysis pass which caches in its result another
/// analysis pass and uses it to serve queries. This requires the result to
/// invalidate itself when its dependency is invalidated.
///
/// FIXME: Currently this doesn't also depend on a function analysis, and if it
/// did we would fail to invalidate it correctly.
struct TestIndirectSCCAnalysis
    : public AnalysisInfoMixin<TestIndirectSCCAnalysis> {
  struct Result {
    Result(TestSCCAnalysis::Result &SCCDep, TestModuleAnalysis::Result &MDep)
        : SCCDep(SCCDep), MDep(MDep) {}
    TestSCCAnalysis::Result &SCCDep;
    TestModuleAnalysis::Result &MDep;

    bool invalidate(LazyCallGraph::SCC &C, const PreservedAnalyses &PA,
                    CGSCCAnalysisManager::Invalidator &Inv) {
      auto PAC = PA.getChecker<TestIndirectSCCAnalysis>();
      return !(PAC.preserved() ||
               PAC.preservedSet<AllAnalysesOn<LazyCallGraph::SCC>>()) ||
             Inv.invalidate<TestSCCAnalysis>(C, PA);
    }
  };

  TestIndirectSCCAnalysis(int &Runs, ModuleAnalysisManager &MAM)
      : Runs(Runs), MAM(MAM) {}

  /// Run the analysis pass over the function and return a result.
  Result run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
             LazyCallGraph &CG) {
    ++Runs;
    auto &SCCDep = AM.getResult<TestSCCAnalysis>(C, CG);

    auto &ModuleProxy = AM.getResult<ModuleAnalysisManagerCGSCCProxy>(C, CG);
    // For the test, we insist that the module analysis starts off in the
    // cache. Getting a cached result that isn't stateless triggers an assert.
    // auto &MDep = *ModuleProxy.getCachedResult<TestModuleAnalysis>(
    //  *C.begin()->getFunction().getParent());
    // Use MAM, for the purposes of this unittest.
    auto &MDep = *MAM.getCachedResult<TestModuleAnalysis>(
        *C.begin()->getFunction().getParent());
    // Register the dependency as module analysis dependencies have to be
    // pre-registered on the proxy.
    ModuleProxy.registerOuterAnalysisInvalidation<TestModuleAnalysis,
                                                  TestIndirectSCCAnalysis>();

    return Result(SCCDep, MDep);
  }

private:
  friend AnalysisInfoMixin<TestIndirectSCCAnalysis>;
  static AnalysisKey Key;

  int &Runs;
  ModuleAnalysisManager &MAM;
};

AnalysisKey TestIndirectSCCAnalysis::Key;

/// A test analysis pass which caches in its result the result from the above
/// indirect analysis pass.
///
/// This allows us to ensure that whenever an analysis pass is invalidated due
/// to dependencies (especially dependencies across IR units that trigger
/// asynchronous invalidation) we correctly detect that this may in turn cause
/// other analysis to be invalidated.
struct TestDoublyIndirectSCCAnalysis
    : public AnalysisInfoMixin<TestDoublyIndirectSCCAnalysis> {
  struct Result {
    Result(TestIndirectSCCAnalysis::Result &IDep) : IDep(IDep) {}
    TestIndirectSCCAnalysis::Result &IDep;

    bool invalidate(LazyCallGraph::SCC &C, const PreservedAnalyses &PA,
                    CGSCCAnalysisManager::Invalidator &Inv) {
      auto PAC = PA.getChecker<TestDoublyIndirectSCCAnalysis>();
      return !(PAC.preserved() ||
               PAC.preservedSet<AllAnalysesOn<LazyCallGraph::SCC>>()) ||
             Inv.invalidate<TestIndirectSCCAnalysis>(C, PA);
    }
  };

  TestDoublyIndirectSCCAnalysis(int &Runs) : Runs(Runs) {}

  /// Run the analysis pass over the function and return a result.
  Result run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
             LazyCallGraph &CG) {
    ++Runs;
    auto &IDep = AM.getResult<TestIndirectSCCAnalysis>(C, CG);
    return Result(IDep);
  }

private:
  friend AnalysisInfoMixin<TestDoublyIndirectSCCAnalysis>;
  static AnalysisKey Key;

  int &Runs;
};

AnalysisKey TestDoublyIndirectSCCAnalysis::Key;

/// A test analysis pass which caches results from three different IR unit
/// layers and requires intermediate layers to correctly propagate the entire
/// distance.
struct TestIndirectFunctionAnalysis
    : public AnalysisInfoMixin<TestIndirectFunctionAnalysis> {
  struct Result {
    Result(TestFunctionAnalysis::Result &FDep, TestModuleAnalysis::Result &MDep,
           TestSCCAnalysis::Result &SCCDep)
        : FDep(FDep), MDep(MDep), SCCDep(SCCDep) {}
    TestFunctionAnalysis::Result &FDep;
    TestModuleAnalysis::Result &MDep;
    TestSCCAnalysis::Result &SCCDep;

    bool invalidate(Function &F, const PreservedAnalyses &PA,
                    FunctionAnalysisManager::Invalidator &Inv) {
      auto PAC = PA.getChecker<TestIndirectFunctionAnalysis>();
      return !(PAC.preserved() ||
               PAC.preservedSet<AllAnalysesOn<Function>>()) ||
             Inv.invalidate<TestFunctionAnalysis>(F, PA);
    }
  };

  TestIndirectFunctionAnalysis(int &Runs, ModuleAnalysisManager &MAM,
                               CGSCCAnalysisManager &CGAM)
      : Runs(Runs), MAM(MAM), CGAM(CGAM) {}

  /// Run the analysis pass over the function and return a result.
  Result run(Function &F, FunctionAnalysisManager &AM) {
    ++Runs;
    auto &FDep = AM.getResult<TestFunctionAnalysis>(F);

    auto &ModuleProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
    // For the test, we insist that the module analysis starts off in the
    // cache. Getting a cached result that isn't stateless triggers an assert.
    // Use MAM, for the purposes of this unittest.
    auto &MDep = *MAM.getCachedResult<TestModuleAnalysis>(*F.getParent());
    // Register the dependency as module analysis dependencies have to be
    // pre-registered on the proxy.
    ModuleProxy.registerOuterAnalysisInvalidation<
        TestModuleAnalysis, TestIndirectFunctionAnalysis>();

    // For the test we assume this is run inside a CGSCC pass manager.
    // Use MAM, for the purposes of this unittest.
    const LazyCallGraph &CG =
        *MAM.getCachedResult<LazyCallGraphAnalysis>(*F.getParent());
    auto &CGSCCProxy = AM.getResult<CGSCCAnalysisManagerFunctionProxy>(F);
    // For the test, we insist that the CGSCC analysis starts off in the cache.
    // Getting a cached result that isn't stateless triggers an assert.
    // Use CGAM, for the purposes of this unittest.
    auto &SCCDep =
        *CGAM.getCachedResult<TestSCCAnalysis>(*CG.lookupSCC(*CG.lookup(F)));
    // Register the dependency as CGSCC analysis dependencies have to be
    // pre-registered on the proxy.
    CGSCCProxy.registerOuterAnalysisInvalidation<
        TestSCCAnalysis, TestIndirectFunctionAnalysis>();

    return Result(FDep, MDep, SCCDep);
  }

private:
  friend AnalysisInfoMixin<TestIndirectFunctionAnalysis>;
  static AnalysisKey Key;

  int &Runs;
  ModuleAnalysisManager &MAM;
  CGSCCAnalysisManager &CGAM;
};

AnalysisKey TestIndirectFunctionAnalysis::Key;

TEST_F(CGSCCPassManagerTest, TestIndirectAnalysisInvalidation) {
  int ModuleAnalysisRuns = 0;
  MAM.registerPass([&] { return TestModuleAnalysis(ModuleAnalysisRuns); });

  int SCCAnalysisRuns = 0, IndirectSCCAnalysisRuns = 0,
      DoublyIndirectSCCAnalysisRuns = 0;
  CGAM.registerPass([&] { return TestSCCAnalysis(SCCAnalysisRuns); });
  CGAM.registerPass(
      [&] { return TestIndirectSCCAnalysis(IndirectSCCAnalysisRuns, MAM); });
  CGAM.registerPass([&] {
    return TestDoublyIndirectSCCAnalysis(DoublyIndirectSCCAnalysisRuns);
  });

  int FunctionAnalysisRuns = 0, IndirectFunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });
  FAM.registerPass([&] {
    return TestIndirectFunctionAnalysis(IndirectFunctionAnalysisRuns, MAM,
                                        CGAM);
  });

  ModulePassManager MPM;

  int FunctionCount = 0;
  CGSCCPassManager CGPM;
  // First just use the analysis to get the function count and preserve
  // everything.
  CGPM.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &) {
        auto &DoublyIndirectResult =
            AM.getResult<TestDoublyIndirectSCCAnalysis>(C, CG);
        auto &IndirectResult = DoublyIndirectResult.IDep;
        FunctionCount += IndirectResult.SCCDep.FunctionCount;
        return PreservedAnalyses::all();
      }));
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(
      RequireAnalysisPass<TestIndirectFunctionAnalysis, Function>()));

  // Next, invalidate
  //   - both analyses for the (f) and (x) SCCs,
  //   - just the underlying (indirect) analysis for (g) SCC, and
  //   - just the direct analysis for (h1,h2,h3) SCC.
  CGPM.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &) {
        auto &DoublyIndirectResult =
            AM.getResult<TestDoublyIndirectSCCAnalysis>(C, CG);
        auto &IndirectResult = DoublyIndirectResult.IDep;
        FunctionCount += IndirectResult.SCCDep.FunctionCount;
        auto PA = PreservedAnalyses::none();
        PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
        PA.preserveSet<AllAnalysesOn<Function>>();
        if (C.getName() == "(g)")
          PA.preserve<TestSCCAnalysis>();
        else if (C.getName() == "(h3, h1, h2)")
          PA.preserve<TestIndirectSCCAnalysis>();
        return PA;
      }));
  // Finally, use the analysis again on each SCC (and function), forcing
  // re-computation for all of them.
  CGPM.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &) {
        auto &DoublyIndirectResult =
            AM.getResult<TestDoublyIndirectSCCAnalysis>(C, CG);
        auto &IndirectResult = DoublyIndirectResult.IDep;
        FunctionCount += IndirectResult.SCCDep.FunctionCount;
        return PreservedAnalyses::all();
      }));
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(
      RequireAnalysisPass<TestIndirectFunctionAnalysis, Function>()));

  // Create a second CGSCC pass manager. This will cause the module-level
  // invalidation to occur, which will force yet another invalidation of the
  // indirect SCC-level analysis as the module analysis it depends on gets
  // invalidated.
  CGSCCPassManager CGPM2;
  CGPM2.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &) {
        auto &DoublyIndirectResult =
            AM.getResult<TestDoublyIndirectSCCAnalysis>(C, CG);
        auto &IndirectResult = DoublyIndirectResult.IDep;
        FunctionCount += IndirectResult.SCCDep.FunctionCount;
        return PreservedAnalyses::all();
      }));
  CGPM2.addPass(createCGSCCToFunctionPassAdaptor(
      RequireAnalysisPass<TestIndirectFunctionAnalysis, Function>()));

  // Add a requires pass to populate the module analysis and then our CGSCC
  // pass pipeline.
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  // Now require the module analysis again (it will have been invalidated once)
  // and then use it again from our second CGSCC pipeline..
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));
  MPM.run(*M, MAM);

  // There are generally two possible runs for each of the four SCCs. But
  // for one SCC, we only invalidate the indirect analysis so the base one
  // only gets run seven times.
  EXPECT_EQ(7, SCCAnalysisRuns);
  // The module analysis pass should be run twice here.
  EXPECT_EQ(2, ModuleAnalysisRuns);
  // The indirect analysis is invalidated (either directly or indirectly) three
  // times for each of four SCCs.
  EXPECT_EQ(3 * 4, IndirectSCCAnalysisRuns);
  EXPECT_EQ(3 * 4, DoublyIndirectSCCAnalysisRuns);

  // We run the indirect function analysis once per function the first time.
  // Then we re-run it for every SCC but "(g)". Then we re-run it for every
  // function again.
  EXPECT_EQ(6 + 5 + 6, IndirectFunctionAnalysisRuns);

  // Four passes count each of six functions once (via SCCs).
  EXPECT_EQ(4 * 6, FunctionCount);
}

TEST_F(CGSCCPassManagerTest, TestAnalysisInvalidationCGSCCUpdate) {
  int ModuleAnalysisRuns = 0;
  MAM.registerPass([&] { return TestModuleAnalysis(ModuleAnalysisRuns); });

  int SCCAnalysisRuns = 0, IndirectSCCAnalysisRuns = 0,
      DoublyIndirectSCCAnalysisRuns = 0;
  CGAM.registerPass([&] { return TestSCCAnalysis(SCCAnalysisRuns); });
  CGAM.registerPass(
      [&] { return TestIndirectSCCAnalysis(IndirectSCCAnalysisRuns, MAM); });
  CGAM.registerPass([&] {
    return TestDoublyIndirectSCCAnalysis(DoublyIndirectSCCAnalysisRuns);
  });

  int FunctionAnalysisRuns = 0, IndirectFunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });
  FAM.registerPass([&] {
    return TestIndirectFunctionAnalysis(IndirectFunctionAnalysisRuns, MAM,
                                        CGAM);
  });

  ModulePassManager MPM;

  CGSCCPassManager CGPM;
  // First just use the analysis to get the function count and preserve
  // everything.
  using RequireTestIndirectFunctionAnalysisPass =
      RequireAnalysisPass<TestIndirectFunctionAnalysis, Function>;
  using RequireTestDoublyIndirectSCCAnalysisPass =
      RequireAnalysisPass<TestDoublyIndirectSCCAnalysis, LazyCallGraph::SCC,
                          CGSCCAnalysisManager, LazyCallGraph &,
                          CGSCCUpdateResult &>;
  CGPM.addPass(RequireTestDoublyIndirectSCCAnalysisPass());
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(
      RequireTestIndirectFunctionAnalysisPass()));

  // Next, we inject an SCC pass that invalidates everything for the `(h3, h1,
  // h2)` SCC but also deletes the call edge from `h2` to `h3` and updates the
  // CG. This should successfully invalidate (and force to be re-run) all the
  // analyses for that SCC and for the functions.
  CGPM.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR) {
        (void)AM.getResult<TestDoublyIndirectSCCAnalysis>(C, CG);
        if (C.getName() != "(h3, h1, h2)")
          return PreservedAnalyses::all();

        // Build the preserved set.
        auto PA = PreservedAnalyses::none();
        PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
        PA.preserve<TestIndirectSCCAnalysis>();
        PA.preserve<TestDoublyIndirectSCCAnalysis>();

        // Delete the call from `h2` to `h3`.
        auto &H2N = *llvm::find_if(
            C, [](LazyCallGraph::Node &N) { return N.getName() == "h2"; });
        auto &H2F = H2N.getFunction();
        auto &H3F = *cast<CallInst>(H2F.begin()->begin())->getCalledFunction();
        assert(H3F.getName() == "h3" && "Wrong called function!");
        H2F.begin()->begin()->eraseFromParent();
        // Insert a bitcast of `h3` so that we retain a ref edge to it.
        (void)CastInst::CreatePointerCast(
            &H3F, PointerType::getUnqual(H2F.getContext()), "dummy",
            H2F.begin()->begin());

        // Now update the call graph.
        auto &NewC =
            updateCGAndAnalysisManagerForFunctionPass(CG, C, H2N, AM, UR, FAM);
        assert(&NewC != &C && "Should get a new SCC due to update!");
        (void)&NewC;

        return PA;
      }));
  // Now use the analysis again on each SCC and function, forcing
  // re-computation for all of them.
  CGPM.addPass(RequireTestDoublyIndirectSCCAnalysisPass());
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(
      RequireTestIndirectFunctionAnalysisPass()));

  // Create another CGSCC pipeline that requires all the analyses again.
  CGSCCPassManager CGPM2;
  CGPM2.addPass(RequireTestDoublyIndirectSCCAnalysisPass());
  CGPM2.addPass(createCGSCCToFunctionPassAdaptor(
      RequireTestIndirectFunctionAnalysisPass()));

  // Next we inject an SCC pass that finds the `(h2)` SCC, adds a call to `h3`
  // back to `h2`, and then invalidates everything for what will then be the
  // `(h3, h1, h2)` SCC again.
  CGSCCPassManager CGPM3;
  CGPM3.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR) {
        (void)AM.getResult<TestDoublyIndirectSCCAnalysis>(C, CG);
        if (C.getName() != "(h2)")
          return PreservedAnalyses::all();

        // Build the preserved set.
        auto PA = PreservedAnalyses::none();
        PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
        PA.preserve<TestIndirectSCCAnalysis>();
        PA.preserve<TestDoublyIndirectSCCAnalysis>();

        // Delete the bitcast of `h3` that we added earlier.
        auto &H2N = *C.begin();
        auto &H2F = H2N.getFunction();
        auto &H3F = *cast<Function>(cast<BitCastInst>(H2F.begin()->begin())->getOperand(0));
        assert(H3F.getName() == "h3" && "Wrong called function!");
        H2F.begin()->begin()->eraseFromParent();
        // And insert a call to `h3`.
        (void)CallInst::Create(&H3F, {}, "", H2F.begin()->begin());

        // Now update the call graph.
        auto &NewC =
            updateCGAndAnalysisManagerForFunctionPass(CG, C, H2N, AM, UR, FAM);
        assert(&NewC != &C && "Should get a new SCC due to update!");
        (void)&NewC;

        return PA;
      }));
  // Now use the analysis again on each SCC and function, forcing
  // re-computation for all of them.
  CGPM3.addPass(RequireTestDoublyIndirectSCCAnalysisPass());
  CGPM3.addPass(createCGSCCToFunctionPassAdaptor(
      RequireTestIndirectFunctionAnalysisPass()));

  // Create a second CGSCC pass manager. This will cause the module-level
  // invalidation to occur, which will force yet another invalidation of the
  // indirect SCC-level analysis as the module analysis it depends on gets
  // invalidated.
  CGSCCPassManager CGPM4;
  CGPM4.addPass(RequireTestDoublyIndirectSCCAnalysisPass());
  CGPM4.addPass(createCGSCCToFunctionPassAdaptor(
      RequireTestIndirectFunctionAnalysisPass()));

  // Add a requires pass to populate the module analysis and then one of our
  // CGSCC pipelines. Repeat for all four CGSCC pipelines.
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM2)));
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM3)));
  MPM.addPass(RequireAnalysisPass<TestModuleAnalysis, Module>());
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM4)));
  MPM.run(*M, MAM);

  // We run over four SCCs the first time. But then we split an SCC into three.
  // And then we merge those three back into one. However, this also
  // invalidates all three SCCs further down in the PO walk.
  EXPECT_EQ(4 + 3 + 3, SCCAnalysisRuns);
  // The module analysis pass should be run three times.
  EXPECT_EQ(3, ModuleAnalysisRuns);
  // We run over four SCCs the first time. Then over the two new ones. Then the
  // entire module is invalidated causing a full run over all seven. Then we
  // fold three SCCs back to one, re-compute for it and the two SCCs above it
  // in the graph, and then run over the whole module again.
  EXPECT_EQ(4 + 2 + 7 + 3 + 4, IndirectSCCAnalysisRuns);
  EXPECT_EQ(4 + 2 + 7 + 3 + 4, DoublyIndirectSCCAnalysisRuns);

  // First we run over all six functions. Then we re-run it over three when we
  // split their SCCs. Then we re-run over the whole module. Then we re-run
  // over three functions merged back into a single SCC, then those three
  // functions again, the two functions in SCCs above it in the graph, and then
  // over the whole module again.
  EXPECT_EQ(6 + 3 + 6 + 3 + 2 + 6, FunctionAnalysisRuns);

  // Re run the function analysis over the entire module, and then re-run it
  // over the `(h3, h1, h2)` SCC due to invalidation. Then we re-run it over
  // the entire module, then the three functions merged back into a single SCC,
  // those three functions again, then the two functions in SCCs above it in
  // the graph, and then over the whole module.
  EXPECT_EQ(6 + 3 + 6 + 3 + 2 + 6, IndirectFunctionAnalysisRuns);
}

// The (negative) tests below check for assertions so we only run them if NDEBUG
// is not defined.
#ifndef NDEBUG

struct LambdaSCCPassNoPreserve : public PassInfoMixin<LambdaSCCPassNoPreserve> {
  template <typename T>
  LambdaSCCPassNoPreserve(T &&Arg) : Func(std::forward<T>(Arg)) {}

  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR) {
    Func(C, AM, CG, UR);
    PreservedAnalyses PA;
    // We update the core CGSCC data structures and so can preserve the proxy to
    // the function analysis manager.
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
    return PA;
  }

  std::function<void(LazyCallGraph::SCC &, CGSCCAnalysisManager &,
                     LazyCallGraph &, CGSCCUpdateResult &)>
      Func;
};

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses0) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (C.getName() != "(h3, h1, h2)")
          return;

        auto &FAM =
            AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
        Function *FnX = M->getFunction("x");
        Function *FnH1 = M->getFunction("h1");
        Function *FnH2 = M->getFunction("h2");
        Function *FnH3 = M->getFunction("h3");
        ASSERT_NE(FnX, nullptr);
        ASSERT_NE(FnH1, nullptr);
        ASSERT_NE(FnH2, nullptr);
        ASSERT_NE(FnH3, nullptr);

        // And insert a call to `h1`, `h2`, and `h3`.
        BasicBlock::iterator IP = FnH2->getEntryBlock().begin();
        (void)CallInst::Create(FnH1, {}, "", IP);
        (void)CallInst::Create(FnH2, {}, "", IP);
        (void)CallInst::Create(FnH3, {}, "", IP);

        auto &H2N = *llvm::find_if(
            C, [](LazyCallGraph::Node &N) { return N.getName() == "h2"; });
        ASSERT_NO_FATAL_FAILURE(
            updateCGAndAnalysisManagerForCGSCCPass(CG, C, H2N, AM, UR, FAM));
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses1) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve([&](LazyCallGraph::SCC &C,
                                           CGSCCAnalysisManager &AM,
                                           LazyCallGraph &CG,
                                           CGSCCUpdateResult &UR) {
    if (C.getName() != "(h3, h1, h2)")
      return;

    auto &FAM =
        AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
    Function *FnX = M->getFunction("x");
    Function *FnH1 = M->getFunction("h1");
    Function *FnH2 = M->getFunction("h2");
    Function *FnH3 = M->getFunction("h3");
    ASSERT_NE(FnX, nullptr);
    ASSERT_NE(FnH1, nullptr);
    ASSERT_NE(FnH2, nullptr);
    ASSERT_NE(FnH3, nullptr);

    // And insert a call to `h1`, `h2`, and `h3`.
    BasicBlock::iterator IP = FnH2->getEntryBlock().begin();
    (void)CallInst::Create(FnH1, {}, "", IP);
    (void)CallInst::Create(FnH2, {}, "", IP);
    (void)CallInst::Create(FnH3, {}, "", IP);

    auto &H2N = *llvm::find_if(
        C, [](LazyCallGraph::Node &N) { return N.getName() == "h2"; });
    ASSERT_DEATH(
        updateCGAndAnalysisManagerForFunctionPass(CG, C, H2N, AM, UR, FAM),
        "Any new calls should be modeled as");
  }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses2) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (C.getName() != "(f)")
          return;

        auto &FAM =
            AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
        Function *FnF = M->getFunction("f");
        Function *FnH2 = M->getFunction("h2");
        ASSERT_NE(FnF, nullptr);
        ASSERT_NE(FnH2, nullptr);

        // And insert a call to `h2`
        BasicBlock::iterator IP = FnF->getEntryBlock().begin();
        (void)CallInst::Create(FnH2, {}, "", IP);

        auto &FN = *llvm::find_if(
            C, [](LazyCallGraph::Node &N) { return N.getName() == "f"; });
        ASSERT_NO_FATAL_FAILURE(
            updateCGAndAnalysisManagerForCGSCCPass(CG, C, FN, AM, UR, FAM));
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses3) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve([&](LazyCallGraph::SCC &C,
                                           CGSCCAnalysisManager &AM,
                                           LazyCallGraph &CG,
                                           CGSCCUpdateResult &UR) {
    if (C.getName() != "(f)")
      return;

    auto &FAM =
        AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
    Function *FnF = M->getFunction("f");
    Function *FnH2 = M->getFunction("h2");
    ASSERT_NE(FnF, nullptr);
    ASSERT_NE(FnH2, nullptr);

    // And insert a call to `h2`
    BasicBlock::iterator IP = FnF->getEntryBlock().begin();
    (void)CallInst::Create(FnH2, {}, "", IP);

    auto &FN = *llvm::find_if(
        C, [](LazyCallGraph::Node &N) { return N.getName() == "f"; });
    ASSERT_DEATH(
        updateCGAndAnalysisManagerForFunctionPass(CG, C, FN, AM, UR, FAM),
        "Any new calls should be modeled as");
  }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses4) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (C.getName() != "(f)")
          return;

        auto &FAM =
            AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
        Function *FnF = M->getFunction("f");
        Function *FnewF = Function::Create(FnF->getFunctionType(),
                                           FnF->getLinkage(), "newF", *M);
        BasicBlock *BB = BasicBlock::Create(FnewF->getContext(), "", FnewF);
        ReturnInst::Create(FnewF->getContext(), BB);

        // And insert a call to `newF`
        BasicBlock::iterator IP = FnF->getEntryBlock().begin();
        (void)CallInst::Create(FnewF, {}, "", IP);

        // Use the CallGraphUpdater to update the call graph for the new
        // function.
        CallGraphUpdater CGU;
        CGU.initialize(CG, C, AM, UR);
        CGU.registerOutlinedFunction(*FnF, *FnewF);

        auto &FN = *llvm::find_if(
            C, [](LazyCallGraph::Node &N) { return N.getName() == "f"; });

        ASSERT_NO_FATAL_FAILURE(
            updateCGAndAnalysisManagerForCGSCCPass(CG, C, FN, AM, UR, FAM));
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses5) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve([&](LazyCallGraph::SCC &C,
                                           CGSCCAnalysisManager &AM,
                                           LazyCallGraph &CG,
                                           CGSCCUpdateResult &UR) {
    if (C.getName() != "(f)")
      return;

    auto &FAM =
        AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
    Function *FnF = M->getFunction("f");
    Function *FnewF =
        Function::Create(FnF->getFunctionType(), FnF->getLinkage(), "newF", *M);
    BasicBlock *BB = BasicBlock::Create(FnewF->getContext(), "", FnewF);
    ReturnInst::Create(FnewF->getContext(), BB);

    // Use the CallGraphUpdater to update the call graph for the new
    // function.
    CallGraphUpdater CGU;
    CGU.initialize(CG, C, AM, UR);

    // And insert a call to `newF`
    BasicBlock::iterator IP = FnF->getEntryBlock().begin();
    (void)CallInst::Create(FnewF, {}, "", IP);

    auto &FN = *llvm::find_if(
        C, [](LazyCallGraph::Node &N) { return N.getName() == "f"; });

    ASSERT_DEATH(updateCGAndAnalysisManagerForCGSCCPass(CG, C, FN, AM, UR, FAM),
                 "should already have an associated node");
  }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses6) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (C.getName() != "(h3, h1, h2)")
          return;

        Function *FnX = M->getFunction("x");
        Function *FnH1 = M->getFunction("h1");
        Function *FnH2 = M->getFunction("h2");
        Function *FnH3 = M->getFunction("h3");
        ASSERT_NE(FnX, nullptr);
        ASSERT_NE(FnH1, nullptr);
        ASSERT_NE(FnH2, nullptr);
        ASSERT_NE(FnH3, nullptr);

        // And insert a call to `h1`, `h2`, and `h3`.
        BasicBlock::iterator IP = FnH2->getEntryBlock().begin();
        (void)CallInst::Create(FnH1, {}, "", IP);
        (void)CallInst::Create(FnH2, {}, "", IP);
        (void)CallInst::Create(FnH3, {}, "", IP);

        // Use the CallGraphUpdater to update the call graph for the new
        // function.
        CallGraphUpdater CGU;
        CGU.initialize(CG, C, AM, UR);
        ASSERT_NO_FATAL_FAILURE(CGU.reanalyzeFunction(*FnH2));
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses7) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (C.getName() != "(f)")
          return;

        Function *FnF = M->getFunction("f");
        Function *FnH2 = M->getFunction("h2");
        ASSERT_NE(FnF, nullptr);
        ASSERT_NE(FnH2, nullptr);

        // And insert a call to `h2`
        BasicBlock::iterator IP = FnF->getEntryBlock().begin();
        (void)CallInst::Create(FnH2, {}, "", IP);

        // Use the CallGraphUpdater to update the call graph for the new
        // function.
        CallGraphUpdater CGU;
        CGU.initialize(CG, C, AM, UR);
        ASSERT_NO_FATAL_FAILURE(CGU.reanalyzeFunction(*FnF));
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses8) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (C.getName() != "(f)")
          return;

        Function *FnF = M->getFunction("f");
        Function *FnewF = Function::Create(FnF->getFunctionType(),
                                           FnF->getLinkage(), "newF", *M);
        BasicBlock *BB = BasicBlock::Create(FnewF->getContext(), "", FnewF);
        auto *RI = ReturnInst::Create(FnewF->getContext(), BB);
        while (FnF->getEntryBlock().size() > 1)
          FnF->getEntryBlock().front().moveBefore(RI);
        ASSERT_NE(FnF, nullptr);

        // Create an unused constant that is referencing the old (=replaced)
        // function.
        ConstantExpr::getPtrToInt(FnF, Type::getInt64Ty(FnF->getContext()));

        // Use the CallGraphUpdater to update the call graph.
        CallGraphUpdater CGU;
        CGU.initialize(CG, C, AM, UR);
        ASSERT_NO_FATAL_FAILURE(CGU.replaceFunctionWith(*FnF, *FnewF));
        ASSERT_TRUE(FnF->isDeclaration());
        ASSERT_EQ(FnF->getNumUses(), 0U);
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses9) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (C.getName() != "(f)")
          return;

        Function *FnF = M->getFunction("f");

        // Use the CallGraphUpdater to update the call graph.
        CallGraphUpdater CGU;
        CGU.initialize(CG, C, AM, UR);
        ASSERT_NO_FATAL_FAILURE(CGU.removeFunction(*FnF));
        ASSERT_EQ(M->getFunctionList().size(), 6U);
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
  ASSERT_EQ(M->getFunctionList().size(), 5U);
}

TEST_F(CGSCCPassManagerTest, TestUpdateCGAndAnalysisManagerForPasses10) {
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (C.getName() != "(h3, h1, h2)")
          return;

        Function *FnX = M->getFunction("x");
        Function *FnH1 = M->getFunction("h1");
        Function *FnH2 = M->getFunction("h2");
        Function *FnH3 = M->getFunction("h3");
        ASSERT_NE(FnX, nullptr);
        ASSERT_NE(FnH1, nullptr);
        ASSERT_NE(FnH2, nullptr);
        ASSERT_NE(FnH3, nullptr);

        // And insert a call to `h1`, and `h3`.
        BasicBlock::iterator IP = FnH1->getEntryBlock().begin();
        (void)CallInst::Create(FnH1, {}, "", IP);
        (void)CallInst::Create(FnH3, {}, "", IP);

        // Remove the `h2` call.
        ASSERT_TRUE(isa<CallBase>(IP));
        ASSERT_EQ(cast<CallBase>(IP)->getCalledFunction(), FnH2);
        IP->eraseFromParent();

        // Use the CallGraphUpdater to update the call graph.
        CallGraphUpdater CGU;
        CGU.initialize(CG, C, AM, UR);
        ASSERT_NO_FATAL_FAILURE(CGU.reanalyzeFunction(*FnH1));
        ASSERT_NO_FATAL_FAILURE(CGU.removeFunction(*FnH2));
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
}

// Returns a vector containing the SCC's nodes. Useful for not iterating over an
// SCC while mutating it.
static SmallVector<LazyCallGraph::Node *> SCCNodes(LazyCallGraph::SCC &C) {
  SmallVector<LazyCallGraph::Node *> Nodes;
  for (auto &N : C)
    Nodes.push_back(&N);

  return Nodes;
}

// Start with call recursive f, create f -> g and ref recursive f.
TEST_F(CGSCCPassManagerTest, TestInsertionOfNewFunctions1) {
  std::unique_ptr<Module> M = parseIR("define void @f() {\n"
                                      "entry:\n"
                                      "  call void @f()\n"
                                      "  ret void\n"
                                      "}\n");

  bool Ran = false;

  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (Ran)
          return;

        auto &FAM =
            AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

        for (LazyCallGraph::Node *N : SCCNodes(C)) {
          Function &F = N->getFunction();
          if (F.getName() != "f")
            continue;

          // Create a new function 'g'.
          auto *G = Function::Create(F.getFunctionType(), F.getLinkage(),
                                     F.getAddressSpace(), "g", F.getParent());
          auto *GBB =
              BasicBlock::Create(F.getParent()->getContext(), "entry", G);
          (void)ReturnInst::Create(G->getContext(), GBB);
          // Instruct the LazyCallGraph to create a new node for 'g', as the
          // single node in a new SCC, into the call graph. As a result
          // the call graph is composed of a single RefSCC with two SCCs:
          // [(f), (g)].

          // "Demote" the 'f -> f' call edge to a ref edge.
          // 1. Erase the call edge from 'f' to 'f'.
          F.getEntryBlock().front().eraseFromParent();
          // 2. Insert a ref edge from 'f' to 'f'.
          (void)CastInst::CreatePointerCast(
              &F, PointerType::getUnqual(F.getContext()), "f.ref",
              F.getEntryBlock().begin());
          // 3. Insert a ref edge from 'f' to 'g'.
          (void)CastInst::CreatePointerCast(
              G, PointerType::getUnqual(F.getContext()), "g.ref",
              F.getEntryBlock().begin());

          CG.addSplitFunction(F, *G);

          ASSERT_FALSE(verifyModule(*F.getParent(), &errs()));

          ASSERT_NO_FATAL_FAILURE(
              updateCGAndAnalysisManagerForCGSCCPass(CG, C, *N, AM, UR, FAM))
              << "Updating the call graph with a demoted, self-referential "
                 "call edge 'f -> f', and a newly inserted ref edge 'f -> g', "
                 "caused a fatal failure";

          Ran = true;
        }
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
  ASSERT_TRUE(Ran);
}

// Start with f, end with f -> g1, f -> g2, and f -ref-> (h1 <-ref-> h2).
TEST_F(CGSCCPassManagerTest, TestInsertionOfNewFunctions2) {
  std::unique_ptr<Module> M = parseIR("define void @f() {\n"
                                      "entry:\n"
                                      "  ret void\n"
                                      "}\n");

  bool Ran = false;

  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve([&](LazyCallGraph::SCC &C,
                                           CGSCCAnalysisManager &AM,
                                           LazyCallGraph &CG,
                                           CGSCCUpdateResult &UR) {
    if (Ran)
      return;

    auto &FAM =
        AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

    for (LazyCallGraph::Node *N : SCCNodes(C)) {
      Function &F = N->getFunction();
      if (F.getName() != "f")
        continue;

      // Create g1 and g2.
      auto *G1 = Function::Create(F.getFunctionType(), F.getLinkage(),
                                  F.getAddressSpace(), "g1", F.getParent());
      auto *G2 = Function::Create(F.getFunctionType(), F.getLinkage(),
                                  F.getAddressSpace(), "g2", F.getParent());
      BasicBlock *G1BB =
          BasicBlock::Create(F.getParent()->getContext(), "entry", G1);
      BasicBlock *G2BB =
          BasicBlock::Create(F.getParent()->getContext(), "entry", G2);
      (void)ReturnInst::Create(G1->getContext(), G1BB);
      (void)ReturnInst::Create(G2->getContext(), G2BB);

      // Add 'f -> g1' call edge.
      (void)CallInst::Create(G1, {}, "", F.getEntryBlock().begin());
      // Add 'f -> g2' call edge.
      (void)CallInst::Create(G2, {}, "", F.getEntryBlock().begin());

      CG.addSplitFunction(F, *G1);
      CG.addSplitFunction(F, *G2);

      // Create mutually recursive functions (ref only) 'h1' and 'h2'.
      auto *H1 = Function::Create(F.getFunctionType(), F.getLinkage(),
                                  F.getAddressSpace(), "h1", F.getParent());
      auto *H2 = Function::Create(F.getFunctionType(), F.getLinkage(),
                                  F.getAddressSpace(), "h2", F.getParent());
      BasicBlock *H1BB =
          BasicBlock::Create(F.getParent()->getContext(), "entry", H1);
      BasicBlock *H2BB =
          BasicBlock::Create(F.getParent()->getContext(), "entry", H2);
      (void)CastInst::CreatePointerCast(
          H2, PointerType::getUnqual(F.getContext()), "h2.ref", H1BB);
      (void)ReturnInst::Create(H1->getContext(), H1BB);
      (void)CastInst::CreatePointerCast(
          H1, PointerType::getUnqual(F.getContext()), "h1.ref", H2BB);
      (void)ReturnInst::Create(H2->getContext(), H2BB);

      // Add 'f -> h1' ref edge.
      (void)CastInst::CreatePointerCast(H1,
                                        PointerType::getUnqual(F.getContext()),
                                        "h1.ref", F.getEntryBlock().begin());
      // Add 'f -> h2' ref edge.
      (void)CastInst::CreatePointerCast(H2,
                                        PointerType::getUnqual(F.getContext()),
                                        "h2.ref", F.getEntryBlock().begin());

      CG.addSplitRefRecursiveFunctions(F, SmallVector<Function *, 2>({H1, H2}));

      ASSERT_FALSE(verifyModule(*F.getParent(), &errs()));

      ASSERT_NO_FATAL_FAILURE(
          updateCGAndAnalysisManagerForCGSCCPass(CG, C, *N, AM, UR, FAM))
          << "Updating the call graph with mutually recursive g1 <-> g2, h1 "
             "<-> h2 caused a fatal failure";

      Ran = true;
    }
  }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
  ASSERT_TRUE(Ran);
}

TEST_F(CGSCCPassManagerTest, TestDeletionOfFunctionInNonTrivialRefSCC) {
  std::unique_ptr<Module> M = parseIR("define void @f1() {\n"
                                      "entry:\n"
                                      "  call void @f2()\n"
                                      "  ret void\n"
                                      "}\n"
                                      "define void @f2() {\n"
                                      "entry:\n"
                                      "  call void @f1()\n"
                                      "  ret void\n"
                                      "}\n");

  bool Ran = false;
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve(
      [&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM, LazyCallGraph &CG,
          CGSCCUpdateResult &UR) {
        if (Ran)
          return;

        LazyCallGraph::Node *N1 = nullptr;

        for (LazyCallGraph::Node *N : SCCNodes(C)) {
          Function &F = N->getFunction();
          if (F.getName() != "f1")
            continue;
          N1 = N;

          Function &F2 = *F.getParent()->getFunction("f2");

          // Remove f1 <-> f2 references
          F.getEntryBlock().front().eraseFromParent();
          F2.getEntryBlock().front().eraseFromParent();

          CallGraphUpdater CGU;
          CGU.initialize(CG, C, AM, UR);
          CGU.removeFunction(F2);
          CGU.reanalyzeFunction(F);

          Ran = true;
        }

        // Check that updateCGAndAnalysisManagerForCGSCCPass() after
        // CallGraphUpdater::removeFunction() succeeds.
        updateCGAndAnalysisManagerForCGSCCPass(CG, *CG.lookupSCC(*N1), *N1, AM,
                                               UR, FAM);
      }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);

  ASSERT_TRUE(Ran);
}

TEST_F(CGSCCPassManagerTest, TestInsertionOfNewNonTrivialCallEdge) {
  std::unique_ptr<Module> M = parseIR("define void @f1() {\n"
                                      "entry:\n"
                                      "  %a = bitcast void ()* @f4 to i8*\n"
                                      "  %b = bitcast void ()* @f2 to i8*\n"
                                      "  ret void\n"
                                      "}\n"
                                      "define void @f2() {\n"
                                      "entry:\n"
                                      "  %a = bitcast void ()* @f1 to i8*\n"
                                      "  %b = bitcast void ()* @f3 to i8*\n"
                                      "  ret void\n"
                                      "}\n"
                                      "define void @f3() {\n"
                                      "entry:\n"
                                      "  %a = bitcast void ()* @f2 to i8*\n"
                                      "  %b = bitcast void ()* @f4 to i8*\n"
                                      "  ret void\n"
                                      "}\n"
                                      "define void @f4() {\n"
                                      "entry:\n"
                                      "  %a = bitcast void ()* @f3 to i8*\n"
                                      "  %b = bitcast void ()* @f1 to i8*\n"
                                      "  ret void\n"
                                      "}\n");

  bool Ran = false;
  CGSCCPassManager CGPM;
  CGPM.addPass(LambdaSCCPassNoPreserve([&](LazyCallGraph::SCC &C,
                                           CGSCCAnalysisManager &AM,
                                           LazyCallGraph &CG,
                                           CGSCCUpdateResult &UR) {
    if (Ran)
      return;

    auto &FAM =
        AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

    for (LazyCallGraph::Node *N : SCCNodes(C)) {
      Function &F = N->getFunction();
      if (F.getName() != "f1")
        continue;

      Function *F3 = F.getParent()->getFunction("f3");
      ASSERT_TRUE(F3 != nullptr);

      // Create call from f1 to f3.
      (void)CallInst::Create(F3, {}, "",
                             F.getEntryBlock().getTerminator()->getIterator());

      ASSERT_NO_FATAL_FAILURE(
          updateCGAndAnalysisManagerForCGSCCPass(CG, C, *N, AM, UR, FAM))
          << "Updating the call graph with mutually recursive g1 <-> g2, h1 "
             "<-> h2 caused a fatal failure";

      Ran = true;
    }
  }));

  ModulePassManager MPM;
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);

  ASSERT_TRUE(Ran);
}

TEST_F(CGSCCPassManagerTest, TestFunctionPassesAreQueriedForInvalidation) {
  std::unique_ptr<Module> M = parseIR("define void @f() { ret void }");
  CGSCCPassManager CGPM;
  bool SCCCalled = false;
  FunctionPassManager FPM;
  int ImmRuns = 0;
  FAM.registerPass([&] { return TestImmutableFunctionAnalysis(ImmRuns); });
  FPM.addPass(RequireAnalysisPass<TestImmutableFunctionAnalysis, Function>());
  CGPM.addPass(
      LambdaSCCPass([&](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR) {
        SCCCalled = true;
        return PreservedAnalyses::none();
      }));
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(
      RequireAnalysisPass<TestImmutableFunctionAnalysis, Function>()));
  ModulePassManager MPM;

  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  MPM.run(*M, MAM);
  ASSERT_EQ(ImmRuns, 1);
  ASSERT_TRUE(SCCCalled);
}

#endif
} // namespace
