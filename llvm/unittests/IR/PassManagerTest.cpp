//===- llvm/unittest/IR/PassManager.cpp - PassManager tests ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
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

  /// \brief Run the analysis pass over the function and return a result.
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
  static char PassID;

  int &Runs;
};

char TestFunctionAnalysis::PassID;

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
  static char PassID;

  int &Runs;
};

char TestModuleAnalysis::PassID;

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

struct TestMinPreservingModulePass
    : PassInfoMixin<TestMinPreservingModulePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    PreservedAnalyses PA;

    // Force running an analysis.
    (void)AM.getResult<TestModuleAnalysis>(M);

    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    return PA;
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

TEST_F(PassManagerTest, BasicPreservedAnalyses) {
  PreservedAnalyses PA1 = PreservedAnalyses();
  EXPECT_FALSE(PA1.preserved<TestFunctionAnalysis>());
  EXPECT_FALSE(PA1.preserved<TestModuleAnalysis>());
  PreservedAnalyses PA2 = PreservedAnalyses::none();
  EXPECT_FALSE(PA2.preserved<TestFunctionAnalysis>());
  EXPECT_FALSE(PA2.preserved<TestModuleAnalysis>());
  PreservedAnalyses PA3 = PreservedAnalyses::all();
  EXPECT_TRUE(PA3.preserved<TestFunctionAnalysis>());
  EXPECT_TRUE(PA3.preserved<TestModuleAnalysis>());
  PreservedAnalyses PA4 = PA1;
  EXPECT_FALSE(PA4.preserved<TestFunctionAnalysis>());
  EXPECT_FALSE(PA4.preserved<TestModuleAnalysis>());
  PA4 = PA3;
  EXPECT_TRUE(PA4.preserved<TestFunctionAnalysis>());
  EXPECT_TRUE(PA4.preserved<TestModuleAnalysis>());
  PA4 = std::move(PA2);
  EXPECT_FALSE(PA4.preserved<TestFunctionAnalysis>());
  EXPECT_FALSE(PA4.preserved<TestModuleAnalysis>());
  PA4.preserve<TestFunctionAnalysis>();
  EXPECT_TRUE(PA4.preserved<TestFunctionAnalysis>());
  EXPECT_FALSE(PA4.preserved<TestModuleAnalysis>());
  PA1.preserve<TestModuleAnalysis>();
  EXPECT_FALSE(PA1.preserved<TestFunctionAnalysis>());
  EXPECT_TRUE(PA1.preserved<TestModuleAnalysis>());
  PA1.preserve<TestFunctionAnalysis>();
  EXPECT_TRUE(PA1.preserved<TestFunctionAnalysis>());
  EXPECT_TRUE(PA1.preserved<TestModuleAnalysis>());
  PA1.intersect(PA4);
  EXPECT_TRUE(PA1.preserved<TestFunctionAnalysis>());
  EXPECT_FALSE(PA1.preserved<TestModuleAnalysis>());
}

TEST_F(PassManagerTest, Basic) {
  FunctionAnalysisManager FAM;
  int FunctionAnalysisRuns = 0;
  FAM.registerPass([&] { return TestFunctionAnalysis(FunctionAnalysisRuns); });

  ModuleAnalysisManager MAM;
  int ModuleAnalysisRuns = 0;
  MAM.registerPass([&] { return TestModuleAnalysis(ModuleAnalysisRuns); });
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

  ModulePassManager MPM;

  // Count the runs over a Function.
  int FunctionPassRunCount1 = 0;
  int AnalyzedInstrCount1 = 0;
  int AnalyzedFunctionCount1 = 0;
  {
    // Pointless scoped copy to test move assignment.
    ModulePassManager NestedMPM;
    FunctionPassManager FPM;
    {
      // Pointless scope to test move assignment.
      FunctionPassManager NestedFPM;
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
    FunctionPassManager FPM;
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
    FunctionPassManager FPM;
    FPM.addPass(TestFunctionPass(FunctionPassRunCount3, AnalyzedInstrCount3,
                                 AnalyzedFunctionCount3));
    FPM.addPass(TestInvalidationFunctionPass("f"));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  // A fourth function pass manager but with a minimal intervening passes.
  MPM.addPass(TestMinPreservingModulePass());
  int FunctionPassRunCount4 = 0;
  int AnalyzedInstrCount4 = 0;
  int AnalyzedFunctionCount4 = 0;
  {
    FunctionPassManager FPM;
    FPM.addPass(TestFunctionPass(FunctionPassRunCount4, AnalyzedInstrCount4,
                                 AnalyzedFunctionCount4));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  // A fifth function pass manager but which uses only cached results.
  int FunctionPassRunCount5 = 0;
  int AnalyzedInstrCount5 = 0;
  int AnalyzedFunctionCount5 = 0;
  {
    FunctionPassManager FPM;
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
  EXPECT_EQ(0, AnalyzedFunctionCount4);
  EXPECT_EQ(3, FunctionPassRunCount5);
  EXPECT_EQ(2, AnalyzedInstrCount5); // Only 'g' and 'h' were cached.
  EXPECT_EQ(0, AnalyzedFunctionCount5);

  // Validate the analysis counters:
  //   first run over 3 functions, then module pass invalidates
  //   second run over 3 functions, nothing invalidates
  //   third run over 0 functions, but 1 function invalidated
  //   fourth run over 1 function
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
  static char PassID;
};

char CustomizedAnalysis::PassID;

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

  CustomizedPassManager PM;

  // Add an instance of the customized pass that just accumulates the input
  // after it is round-tripped through the analysis.
  int Result = 0;
  PM.addPass(CustomizedPass::CustomizedPass(
      [](CustomizedAnalysis::Result &R, int &O) { O += R.I; }));

  // Run this over every function with the input of 42.
  for (Function &F : *M)
    PM.run(F, AM, 42, Result);

  // And ensure that we accumulated the correct result.
  EXPECT_EQ(42 * (int)M->size(), Result);
}
}
