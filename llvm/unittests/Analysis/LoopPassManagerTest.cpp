//===- llvm/unittest/Analysis/LoopPassManagerTest.cpp - LPM tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Analysis/LoopPassManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

namespace {

class TestLoopAnalysis {
  /// \brief Private static data to provide unique ID.
  static char PassID;

  int &Runs;

public:
  struct Result {
    Result(int Count) : BlockCount(Count) {}
    int BlockCount;
  };

  /// \brief Returns an opaque, unique ID for this pass type.
  static void *ID() { return (void *)&PassID; }

  /// \brief Returns the name of the analysis.
  static StringRef name() { return "TestLoopAnalysis"; }

  TestLoopAnalysis(int &Runs) : Runs(Runs) {}

  /// \brief Run the analysis pass over the loop and return a result.
  Result run(Loop &L, AnalysisManager<Loop> &AM) {
    ++Runs;
    int Count = 0;

    for (auto I = L.block_begin(), E = L.block_end(); I != E; ++I)
      ++Count;
    return Result(Count);
  }
};

char TestLoopAnalysis::PassID;

class TestLoopPass {
  std::vector<StringRef> &VisitedLoops;
  int &AnalyzedBlockCount;
  bool OnlyUseCachedResults;

public:
  TestLoopPass(std::vector<StringRef> &VisitedLoops, int &AnalyzedBlockCount,
               bool OnlyUseCachedResults = false)
      : VisitedLoops(VisitedLoops), AnalyzedBlockCount(AnalyzedBlockCount),
        OnlyUseCachedResults(OnlyUseCachedResults) {}

  PreservedAnalyses run(Loop &L, AnalysisManager<Loop> &AM) {
    VisitedLoops.push_back(L.getName());

    if (OnlyUseCachedResults) {
      // Hack to force the use of the cached interface.
      if (auto *AR = AM.getCachedResult<TestLoopAnalysis>(L))
        AnalyzedBlockCount += AR->BlockCount;
    } else {
      // Typical path just runs the analysis as needed.
      auto &AR = AM.getResult<TestLoopAnalysis>(L);
      AnalyzedBlockCount += AR.BlockCount;
    }

    return PreservedAnalyses::all();
  }

  static StringRef name() { return "TestLoopPass"; }
};

// A test loop pass that invalidates the analysis for loops with the given name.
class TestLoopInvalidatingPass {
  StringRef Name;

public:
  TestLoopInvalidatingPass(StringRef LoopName) : Name(LoopName) {}

  PreservedAnalyses run(Loop &L, AnalysisManager<Loop> &AM) {
    return L.getName() == Name ? getLoopPassPreservedAnalyses()
                               : PreservedAnalyses::all();
  }

  static StringRef name() { return "TestLoopInvalidatingPass"; }
};

std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, C);
}

class LoopPassManagerTest : public ::testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;

public:
  LoopPassManagerTest()
      : M(parseIR(Context, "define void @f() {\n"
                           "entry:\n"
                           "  br label %loop.0\n"
                           "loop.0:\n"
                           "  br i1 undef, label %loop.0.0, label %end\n"
                           "loop.0.0:\n"
                           "  br i1 undef, label %loop.0.0, label %loop.0.1\n"
                           "loop.0.1:\n"
                           "  br i1 undef, label %loop.0.1, label %loop.0\n"
                           "end:\n"
                           "  ret void\n"
                           "}\n"
                           "\n"
                           "define void @g() {\n"
                           "entry:\n"
                           "  br label %loop.g.0\n"
                           "loop.g.0:\n"
                           "  br i1 undef, label %loop.g.0, label %end\n"
                           "end:\n"
                           "  ret void\n"
                           "}\n")) {}
};

#define EXPECT_N_ELEMENTS_EQ(N, EXPECTED, ACTUAL)                              \
  do {                                                                         \
    EXPECT_EQ(N##UL, ACTUAL.size());                                           \
    for (int I = 0; I < N; ++I)                                                \
      EXPECT_TRUE(EXPECTED[I] == ACTUAL[I]) << "Element " << I << " is "       \
                                            << ACTUAL[I] << ". Expected "      \
                                            << EXPECTED[I] << ".";             \
  } while (0)

TEST_F(LoopPassManagerTest, Basic) {
  LoopAnalysisManager LAM(true);
  int LoopAnalysisRuns = 0;
  LAM.registerPass([&] { return TestLoopAnalysis(LoopAnalysisRuns); });

  FunctionAnalysisManager FAM(true);
  // We need DominatorTreeAnalysis for LoopAnalysis.
  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
  LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(FAM); });

  ModuleAnalysisManager MAM(true);
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

  ModulePassManager MPM(true);
  FunctionPassManager FPM(true);

  // Visit all of the loops.
  std::vector<StringRef> VisitedLoops1;
  int AnalyzedBlockCount1 = 0;
  {
    LoopPassManager LPM;
    LPM.addPass(TestLoopPass(VisitedLoops1, AnalyzedBlockCount1));

    FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
  }

  // Only use cached analyses.
  std::vector<StringRef> VisitedLoops2;
  int AnalyzedBlockCount2 = 0;
  {
    LoopPassManager LPM;
    LPM.addPass(TestLoopInvalidatingPass("loop.g.0"));
    LPM.addPass(TestLoopPass(VisitedLoops2, AnalyzedBlockCount2,
                             /*OnlyUseCachedResults=*/true));

    FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
  }

  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.run(*M, MAM);

  StringRef ExpectedLoops[] = {"loop.0.0", "loop.0.1", "loop.0", "loop.g.0"};

  // Validate the counters and order of loops visited.
  // loop.0 has 3 blocks whereas loop.0.0, loop.0.1, and loop.g.0 each have 1.
  EXPECT_N_ELEMENTS_EQ(4, ExpectedLoops, VisitedLoops1);
  EXPECT_EQ(6, AnalyzedBlockCount1);

  EXPECT_N_ELEMENTS_EQ(4, ExpectedLoops, VisitedLoops2);
  // The block from loop.g.0 won't be counted, since it wasn't cached.
  EXPECT_EQ(5, AnalyzedBlockCount2);

  // The first LPM runs the loop analysis for all four loops, the second uses
  // cached results for everything.
  EXPECT_EQ(4, LoopAnalysisRuns);
}
}
