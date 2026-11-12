//===- SandboxVectorizerTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizer.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxVectorizerTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxVectorizerTest", errs());
  }
};

// Check that we can run the pass on the same function more than once without
// issues. This basically checks that Sandbox IR Context gets cleared after we
// run the function pass.
TEST_F(SandboxVectorizerTest, ContextCleared) {
  parseIR(C, R"IR(
define void @foo() {
  ret void
}
)IR");
  auto &LLVMF = *M->getFunction("foo");
  SandboxVectorizerPass SVecPass;
  FunctionAnalysisManager AM;
  AM.registerPass([] { return TargetIRAnalysis(); });
  AM.registerPass([] { return AAManager(); });
  AM.registerPass([] { return ScalarEvolutionAnalysis(); });
  AM.registerPass([] { return PassInstrumentationAnalysis(); });
  AM.registerPass([] { return TargetLibraryAnalysis(); });
  AM.registerPass([] { return AssumptionAnalysis(); });
  AM.registerPass([] { return DominatorTreeAnalysis(); });
  AM.registerPass([] { return LoopAnalysis(); });
  SVecPass.run(LLVMF, AM);
  // This shouldn't crash.
  SVecPass.run(LLVMF, AM);
}
