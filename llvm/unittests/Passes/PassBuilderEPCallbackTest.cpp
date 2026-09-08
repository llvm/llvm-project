//===- unittests/Passes/PassBuilderEPCallbackTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, C);
}

class PassBuilderEPCallbackTest : public testing::Test {
protected:
  LLVMContext C;
  std::unique_ptr<Module> M;
  PassBuilder PB;

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  SmallVector<StringRef, 2> Order;

  PassBuilderEPCallbackTest() {
    M = parseIR(C, "define void @f() { ret void }");

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    PB.registerThinLinkTimeOptimizationEarlyEPCallback(
        [this](ModulePassManager &, OptimizationLevel) {
          Order.push_back("Early");
        });
    PB.registerThinLinkTimeOptimizationLastEPCallback(
        [this](ModulePassManager &, OptimizationLevel) {
          Order.push_back("Last");
        });
  }
};

TEST_F(PassBuilderEPCallbackTest, ThinLTO_O0) {
  ModulePassManager MPM =
      PB.buildThinLTODefaultPipeline(OptimizationLevel::O0,
                                     /*ImportSummary=*/nullptr);
  MPM.run(*M, MAM);

  EXPECT_EQ(Order, (SmallVector<StringRef, 2>{"Early", "Last"}));
}

TEST_F(PassBuilderEPCallbackTest, ThinLTO_O2) {
  ModulePassManager MPM =
      PB.buildThinLTODefaultPipeline(OptimizationLevel::O2,
                                     /*ImportSummary=*/nullptr);
  MPM.run(*M, MAM);

  EXPECT_EQ(Order, (SmallVector<StringRef, 2>{"Early", "Last"}));
}

} // namespace
