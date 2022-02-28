//===- InlineCostTest.cpp - test for InlineCost ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace {

// Tests that we can retrieve the CostFeatures without an error
TEST(InlineCostTest, CostFeatures) {
  using namespace llvm;

  const auto *const IR = R"IR(
define i32 @f(i32) {
  ret i32 4
}

define i32 @g(i32) {
  %2 = call i32 @f(i32 0)
  ret i32 %2
}
)IR";

  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, C);
  ASSERT_TRUE(M);

  auto *G = M->getFunction("g");
  ASSERT_TRUE(G);

  // find the call to f in g
  CallBase *CB = nullptr;
  for (auto &BB : *G) {
    for (auto &I : BB) {
      if ((CB = dyn_cast<CallBase>(&I)))
        break;
    }
  }
  ASSERT_TRUE(CB);

  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return TargetIRAnalysis(); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  FAM.registerPass([&] { return AssumptionAnalysis(); });
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });

  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });

  ModulePassManager MPM;
  MPM.run(*M, MAM);

  auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto &TIR = FAM.getResult<TargetIRAnalysis>(*G);

  const auto Features =
      llvm::getInliningCostFeatures(*CB, TIR, GetAssumptionCache);

  // Check that the optional is not empty
  ASSERT_TRUE(Features);
}

} // namespace
