//===- InlineCostTest.cpp - test for InlineCost ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InlineModelFeatureMaps.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;

CallBase *getCallInFunction(Function *F) {
  for (auto &I : instructions(F)) {
    if (auto *CB = dyn_cast<llvm::CallBase>(&I))
      return CB;
  }
  return nullptr;
}

std::optional<InlineCostFeatures> getInliningCostFeaturesForCall(CallBase &CB) {
  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return TargetIRAnalysis(); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  FAM.registerPass([&] { return AssumptionAnalysis(); });
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });

  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });

  ModulePassManager MPM;
  MPM.run(*CB.getModule(), MAM);

  auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto &TIR = FAM.getResult<TargetIRAnalysis>(*CB.getFunction());

  return getInliningCostFeatures(CB, TIR, GetAssumptionCache);
}

// Tests that we can retrieve the CostFeatures without an error
TEST(InlineCostTest, CostFeatures) {
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
  CallBase *CB = getCallInFunction(G);
  ASSERT_TRUE(CB);

  const auto Features = getInliningCostFeaturesForCall(*CB);

  // Check that the optional is not empty
  ASSERT_TRUE(Features);
}

// Tests the calculated SROA cost
TEST(InlineCostTest, SROACost) {
  using namespace llvm;

  const auto *const IR = R"IR(
define void @f_savings(ptr %var) {
  %load = load i32, ptr %var
  %inc = add i32 %load, 1
  store i32 %inc, ptr %var
  ret void
}

define void @g_savings(i32) {
  %var = alloca i32
  call void @f_savings(ptr %var)
  ret void
}

define void @f_losses(ptr %var) {
  %load = load i32, ptr %var
  %inc = add i32 %load, 1
  store i32 %inc, ptr %var
  call void @prevent_sroa(ptr %var)
  ret void
}

define void @g_losses(i32) {
  %var = alloca i32
  call void @f_losses(ptr %var)
  ret void
}

declare void @prevent_sroa(ptr)
)IR";

  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, C);
  ASSERT_TRUE(M);

  const int DefaultInstCost = 5;
  const int DefaultAllocaCost = 0;

  const char *GName[] = {"g_savings", "g_losses", nullptr};
  const int Savings[] = {2 * DefaultInstCost + DefaultAllocaCost, 0};
  const int Losses[] = {0, 2 * DefaultInstCost + DefaultAllocaCost};

  for (unsigned i = 0; GName[i]; ++i) {
    auto *G = M->getFunction(GName[i]);
    ASSERT_TRUE(G);

    // find the call to f in g
    CallBase *CB = getCallInFunction(G);
    ASSERT_TRUE(CB);

    const auto Features = getInliningCostFeaturesForCall(*CB);
    ASSERT_TRUE(Features);

    // Check the predicted SROA cost
    auto GetFeature = [&](InlineCostFeatureIndex I) {
      return (*Features)[static_cast<size_t>(I)];
    };
    ASSERT_EQ(GetFeature(InlineCostFeatureIndex::sroa_savings), Savings[i]);
    ASSERT_EQ(GetFeature(InlineCostFeatureIndex::sroa_losses), Losses[i]);
  }
}

} // namespace
