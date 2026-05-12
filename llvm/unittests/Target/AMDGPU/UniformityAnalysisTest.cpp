//===- UniformityAnalysisTest.cpp - Conservative divergence query test ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that values created after Uniformity analysis are conservatively
// reported as divergent, since they are not present in UniformValues.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/UniformityAnalysis.h"
#include "AMDGPUUnitTests.h"
#include "llvm/ADT/GenericUniformityImpl.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

static UniformityInfo computeUniformity(const TargetTransformInfo *TTI,
                                        Function *F) {
  DominatorTree DT(*F);
  CycleInfo CI;
  CI.compute(*F);
  UniformityInfo UI(DT, CI, TTI);
  if (TTI->hasBranchDivergence(F))
    UI.compute();
  return UI;
}

TEST_F(AMDGPUTestBase, NewValueIsConservativelyDivergent) {

  StringRef ModuleString = R"(
  target triple = "amdgcn-unknown-amdhsa"
  define amdgpu_kernel void @test(i32 inreg %a, i32 inreg %b) {
    %add = add i32 %a, %b
    ret void
  }
  )";
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, Context);
  ASSERT_TRUE(M) << Err.getMessage();

  Function *F = M->getFunction("test");
  ASSERT_TRUE(F);

  auto TM =
      createAMDGPUTargetMachine("amdgcn-amd-", "gfx1010", "+wavefrontsize32");
  ASSERT_TRUE(TM);
  TargetTransformInfo TTI = TM->getTargetTransformInfo(*F);

  UniformityInfo UI = computeUniformity(&TTI, F);

  // Existing values from the analysis are uniform (kernel args are inreg).
  Instruction *AddInst = &*F->getEntryBlock().begin();
  ASSERT_TRUE(isa<BinaryOperator>(AddInst));
  EXPECT_FALSE(UI.isDivergentAtDef(AddInst)) << "%add should be uniform";
  EXPECT_FALSE(UI.isDivergentAtDef(F->getArg(0))) << "%a should be uniform";
  EXPECT_FALSE(UI.isDivergentAtDef(F->getArg(1))) << "%b should be uniform";

  // Create a new instruction after analysis. It was not present during
  // analysis, so it is not in UniformValues and must be conservatively
  // reported as divergent.
  IRBuilder<> Builder(AddInst->getNextNode());
  Value *NewInst = Builder.CreateMul(F->getArg(0), F->getArg(1), "new_mul");

  EXPECT_TRUE(UI.isDivergentAtDef(NewInst))
      << "New instruction created after analysis must be reported divergent";
}
